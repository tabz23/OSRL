import os
import sys
import uuid
import types
from dataclasses import asdict
from typing import Any

sys.path.append("/Users/i.k.tabbara/Documents/python directory/OSRL")

import bullet_safety_gym  # noqa
import dsrl
import safety_gymnasium as gym
import numpy as np
import pyrallis
import torch
import wandb
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from torch.utils.data import DataLoader
from tqdm.auto import trange
from examples.configs.bc_configs import BC_DEFAULT_CONFIG, BCTrainConfig

from osrl.algorithms import BC, BCTrainer
from osrl.common.dataset import process_bc_dataset
from osrl.common.exp_util import auto_name, seed_all

from network_ihab import AffineDynamics, CBF
from dataset_ihab import TransitionDataset
import torch.nn.functional as F


class CBF_dyn_trainer:
    def __init__(self, cbf, train_dataset, val_dataset, dt=0.1, lr=1e-4, device="cpu", train_steps=10000, 
                 eval_every_n_steps=100, eval_steps=100, without_dynamic=True, args=None,
                 eps_safe=0.1, eps_unsafe=0.1, eps_grad=0.1, w_safe=1, w_unsafe=1, w_grad=0.1,
                 dynamics=None, dynamics_checkpoint=None, train_dynamics=False, dynamics_lr=1e-4):
        
        self.cbf = cbf.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.device = device
        self.args = args  
        self.train_steps = train_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_steps = eval_steps
        self.without_dynamic = without_dynamic
        self.dt = dt
        self.train_dynamics = train_dynamics
        
        # Initialize or load dynamics model
        if not without_dynamic:
            if dynamics is None:
                # Create a new dynamics model if none is provided
                self.dynamics = AffineDynamics(
                    num_action=args.num_action if hasattr(args, 'num_action') else cbf.num_action,
                    state_dim=args.state_dim if hasattr(args, 'state_dim') else cbf.state_dim,
                    hidden_dim=args.dynamics_hidden_dim if hasattr(args, 'dynamics_hidden_dim') else args.hidden_dim,
                    num_layers=args.dynamics_num_layers if hasattr(args, 'dynamics_num_layers') else args.num_layers,
                    dt=dt
                ).to(device)
            else:
                self.dynamics = dynamics.to(device)
            
            # Load dynamics from checkpoint if provided
            if dynamics_checkpoint is not None:
                print(f"Loading dynamics model from checkpoint: {dynamics_checkpoint}")
                self.dynamics.load_state_dict(torch.load(dynamics_checkpoint, map_location=device))
                if not self.train_dynamics:
                    print("Dynamics model loaded and frozen (not training)")
                    for param in self.dynamics.parameters():
                        param.requires_grad = False
                else:
                    print("Dynamics model loaded and will be fine-tuned")
            
            self.dynamics_lr = dynamics_lr
        
        self.eps_safe = eps_safe
        self.eps_unsafe = eps_unsafe
        self.eps_grad = eps_grad
        self.w_safe = w_safe
        self.w_unsafe = w_unsafe
        self.w_grad = w_grad

        self.setup_optimizer()
        
        # Initialize WandB with expanded config
        config = {
            "learning_rate": lr,
            "train_steps": train_steps,
            "eps_safe": eps_safe,
            "eps_unsafe": eps_unsafe,
            "eps_grad": eps_grad,
            "w_safe": w_safe,
            "w_unsafe": w_unsafe,
            "w_grad": w_grad,
            "dt": dt,
            "hidden_dim": args.hidden_dim if hasattr(args, 'hidden_dim') else None,
            "num_layers": args.num_layers if hasattr(args, 'num_layers') else None,
            "batch_size": args.batch_size if hasattr(args, 'batch_size') else None,
            "lr": self.lr,
            "seed": args.seed if hasattr(args, 'seed') else None,
            "without_dynamic": without_dynamic,
            "train_dynamics": train_dynamics,
        }
        
        if not without_dynamic:
            config.update({
                "dynamics_checkpoint": dynamics_checkpoint,
                "dynamics_lr": dynamics_lr,
                "dynamics_hidden_dim": self.dynamics.hidden_dim,
                "dynamics_num_layers": self.dynamics.num_layers,
            })
            
        wandb.init(project="cbf_dynamics_training", config=config)
            
    def compute_loss(self, observations, next_observations, actions, costs, training_bool=None):
        # Determine safe and unsafe states
        safe_mask = (costs <= 0).reshape(-1, 1)
        unsafe_mask = (costs > 0).reshape(-1, 1)

        # Calculate CBF values for current states
        B = self.cbf(observations).reshape(-1, 1)

        # Safe loss computation
        loss_safe_vector = self.w_safe * F.relu(self.eps_safe - B) * safe_mask
        num_safe_elements = safe_mask.sum()
        loss_safe = loss_safe_vector.sum() / (num_safe_elements + 1e-8)

        # Unsafe loss computation
        loss_unsafe_vector = self.w_unsafe * F.relu(self.eps_unsafe + B) * unsafe_mask
        num_unsafe_elements = unsafe_mask.sum()
        loss_unsafe = loss_unsafe_vector.sum() / (num_unsafe_elements + 1e-8)
        
        # Calculate gradient based on whether to use dynamics or not
        if self.without_dynamic:
            # Use finite difference approximation for gradient
            B_next = self.cbf(next_observations).reshape(-1, 1)
            B_dot = (B_next - B) / self.dt
            gradient = B_dot + B
        else:
            # Use dynamics model to calculate gradient
            # Enable grad for observations temporarily if in training mode
            observations.requires_grad_(True)

            # Compute CBF value with gradient tracking
            B = self.cbf(observations).reshape(-1, 1)
            
            # Get state derivative from dynamics model
            x_dot = self.dynamics.forward_x_dot(observations, actions)
            
            # Compute gradient of CBF with respect to state
            grad_b = torch.autograd.grad(B.sum(), observations, create_graph=True)[0]
            
            # Compute Lie derivative (inner product of grad_B and x_dot)
            b_dot = torch.sum(grad_b * x_dot, dim=1, keepdim=True)
            gradient = b_dot + B
        
        # Gradient loss computation
        loss_grad_vector = self.w_grad * F.relu(self.eps_grad - gradient) * safe_mask
        num_grad_elements = safe_mask.sum()
        loss_grad = loss_grad_vector.sum() / (num_grad_elements + 1e-8)

        # Dynamics loss computation if we're also training dynamics
        dynamics_loss = 0.0
        if not self.without_dynamic and self.train_dynamics and training_bool:
            predicted_next_observations = self.dynamics.forward_next_state(observations, actions)
            loss_fn = torch.nn.MSELoss()
            dynamics_loss = loss_fn(predicted_next_observations, next_observations)

        # Total loss
        cbf_loss = loss_safe + loss_unsafe + loss_grad
        total_loss = cbf_loss
        
        # Add dynamics loss if applicable
        if not self.without_dynamic and self.train_dynamics:
            total_loss = total_loss + dynamics_loss

        # Backward pass during training
        if training_bool:
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()
        
        observations.requires_grad_(False)

        # Computing averages for safe and unsafe regions
        avg_safe_B = (B * safe_mask).sum() / (safe_mask.sum() + 1e-8)
        avg_unsafe_B = (B * unsafe_mask).sum() / (unsafe_mask.sum() + 1e-8)

        # Compute accuracy metrics
        safe_acc = ((B >= 0) * safe_mask).sum() / (num_safe_elements + 1e-8)
        unsafe_acc = ((B < 0) * unsafe_mask).sum() / (num_unsafe_elements + 1e-8)
        
        if not self.without_dynamic and self.train_dynamics:
            return loss_safe.item(), loss_unsafe.item(), loss_grad.item(), dynamics_loss.item(), avg_safe_B.item(), avg_unsafe_B.item(), safe_acc.item(), unsafe_acc.item()
        else:
            return loss_safe.item(), loss_unsafe.item(), loss_grad.item(), avg_safe_B.item(), avg_unsafe_B.item(), safe_acc.item(), unsafe_acc.item()

    def validate(self):
        self.cbf.eval()
        if not self.without_dynamic:
            self.dynamics.eval()
            
        valloader_iter = iter(self.val_dataset)
        total_loss_safe = 0.0
        total_loss_unsafe = 0.0
        total_loss_grad = 0.0
        total_dynamics_loss = 0.0
        total_avg_safe_B = 0.0
        total_avg_unsafe_B = 0.0
        total_safe_acc = 0.0
        total_unsafe_acc = 0.0

        print("\nStarting validation...")
        with torch.no_grad():
            for step in range(self.eval_steps):
                batch = next(valloader_iter)
                observations, next_observations, actions, _, costs, done = [b.to(torch.float32).to(self.device) for b in batch]
                
                if not self.without_dynamic and self.train_dynamics:
                    loss_safe, loss_unsafe, loss_grad, dynamics_loss, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc = self.compute_loss(
                        observations, next_observations, actions, costs, training_bool=False
                    )
                    total_dynamics_loss += dynamics_loss
                else:
                    loss_safe, loss_unsafe, loss_grad, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc = self.compute_loss(
                        observations, next_observations, actions, costs, training_bool=False
                    )
                
                total_loss_safe += loss_safe
                total_loss_unsafe += loss_unsafe
                total_loss_grad += loss_grad
                total_avg_safe_B += avg_safe_B
                total_avg_unsafe_B += avg_unsafe_B
                total_safe_acc += safe_acc
                total_unsafe_acc += unsafe_acc

        # Calculate averages
        avg_loss_safe = total_loss_safe / self.eval_steps
        avg_loss_unsafe = total_loss_unsafe / self.eval_steps
        avg_loss_grad = total_loss_grad / self.eval_steps
        avg_dynamics_loss = total_dynamics_loss / self.eval_steps if not self.without_dynamic and self.train_dynamics else 0.0
        avg_safe_B = total_avg_safe_B / self.eval_steps
        avg_unsafe_B = total_avg_unsafe_B / self.eval_steps
        
        total_cbf_loss = avg_loss_safe + avg_loss_unsafe + avg_loss_grad
        total_loss = total_cbf_loss + avg_dynamics_loss if not self.without_dynamic and self.train_dynamics else total_cbf_loss
        
        avg_safe_acc = total_safe_acc / self.eval_steps
        avg_unsafe_acc = total_unsafe_acc / self.eval_steps

        # Log validation metrics
        log_dict = {
            "val_loss_safe": avg_loss_safe,
            "val_loss_unsafe": avg_loss_unsafe,
            "val_loss_grad": avg_loss_grad,
            "val_cbf_loss": total_cbf_loss,
            "val_avg_safe_B": avg_safe_B,
            "val_avg_unsafe_B": avg_unsafe_B,
            "val_safe_acc": avg_safe_acc,
            "val_unsafe_acc": avg_unsafe_acc,
            "val_total_loss": total_loss
        }
        
        if not self.without_dynamic and self.train_dynamics:
            log_dict["val_dynamics_loss"] = avg_dynamics_loss
            
        wandb.log(log_dict)

        # Print validation results
        print("\nValidation Results:")
        print(f"Average Safe Loss: {avg_loss_safe:.4f}")
        print(f"Average Unsafe Loss: {avg_loss_unsafe:.4f}")
        print(f"Average Gradient Loss: {avg_loss_grad:.4f}")
        if not self.without_dynamic and self.train_dynamics:
            print(f"Average Dynamics Loss: {avg_dynamics_loss:.4f}")
        print(f"Average Safe B Value: {avg_safe_B:.4f}")
        print(f"Average Unsafe B Value: {avg_unsafe_B:.4f}")
        print(f"Safe Accuracy: {avg_safe_acc:.4f}")
        print(f"Unsafe Accuracy: {avg_unsafe_acc:.4f}")       
        print(f"Total Loss: {total_loss:.4f}")

        self.cbf.train()
        if not self.without_dynamic:
            self.dynamics.train()
            
        return total_loss
                
    def train(self):
        trainloader_iter = iter(self.train_dataset)
        lowest_eval_loss = float("inf")
        
        # Setup model save paths
        base_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/"
        cbf_path = f"{base_path}best_cbf.pth"
        dynamics_path = f"{base_path}best_dynamics.pth"
        os.makedirs(os.path.dirname(cbf_path), exist_ok=True)

        print("\nStarting training CBF" + (" with dynamics" if not self.without_dynamic else "") + "...")
        for step in trange(self.train_steps, desc="Training"):
            batch = next(trainloader_iter)
            observations, next_observations, actions, _, costs, done = [b.to(torch.float32).to(self.device) for b in batch]
            
            if not self.without_dynamic and self.train_dynamics:
                loss_safe, loss_unsafe, loss_grad, dynamics_loss, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc = self.compute_loss(
                    observations, next_observations, actions, costs, training_bool=True
                )
                
                # Log training metrics
                log_dict = {
                    "train_loss_safe": loss_safe,
                    "train_loss_unsafe": loss_unsafe,
                    "train_loss_grad": loss_grad,
                    "train_dynamics_loss": dynamics_loss,
                    "train_cbf_loss": loss_safe + loss_unsafe + loss_grad,
                    "train_total_loss": loss_safe + loss_unsafe + loss_grad + dynamics_loss,
                    "train_avg_safe_B": avg_safe_B,
                    "train_avg_unsafe_B": avg_unsafe_B,
                    "train_safe_acc": safe_acc,
                    "train_unsafe_acc": unsafe_acc,
                    "step": step
                }
            else:
                loss_safe, loss_unsafe, loss_grad, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc = self.compute_loss(
                    observations, next_observations, actions, costs, training_bool=True
                )
                
                # Log training metrics
                log_dict = {
                    "train_loss_safe": loss_safe,
                    "train_loss_unsafe": loss_unsafe,
                    "train_loss_grad": loss_grad,
                    "train_cbf_loss": loss_safe + loss_unsafe + loss_grad,
                    "train_total_loss": loss_safe + loss_unsafe + loss_grad,
                    "train_avg_safe_B": avg_safe_B,
                    "train_avg_unsafe_B": avg_unsafe_B,
                    "train_safe_acc": safe_acc,
                    "train_unsafe_acc": unsafe_acc,
                    "step": step
                }
                
            wandb.log(log_dict)

            if (step+1) % self.eval_every_n_steps == 0:
                total_eval_loss = self.validate()
                
                if total_eval_loss < lowest_eval_loss:
                    lowest_eval_loss = total_eval_loss
                    
                    # Save CBF model
                    cbf_save_path = f"{cbf_path}_task{self.args.task}_layers{self.args.num_layers}_dim{self.args.hidden_dim}_batch{self.args.batch_size}_seed{self.args.seed}"
                    if not self.without_dynamic:
                        cbf_save_path += "_with_dynamics"
                    torch.save(self.cbf.state_dict(), cbf_save_path)
                    
                    # Save dynamics model if applicable
                    if not self.without_dynamic and self.train_dynamics:
                        dynamics_save_path = f"{dynamics_path}_task{self.args.task}_layers{self.dynamics.num_layers}_dim{self.dynamics.hidden_dim}_batch{self.args.batch_size}_seed{self.args.seed}"
                        torch.save(self.dynamics.state_dict(), dynamics_save_path)
                        print(f"\nBest models saved at step {step} with eval loss {total_eval_loss:.4f}")
                    else:
                        print(f"\nBest CBF model saved at step {step} with eval loss {total_eval_loss:.4f}")
        
    def setup_optimizer(self):
        if not self.without_dynamic and self.train_dynamics:
            # Combined optimizer for both CBF and dynamics
            self.optim = torch.optim.Adam([
                {'params': self.cbf.parameters(), 'lr': self.lr, 'weight_decay': 2e-5},
                {'params': self.dynamics.parameters(), 'lr': self.dynamics_lr, 'weight_decay': 1e-5}
            ])
        else:
            # Optimizer for CBF only
            self.optim = torch.optim.Adam(self.cbf.parameters(), lr=self.lr, weight_decay=2e-5)
        
    def load_cbf(self, model_path):
        self.cbf.load_state_dict(torch.load(model_path, map_location=self.device))
        return self.cbf
        
    def load_dynamics(self, model_path):
        self.dynamics.load_state_dict(torch.load(model_path, map_location=self.device))
        return self.dynamics


@pyrallis.wrap()
def main(args: BCTrainConfig):
    cfg, old_cfg = asdict(args), asdict(BCTrainConfig())
    differing_values = {key: cfg[key] for key in cfg if cfg[key] != old_cfg[key]}
    cfg = asdict(BC_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)
    
    # Set seed
    args.seed = 7
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    
    import gymnasium as gym
    env = gym.make(args.task)
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)
    
    process_bc_dataset(data, args.cost_limit, args.gamma, args.bc_mode)
    
    # Set model parameters
    args.hidden_dim = 256
    args.num_layers = 5
    args.batch_size = 128
    args.num_action = env.action_space.shape[0]
    args.state_dim = env.observation_space.shape[0]
    
    # Choose whether to use dynamics or not
    args.without_dynamic = False  # Set to False to use dynamics
    args.train_dynamics = True
    
    # Create data loaders
    trainloader = DataLoader(TransitionDataset(data, split='train'), batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    valloader = DataLoader(TransitionDataset(data, split='val'), batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    

    
    # Create models
    cbf = CBF(
        num_action=args.num_action,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dt=0.1
    )
    
    dynamics = None
    dynamics_checkpoint = None
    
    if not args.without_dynamic:
        dynamics = AffineDynamics(
            num_action=args.num_action,
            state_dim=args.state_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dt=0.1
        )
        
        # Uncomment to load dynamics from a checkpoint
        # dynamics_checkpoint = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/best_model.pth_taskOfflineCarGoal1Gymnasium-v0_layers5_dim256"
    
    # Create trainer
    cbf_dynamics_trainer = CBF_dyn_trainer(
        cbf=cbf,
        lr=5e-5,
        device=args.device,
        train_steps=10000,
        eval_every_n_steps=200,
        train_dataset=trainloader,
        val_dataset=valloader,
        eval_steps=20,
        args=args,
        without_dynamic=args.without_dynamic,
        eps_safe=0.1,
        eps_unsafe=0.15, 
        eps_grad=0.05,
        w_safe=1,
        w_unsafe=1.3,
        w_grad=1,
        dynamics=dynamics,
        dynamics_checkpoint=dynamics_checkpoint,
        train_dynamics=args.train_dynamics,  # Set to True to train dynamics along with CBF
        dynamics_lr=1e-4
    )
    
    # Train models
    cbf_dynamics_trainer.train()
    
    wandb.finish()


if __name__ == "__main__":
    main()
#python examples/research/CBF_dynamics_new.py --task OfflineCarGoal1Gymnasium-v0 --device="mps"