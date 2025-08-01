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


class CBFtrainer:
    def __init__(self, cbf, train_dataset, val_dataset, dt=0.1, lr=1e-4, device="cpu", train_steps=10000, eval_every_n_steps=100, eval_steps=100, without_dynamic=True, args=None,
                 eps_safe=0.1, eps_unsafe=0.1, eps_grad=0.1, w_safe=1, w_unsafe=1, w_grad=0.1):
        
        self.cbf = cbf.to(device)  # Fixed: Changed AffineDynamics to cbf
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
        
        self.eps_safe = eps_safe
        self.eps_unsafe = eps_unsafe
        self.eps_grad = eps_grad
        self.w_safe = w_safe
        self.w_unsafe = w_unsafe
        self.w_grad = w_grad

        self.setup_optimizer()
        wandb.init(project="cbf_training", config={
            "learning_rate": lr,
            "train_steps": train_steps,
            "eps_safe": eps_safe,
            "eps_unsafe": eps_unsafe,
            "eps_grad": eps_grad,
            "w_safe": w_safe,
            "w_unsafe": w_unsafe,
            "w_grad": w_grad,
            "dt": dt,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "lr": self.lr,
            "seed": args.seed,
            "without dynamic:":without_dynamic

        })
            
    def compute_loss(self, observations, next_observations, actions, costs, training_bool=None):
        safe_mask = (costs <= 0).reshape(-1, 1)
        unsafe_mask = (costs > 0).reshape(-1, 1)

        B = self.cbf(observations).reshape(-1, 1)

        # Safe loss computation
        loss_safe_vector = self.w_safe * F.relu(self.eps_safe - B) * safe_mask
        num_safe_elements = safe_mask.sum()
        loss_safe = loss_safe_vector.sum() / (num_safe_elements + 1e-8)

        # Unsafe loss computation
        loss_unsafe_vector = self.w_unsafe * F.relu(self.eps_unsafe + B) * unsafe_mask
        num_unsafe_elements = unsafe_mask.sum()
        loss_unsafe = loss_unsafe_vector.sum() / (num_unsafe_elements + 1e-8)
        
        if (self.without_dynamic==True):
            # Gradient computation
            B_next = self.cbf(next_observations).reshape(-1, 1)
            B_dot = (B_next - B) / self.dt
            gradient = B_dot + B
            
        # if (self.without_dynamic==False):
        #     with torch.no_grad():
        #         x_dot=dynamics.forward_x_dot(observations)#reshape(-1,self.state_dim) #batchsize, state_dim size
        #     B=self.cbf(observations)
        #     grad_b = torch.autograd.grad(B,observations,retain_graph=True)[0]#.reshape(-1,self.state_dim)
        #     b_dot=torch.einsum("bd,bd->b",x_dot, grad_b)
        #     gradient=b_dot+B

        # Gradient loss computation
        loss_grad_vector = self.w_grad * F.relu(self.eps_grad - gradient) * safe_mask
        num_grad_elements = safe_mask.sum()
        loss_grad = loss_grad_vector.sum() / (num_grad_elements + 1e-8)

        # Total loss
        cbf_loss = loss_safe + loss_unsafe + loss_grad

        if training_bool:
            self.optim.zero_grad()
            cbf_loss.backward()
            self.optim.step()

        # Computing averages for safe and unsafe regions
        avg_safe_B = (B * safe_mask).sum() / (safe_mask.sum() + 1e-8)
        avg_unsafe_B = (B * unsafe_mask).sum() / (unsafe_mask.sum() + 1e-8)

        safe_acc= ((B>=0) * safe_mask).sum() / (num_safe_elements+ 1e-8)
        unsafe_acc= ((B<0) * unsafe_mask).sum() / (num_unsafe_elements+ 1e-8)
        return loss_safe.item(), loss_unsafe.item(), loss_grad.item(), avg_safe_B.item(), avg_unsafe_B.item(), safe_acc.item(), unsafe_acc.item()

    def validate(self):
        self.cbf.eval()  # Fixed: Changed to cbf
        valloader_iter = iter(self.val_dataset)
        total_loss_safe = 0.0
        total_loss_unsafe = 0.0
        total_loss_grad = 0.0
        total_avg_safe_B = 0.0
        total_avg_unsafe_B = 0.0
        total_safe_acc=0.0
        total_unsafe_acc=0.0

        print("\nStarting validation...")
        with torch.no_grad():
            for step in range(self.eval_steps):
                batch = next(valloader_iter)
                observations, next_observations, actions, _, costs, done = [b.to(torch.float32).to(self.device) for b in batch]
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
        avg_safe_B = total_avg_safe_B / self.eval_steps
        avg_unsafe_B = total_avg_unsafe_B / self.eval_steps
        total_loss = avg_loss_safe + avg_loss_unsafe + avg_loss_grad
        avg_safe_acc = total_safe_acc/self.eval_steps
        avg_unsafe_acc= total_unsafe_acc/self.eval_steps

        # Log validation metrics
        wandb.log({
            "val_loss_safe": avg_loss_safe,
            "val_loss_unsafe": avg_loss_unsafe,
            "val_loss_grad": avg_loss_grad,
            "val_avg_safe_B": avg_safe_B,
            "val_avg_unsafe_B": avg_unsafe_B,
            "val_total_loss": total_loss,
            "val_safe_acc":avg_safe_acc,
            "val_unsafe_acc":avg_unsafe_acc
        })

        # Print validation results
        print("\nValidation Results:")
        print(f"Average Safe Loss: {avg_loss_safe:.4f}")
        print(f"Average Unsafe Loss: {avg_loss_unsafe:.4f}")
        print(f"Average Gradient Loss: {avg_loss_grad:.4f}")
        print(f"Average Safe B Value: {avg_safe_B:.4f}")
        print(f"Average Unsafe B Value: {avg_unsafe_B:.4f}")
        print(f"val_safe_acc: {avg_safe_acc:.4f}")
        print(f"val_unsafe_acc: {avg_unsafe_acc:.4f}")       
        print(f"Total Loss: {total_loss:.4f}")

        self.cbf.train()  # Fixed: Changed to cbf
        return total_loss
                
    def train(self):
        trainloader_iter = iter(self.train_dataset)
        lowest_eval_loss = float("inf")
        best_model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/best_cbf.pth"
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        print("\nStarting training CBF...")
        for step in trange(self.train_steps, desc="Training"):
            batch = next(trainloader_iter)
            observations, next_observations, actions, _, costs, done = [b.to(torch.float32).to(self.device) for b in batch]
            
            loss_safe, loss_unsafe, loss_grad, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc= self.compute_loss(
                observations, next_observations, actions, costs, training_bool=True
            )

            # Log training metrics
            wandb.log({
                "train_loss_safe": loss_safe,
                "train_loss_unsafe": loss_unsafe,
                "train_loss_grad": loss_grad,
                "train_avg_safe_B": avg_safe_B,
                "train_avg_unsafe_B": avg_unsafe_B,
                "train_total_loss": loss_safe + loss_unsafe + loss_grad,
                "train_safe_acc":safe_acc,
                "train_unsafe_acc":unsafe_acc,
                "step": step
            })

            if step % self.eval_every_n_steps == 0:
                total_eval_loss = self.validate()
                
                if total_eval_loss < lowest_eval_loss:
                    lowest_eval_loss = total_eval_loss
                    torch.save(self.cbf.state_dict(), 
                             f"{best_model_path}_task{self.args.task}_layers{self.args.num_layers}_dim{self.args.hidden_dim}_batch{self.args.batch_size}_seed{self.args.seed}")
                    print(f"\nBest model saved at step {step} with eval loss {total_eval_loss:.4f}")
        
    def setup_optimizer(self):
        self.optim = torch.optim.Adam(self.cbf.parameters(), lr=self.lr, weight_decay=2e-5)
        
    def load_cbf(self, model_path):
        model = self.cbf().to(self.device)  # Initialize model
        model.load_state_dict(torch.load(model_path, map_location=self.device))  # Load weights
        model.eval()  # Set to evaluation mode
        return model
        

@pyrallis.wrap()
def main(args: BCTrainConfig):
    cfg, old_cfg = asdict(args), asdict(BCTrainConfig())
    differing_values = {key: cfg[key] for key in cfg if cfg[key] != old_cfg[key]}
    cfg = asdict(BC_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)
    

    args.seed=7 ##i changed the seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    import gymnasium as gym
    env = gym.make(args.task)
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)
    
    process_bc_dataset(data, args.cost_limit, args.gamma, args.bc_mode)
 
 
    
    args.hidden_dim = 256   # Example: set hidden_dim to 128
    args.num_layers = 5     # Example: set num_layers to 4
    args.batch_size = 128
    
    trainloader = DataLoader(TransitionDataset(data, split='train'), batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    valloader = DataLoader(TransitionDataset(data, split='val'), batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
 
    
    cbf=CBF(num_action=env.action_space.shape[0],
        state_dim=env.observation_space.shape[0],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dt=0.1
        )

    cbftrainer = CBFtrainer(
        cbf=cbf,
        lr=5e-5,
        device=args.device,
        train_steps=10000,
        eval_every_n_steps=200,
        train_dataset=trainloader,
        val_dataset=valloader,
        eval_steps=20,
        args=args,
        without_dynamic=True,
        eps_safe=0.1,
        eps_unsafe=0.15, 
        eps_grad=0.05,
        w_safe=1,
        w_unsafe=1.3,
        w_grad=1
    )
        
    cbftrainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
#python examples/research/CBF_dynamics_new.py --task OfflineCarGoal1Gymnasium-v0 --device="mps"