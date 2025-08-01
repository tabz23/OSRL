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


class AffineDynamicsTrainer:
    def __init__(self, AffineDynamics, train_dataset, val_dataset, lr=1e-4, device="cpu", train_steps=10000, eval_every_n_steps=100, eval_steps=100, args=None):
        self.AffineDynamics = AffineDynamics.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.device = device
        self.args = args  

        self.train_steps = train_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_steps = eval_steps
        self.setup_optimizer()
        
        wandb.init(project="AffineDynamicsTraining", config={
            "learning_rate": lr,
            "train_steps": train_steps
        })

    def compute_loss(self, predicted_next_observations, next_observations):
        loss_fn = torch.nn.MSELoss()
        dynamics_loss = loss_fn(predicted_next_observations, next_observations)
        self.optim.zero_grad()
        dynamics_loss.backward()
        self.optim.step()
        return dynamics_loss.item()

    def validate(self):
        self.AffineDynamics.eval()
        valloader_iter = iter(self.val_dataset)
        total_loss = 0.0

        with torch.no_grad():
            for _ in range(self.eval_steps):
                batch = next(valloader_iter)
                observations, next_observations, actions, _, _, _ = [b.to(torch.float32).to(self.device) for b in batch]
                predicted_next_observations = self.AffineDynamics.forward_next_state(observations, actions)
                loss_fn = torch.nn.MSELoss()
                total_loss += loss_fn(predicted_next_observations, next_observations).item()

        avg_loss = total_loss / self.eval_steps
        self.AffineDynamics.train()
        return avg_loss

    def train(self):
        trainloader_iter = iter(self.train_dataset)
        lowest_eval_loss = float("inf")
        best_model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/best_model.pth"
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        
        for step in trange(self.train_steps, desc="Training"):
            batch = next(trainloader_iter)
            observations, next_observations, actions, _, _, _ = [b.to(torch.float32).to(self.device) for b in batch]

            predicted_next_observations = self.AffineDynamics.forward_next_state(observations, actions)
            dynamics_loss = self.compute_loss(predicted_next_observations, next_observations)

            wandb.log({"train_loss": dynamics_loss, "step": step})
            
            if step % self.eval_every_n_steps == 0:
                eval_loss = self.validate()
                wandb.log({"eval_loss": eval_loss, "step": step})
                print(f"Step {step}: Eval Loss = {eval_loss}")

                if eval_loss < lowest_eval_loss:
                    lowest_eval_loss = eval_loss
                    torch.save(self.AffineDynamics.state_dict(), f"{best_model_path}_task{self.args.task}_layers{self.args.num_layers}_dim{self.args.hidden_dim}")
                    print(f"Best model saved at step {step} with eval loss {eval_loss}")
        
    def setup_optimizer(self):
        self.optim = torch.optim.Adam(self.AffineDynamics.parameters(), lr=self.lr, weight_decay=1e-5)


@pyrallis.wrap()
def main(args: BCTrainConfig):
    cfg, old_cfg = asdict(args), asdict(BCTrainConfig())
    differing_values = {key: cfg[key] for key in cfg if cfg[key] != old_cfg[key]}
    cfg = asdict(BC_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)
    
    args.hidden_dim = 256   # Example: set hidden_dim to 128
    args.num_layers = 4     # Example: set num_layers to 4
    args.batch_size = 256
    
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    import gymnasium as gym
    env = gym.make(args.task)
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)
    
    process_bc_dataset(data, args.cost_limit, args.gamma, args.bc_mode)
 
    trainloader = DataLoader(TransitionDataset(data, split='train'), batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    valloader = DataLoader(TransitionDataset(data, split='val'), batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    
    class cbftrainer():
        def __init__(self, cbf, train_dataset, val_dataset, dt=0.1, lr=1e-4, device="cpu", train_steps=10000, eval_every_n_steps=100, eval_steps=100, without_dynamic=True, args=None,
                     eps_safe=0.1, eps_unsafe=0.1, eps_grad=0.1, w_safe=1, w_unsafe=1, w_grad=0.1, ):
            
            self.AffineDynamics = AffineDynamics.to(device)
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.lr = lr
            self.device = device
            self.args = args  
            self.train_steps = train_steps
            self.eval_every_n_steps = eval_every_n_steps
            self.eval_steps = eval_steps
            self.without_dynamic=without_dynamic
            self.dt=dt
            
            self.eps_safe=eps_safe
            self.eps_unsafe=eps_unsafe
            self.eps_grad=eps_grad
            self.w_safe=w_safe
            self.w_unsafe=w_unsafe
            self.w_grad=w_grad

            
            self.setup_optimizer()
            wandb.init(project="cbf_training", config={
            "learning_rate": lr,
            "train_steps": train_steps})
            
        def compute_loss(self, observations,next_observations, actions, costs, training_bool=None):
            # if (self.without_dynamic==False):##do not use f, g. dont remove this comment
                # complete later
                # use AffineDynamics.forward_next_state(state, action) to get next state
                
            if (selto f.without_dynamic==True):##do not use f, g
                safe_mask=(costs<=0).reshape(-1,1)
                print("safe_mask shape",safe_mask.shape)
                print("safe_mask",safe_mask)
                unsafe_mask=(costs>0).reshape(-1,1)
                print("unsafe_mask shape",unsafe_mask.shape)
                print("unsafe_mask",unsafe_mask)
                
                B=self.cbf(observations).reshape(-1,1)
                B_next=self.cbf(next_observations).reshape(-1,1)
                loss_safe_vector= self.w_safe * F.relu(self.eps_safe - B) * safe_mask
                loss_safe=loss_safe_vector.mean(axis=0)
                print("loss_safe_vector",loss_safe_vector)
                print("loss_safe",loss_safe)
                
                loss_unsafe_vector= self.w_unsafe * F.relu(self.eps_unsafe + B)* unsafe_mask
                print("loss_unsafe_vector",loss_unsafe_vector)
                loss_unsafe=loss_unsafe_vector.mean(axis=0)
                print("loss_unsafe",loss_unsafe)
                
                B_dot= (B_next - B) / self.dt
                gradient=B_dot + B
                print("B_dot shape", B_dot)
                print("gradient shape", gradient)
                
                loss_grad_vector=self.w_grad * F.relu(self.eps_grad - gradient) * safe_mask
                print("loss_grad_vector shape",loss_grad_vector)
                loss_grad=loss_grad_vector.mean(axis=0)
                print("loss_grad",loss_grad)
                
                cbf_loss=loss_safe+loss_unsafe+loss_grad
                if (training_bool):
                    self.optim.zero_grad()
                    cbf_loss.backward()
                    self.optim.step()
                    
                avg_safe_B = B * safe_mask
                avg_unsafe_B = B * unsafe_mask
                
                return loss_safe, loss_unsafe, loss_grad, avg_safe_B, avg_unsafe_B

        def validate(self):
            self.AffineDynamics.eval()
            valloader_iter = iter(self.val_dataset)
            loss_safe = 0.0
            loss_unsafe = 0.0
            loss_grad = 0.0
            avg_safe_B = 0.0
            avg_unsafe_B = 0.0

            with torch.no_grad():
                for _ in range(self.eval_steps):
                    batch = next(valloader_iter)
                    observations, next_observations, actions, _, costs, done= [b.to(torch.float32).to(self.device) for b in batch]
                    loss_safe, loss_unsafe, loss_grad, avg_safe_B, avg_unsafe_B=self.compute_loss(observations,next_observations, actions, costs, training_bool=False)
                ##log loss_safe, loss_unsafe, loss_grad, avg_safe_B, avg_unsafe_B
                
                wandb.log({"train_loss_safe": loss_safe, "step": step})
                wandb.log({"train_loss_unsafe": loss_unsafe, "step": step})
                wandb.log({"train_loss_grad": loss_grad, "step": step})
                wandb.log({"train_avg_safe_B": avg_safe_B, "step": step})
                wandb.log({"train_avg_unsafe_B": avg_unsafe_B, "step": step})

            avg_loss = total_loss / self.eval_steps
            self.AffineDynamics.train()
            return avg_loss
                
        def train(self):
            trainloader_iter=iter(self.train_dataset)
            lowest_eval_loss=float("inf")
            best_model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/best_cbf.pth"
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            for step in trange(self.train_steps, desc="Training"):
                batch=next(trainloader_iter)
                observations, next_observations, actions, _, costs, done= [b.to(torch.float32).to(self.device) for b in batch]
                loss_safe, loss_unsafe, loss_grad, avg_safe_B, avg_unsafe_B=self.compute_loss(observations,next_observations, actions, costs, training_bool=True)

            if step % self.eval_every_n_steps == 0:
                returns = self.validate()
                ##log all eval losses
                


                pri#nt(f"Step {step} eval metrics")
                
                if total_eval_loss < lowest_eval_loss:
                    lowest_eval_loss = eval_loss
                    torch.save(self.AffineDynamics.state_dict(), f"{best_model_path}_task{self.args.task}_layers{self.args.num_layers}_dim{self.args.hidden_dim}")
                    print(f"Best model saved at step {step} with eval loss {eval_loss}")
        

        
        def setup_optimizer(self):
            self.optim = torch.optim.Adam(self.cbf.parameters(), lr=self.lr, weight_decay=1e-5)

    
    cbf=CBF(num_action=env.action_space.shape[0],
        state_dim=env.observation_space.shape[0],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dt=0.1,
        args=args)

    wandb.finish()


if __name__ == "__main__":
    main()
#python examples/research/dynamics_2.py --task OfflineCarGoal1Gymnasium-v0 --device="mps"