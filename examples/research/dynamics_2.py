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

from network_ihab import AffineDynamics
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
        best_model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/best_dynamic.pth"
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        
        for step in trange(self.train_steps, desc="Training"):
            batch = next(trainloader_iter)
            observations, next_observations, actions, _, _, _ = [b.to(torch.float32).to(self.device) for b in batch]
            # print(observations.shape)
            # print(next_observations.shape)
            # print(actions.shape)
            # print(observations[:2,:])
            # print(next_observations[:2,:])
            # print(actions[:2,:])
            diff = next_observations[:-1] - observations[1:]
            # print(diff.abs().mean().item())  # Check mean absolute difference

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
    
    ##CHECK THE BATCH SIZE BEING GIVEN ITS CURRENTLY 512
    trainloader = DataLoader(TransitionDataset(data, split='train'), batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    valloader = DataLoader(TransitionDataset(data, split='val'), batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    affinedynamics = AffineDynamics(
        num_action=env.action_space.shape[0],
        state_dim=env.observation_space.shape[0],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dt=0.1
    )
    
    dynamicstrainer = AffineDynamicsTrainer(
        AffineDynamics=affinedynamics,
        lr=1e-4,
        device=args.device,
        train_steps=10000,
        eval_every_n_steps=100,
        train_dataset=trainloader,
        val_dataset=valloader,
        eval_steps=100,
        args=args
    )
    
    dynamicstrainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
#python examples/research/dynamics_2.py --task OfflineCarGoal1Gymnasium-v0 --device="mps"