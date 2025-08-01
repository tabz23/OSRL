import os
import uuid
import types
import sys
sys.path.append("/Users/i.k.tabbara/Documents/python directory/OSRL")
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import bullet_safety_gym  # noqa
import dsrl
# import gymnasium as gym  # noqa
import safety_gymnasium as gym
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa
from examples.configs.bc_configs import BC_DEFAULT_CONFIG, BCTrainConfig

from osrl.algorithms import BC, BCTrainer
# from osrl.common import TransitionDataset
from osrl.common.dataset import process_bc_dataset
from osrl.common.exp_util import auto_name, seed_all

from network_ihab import AffineDynamics
from dataset_ihab import TransitionDataset
import torch.nn.functional as F


class AffineDynamicsTrainer():
    def __init__(self,
                AffineDynamics,
                train_dataset,
                val_dataset,
                lr=1e-4,
                device="cpu",
                batch_size=64,
                train_steps=10000,
                eval_every_n_steps=100,
                eval_steps=100
                ):
        
        self.AffineDynamics=AffineDynamics
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset
        self.lr=lr
        self.device=device
        self.batch_size=batch_size
        self.train_steps=train_steps
        self.eval_every_n_steps=eval_every_n_steps
        self.eval_steps=eval_steps
        self.setup_optimzer()
    
    def compute_loss(self,predicted_next_observations, next_observations):
        loss_fn = torch.nn.MSELoss()   
        dynamics_loss = loss_fn(predicted_next_observations, next_observations)
        self.optim.zero_grad()
        dynamics_loss.backward()
        self.optim.step()
        return dynamics_loss
        
    def validate(self):
        print(eval)
        self.AffineDynamics.eval()
        valloader_iter = iter(self.val_dataset)
        dynamics_loss=0
        for i in range(self.eval_steps):
            batch = next(valloader_iter)
            observations, next_observations, actions, _, _, _ = [b.to(self.device) for b in batch] #observations, next_observations, actions, rewards, costs, done
            predicted_next_observations=self.AffineDynamics.forward_next_state(observations,actions)
            loss_fn = torch.nn.MSELoss()   
            dynamics_loss = loss_fn(predicted_next_observations, next_observations)
            dynamics_loss+=dynamics_loss
        self.AffineDynamics.train()
        return dynamics_loss/self.eval_steps
            
    def train(self):
        trainloader_iter = iter(self.train_dataset)
        lowest_eval_loss= float("inf")
        for step in trange( self.train_steps, desc="Training"):
            batch = next(trainloader_iter)
            observations, next_observations, actions, _, _, _ = [b.to(self.device) for b in batch] #observations, next_observations, actions, rewards, costs, done
            predicted_next_observations=self.AffineDynamics.forward_next_state(observations,actions)
            dynamics_loss=self.compute_loss(predicted_next_observations,next_observations)

            # evaluation
            if (step) % self.eval_every_n_steps == 0:
                eval_loss = self.validate()

                # save the best weight
                if eval_loss<lowest_eval_loss:
                    lowest_eval_loss=eval_loss
                    ##log stuff and save best weight to /Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/logs along with the logs
 
    def setup_optimizer(self):
        self.optim=torch.optim.Adam(self.AffineDynamics.parameters(), lr=self.lr, weight_decay=1e-5)
 
@pyrallis.wrap()##check what means
def main(args:BCTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(BCTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}#dictionary that holds only the overridden values.
    cfg = asdict(BC_DEFAULT_CONFIG[args.task]())##args takes BCTrainConfig class which has task as method so can access it liek this
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)#More readable: args.learning_rate instead of args["learning_rate"].



    # set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # the cost scale is down in trainer rollout
    if "Metadrive" in args.task:
        import gym
    import gymnasium as gym
    env = gym.make(args.task)
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

    process_bc_dataset(data, args.cost_limit, args.gamma, args.bc_mode)

    trainloader = DataLoader(
        TransitionDataset(data,split='train'),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers
    )
    valloader = DataLoader(
        TransitionDataset(data,split='val'),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers
    )

    
    AffineDynamics=AffineDynamics( 
        num_action=env.action_space.shape[0],
        state_dim=env.observation_space.shape[0],
        hidden_dim=64,
        num_layers=3,
        dt=0.1)
    AffineDynamicsTrainer=AffineDynamicsTrainer(
                AffineDynamics=AffineDynamics,
                lr=1e-4,
                device="mps",
                batch_size=64,
                train_steps=10000,
                eval_every_n_steps=100,
                train_dataset=trainloader,
                val_dataset=valloader,
                eval_steps=100
                )
    
    AffineDynamicsTrainer.train()
if __name__ == "__main__":
    main()
    