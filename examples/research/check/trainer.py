import os
import sys
import uuid
import types
from dataclasses import asdict
from typing import Any
import json
import random

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

from network_ihab import AffineDynamics, CBF, CombinedCBFDynamics
from dataset_ihab import TransitionDataset
import torch.nn.functional as F

class CombinedCBFTrainer:
    def __init__(self, model, train_dataset, val_dataset, dt=0.1, lr=1e-4, device="cpu", train_steps=10000, 
                 eval_every_n_steps=100, eval_steps=100, args=None,
                 eps_safe=0.1, eps_unsafe=0.1, eps_grad=0.1, w_safe=1, w_unsafe=1, w_grad=0.1,lambda_lip=20,w_CQL=1,
                 train_dynamics=True, dynamics_lr=1e-4,num_action_samples=10,temp=0.9,detach=False):
        
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.device = device
        self.args = args  
        self.train_steps = train_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_steps = eval_steps
        self.dt = dt
        self.train_dynamics = train_dynamics
        
        self.eps_safe = eps_safe
        self.eps_unsafe = eps_unsafe
        self.eps_grad = eps_grad
        self.w_safe = w_safe
        self.w_unsafe = w_unsafe
        self.w_grad = w_grad
        self.w_CQL=w_CQL
        self.num_action_samples=num_action_samples
        self.temp=temp
        self.dynamics_lr = dynamics_lr
        rng = random.Random()  # This uses a new random state
        self.random_value = rng.randint(100, 999)
        self.detach=detach
        self.setup_optimizer()
        
        # Initialize WandB with expanded config
        
        self.lambda_lip =lambda_lip ##ADD THIS AND LOG PROPERLY LATER
        
        config = {
            "learning_rate": lr,
            "train_steps": train_steps,
            "eps_safe": eps_safe,
            "eps_unsafe": eps_unsafe,
            "eps_grad": eps_grad,
            "w_safe": w_safe,
            "w_unsafe": w_unsafe,
            "w_grad": w_grad,
            "w_CQL":w_CQL,
            "dt": dt,
            "cbf_hidden_dim": model.cbf_hidden_dim,
            "dynamics_hidden_dim": model.dynamics_hidden_dim,
            "cbf_num_layers": model.cbf_num_layers,
            "dynamics_num_layers": model.dynamics_num_layers,
            "batch_size": args.batch_size if hasattr(args, 'batch_size') else None,
            "seed": args.seed if hasattr(args, 'seed') else None,
            "train_dynamics": train_dynamics,
            "dynamics_lr": dynamics_lr,
            "task":args.task,
            "lambda_lip":self.lambda_lip,
            "num_action_samples":num_action_samples,
            "temp":self.temp,
            "detach":self.detach
        }
        
        wandb.init(project="combined_cbf_dynamics_training",name=f"run_{self.args.task}_{self.random_value}", config=config)
                
    def compute_loss(self, observations, next_observations, actions, costs, training_bool=None):
        # Determine safe and unsafe states
        safe_mask = (costs <= 0).reshape(-1, 1)
        unsafe_mask = (costs > 0).reshape(-1, 1)

        # Calculate CBF values for current states
        B = self.model.forward_cbf(next_observations).reshape(-1, 1)##CHANGED TO NEXT_OBSERVATION since safe mask saying next observation is safe

        # Safe loss computation
        loss_safe_vector = self.w_safe * F.relu(self.eps_safe - B) * safe_mask
        num_safe_elements = safe_mask.sum()
        loss_safe = loss_safe_vector.sum() / (num_safe_elements + 1e-8)

        # Unsafe loss computation
        loss_unsafe_vector = self.w_unsafe * F.relu(self.eps_unsafe + B) * unsafe_mask
        num_unsafe_elements = unsafe_mask.sum()
        loss_unsafe = loss_unsafe_vector.sum() / (num_unsafe_elements + 1e-8)
        # print(num_unsafe_elements)
        # print(num_safe_elements)
        
        
        
        
    ##ADDED THIS LOSS BE CAREFUL ##ADDED THIS LOSS BE CAREFUL ##ADDED THIS LOSS BE CAREFUL ##ADDED THIS LOSS BE CAREFUL
        # Compute B values
        B_curr = self.model.forward_cbf(observations).reshape(-1, 1) 
        B_next = self.model.forward_cbf(next_observations).reshape(-1, 1) 
        # Lipschitz continuity loss (smoothness constraint)
        # Adjust this hyperparameter as needed
        loss_lip = self.lambda_lip * torch.mean(torch.abs(B_next - B_curr)) 
  
        wandb.log({"loss_lip":loss_lip})
        
        
        
    
        # Gradient loss computation
        
        
        loss_grad = self.compute_gradient_loss(observations, actions, safe_mask)##safe mask means next state is safe 
        avg_random_cbf = 0.0  #i added this

        loss_cql, logsumexp_h, avg_random_cbf=self.compute_CQL_loss(observations,next_observations,actions,safe_mask)  #i added this - added avg_random_cbf
        wandb.log({"logsumexp_h": logsumexp_h.mean().item() / self.temp})
        if (self.w_CQL==0):
            loss_cql=torch.tensor(0.0)
            
        
        # Dynamics loss computation if we're also training dynamics
        dynamics_loss = torch.tensor(0.0)
        if self.train_dynamics:
            predicted_next_observations = self.model.forward_next_state(observations, actions)
            loss_fn = torch.nn.MSELoss()
            dynamics_loss = loss_fn(predicted_next_observations, next_observations)

        # Total loss
        cbf_loss = loss_safe + loss_unsafe + loss_grad + loss_lip + loss_cql##added cql loss
        total_loss = cbf_loss + (dynamics_loss if self.train_dynamics else 0.0) 
        
        

        
        # Backward pass during training
        if training_bool:
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        # Computing averages for safe and unsafe regions
        avg_safe_B = (B * safe_mask).sum() / (safe_mask.sum() + 1e-8)
        avg_unsafe_B = (B * unsafe_mask).sum() / (unsafe_mask.sum() + 1e-8)

        # Compute accuracy metrics
        safe_acc = ((B >= 0) * safe_mask).sum() / (num_safe_elements + 1e-8)
        unsafe_acc = ((B < 0) * unsafe_mask).sum() / (num_unsafe_elements + 1e-8)
    
        if self.train_dynamics:
            return loss_safe.item(), loss_unsafe.item(), loss_grad.item(),loss_cql.item(), dynamics_loss.item(), avg_safe_B.item(), avg_unsafe_B.item(), safe_acc.item(), unsafe_acc.item(), avg_random_cbf  #i added this - added avg_random_cbf to return
        else: 
            return loss_safe.item(), loss_unsafe.item(), loss_grad.item(),loss_cql.item(), avg_safe_B.item(), avg_unsafe_B.item(), safe_acc.item(), unsafe_acc.item(), avg_random_cbf  #i added this - added avg_random_cbf to return

 
    def compute_next_states(self, observation, action):
        with torch.no_grad():
            return self.model.forward_next_state(observation,action)
    def sample_random_actions(self, batch_size):
        return 2 * torch.rand(batch_size, self.args.num_action, device=self.device) - 1  # IMPORTANT Uniform in [-1,1] MAKE SURE ENVIRONMENTS HAVE THAT CONSTRAINT. MOST GYM ENVS DO THAT
    
    def compute_CQL_loss(self, observations, next_observations, actions, safe_mask):### ASSUMPTION IMP:i will assume observations are safe in a datapoints if cost is 0 even though cost 0 means next obs is safe.
        observations_safe = observations[safe_mask.reshape(-1,)]##SAFE_MASK INITIALLY B,1 so reshape to properly use it as boolean mask as we are not using * anymore but rather using it as mask
        # print("observations.shape", observations.shape)
        # print("observations_safe.shape", observations_safe.shape)

        next_observations_safe = next_observations[safe_mask.reshape(-1,)]
        # print("next_observations_.shape", next_observations.shape)
        # print("next_observations_safe.shape", next_observations_safe.shape)
        next_observation_h = self.model.forward_cbf(next_observations_safe)
        all_random_next_h = []
        for _ in range(self.num_action_samples):
            random_actions = self.sample_random_actions(observations_safe.shape[0])
            random_next_states = self.compute_next_states(observations_safe, random_actions)
            random_next_h = self.model.forward_cbf(random_next_states)
            all_random_next_h.append(random_next_h.squeeze())
            
        avg_random_cbf = 0.0  #i added this
        if all_random_next_h:
            stacked_h_values = torch.stack(all_random_next_h, dim=1)
            avg_random_cbf = torch.mean(stacked_h_values).item()  #i added this
            combined_h_values = torch.cat([stacked_h_values, next_observation_h.squeeze().unsqueeze(1)], dim=1)
            logsumexp_h = self.temp * torch.logsumexp(combined_h_values/self.temp, dim=1)
           
            
            # print(logsumexp_h[0:10])
            # print(next_observation_h[0:10])
            if not self.detach:
                cql_actions_term = logsumexp_h - next_observation_h.squeeze()
            else:
                cql_actions_term = logsumexp_h - next_observation_h.squeeze().detach()
            loss_cql_actions = self.w_CQL * torch.mean(cql_actions_term)
            return loss_cql_actions,logsumexp_h, avg_random_cbf  #i added this - added avg_random_cbf to return

        
    def compute_gradient_loss(self, observations, actions, safe_mask):
        """
        Custom method to compute gradient loss using a more stable approach.
        """
        '''
        CHANGED TO NEXT_OBSERVATION SINCE MASK SAFE AND UNSAFE USING COSTS WOULD CORRESPOND TO THE NEXT STATE, NOT THE CURRENT STATE
        I AM USING ACTION OF CURRENT STATE AS ACTION OF THE NEXT STATE
        '''
        # Forward pass CBF
        observations.requires_grad = True
        B = self.model.forward_cbf(observations).reshape(-1, 1)##THIS IS TAKING NEXT_OBSERVATION NOT OBSERVATION
        # print(B.shape)
        # print(B.sum().shape)
        grad_b = torch.autograd.grad(B, observations,grad_outputs=torch.ones_like(B),retain_graph=True)[0] ###old

        
        with torch.no_grad():   ##dont want gradient of CBF to propagate into the dynamics
            x_dot = self.model.forward_x_dot(observations, actions)##
        
        # Compute Lie derivative (inner product of grad_B and x_dot)
        # b_dot = torch.sum(grad_b * x_dot, axis=1)## grad_b * x_dot gives batch_size,state_dim. then sum over entries along the state_dim to simulate dot product
        b_dot = torch.einsum('bo,bo->b',grad_b,x_dot).reshape(-1,1)#compute dot product between grad B and x_dot in order to get b_dot

        
        # Compute gradient
        gradient = b_dot + 1*B

        # print("gradient.shape",gradient.shape)
        
        # Compute loss
        loss_grad_vector = self.w_grad * F.relu(self.eps_grad - gradient) * safe_mask
        
   
        
        num_grad_elements = safe_mask.sum()
        loss_grad = loss_grad_vector.sum() / (num_grad_elements + 1e-8)
        # print("loss_grad_vector.shape",loss_grad_vector.shape)
        # print("loss_grad",loss_grad)

        return loss_grad
    '''
B.shape torch.Size([128, 1])
next_observations.shape torch.Size([128, 72])
grad_b shape torch.Size([128, 72])
x_dot shape torch.Size([128, 72])
bdot shape torch.Size([128, 1])
b_dot.shape torch.Size([128, 1])
loss_grad_vector.shape torch.Size([128, 1])
'''
    def validate(self):
        self.model.eval()
            
        valloader_iter = iter(self.val_dataset)
        total_loss_safe = 0.0
        total_loss_unsafe = 0.0
        total_loss_grad = 0.0
        total_loss_cql=0.0
        total_dynamics_loss = 0.0
        total_avg_safe_B = 0.0
        total_avg_unsafe_B = 0.0
        total_safe_acc = 0.0
        total_unsafe_acc = 0.0
        total_avg_random_cbf = 0.0  #i added this

        print("\nStarting validation...")
        # with torch.no_grad():##dont disable grad because need gradient through the barrier. 
        for step in range(self.eval_steps):
            batch = next(valloader_iter)
            observations, next_observations, actions, _, costs, done = [b.to(torch.float32).to(self.device) for b in batch]
            
            if self.train_dynamics:
                loss_safe, loss_unsafe, loss_grad,loss_cql, dynamics_loss, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc, avg_random_cbf = self.compute_loss(  #i added this - added avg_random_cbf
                    observations, next_observations, actions, costs, training_bool=False
                )
                total_dynamics_loss += dynamics_loss
            else:
                loss_safe, loss_unsafe, loss_grad, loss_cql, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc, avg_random_cbf = self.compute_loss(  #i added this - added avg_random_cbf
                    observations, next_observations, actions, costs, training_bool=False
                )
            
            total_loss_safe += loss_safe
            total_loss_unsafe += loss_unsafe
            total_loss_grad += loss_grad
            total_loss_cql+=loss_cql
            total_avg_safe_B += avg_safe_B
            total_avg_unsafe_B += avg_unsafe_B
            total_safe_acc += safe_acc
            total_unsafe_acc += unsafe_acc
            total_avg_random_cbf += avg_random_cbf  #i added this

        # Calculate averages
        avg_loss_safe = total_loss_safe / self.eval_steps
        avg_loss_unsafe = total_loss_unsafe / self.eval_steps
        avg_loss_grad = total_loss_grad / self.eval_steps
        avg_loss_cql = total_loss_cql /self.eval_steps
        avg_dynamics_loss = total_dynamics_loss / self.eval_steps if self.train_dynamics else 0.0
        avg_safe_B = total_avg_safe_B / self.eval_steps
        avg_unsafe_B = total_avg_unsafe_B / self.eval_steps
        avg_random_cbf = total_avg_random_cbf / self.eval_steps  #i added this
        
        total_cbf_loss = avg_loss_safe + avg_loss_unsafe + avg_loss_grad #+ avg_loss_cql##TODO ADD HERE LIPCHITZ LOSS 
        total_loss = total_cbf_loss + (avg_dynamics_loss if self.train_dynamics else 0.0)
        
        avg_safe_acc = total_safe_acc / self.eval_steps
        avg_unsafe_acc = total_unsafe_acc / self.eval_steps

        # Log validation metrics
        log_dict = {
            "val_loss_safe": avg_loss_safe,
            "val_loss_unsafe": avg_loss_unsafe,
            "val_loss_grad": avg_loss_grad,
            "val_loss_cql": avg_loss_cql,
            "val_cbf_loss": total_cbf_loss,
            "val_avg_safe_B": avg_safe_B,
            "val_avg_unsafe_B": avg_unsafe_B,
            "val_safe_acc": avg_safe_acc,
            "val_unsafe_acc": avg_unsafe_acc,
            "val_total_loss": total_loss,
            "val_avg_random_cbf": avg_random_cbf  #i added this
        }
        
        if self.train_dynamics:
            log_dict["val_dynamics_loss"] = avg_dynamics_loss
            
        wandb.log(log_dict)

        # Print validation results
        print("\nValidation Results:")
        print(f"Average Safe Loss: {avg_loss_safe:.4f}")
        print(f"Average Unsafe Loss: {avg_loss_unsafe:.4f}")
        print(f"Average cql Loss: {avg_loss_cql:.4f}")
        print(f"Average Gradient Loss: {avg_loss_grad:.4f}")
        if self.train_dynamics:
            print(f"Average Dynamics Loss: {avg_dynamics_loss:.4f}")
        print(f"Average Safe B Value: {avg_safe_B:.4f}")
        print(f"Average Unsafe B Value: {avg_unsafe_B:.4f}")
        print(f"Safe Accuracy: {avg_safe_acc:.4f}")
        print(f"Unsafe Accuracy: {avg_unsafe_acc:.4f}")       
        print(f"Total Loss: {total_loss:.4f}")

        self.model.train()
            
        return total_loss, avg_safe_acc, avg_unsafe_acc
                    
    def train(self):
        trainloader_iter = iter(self.train_dataset)
        lowest_eval_loss = float("inf")
        
        # Setup model save path
        base_path = f"/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/{self.args.task}_{self.random_value}"
        os.makedirs(base_path, exist_ok=True)

        print("\nStarting training combined CBF and dynamics...")
        for step in trange(self.train_steps, desc="Training"):
            batch = next(trainloader_iter)
            observations, next_observations, actions, _, costs, done = [b.to(torch.float32).to(self.device) for b in batch]
            
            if self.train_dynamics:
                loss_safe, loss_unsafe, loss_grad,loss_cql, dynamics_loss, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc, avg_random_cbf = self.compute_loss(  #i added this - added avg_random_cbf
                    observations, next_observations, actions, costs, training_bool=True
                )
                
                # Log training metrics
                log_dict = {
                    "train_loss_safe": loss_safe,
                    "train_loss_unsafe": loss_unsafe,
                    "train_loss_grad": loss_grad,
                    "train_loss_cql":loss_cql,
                    "train_dynamics_loss": dynamics_loss,
                    "train_cbf_loss": loss_safe + loss_unsafe + loss_grad + loss_cql,
                    "train_total_loss": loss_safe + loss_unsafe + loss_grad + dynamics_loss + loss_cql,
                    "train_avg_safe_B": avg_safe_B,
                    "train_avg_unsafe_B": avg_unsafe_B,
                    "train_safe_acc": safe_acc,
                    "train_unsafe_acc": unsafe_acc,
                    "train_avg_random_cbf": avg_random_cbf,  #i added this
                    "step": step
                }
            else:
                loss_safe, loss_unsafe, loss_grad,loss_cql, avg_safe_B, avg_unsafe_B, safe_acc, unsafe_acc, avg_random_cbf = self.compute_loss(  #i added this - added avg_random_cbf
                    observations, next_observations, actions, costs, training_bool=True
                )
                
                # Log training metrics
                log_dict = {
                    "train_loss_safe": loss_safe,
                    "train_loss_unsafe": loss_unsafe,
                    "train_loss_grad": loss_grad,
                    "train_loss_cql":loss_cql,
                    "train_cbf_loss": loss_safe + loss_unsafe + loss_grad,
                    "train_total_loss": loss_safe + loss_unsafe + loss_grad,
                    "train_avg_safe_B": avg_safe_B,
                    "train_avg_unsafe_B": avg_unsafe_B,
                    "train_safe_acc": safe_acc,
                    "train_unsafe_acc": unsafe_acc,
                    "train_avg_random_cbf": avg_random_cbf,  #i added this
                    "step": step
                }
                
            wandb.log(log_dict)

            if (step+1) % self.train_steps == 0:
                model_save_path = os.path.join(base_path, "combined_model_laststep.pth")
                torch.save(self.model.state_dict(), model_save_path)
                hyperparameters = {
                            "task": self.args.task,
                            "seed": self.args.seed,
                            "cbf_hidden_dim": self.args.cbf_hidden_dim,
                            "dynamics_hidden_dim": self.args.dynamics_hidden_dim,
                            "cbf_num_layers": self.args.cbf_num_layers,
                            "dynamics_num_layers": self.args.dynamics_num_layers,
                            "batch_size": self.args.batch_size,
                            "learning_rate": self.lr,
                            "dynamics_learning_rate": self.dynamics_lr,
                            "eps_safe": self.eps_safe,
                            "eps_unsafe": self.eps_unsafe,
                            "eps_grad": self.eps_grad,
                            "w_safe": self.w_safe,
                            "w_unsafe": self.w_unsafe,
                            "w_grad": self.w_grad,
                            "train_dynamics": self.train_dynamics,
                            "dt": self.dt,
                            "num_action":self.args.num_action,
                            "state_dim":self.args.state_dim,
                            "best_safe acc":val_avg_safe_acc,
                            "best_unsafe_acc":val_avg_unsafe_acc
                        }
                hyperparameters_path = os.path.join(base_path, "hyperparameters.json") #f"
                    # Save the hyperparameters to a JSON file
                with open(hyperparameters_path, 'w') as f:
                    json.dump(hyperparameters, f, indent=4)
                print(f"\last combined model saved at step {step} with eval loss {total_eval_loss:.4f}")
                print(f"Hyperparameters saved to {hyperparameters_path}")
                
            if (step+1) % self.eval_every_n_steps == 0:
                total_eval_loss,val_avg_safe_acc, val_avg_unsafe_acc = self.validate()
                
                if (total_eval_loss < lowest_eval_loss) and (val_avg_safe_acc>0.86) and (val_avg_unsafe_acc>0.86): ###FIX THS LATER
                # if (total_eval_loss < lowest_eval_loss) and (val_avg_safe_acc>0.7) and (val_avg_unsafe_acc>0.7):
                    lowest_eval_loss = total_eval_loss
                    
                    # Save combined model
                    model_save_path = os.path.join(base_path, "combined_model.pth")
                    torch.save(self.model.state_dict(), model_save_path)
                    hyperparameters = {
                            "task": self.args.task,
                            "seed": self.args.seed,
                            "cbf_hidden_dim": self.args.cbf_hidden_dim,
                            "dynamics_hidden_dim": self.args.dynamics_hidden_dim,
                            "cbf_num_layers": self.args.cbf_num_layers,
                            "dynamics_num_layers": self.args.dynamics_num_layers,
                            "batch_size": self.args.batch_size,
                            "learning_rate": self.lr,
                            "dynamics_learning_rate": self.dynamics_lr,
                            "eps_safe": self.eps_safe,
                            "eps_unsafe": self.eps_unsafe,
                            "eps_grad": self.eps_grad,
                            "w_safe": self.w_safe,
                            "w_unsafe": self.w_unsafe,
                            "w_grad": self.w_grad,
                            "train_dynamics": self.train_dynamics,
                            "dt": self.dt,
                            "num_action":self.args.num_action,
                            "state_dim":self.args.state_dim,
                            "best_safe acc":val_avg_safe_acc,
                            "best_unsafe_acc":val_avg_unsafe_acc
                        }
                    hyperparameters_path = os.path.join(base_path, "hyperparameters.json") #f"
                    # Save the hyperparameters to a JSON file
                    with open(hyperparameters_path, 'w') as f:
                        json.dump(hyperparameters, f, indent=4)
                    print(f"\nBest combined model saved at step {step} with eval loss {total_eval_loss:.4f}")
                    print(f"Hyperparameters saved to {hyperparameters_path}")
                    
    def setup_optimizer(self):
        # Create parameter groups with different learning rates if needed
        if self.train_dynamics:
            # Identify which parameters belong to CBF vs dynamics
            cbf_params = list(self.model.cbf.parameters())
            dynamics_params = list(self.model.f.parameters()) + list(self.model.g.parameters())
            
            self.optim = torch.optim.Adam([
                {'params': cbf_params, 'lr': self.lr, 'weight_decay': 2e-5},
                {'params': dynamics_params, 'lr': self.dynamics_lr, 'weight_decay': 1e-5}
            ])
        else:
            # Only optimize CBF parameters
            self.optim = torch.optim.Adam(self.model.cbf.parameters(), lr=self.lr, weight_decay=2e-5)
        
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        return self.model
    
    
@pyrallis.wrap()
def main(args: BCTrainConfig):
    cfg, old_cfg = asdict(args), asdict(BCTrainConfig())
    differing_values = {key: cfg[key] for key in cfg if cfg[key] != old_cfg[key]}
    cfg = asdict(BC_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)
    

    # args.seed=7 ##i changed the seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    import gymnasium as gym
    env = gym.make(args.task)
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)
     
    process_bc_dataset(data, args.cost_limit, args.gamma, "all")
    
    # Set model parameters
    
    args.num_action = env.action_space.shape[0]
    args.state_dim = env.observation_space.shape[0]
    
    # Create data loaders
    trainloader = DataLoader(TransitionDataset(data, split='train'), batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    valloader = DataLoader(TransitionDataset(data, split='val'), batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    
    # Create the combined model
    combined_model = CombinedCBFDynamics(
        num_action=args.num_action,
        state_dim=args.state_dim,
        cbf_hidden_dim=args.cbf_hidden_dim,
        dynamics_hidden_dim=args.dynamics_hidden_dim,
        cbf_num_layers=args.cbf_num_layers,
        dynamics_num_layers=args.dynamics_num_layers,
        dt=0.1
    )
    
    # Optionally load from a previous checkpoint
    # combined_model_checkpoint = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/combined_model.pth_taskOfflineCarGoal1Gymnasium-v0_seed7"
    # if os.path.exists(combined_model_checkpoint):
    #     combined_model.load_state_dict(torch.load(combined_model_checkpoint, map_location=args.device))
    #     print(f"Loaded combined model from {combined_model_checkpoint}")
    
    # Create trainer
    trainer = CombinedCBFTrainer(
        model=combined_model,
        lr=1e-5,
        device=args.device,
        
        train_steps=args.train_steps,##change 50k hopper and 15k swimmer
        
        eval_every_n_steps=300,
        train_dataset=trainloader,
        val_dataset=valloader,
        eval_steps=20,
        args=args,
        # eps_safe=0.08,
        # eps_unsafe=0.15, 
        # eps_grad=0.02,
        # w_safe=1,
        # w_unsafe=1.2,
        # w_grad=1,##change 1 or 2
        # lambda_lip=1,
        
        eps_safe=args.eps_safe,
        eps_unsafe=args.eps_unsafe, 
        eps_grad=args.eps_grad,
        w_safe=args.w_safe,
        w_unsafe=args.w_unsafe,
        w_grad=args.w_grad,##change 1 or 2
        lambda_lip=args.lambda_lip,
        
        train_dynamics=True,
        dynamics_lr=1e-4,
        w_CQL=args.cql,
        num_action_samples=args.num_action_samples_cql,
        temp=args.temp,
        detach=args.detach
    )
    
    # Train models
    trainer.train()
    
    # Optionally extract individual models
    # This can be useful if you need to use them separately later
    standalone_cbf = combined_model.get_cbf_model()
    standalone_dynamics = combined_model.get_dynamics_model()
    
    # Save individual models if needed
    base_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/"
    torch.save(standalone_cbf.state_dict(), f"{base_path}extracted_cbf_model_task{args.task}_seed{args.seed}.pth")
    torch.save(standalone_dynamics.state_dict(), f"{base_path}extracted_dynamics_model_task{args.task}_seed{args.seed}.pth")
    
    wandb.finish()


if __name__ == "__main__":
    main()
    
  # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps"
  #python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/trainer.py" --task OfflineHopperVelocityGymnasium-v1 --device="mps"
  
  
  
  
    """
Looking at the CQL loss in your code, it appears that the loss is currently promoting higher CBF values for safe states.
Let me explain what I see in the compute_CQL_loss function:

The function isolates the safe observations (where safe_mask is True)
It computes the CBF value for the actual next observations: next_observation_h
It samples random actions and computes CBF values for the resulting next states: random_next_h
It combines these values and applies a logsumexp operation
The final loss is computing: logsumexp_h - next_observation_h.squeeze()

Since this is being used as a loss to minimize, the objective is to make this difference smaller. This means:

The model is encouraged to make next_observation_h (the CBF value of the actual next state) larger
And/or make the logsumexp_h term smaller (which means making the CBF values of random next states smaller)
    
    """
    
    
    
    ###BELOW PLEASE TRY IT
    
    """def compute_CQL_loss(self, observations, next_observations, actions, safe_mask):
    observations_safe = observations[safe_mask.reshape(-1,)]
    next_observations_safe = next_observations[safe_mask.reshape(-1,)]
    
    # Get in-distribution CBF values WITHOUT gradients
    with torch.no_grad():
        next_observation_h = self.model.forward_cbf(next_observations_safe)
    
    all_random_next_h = []
    for _ in range(self.num_action_samples):
        random_actions = self.sample_random_actions(observations_safe.shape[0])
        random_next_states = self.compute_next_states(observations_safe, random_actions)
        random_next_h = self.model.forward_cbf(random_next_states)
        all_random_next_h.append(random_next_h.squeeze())
        
    if all_random_next_h:
        stacked_h_values = torch.stack(all_random_next_h, dim=1)
        
        # You can either keep your existing formula:
        cql_actions_term = torch.logsumexp(stacked_h_values, dim=1) - next_observation_h.squeeze()
        # Or simplify to just minimize OOD values:
        # cql_actions_term = torch.logsumexp(stacked_h_values, dim=1)
        
        loss_cql_actions = self.w_CQL * torch.mean(cql_actions_term)
        return loss_cql_actions
    """
#for swimmer
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples_cql 10 --seed 7 --w_grad 2 --train_steps 15000
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0.5 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples_cql 10 --seed 7 --w_grad 2 --train_steps 15000
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 1 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples_cql 10 --seed 7 --w_grad 2 --train_steps 15000





# Training: Safe states → CBF learns "states like this are safe" 
#                     → "random actions from safe states are risky"

# Validation: New safe states → CBF recognizes "this looks like a safe state"
#                            → Applies learned rule: "random actions are risky"
                           
# CBF is successfully learning that "random actions from safe states tend to be risky" 

##confirming hypothesis we had in the 2d single integrator case, as w increase and at 1 specifically, we started labeling safe OOD states as unsafe, but when wc=0.5 they are simply less safe than the IND safe states but not seen as unsafe.

#note that at first both safe and ranomly reached states have similar B, but as training progresses the OOD safe states are seen as less safe
#since L_safe only pushes up the safe states that are IND whereas the OOD reached safe states are not pushed up.




#for hopper
# python  examples/research/check/trainer.py --task OfflineHopperVelocityGymnasium-v1  --cql 0 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples_cql 10 --seed 7 --train_steps 50000 --w_grad 2
# python  examples/research/check/trainer.py --task OfflineHopperVelocityGymnasium-v1  --cql 0.5 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples_cql 10 --seed 7 --train_steps 50000 --w_grad 2
# python  examples/research/check/trainer.py --task OfflineHopperVelocityGymnasium-v1  --cql 1 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples_cql 10 --seed 7 --train_steps 50000 --w_grad 2
