from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import dsrl
import numpy as np
import pyrallis
import torch
from pyrallis import field
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from bc_modif_cbf_list import BC, BCTrainer
from osrl.common.exp_util import load_config_and_model, seed_all
# Add to imports at the top
import qpsolvers
import numpy as np
from network_ihab import CombinedCBFDynamics

import random
@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    noise_scale: List[float] = None
    costs: List[float] = field(default=[1, 10, 20, 30, 40], is_mutable=True)
    eval_episodes: int = 20
    best: bool = True
    device: str = "cpu"
    threads: int = 4
    

@pyrallis.wrap()
def eval(args: EvalConfig):
    all_metrics = []

    # model_paths = [
    #     ######final chosen ones
    #     ##idbf
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_269/combined_model.pth",
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_481/combined_model.pth",
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_576/combined_model.pth",
    #     #CCBF
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##gr8 bc all
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_932/combined_model.pth",##gr8 bc all #gr8 for bc safe
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_709/combined_model.pth",##gr8 bc all #greatt for bc safe
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##gr8 bc all
    #     ##CBF
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_933/combined_model.pth",
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_984/combined_model.pth",
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_188/combined_model.pth",       
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_626/combined_model.pth",
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_758/combined_model.pth",


# for the ablation study


    #  ]
    model_paths = [
    ####final ones below
#     #   #idbf
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_271/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_209/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_693/combined_model.pth",##replace later
#     # ##ccbf
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
#     # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_217/combined_model.pth",
#     #   #CBF
#    # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_861/combined_model.pth",
#     ## "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_799/combined_model.pth",
#     ## "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_159/combined_model.pth",
#         "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_409/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_929/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_892/combined_model.pth",
    


# ##for ablation part 1 
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_264/combined_model.pth",#p=0.1, normal size
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_264/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_264/combined_model.pth",
#     # ##ccbf
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_882/combined_model.pth",#p=10e-4, normal size
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_882/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_882/combined_model.pth",
#     #   #CBF
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_270/combined_model.pth",#p=10e-8, normal size
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_270/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_270/combined_model.pth",
     #########################################
#      bc safe
#      Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.5424, std: 0.0363
#   Avg Normalized Cost: 0.1742, std: 0.0706
#   Avg Length: 890.5000, std: 58.7089

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5825, std: 0.0193
#   Avg Normalized Cost: 0.3525, std: 0.0864
#   Avg Length: 936.6167, std: 28.5866

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.6014, std: 0.0098
#   Avg Normalized Cost: 0.2533, std: 0.0646
#   Avg Length: 955.7500, std: 16.0619
  
#   bc
#   Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3335, std: 0.0555
#   Avg Normalized Cost: 3.8683, std: 0.1830
#   Avg Length: 418.9333, std: 73.1204

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.6272, std: 0.1258
#   Avg Normalized Cost: 7.4908, std: 1.2622
#   Avg Length: 752.0167, std: 142.6290

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.2496, std: 0.0614
#   Avg Normalized Cost: 2.2758, std: 0.8827
#   Avg Length: 338.6333, std: 63.9220
     #########################################
##for ablation part 2 
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_562/combined_model.pth",#p=0.1, big size
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_562/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_562/combined_model.pth",
    # # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_216/combined_model.pth",#p=10e-4, big size
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_216/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_216/combined_model.pth",
    # #   #CBF
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",#p=10e-8, big size
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",   
     #########################################
#      bc safe
#      Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.5943, std: 0.0169
#   Avg Normalized Cost: 0.3250, std: 0.0727
#   Avg Length: 917.1333, std: 24.4275

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5117, std: 0.0230
#   Avg Normalized Cost: 0.1592, std: 0.0780
#   Avg Length: 831.8500, std: 34.4025

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4492, std: 0.0176
#   Avg Normalized Cost: 0.0758, std: 0.0453
#   Avg Length: 740.6333, std: 25.6133
  
#   bc
#   Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.0849, std: 0.0300
#   Avg Normalized Cost: 0.9117, std: 0.4845
#   Avg Length: 143.7167, std: 33.4144

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.1475, std: 0.0472
#   Avg Normalized Cost: 2.5642, std: 0.5321
#   Avg Length: 192.5167, std: 54.9030

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.2136, std: 0.0356
#   Avg Normalized Cost: 3.3408, std: 0.5297
#   Avg Length: 262.1333, std: 45.6601
     ######################################### 
# ##for ablation part 3
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_204/combined_model.pth",#p=10, small model
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_204/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_204/combined_model.pth",
#     # ##ccbf
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",#p=10, big model
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",
#     #   #CBF
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",#dont care
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",      
      
########################
# #bc safe
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3483, std: 0.0347
#   Avg Normalized Cost: 0.4258, std: 0.1121
#   Avg Length: 552.6500, std: 64.7598

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.3382, std: 0.0363
#   Avg Normalized Cost: 0.0833, std: 0.0621
#   Avg Length: 570.6833, std: 60.3638
  
#   #bc 
  
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.2145, std: 0.0678
#   Avg Normalized Cost: 2.0050, std: 0.2819
#   Avg Length: 283.1000, std: 81.4236

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.1672, std: 0.0244
#   Avg Normalized Cost: 4.7492, std: 0.4654
#   Avg Length: 156.5833, std: 25.3083
########################
##for ablation with and without detach
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_559/combined_model.pth",#this is without detach. dont care about the below
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_559/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_559/combined_model.pth",
    # ##ccbf
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",#p=10, big model
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",
    #   #CBF
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",#dont care
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",  
#bc
#     Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.0434, std: 0.0054
#   Avg Normalized Cost: 0.0975, std: 0.0499
#   Avg Length: 81.5000, std: 7.2320

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.1582, std: 0.0321
#   Avg Normalized Cost: 4.7292, std: 0.4122
#   Avg Length: 138.7000, std: 39.8946

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.2138, std: 0.0584
#   Avg Normalized Cost: 3.5767, std: 1.0941
#   Avg Length: 273.7167, std: 58.2263

#bc safe
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.5852, std: 0.0229
#   Avg Normalized Cost: 0.1617, std: 0.1196
#   Avg Length: 927.5167, std: 37.4683

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.3743, std: 0.0179
#   Avg Normalized Cost: 0.1042, std: 0.1213
#   Avg Length: 630.6000, std: 31.1103

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4728, std: 0.0383
#   Avg Normalized Cost: 0.1133, std: 0.0541
#   Avg Length: 774.5000, std: 62.1148
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_409/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_929/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_892/combined_model.pth",
    ####final ones below
    #   #idbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_929/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_929/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_929/combined_model.pth",
    
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_892/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_892/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_892/combined_model.pth",
    
    #    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
    
    
]
    
    
        # # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
    # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_217/combined_model.pth",
    for i, model_path in enumerate(model_paths):
        # print(f"\n[{i+1}/{len(model_paths)}] Evaluating model: {os.path.basename(os.path.dirname(model_path))}")
        model_dir = os.path.dirname(model_path)
        hyperparam_paths = os.path.join(model_dir, "hyperparameters.json")
        cfg, model = load_config_and_model(args.path, args.best)
        seed_all(cfg["seed"])
        if args.device == "cpu":
            torch.set_num_threads(args.threads)

        if "Metadrive" in cfg["task"]:
            import gym
        else:
            import gymnasium as gym  # noqa

        # env = wrap_env(env = gym.make("SafetySwimmerVelocityGymnasium-v1",render_mode="human"))#uncomment this and go uncomment env.render in bc_modif to view render. 
        env = wrap_env(env = gym.make("OfflineHopperVelocityGymnasium-v1"))
        # env = OfflineEnvWrapper(env)
        env.set_target_cost(cfg["cost_limit"])

        # model & optimizer & scheduler setup
        state_dim = env.observation_space.shape[0]
        if cfg["bc_mode"] == "multi-task":
            state_dim += 1
        bc_model = BC(
            state_dim=state_dim,
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            a_hidden_sizes=cfg["a_hidden_sizes"],
            episode_len=cfg["episode_len"],
            device=args.device,
        )
        bc_model.load_state_dict(model["model_state"])
        bc_model.to(args.device)

        trainer = BCTrainer(bc_model,
                            env,
                            bc_mode=cfg["bc_mode"],
                            cost_limit=cfg["cost_limit"],
                            device=args.device,
                            model_path=model_path,
                            hyperparams_path=hyperparam_paths)

        if cfg["bc_mode"] == "multi-task":
            for target_cost in args.costs:
                env.set_target_cost(target_cost)
                trainer.set_target_cost(target_cost)
                ret, cost, length = trainer.evaluate(args.eval_episodes)
                normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
                print(
                    f"Eval reward: {ret}, normalized reward: {normalized_ret}; target cost {target_cost}, real cost {cost}, normalized cost: {normalized_cost}"
                )
        else:
            ret, cost, length = trainer.evaluate(args.eval_episodes)
            normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        all_metrics.append((ret, normalized_ret, cost, normalized_cost, length))
        print(
            f"Eval reward: {ret}, normalized reward: {normalized_ret}; cost: {cost}, normalized cost: {normalized_cost}; length: {length}"
        )
        # Compute statistics for groups
    group_names = ["idbf", "ccbf", "cbf"]

    for idx, group in enumerate(range(0, 9, 3)):
        group_metrics = np.array(all_metrics[group:group+3])  # Extract the subset
        mean_metrics = np.mean(group_metrics, axis=0)
        var_metrics = np.std(group_metrics, axis=0)

        group_name = group_names[idx]  # Assign group name
        
        print(f"\nGroup {group_name} (Models {group+1}-{group+3}):")
        print(f"  Avg Normalized Reward: {mean_metrics[1]:.4f}, std: {var_metrics[1]:.4f}")
        print(f"  Avg Normalized Cost: {mean_metrics[3]:.4f}, std: {var_metrics[3]:.4f}")
        print(f"  Avg Length: {mean_metrics[4]:.4f}, std: {var_metrics[4]:.4f}")

if __name__ == "__main__":
    
    eval()
#python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/eval/eval_bc.py" --device="mps" --path="/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflinePointGoal1Gymnasium-v0-cost-80/BC-safe_bc_modesafe_cost80_seed10-60dd" 

#python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_bc_modif.py" --device="mps" --path="/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflinePointGoal1Gymnasium-v0-cost-80/BC-safe_bc_modesafe_cost80_seed10-60dd" --device="mps" --eval_episodes 5

# python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_bc_cbf.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78" --eval_episode 1





#python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_bc_cbf_modif_list.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78" --eval_episode 50 --device mps
#python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_bc_cbf_modif_list.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180" --eval_episode 50 --device mps

#python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_bc_cbf_modif_list.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-0912"  --eval_episode 50 --device mps
#python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_bc_cbf_modif_list.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-70bc" --eval_episode 50 --device mps



'''
python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/hopper_random/eval_bc_cbf_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-0912"  --eval_episode 20 --device cpu

hopper bc 
Group idbf (Models 1-3):
  Avg Normalized Reward: 0.3266, std: 0.2379
  Avg Normalized Cost: 4.7075, std: 2.2583
  Avg Length: 388.0167, std: 277.1884

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.0377, std: 0.0042
  Avg Normalized Cost: 0.0900, std: 0.0629
  Avg Length: 74.1667, std: 5.1274

Group cbf (Models 7-9):
  Avg Normalized Reward: 0.0460, std: 0.0127
  Avg Normalized Cost: 0.2323, std: 0.1817
  Avg Length: 83.4333, std: 14.5257
  
none:
  Avg Normalized Reward: 0.0430, std: 0.02
  Avg Normalized Cost: 0.1925, std: 0.27
  Avg Length: 78.3833
  

python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/hopper_random/eval_bc_cbf_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-70bc" --eval_episode 20 --device cpu

hopper bc safe
Group idbf (Models 1-3):
  Avg Normalized Reward: 0.5583, std: 0.0268
  Avg Normalized Cost: 0.4833, std: 0.2195
  Avg Length: 901.1833, std: 54.9389

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.5583, std: 0.0484
  Avg Normalized Cost: 0.0467, std: 0.0344
  Avg Length: 897.0833, std: 83.1093

Group cbf (Models 7-9)
  Avg Normalized Reward: 0.6093, std: 0.0145
  Avg Normalized Cost: 0.1425, std: 0.1006
  Avg Length: 970.1333, std: 28.0180
none:#none means just set the use cbf param to false and check the policy without cbf how it does

  Avg Normalized Reward: 0.5664, std: 0.0026
  Avg Normalized Cost: 0.0308, std: 0.020
  Avg Length: 910.7833, 
  
  '''