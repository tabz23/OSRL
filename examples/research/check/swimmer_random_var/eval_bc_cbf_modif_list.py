from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import dsrl
import numpy as np
import pyrallis
import torch
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
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
    model_paths = [
        ######final chosen ones
        ######final chosen ones
        ##idbf
        "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_269/combined_model.pth",
        "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_481/combined_model.pth",
        "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_576/combined_model.pth",
        # #CCBF
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
        "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##gr8 bc all
        "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_932/combined_model.pth",##gr8 bc all #gr8 for bc safe
        "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_709/combined_model.pth",##gr8 bc all #greatt for bc safe
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##gr8 bc all
        # ##CBF
        "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_933/combined_model.pth",
        "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_278/combined_model.pth",
        "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_619/combined_model.pth",     
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_626/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_758/combined_model.pth",


        # model paths from compute
        #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_829/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_354/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_795/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_863/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_146/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_758/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_126/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_226/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_640/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_569/combined_model.pth",
        # # Model paths from the attached document
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_592/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_202/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_456/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_847/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_666/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_465/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_321/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_353/combined_model_laststep.pth",###good bc all
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_188/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_878/combined_model.pth",##good bc all
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_262/combined_model_laststep.pth",#gr8 for bc safe
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_743/combined_model_laststep.pth",##gr8 bc all
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_266/combined_model.pth",##gr8 bc all
        # # Original model paths
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_984/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_932/combined_model.pth",##gr8 bc all #gr8 for bc safe
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_709/combined_model.pth",##gr8 bc all #greatt for bc safe
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##gr8 bc all
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_344/combined_model.pth",##gr8 bc all
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_198/combined_model.pth",##gr8 bc all
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_933/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_982/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_415/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_400/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_147/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_217/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_938/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_626/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_199/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_873/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_564/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_541/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_825/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_614/combined_model.pth"
     ]
    # model_paths = [
    # ####final ones below
    #   #idbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_271/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_209/combined_model.pth",
    # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_217/combined_model.pth",
    #   #CBF
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_861/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_925/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_266/combined_model.pth",

# # # 
# 
# ##Appended Hopper model paths from compute
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_668/combined_model_laststep.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_357/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_662/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_161/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_346/combined_model_laststep.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_988/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_217/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_492/combined_model_laststep.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_741/combined_model_laststep.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",

#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_181/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_287/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_773/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_215/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_948/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_139/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_861/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_213/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_978/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_688/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_188/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_352/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_421/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_925/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_820/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_107/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_920/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_658/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_387/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_266/combined_model.pth"
# ]
    for i, model_path in enumerate(model_paths):
        print(f"\n[{i+1}/{len(model_paths)}] Evaluating model: {os.path.basename(os.path.dirname(model_path))}")
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
        env = wrap_env(env = gym.make("OfflineSwimmerVelocityGymnasium-v1"))
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
    for idx, group in enumerate(range(0, 9, 3)):
        group_metrics = np.array(all_metrics[group:group+3])  # Extract the subset
        mean_metrics = np.mean(group_metrics, axis=0)
        var_metrics = np.var(group_metrics, axis=0)
        
        print(f"\nGroup {idx+1} (Models {group+1}-{group+3}):")
        print(f"  Avg Reward: {mean_metrics[0]:.4f}, Var: {var_metrics[0]:.4f}")
        print(f"  Avg Normalized Reward: {mean_metrics[1]:.4f}, Var: {var_metrics[1]:.4f}")
        print(f"  Avg Cost: {mean_metrics[2]:.4f}, Var: {var_metrics[2]:.4f}")
        print(f"  Avg Normalized Cost: {mean_metrics[3]:.4f}, Var: {var_metrics[3]:.4f}")
        print(f"  Avg Length: {mean_metrics[4]:.4f}, Var: {var_metrics[4]:.4f}")


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
(osrl) (base) i.k.tabbara@1J000AL-FYYVJW7 python directory % python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/swimmer_random/eval_bc_cbf_modif_list.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567"  --eval_episode 20 --device cpu


[1/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_269
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/checkpoint/model_best.pt
loaded cbf
Eval reward: 95.84374502142552, normalized reward: 0.40091144735424644; cost: 116.55, normalized cost: 5.8275; length: 1000.0                        

[2/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_481
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/checkpoint/model_best.pt
loaded cbf
Eval reward: 112.72422902539884, normalized reward: 0.4715744474496822; cost: 104.4, normalized cost: 5.220000000000001; length: 1000.0              

[3/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_576
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/checkpoint/model_best.pt
loaded cbf
Eval reward: 110.97327737487652, normalized reward: 0.46424482961296376; cost: 148.75, normalized cost: 7.4375; length: 1000.0                       

[4/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_784
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/checkpoint/model_best.pt
loaded cbf
Eval reward: 89.85100634508385, normalized reward: 0.37582538385283387; cost: 26.75, normalized cost: 1.3375; length: 1000.0                         

[5/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_932
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/checkpoint/model_best.pt
loaded cbf
Eval reward: 70.48523113406024, normalized reward: 0.2947587642427182; cost: 17.9, normalized cost: 0.8949999999999999; length: 1000.0               

[6/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_709
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/checkpoint/model_best.pt
loaded cbf
Eval reward: 74.56060217055264, normalized reward: 0.31181857981867056; cost: 16.4, normalized cost: 0.82; length: 1000.0                            

[7/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_933
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/checkpoint/model_best.pt
loaded cbf
Eval reward: 114.94672165084273, normalized reward: 0.4808779719556668; cost: 152.2, normalized cost: 7.609999999999999; length: 1000.0              

[8/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_278
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/checkpoint/model_best.pt
loaded cbf
Eval reward: 94.03604626951795, normalized reward: 0.393344281800907; cost: 47.85, normalized cost: 2.3925; length: 1000.0                           

[9/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_619
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567/checkpoint/model_best.pt
loaded cbf
Eval reward: 135.15300471019722, normalized reward: 0.5654630218520729; cost: 116.7, normalized cost: 5.835; length: 1000.0                          

Group 1 (Models 1-3):
  Avg Reward: 106.5138, Var: 57.4355
  Avg Normalized Reward: 0.4456, Var: 0.0010
  Avg Cost: 123.2333, Var: 350.1539
  Avg Normalized Cost: 6.1617, Var: 0.8754
  Avg Length: 1000.0000, Var: 0.0000

Group 2 (Models 4-6):
  Avg Reward: 78.2989, Var: 69.4932
  Avg Normalized Reward: 0.3275, Var: 0.0012
  Avg Cost: 20.3500, Var: 20.8550
  Avg Normalized Cost: 1.0175, Var: 0.0521
  Avg Length: 1000.0000, Var: 0.0000

Group 3 (Models 7-9):
  Avg Reward: 114.7119, Var: 281.7949
  Avg Normalized Reward: 0.4799, Var: 0.0049
  Avg Cost: 105.5833, Var: 1876.6106
  Avg Normalized Cost: 5.2792, Var: 4.6915
  Avg Length: 1000.0000, Var: 0.0000
  
  
  
                                                                                                                                                       
Group 1 (Models 1-3):
  Avg Reward: 104.2298, Var: 63.6504
  Avg Normalized Reward: 0.4360, Var: 0.0011
  Avg Cost: 45.1333, Var: 166.1039
  Avg Normalized Cost: 2.2567, Var: 0.4153
  Avg Length: 1000.0000, Var: 0.0000
  
  
  
  
  
  (osrl) (base) i.k.tabbara@1J000AL-FYYVJW7 python directory % python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/swimmer_random/eval_bc_cbf_modif_list.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180" --eval_episode 20 --device cpu

[1/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_269
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/checkpoint/model_best.pt
loaded cbf
Eval reward: 110.38073662244787, normalized reward: 0.46176440859923223; cost: 4.9, normalized cost: 0.24500000000000002; length: 1000.0                       

[2/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_481
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/checkpoint/model_best.pt
loaded cbf
Eval reward: 130.27237406635675, normalized reward: 0.5450323278287078; cost: 4.7, normalized cost: 0.23500000000000001; length: 1000.0                        

[3/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_576
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/checkpoint/model_best.pt
loaded cbf
Eval reward: 114.78055632167616, normalized reward: 0.4801823911495719; cost: 7.4, normalized cost: 0.37; length: 1000.0                                       

[4/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_784
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/checkpoint/model_best.pt
loaded cbf
Eval reward: 103.32507752552206, normalized reward: 0.43222887873397026; cost: 0.75, normalized cost: 0.0375; length: 1000.0                                   

[5/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_932
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/checkpoint/model_best.pt
loaded cbf
Eval reward: 114.49881190913375, normalized reward: 0.4790029874401944; cost: 1.25, normalized cost: 0.0625; length: 1000.0                                    

[6/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_709
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/checkpoint/model_best.pt
loaded cbf
Eval reward: 97.11679869980298, normalized reward: 0.40624054764399564; cost: 1.35, normalized cost: 0.0675; length: 1000.0                                    

[7/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_933
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/checkpoint/model_best.pt
loaded cbf
Eval reward: 106.7214859189889, normalized reward: 0.44644650463706614; cost: 3.45, normalized cost: 0.17250000000000001; length: 1000.0                       

[8/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_278
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/checkpoint/model_best.pt
loaded cbf
Eval reward: 105.33052106145028, normalized reward: 0.4406238191122989; cost: 3.35, normalized cost: 0.1675; length: 1000.0                                    

[9/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_619
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180/checkpoint/model_best.pt
loaded cbf
Eval reward: 118.2002452789086, normalized reward: 0.4944974712772743; cost: 0.25, normalized cost: 0.0125; length: 1000.0                                     

Group 1 (Models 1-3):
  Avg Reward: 118.4779, Var: 72.7813
  Avg Normalized Reward: 0.4957, Var: 0.0013
  Avg Cost: 5.6667, Var: 1.5089
  Avg Normalized Cost: 0.2833, Var: 0.0038
  Avg Length: 1000.0000, Var: 0.0000

Group 2 (Models 4-6):
  Avg Reward: 104.9802, Var: 51.7255
  Avg Normalized Reward: 0.4392, Var: 0.0009
  Avg Cost: 1.1167, Var: 0.0689
  Avg Normalized Cost: 0.0558, Var: 0.0002
  Avg Length: 1000.0000, Var: 0.0000

Group 3 (Models 7-9):
  Avg Reward: 110.0841, Var: 33.2585
  Avg Normalized Reward: 0.4605, Var: 0.001
  Avg Cost: 2.3500, Var: 2.2067
  Avg Normalized Cost: 0.1175, Var: 0.0055
  Avg Length: 1000.0000, Var: 0.0000
  
  
  
  Group 1 (Models 1-3):
  Avg Reward: 103.1262, Var: 0.6326
  Avg Normalized Reward: 0.4314, Var: 0.001
  Avg Cost: 2.3667, Var: 1.1939
  Avg Normalized Cost: 0.1183, Var: 0.0030
  Avg Length: 1000.0000, Var: 0.0000
  '''