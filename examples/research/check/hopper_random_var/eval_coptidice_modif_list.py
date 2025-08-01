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
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field
import sys
import os
from coptidice_modif_cbf_list import COptiDICE, COptiDICETrainer
from osrl.common.exp_util import load_config_and_model, seed_all


@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    noise_scale: List[float] = None
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
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_269/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_481/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_576/combined_model.pth",
    #     #CCBF
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
    
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##gr8 bc all
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_932/combined_model.pth",##gr8 bc all #gr8 for bc safe
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_709/combined_model.pth",##gr8 bc all #greatt for bc safe
    
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##gr8 bc all
    #     ##CBF
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_933/combined_model.pth",
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_984/combined_model.pth",
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_188/combined_model.pth",       
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_626/combined_model.pth",
    #     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_758/combined_model.pth",


        
        # ##from the compute:
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
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
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_569/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_592/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_202/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_456/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_847/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_666/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_465/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_321/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_353/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_188/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_878/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_262/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_743/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_266/combined_model.pth",
        # # Original model paths
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_984/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_932/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_709/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_344/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_198/combined_model.pth",
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
    # ]
    model_paths = [    
    #idbf
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_271/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_209/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_693/combined_model.pth",
    # ##ccbf
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_217/combined_model.pth",
    #   #CBF
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_861/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_799/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_159/combined_model.pth",   #CHANGE       
                   
                   
    ####final ones below
    #   #idbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_271/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_209/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_209/combined_model.pth",
    # # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
    # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_217/combined_model.pth",
    # #   #CBF
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_861/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_925/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_266/combined_model.pth",

        ## Appended Hopper model paths from compute
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_668/combined_model_laststep.pth",
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
]
    for i, model_path in enumerate(model_paths):
        cfg, model = load_config_and_model(args.path, args.best)
        print(f"\n[{i+1}/{len(model_paths)}] Evaluating model: {os.path.basename(os.path.dirname(model_path))}")
        
        # Generate hyperparameter path from model path
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

        env = wrap_env(
        env=gym.make("OfflineHopperVelocityGymnasium"),
            reward_scale=cfg["reward_scale"],
        )
        # env = OfflineEnvWrapper(env)
        env.set_target_cost(cfg["cost_limit"])



        # setup model
        coptidice_model = COptiDICE(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            f_type=cfg["f_type"],
            init_state_propotion=1.0,
            observations_std=np.array([0]),
            actions_std=np.array([0]),
            a_hidden_sizes=cfg["a_hidden_sizes"],
            c_hidden_sizes=cfg["c_hidden_sizes"],
            gamma=cfg["gamma"],
            alpha=cfg["alpha"],
            cost_ub_epsilon=cfg["cost_ub_epsilon"],
            num_nu=cfg["num_nu"],
            num_chi=cfg["num_chi"],
            cost_limit=cfg["cost_limit"],
            episode_len=cfg["episode_len"],
            device=args.device,
        )
        coptidice_model.load_state_dict(model["model_state"])
        coptidice_model.to(args.device)
        trainer = COptiDICETrainer(coptidice_model,
                                env,
                                reward_scale=cfg["reward_scale"],
                                cost_scale=cfg["cost_scale"],
                                device=args.device,
                                model_path=model_path,
                                hyperparams_path=hyperparam_paths)

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



# #### python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed10-09ab/COptiDICE_cost20_seed10-09ab" --eval_episodes 1

# python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-11cb/COptiDICE_cost20_seed20-11cb" --eval_episodes 1
# Eval reward: 59.060403111086195, normalized reward: 0.24693355831333091; cost: 2.74, normalized cost: 0.137; length: 1000.0

# python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20-3187/COptiDICE_cost20-3187" --eval_episodes 1
# Eval reward: 83.84573596036901, normalized reward: 0.35068686165949287; cost: 58.048, normalized cost: 2.9024; length: 1000.0 






# python directory % python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3"  --eval_episodes  50 


'''(osrl) (base) i.k.tabbara@1J000AL-FYYVJW7 python directory % python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/hopper_random/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3"  --eval_episodes 20
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt

[1/9] Evaluating model: OfflineHopperVelocityGymnasium-v1_271
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt
/Users/i.k.tabbara/miniconda3/envs/osrl/lib/python3.9/site-packages/gymnasium/envs/registration.py:531: UserWarning: WARN: Using the latest versioned environment `OfflineHopperVelocityGymnasium-v1` instead of the unversioned environment `OfflineHopperVelocityGymnasium`.
  logger.warn(
loaded cbf
Eval reward: 412.8090586409179, normalized reward: 0.200474018629925; cost: 17.35, normalized cost: 0.8675; length: 288.85                                     
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt

[2/9] Evaluating model: OfflineHopperVelocityGymnasium-v1_209
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt
loaded cbf
Eval reward: 531.7860290010944, normalized reward: 0.2639505711866477; cost: 53.9, normalized cost: 2.695; length: 354.4                                       
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt

[3/9] Evaluating model: OfflineHopperVelocityGymnasium-v1_693
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt
loaded cbf
Eval reward: 382.67225705700787, normalized reward: 0.18439544255052523; cost: 9.0, normalized cost: 0.45; length: 279.9                                       
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt

[4/9] Evaluating model: OfflineHopperVelocityGymnasium-v1_848
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt
loaded cbf
Eval reward: 353.0278405040705, normalized reward: 0.1685795635645383; cost: 0.0, normalized cost: 0.0; length: 252.75                                         
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt

[5/9] Evaluating model: OfflineHopperVelocityGymnasium-v1_703
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt
loaded cbf
Eval reward: 376.30916816784986, normalized reward: 0.18100060954147318; cost: 0.05, normalized cost: 0.0025; length: 271.45                                   
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt

[6/9] Evaluating model: OfflineHopperVelocityGymnasium-v1_887
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt
loaded cbf
Eval reward: 372.1093926752116, normalized reward: 0.17875994675624798; cost: 0.1, normalized cost: 0.005; length: 267.1                                       
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt

[7/9] Evaluating model: OfflineHopperVelocityGymnasium-v1_861
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt
loaded cbf
Eval reward: 391.32057667684995, normalized reward: 0.1890094910778526; cost: 15.05, normalized cost: 0.7525000000000001; length: 273.3                        
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt

[8/9] Evaluating model: OfflineHopperVelocityGymnasium-v1_799
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt
loaded cbf
Eval reward: 293.5867934885413, normalized reward: 0.1368665964795167; cost: 52.0, normalized cost: 2.6; length: 180.95                                        
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt

[9/9] Evaluating model: OfflineHopperVelocityGymnasium-v1_159
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3/checkpoint/model_best.pt
loaded cbf
Eval reward: 354.86500057174294, normalized reward: 0.16955972457584736; cost: 0.85, normalized cost: 0.042499999999999996; length: 255.35                     

Group 1 (Models 1-3):
  Avg Reward: 442.4224, Var: 4144.2959
  Avg Normalized Reward: 0.2163, Var: 0.0012
  Avg Cost: 26.7500, Var: 380.1817
  Avg Normalized Cost: 1.3375, Var: 0.9505
  Avg Length: 307.7167, Var: 1103.0172

Group 2 (Models 4-6):
  Avg Reward: 367.1488, Var: 102.6404
  Avg Normalized Reward: 0.1761, Var: 0.0000
  Avg Cost: 0.0500, Var: 0.0017
  Avg Normalized Cost: 0.0025, Var: 0.0000
  Avg Length: 263.7667, Var: 63.8372

Group 3 (Models 7-9):
  Avg Reward: 346.5908, Var: 1626.2133
  Avg Normalized Reward: 0.1651, Var: 0.0005
  Avg Cost: 22.6333, Var: 464.8072
  Avg Normalized Cost: 1.1317, Var: 1.1620
  Avg Length: 236.5333, Var: 1598.4539
  
  
  
  Group 1 (Models 1-3):
  Avg Reward: 367.6559, Var: 45.9095
  Avg Normalized Reward: 0.1764, Var: 0.0000
  Avg Cost: 0.1000, Var: 0.0067
  Avg Normalized Cost: 0.0050, Var: 0.0000
  Avg Length: 264.1667, Var: 30.7372
  '''