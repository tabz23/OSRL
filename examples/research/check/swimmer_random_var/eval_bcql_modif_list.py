from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import importlib
import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field
import sys
import os
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from bcql_modif_cbf_list import BCQL, BCQLTrainer
from osrl.common.exp_util import load_config_and_model, seed_all


@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"  # This will be ignored in multi-model mode
    noise_scale: List[float] = None
    eval_episodes: int = 20
    best: bool = True  # Previously changed to TRUE
    device: str = "cpu"
    threads: int = 4
    # New field for multi-model evaluation (only need model paths)
   


@pyrallis.wrap()
def eval(args: EvalConfig):
    all_metrics = []
    # Use the model_paths if provided
    
    model_paths = [
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


        ##from the compute:
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
    ]
    # model_paths = [
    
    # ####final ones below
    # #idbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_271/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_209/combined_model.pth",
    # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_217/combined_model.pth",
    # #CBF
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_861/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_925/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_266/combined_model.pth",


#         # Appended Hopper model paths from compute
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

        # env = wrap_env(
        #     env=gym.make("SafetySwimmerVelocityGymnasium-v1", render_mode="human"),## ADDED THIS changed this to "SafetyCarGoal1Gymnasium-v0" from OfflineCarGoal1Gymnasium-v0 as mentioned in yaml file
        #     reward_scale=cfg["reward_scale"],                                   ##changed this from OfflinePointGoal1Gymnasium-v0 as in yaml file to SafetyPointGoal1Gymnasium-v0
        # )
        env = wrap_env(
            env=gym.make("OfflineSwimmerVelocityGymnasium-v1"),## ADDED THIS changed this to "SafetyCarGoal1Gymnasium-v0" from OfflineCarGoal1Gymnasium-v0 as mentioned in yaml file
            reward_scale=cfg["reward_scale"],                                   ##changed this from OfflinePointGoal1Gymnasium-v0 as in yaml file to SafetyPointGoal1Gymnasium-v0
        )

        # env = OfflineEnvWrapper(env)
        
        env.set_target_cost(cfg["cost_limit"])

        bcql_model = BCQL(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            a_hidden_sizes=cfg["a_hidden_sizes"],
            c_hidden_sizes=cfg["c_hidden_sizes"],
            vae_hidden_sizes=cfg["vae_hidden_sizes"],
            sample_action_num=cfg["sample_action_num"],
            PID=cfg["PID"],
            gamma=cfg["gamma"],
            tau=cfg["tau"],
            lmbda=cfg["lmbda"],
            beta=cfg["beta"],
            phi=cfg["phi"],
            num_q=cfg["num_q"],
            num_qc=cfg["num_qc"],
            cost_limit=cfg["cost_limit"],
            episode_len=cfg["episode_len"],
            device=args.device,
        )
        bcql_model.load_state_dict(model["model_state"])
        bcql_model.to(args.device)
        

        trainer = BCQLTrainer(bcql_model,
                            env,
                            reward_scale=cfg["reward_scale"],
                            cost_scale=cfg["cost_scale"],
                            device=args.device,
                            model_path=model_path,
                            hyperparams_path=hyperparam_paths
                            )

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
    # Define model paths

    # Create an EvalConfig object with just the model paths

    
    # Run evaluation
    eval()
    print("All evaluations completed")
    
# python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_bcql_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed10-ff8a/BCQL_cost20_seed10-ff8a" --eval_episodes 50

# python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_bcql_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-257f" --eval_episodes 50



'''
(osrl) (base) i.k.tabbara@1J000AL-FYYVJW7 python directory % python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/swimmer_random/eval_bcql_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5" --eval_episodes 20

[1/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_269
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/checkpoint/model_best.pt
loaded cbf

Eval reward: 91.56125691150184, normalized reward: 0.38298462383135395; cost: 160.1, normalized cost: 8.004999999999999; length: 1000.0                        

[2/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_481
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/checkpoint/model_best.pt
loaded cbf

Eval reward: 105.63596830075292, normalized reward: 0.44190244467154904; cost: 222.7, normalized cost: 11.135; length: 1000.0                                  

[3/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_576
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/checkpoint/model_best.pt
loaded cbf

Eval reward: 99.68494609466036, normalized reward: 0.4169910095125763; cost: 231.65, normalized cost: 11.5825; length: 1000.0                                  

[4/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_784
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/checkpoint/model_best.pt
loaded cbf

Eval reward: 73.35219005936491, normalized reward: 0.3067600740845882; cost: 39.25, normalized cost: 1.9625; length: 1000.0                                    

[5/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_932
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/checkpoint/model_best.pt
loaded cbf

Eval reward: 54.444914510103764, normalized reward: 0.22761276920711; cost: 19.3, normalized cost: 0.9650000000000001; length: 1000.0                          

[6/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_709
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/checkpoint/model_best.pt
loaded cbf

Eval reward: 66.0998912078459, normalized reward: 0.2764013951322316; cost: 39.25, normalized cost: 1.9625; length: 1000.0                                     

[7/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_933
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/checkpoint/model_best.pt
loaded cbf

Eval reward: 42.13494150455247, normalized reward: 0.17608227852560257; cost: 50.45, normalized cost: 2.5225; length: 1000.0                                   

[8/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_278
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/checkpoint/model_best.pt
loaded cbf

Eval reward: 65.50150397483382, normalized reward: 0.2738965003025877; cost: 108.7, normalized cost: 5.4350000000000005; length: 1000.0                        

[9/9] Evaluating model: OfflineSwimmerVelocityGymnasium-v1_619
load config from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/config.yaml
load model from /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BCQL_cost20_seed20-b8c5/BCQL_cost20_seed20-b8c5/checkpoint/model_best.pt
loaded cbf

Eval reward: 69.7623837034511, normalized reward: 0.29173286948483707; cost: 142.65, normalized cost: 7.1325; length: 1000.0                                   

Group 1 (Models 1-3):
  Avg Reward: 98.9607, Var: 33.2785
  Avg Normalized Reward: 0.4140, Var: 0.0006
  Avg Cost: 204.8167, Var: 1013.1406
  Avg Normalized Cost: 10.2408, Var: 2.5329
  Avg Length: 1000.0000, Var: 0.0000

Group 2 (Models 4-6):
  Avg Reward: 64.6323, Var: 60.6577
  Avg Normalized Reward: 0.2703, Var: 0.0011
  Avg Cost: 32.6000, Var: 88.4450
  Avg Normalized Cost: 1.6300, Var: 0.2211
  Avg Length: 1000.0000, Var: 0.0000

Group 3 (Models 7-9):
  Avg Reward: 59.1329, Var: 147.4919
  Avg Normalized Reward: 0.2472, Var: 0.0026
  Avg Cost: 100.6000, Var: 1449.6117
  Avg Normalized Cost: 5.0300, Var: 3.6240
  Avg Length: 1000.0000, Var: 0.0000
  
  
  
  
  
  
  
  Group 1 (Models 1-3):
  Avg Reward: 58.7421, Var: 413.3399
  Avg Normalized Reward: 0.2456, Var: 0.0072
  Avg Cost: 56.3333, Var: 531.5822
  Avg Normalized Cost: 2.8167, Var: 1.3290
  Avg Length: 1000.0000, Var: 0.0000
All evaluations completed
All evaluations completed'''