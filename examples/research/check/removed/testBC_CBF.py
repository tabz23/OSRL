from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import dsrl
import numpy as np
import pyrallis
import torch
from pyrallis import field

from osrl.algorithms import BC, BCTrainer
from osrl.common.exp_util import load_config_and_model, seed_all
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import safety_gymnasium as safety_gym
from torch.utils.data import DataLoader
from osrl.common.exp_util import seed_all
from IPython.display import display, clear_output
@dataclass
# /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed20-4653/BC-all_cost20_seed20-4653
# /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78
# /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567
#/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180
class EvalConfig:
    path: str = "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed20-4653/BC-all_cost20_seed20-4653"
    noise_scale: List[float] = None
    costs: List[float] = field(default=[1, 10, 20, 30, 40], is_mutable=True)
    eval_episodes: int = 20
    best: bool = False
    device: str = "cpu"
    threads: int = 4


@pyrallis.wrap()
def eval(args: EvalConfig):
    
    cfg, model = load_config_and_model(args.path, args.best)
    # seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    if "Metadrive" in cfg["task"]:
        import gym
    else:
        import gymnasium as gym  # noqa

    env = gym.make(cfg["task"])
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
                        device=args.device)
    

  
 

    # Assuming these imports are correct and the modules are available
    from network_ihab import CombinedCBFDynamics
    from dataset_ihab import TransitionDataset

    # Load model and hyperparameters
    model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_847/combined_model.pth"
    hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_847/hyperparameters.json"


    # Set fixed seed for reproducibility
    # seed_all(115)

    # Load hyperparameters
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)

    # Initialize and load the model
    combinedcbfdynamics = CombinedCBFDynamics(
        num_action=hyperparams['num_action'],
        state_dim=hyperparams['state_dim'],
        cbf_hidden_dim=hyperparams['cbf_hidden_dim'],
        dynamics_hidden_dim=hyperparams['dynamics_hidden_dim'],
        cbf_num_layers=hyperparams['cbf_num_layers'],
        dynamics_num_layers=hyperparams['dynamics_num_layers'],
        dt=hyperparams['dt']
    )

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    combinedcbfdynamics.load_state_dict(checkpoint)
    combinedcbfdynamics.eval()
    print("Model loaded successfully!")

    # Initialize environment and get dataset from gymnasium
    std_env = gym.make('OfflineSwimmerVelocityGymnasium-v1')
    dataset = std_env.get_dataset()
    std_env.close()

    # Create a dataset and dataloader
    dataset = TransitionDataset(dataset, split='val')
    dataloader = DataLoader(dataset, batch_size=128)

    # Get a batch of data
    batch = next(iter(dataloader))
    observations, next_observations, actions, _, costs, done = [b.to(torch.float32) for b in batch]

    # Initialize data storage for plotting
    cbf_values = []
    cost_values = []
    steps = []
    all_observations = []
    episode_count = 0
    max_episodes = 1

    # Create the Safety Gymnasium environment
    env = safety_gym.make('SafetySwimmerVelocity-v1', render_mode=None)
    observation, info = env.reset()#seed=115
    # Start the environment

    all_observations.append(observation)

    print("Initial observation:", observation)

    while episode_count < max_episodes:
        for step_index in range(len(actions)):
            

            # Compute CBF value
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            with torch.no_grad():
                cbf_value = combinedcbfdynamics.forward_cbf(obs_tensor).item()

            # Step in the environment
            
            action = trainer.model.act(observation)   ###ACTION FROM BC POLICY
            # action = actions[step_index].numpy()###ACTION FROM DATASET
            next_obs, reward, cost, terminated, truncated, info = env.step(action)
        # obs_next, reward, terminated, truncated, info 
            all_observations.append(next_obs)
##NEW CBF NOW
            # print(info)
            # Store history
            cbf_values.append(cbf_value)
            cost_values.append(cost)
            steps.append(step_index)


            # Update observation for the next step
            observation = next_obs

            if terminated or truncated:
                print("Episode terminated early")
                break

        episode_count += 1
        print(f"Episode {episode_count} finished!")

    # Close the environment
    env.close()

    # ... (previous code remains the same)

    # After closing the environment and before saving results

    # Create and display overlapping plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, cbf_values, label='CBF Values', color='blue')
    plt.plot(steps, cost_values, label='Cost Values', color='red')
    plt.title('CBF Values and Cost Values over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # ... (rest of the code remains the same)

    # Save results for comparison (optional)
    results = {
        'steps': steps,
        'cbf_values': cbf_values,
        'cost_values': cost_values,
        'observations': all_observations
    }

    # Optionally save the results
    import pickle
    with open('safety_gymnasium_data.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Results saved for comparison.")





if __name__ == "__main__" and "ipykernel" not in __import__("sys").argv[0]:
    eval()
    # main()