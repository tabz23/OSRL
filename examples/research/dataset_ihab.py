from torch.utils.data import DataLoader, Dataset
import torch
import gymnasium as gym
import dsrl
import numpy as np
from torch.utils.data import IterableDataset
import numpy as np
from torch.utils.data import IterableDataset

class TransitionDataset(IterableDataset):
    """
    A dataset of transitions (state, action, reward, next state) used for training RL agents.
    
    Args:
        dataset (dict): A dictionary of NumPy arrays containing the observations, actions, rewards, etc.
        reward_scale (float): The scale factor for the rewards.
        cost_scale (float): The scale factor for the costs.
        state_init (bool): If True, the dataset will include an "is_init" flag indicating if a transition
            corresponds to the initial state of an episode.
        split (str): One of 'train' or 'val', to specify which subset of the data to use.
        train_ratio (float): Ratio of data to be used for training (0.8 means 80% for training and 20% for validation).
    """

    def __init__(self,
                 dataset: dict,
                 reward_scale: float = 1.0,
                 cost_scale: float = 1.0,
                 state_init: bool = False,
                 split: str = 'train',
                 train_ratio: float = 0.8):
        self.dataset = dataset
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.state_init = state_init
        self.split = split
        self.train_ratio = train_ratio

        self.dataset_size = self.dataset["observations"].shape[0]
        self.train_size = int(self.dataset_size * self.train_ratio)
        
        self.dataset["done"] = np.logical_or(self.dataset["terminals"], 
                                             self.dataset["timeouts"]).astype(np.float32)
        if self.state_init:
            self.dataset["is_init"] = self.dataset["done"].copy()
            self.dataset["is_init"][1:] = self.dataset["is_init"][:-1]
            self.dataset["is_init"][0] = 1.0

        ##shuffle indices before splitting into training and validation set. 
        indices = np.arange(self.dataset_size)
        np.random.shuffle(indices)  # Shuffle the indices

        self.train_indices = indices[:self.train_size]
        self.val_indices = indices[self.train_size:]

        if self.split == 'train':
            self.indices = self.train_indices
        elif self.split == 'val':
            self.indices = self.val_indices

        else:
            raise ValueError("split must be one of 'train' or 'val'")

    def get_dataset_states(self):
        """
        Returns the proportion of initial states in the dataset, 
        as well as the standard deviations of the observation and action spaces.
        """
        init_state_propotion = self.dataset["is_init"].mean()
        obs_std = self.dataset["observations"].std(0, keepdims=True)
        act_std = self.dataset["actions"].std(0, keepdims=True)
        return init_state_propotion, obs_std, act_std

    def __prepare_sample(self, idx):
        observations = self.dataset["observations"][idx, :]
        next_observations = self.dataset["next_observations"][idx, :]
        actions = self.dataset["actions"][idx, :]
        rewards = self.dataset["rewards"][idx] * self.reward_scale
        costs = self.dataset["costs"][idx] * self.cost_scale
        done = self.dataset["done"][idx]
        if self.state_init:
            is_init = self.dataset["is_init"][idx]
            return observations, next_observations, actions, rewards, costs, done, is_init
        return observations, next_observations, actions, rewards, costs, done

    def __iter__(self):
        """
        Iterates over the dataset, yielding samples based on the chosen split.
        """
        while True:
            idx = np.random.choice(self.indices)
            yield self.__prepare_sample(idx)
import numpy as np
from torch.utils.data import DataLoader




def test_TransitionDataset():
    # Create dummy data for testing
    dummy_data = {
        "observations": np.random.randn(100, 4),  # 100 samples, 4 features (e.g., state size)
        "next_observations": np.random.randn(100, 4),
        "actions": np.random.randn(100, 2),  # 100 samples, 2 actions
        "rewards": np.random.randn(100),  # 100 rewards
        "costs": np.random.randn(100),  # 100 costs
        "terminals": np.random.choice([0, 1], size=100),  # 0 or 1 indicating if episode ended
        "timeouts": np.random.choice([0, 1], size=100),  # 0 or 1 indicating if time limit reached
    }

    # Initialize the TransitionDataset for train and validation
    train_dataset = TransitionDataset(dataset=dummy_data, reward_scale=1.0, cost_scale=1.0, state_init=True, split='train')
    val_dataset = TransitionDataset(dataset=dummy_data, reward_scale=1.0, cost_scale=1.0, state_init=True, split='val')

    # Check the size of both datasets
    print(f"Train dataset size: {len(train_dataset.indices)}")
    print(f"Validation dataset size: {len(val_dataset.indices)}")

    # Create DataLoaders for both train and val datasets
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    val_dataloader = DataLoader(val_dataset, batch_size=2)

    # Iterate over the DataLoaders and print a sample batch for both train and val
    print("\nTraining Batch Example:")
    for batch in train_dataloader:
        print("Observations:", batch[0])
        print("Next Observations:", batch[1])
        print("Actions:", batch[2])
        print("Rewards:", batch[3])
        print("Costs:", batch[4])
        print("Done:", batch[5])

        break  # Stop after one batch for testing

    print("\nValidation Batch Example:")
    for batch in val_dataloader:
        print("Observations:", batch[0])
        print("Next Observations:", batch[1])
        print("Actions:", batch[2])
        print("Rewards:", batch[3])
        print("Costs:", batch[4])
        print("Done:", batch[5])

        break  # Stop after one batch for testing
# test_TransitionDataset()














# class SafeGymnasium(Dataset):
#     def __init__(self, env_name, data, split='train', subtraj_len=100, traj_len=1000, **kwargs):
#         # Extracting dimensions
#         action_dim = data['actions'].shape[1]
#         ob_dim = data['observations'].shape[1]
        
#         # Store observations and actions
#         self.observations = data['observations']
#         self.next_observations = data['next_observations']
#         self.actions = data['actions']
        
#         # Split data into training and validation sets
#         num_traj = self.observations.shape[0]
#         num_train = int(num_traj * 0.9)  # 90% for training
        
#         if split == 'train':
#             self.num_traj = num_train
#             self.offset = 0  # No offset for training
#         else:
#             self.num_traj = num_traj - num_train  # 10% for validation
#             self.offset = num_train  # Offset for validation
        
#     def __len__(self):
#         return self.num_traj
    
#     def __getitem__(self, idx):
#         # Return one data sample
#         state = self.observations[self.offset + idx]
#         next_state = self.next_observations[self.offset + idx]
#         action = self.actions[self.offset + idx]
        
#         return torch.tensor(state), torch.tensor(next_state), torch.tensor(action)


# # Train and Validation data loading
# def get_dataloaders(env_name, subtraj_len=100, train_batch_size=16, val_batch_size=8, num_workers=4, pin_memory=True, drop_last=True):
#     # Load dataset once outside the DataLoader to avoid redundant calls
#     env = gym.make(env_name)
#     data = env.get_dataset()  # Load dataset only once here
    
#     # Create training and validation datasets
#     train_dataset = SafeGymnasium(
#         env_name=env_name,
#         data=data,
#         subtraj_len=subtraj_len,
#         split='train',
#     )
    
#     val_dataset = SafeGymnasium(
#         env_name=env_name,
#         data=data,
#         subtraj_len=subtraj_len,
#         split='val',
#     )
    
#     # Create DataLoader for training and validation
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=train_batch_size,
#         num_workers=num_workers,
#         shuffle=True,
#         pin_memory=pin_memory,
#         drop_last=drop_last,
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=val_batch_size,
#         num_workers=num_workers,
#         shuffle=False,
#         pin_memory=pin_memory,
#         drop_last=drop_last,
#     )
    
#     return train_loader, val_loader


# # Example Usage
# if __name__ == '__main__':
#     # Get the dataloaders
#     train_loader, val_loader = get_dataloaders(
#         env_name='OfflineCarGoal2-v0', 
#         subtraj_len=100,
#         train_batch_size=16,
#         val_batch_size=8,
#     )

#     # Iterate through the dataloaders
#     for batch in train_loader:
#         states, next_states, actions = batch
#         print(states.shape, next_states.shape, actions.shape)

#     for batch in val_loader:
#         states, next_states, actions = batch
#         print(states.shape, next_states.shape, actions.shape)
