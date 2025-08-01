import gymnasium as gym
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa
from osrl.common.exp_util import load_config_and_model, seed_all
from osrl.common.net import MLPActor
from network_ihab import CombinedCBFDynamics
from dataset_ihab import TransitionDataset
from gymnasium.utils import RecordConstructorArgs, seeding
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import cvxpy as cp
import json
# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_592/combined_model.pth"
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_592/hyperparameters.json"


# BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt no cbf  1000 runsEval reward: 136.66618086583378, normalized reward: 0.5717972932393737; cost: 134.05, normalized cost: 6.702500000000001; length: 1000.0 

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_202/combined_model.pth" #Eval reward(1000runs): 129.85221880266994, normalized reward: 0.5432735256858895; cost: 201.501, normalized cost: 10.075050000000001; length: 1000.0  #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_202/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_456/combined_model.pth" #Eval reward(100 runs): 127.09601365655662, normalized reward: 0.5317358396540789; cost: 187.62, normalized cost: 9.381; length: 1000.0.  another run where alpha is 10, val reward: 130.00855869466093, normalized reward: 0.5439279764587512; cost: 178.87, normalized cost: 8.9435; length: 1000.0. #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_456/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_847/combined_model.pth" #Eval reward(1000 runs): Eval reward: 102.76241560934672, normalized reward: 0.42987353281930113; cost: 51.94, normalized cost: 2.597; length: 1000.0  #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_847/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_666/combined_model_laststep.pth" #(struggles MASSIVELY with finding optimal) Eval reward: 103.730752132362, normalized reward: 0.4339270637371431; cost: 98.39, normalized cost: 4.9195; length: 1000.0  100 runs #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_666/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_465/combined_model.pth" # Eval reward:100runsEval reward: 129.4644023879244, normalized reward: 0.5416500964438296; cost: 197.58, normalized cost: 9.879000000000001; length: 1000.0  #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_465/hyperparameters.json"  

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_321/combined_model.pth" # Eval reward:100 runs Eval reward: 135.8966823520755, normalized reward: 0.5685761134711668; cost: 210.33, normalized cost: 10.5165; length: 1000.0   runs #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_321/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_353/combined_model_laststep.pth" # Eval reward:100 runs  80.57772286605532, normalized reward: 0.33700670832141216; cost: 18.43, normalized cost: 0.9215; length: 1000.0 #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_353/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_188/combined_model.pth" #NO CQL LOSS(NEED MORE TRAINING) Eval 100 runsEval reward: 139.35309338606856, normalized reward: 0.5830449150272968; cost: 203.97, normalized cost: 10.1985; length: 1000.0  #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_188/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_878/combined_model.pth" #100 runs Eval reward: 107.69505204722573, normalized reward: 0.45052192708326605; cost: 57.02, normalized cost: 2.851; length: 1000.0 #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_878/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_262/combined_model_laststep.pth" #(struggles greatly with finding optimal) 100 runs Eval reward: 85.01959964127265, normalized reward: 0.35560074501592964; cost: 17.8, normalized cost: 0.89; length: 1000.0  #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_262/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_743/combined_model_laststep.pth" # 100 runsEval reward: 85.55359764361344, normalized reward: 0.357836101589917; cost: 19.23, normalized cost: 0.9615; length: 1000.0      #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_743/hyperparameters.json"


# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_266/combined_model.pth" # 100 runs Eval reward: 107.79899795074621, normalized reward: 0.45095705260387664; cost: 63.83, normalized cost: 3.1915; length: 1000.0   #OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20_seed10-3a78/BC-all_cost20_seed10-3a78/checkpoint/model.pt
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_266/hyperparameters.json"
##878 262
#compare 266,743


##below are for safe BC: /Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180
#safe BC alone Eval reward: 115.55801204940916, normalized reward: 0.4834368804254755; cost: 6.703, normalized cost: 0.33515; length: 1000.0

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_202/combined_model.pth" #100 runsEval reward: 104.9022106570007, normalized reward: 0.438830878925076; cost: 12.89, normalized cost: 0.6445000000000001; length: 1000.0  
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_202/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_456/combined_model.pth" #100 runs  Eval reward: 106.30639811322631, normalized reward: 0.44470891526158896; cost: 10.15, normalized cost: 0.5075000000000001; length: 1000.0  
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_456/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_847/combined_model.pth" #Eval reward(100 runs): Eval reward: 85.8117064741224, normalized reward: 0.35891656494100305; cost: 14.86, normalized cost: 0.743; length: 1000.0
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_847/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_666/combined_model_laststep.pth" #100 runs MASSIVELY CANT FIND SOLEval reward: 111.71183401698569, normalized reward: 0.4673364843344493; cost: 79.24, normalized cost: 3.9619999999999997; length: 1000.0
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_666/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_465/combined_model.pth" # 100 runs Eval reward: 110.17100246600927, normalized reward: 0.4608864453421983; cost: 9.8, normalized cost: 0.49000000000000005; length: 1000.0     
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_465/hyperparameters.json"  

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_321/combined_model.pth" #100 runs Eval reward: 102.28394783172367, normalized reward: 0.4278706300229191; cost: 11.74, normalized cost: 0.587; length: 1000.0  
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_321/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_353/combined_model_laststep.pth" #100 runs Eval reward: 77.87331669293897, normalized reward: 0.32568585676146655; cost: 4.86, normalized cost: 0.24300000000000002; length: 1000.0  
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_353/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_188/combined_model.pth" #NO CQL LOSS(NEED MORE TRAINING) Eval reward: 111.02024530427032, normalized reward: 0.4644414409659591; cost: 10.42, normalized cost: 0.521; length: 1000.0
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_188/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_878/combined_model.pth" #100 runs Eval reward: 86.49648026930561, normalized reward: 0.3617830805422733; cost: 14.69, normalized cost: 0.7344999999999999; length: 1000.0 
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_878/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_262/combined_model_laststep.pth" #100 runs Eval reward: 78.4937068937873, normalized reward: 0.32828285770121657; cost: 4.25, normalized cost: 0.2125; length: 1000.0
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_262/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_743/combined_model_laststep.pth" # 100 runs Eval reward: 79.51770898593595, normalized reward: 0.33256940895876314; cost: 4.39, normalized cost: 0.21949999999999997; length: 1000.0  
#                                                                                                                                                      ## ///for alpha = 1.5: Eval reward: 81.5406568620223, normalized reward: 0.3410376238468732; cost: 3.83, normalized cost: 0.1915; length: 1000.0
#                                                                                                                                                      ##///for alpha = 0.8:  Eval reward: 83.27117773422475, normalized reward: 0.34828171687932485; cost: 4.78, normalized cost: 0.23900000000000002; length: 1000.0 
#                                                                                                                                                      ##for alpha=0.5        Eval reward: 86.12528625940713, normalized reward: 0.3602292339611737; cost: 6.05, normalized cost: 0.3025; length: 1000.0  
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_743/hyperparameters.json"

# model_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_266/combined_model.pth" # 100 runs Eval reward: 87.1543135347434, normalized reward: 0.3645368210207732; cost: 12.87, normalized cost: 0.6435; length: 1000.0
# hyperparams_path = "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_266/hyperparameters.json"

##878 262
#compare 266,743



#353 743 gd for both



##126 449 645
class BC(nn.Module):
    """
    Behavior Cloning (BC)
    
    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list, optional): List of integers specifying the sizes 
            of the layers in the actor network.
        episode_len (int, optional): Maximum length of an episode.
        device (str, optional): Device to run the model on (e.g. 'cpu' or 'cuda:0'). 
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 episode_len: int = 300,
                 device: str = "cpu"):

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.a_hidden_sizes = a_hidden_sizes
        self.episode_len = episode_len
        self.device = device

        self.actor = MLPActor(self.state_dim, self.action_dim, self.a_hidden_sizes,
                              nn.ReLU, self.max_action).to(self.device)

    def actor_loss(self, observations, actions):
        pred_actions = self.actor(observations)
        loss_actor = F.mse_loss(pred_actions, actions)
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        stats_actor = {"loss/actor_loss": loss_actor.item()}
        return loss_actor, stats_actor

    def setup_optimizers(self, actor_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def act(self, obs):
        '''
        Given a single obs, return the action.
        '''
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        act = self.actor(obs)
        act = act.data.numpy() if self.device == "cpu" else act.data.cpu().numpy()
        return np.squeeze(act, axis=0)


class BCTrainer:
    """
    Behavior Cloning Trainer
    
    Args:
        model (BC): The BC model to be trained.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        bc_mode (str): specify bc mode
        cost_limit (int): Upper limit on the cost per episode.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """

    def __init__(
            self,
            model: BC,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            actor_lr: float = 1e-4,
            bc_mode: str = "all",
            cost_limit: int = 10,
            device="cpu",
            model_path=None,
            hyperparams_path=None):

        self.model = model
        self.logger = logger
        self.env = env
        self.device = device
        self.bc_mode = bc_mode
        self.cost_limit = cost_limit
        self.model.setup_optimizers(actor_lr)
        
        ###new code ###new code ###new code ###new code ###new code ###new code ###new code
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)


        # Initialize and load the model
        self.combinedcbfdynamics = CombinedCBFDynamics(
            num_action=hyperparams['num_action'],
            state_dim=hyperparams['state_dim'],
            cbf_hidden_dim=hyperparams['cbf_hidden_dim'],
            dynamics_hidden_dim=hyperparams['dynamics_hidden_dim'],
            cbf_num_layers=hyperparams['cbf_num_layers'],
            dynamics_num_layers=hyperparams['dynamics_num_layers'],
            dt=hyperparams['dt']
        )
        self.seed=6
        print("seed: ",self.seed)
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.combinedcbfdynamics.load_state_dict(checkpoint)
        self.combinedcbfdynamics.to(self.device)
        self.combinedcbfdynamics.eval()
        print("loaded cbf")

         ###new code ###new code ###new code ###new code ###new code ###new code ###new code

    def set_target_cost(self, target_cost):
        self.cost_limit = target_cost

    def train_one_step(self, observations, actions):
        """
        Trains the model by updating the actor.
        """
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations, actions)
        self.logger.store(**stats_actor)

    def evaluate(self, eval_episodes):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for step in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(seed=step)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets), np.mean(episode_costs), np.mean(episode_lens)

    
    def rollout(self,seed=None, render=False, plot=False,cbf=True):
        """
        Evaluates the performance of the model on a single episode and optionally plots the results.
        
        Args:
            render (bool): Whether to render the environment during rollout
            plot (bool): Whether to plot the results after rollout
            
        Returns:
            episode_ret (float): Cumulative episode return
            episode_len (int): Episode length
            episode_cost (float): Cumulative episode cost
        """
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        
        # For plotting
        states_history = []
        nominal_actions_history = []
        safe_actions_history = []
        costs_history = []
        cbf_values_history = []

        obs, info = self.env.reset(seed=seed)
        
        if self.bc_mode == "multi-task":
            obs = np.append(obs, self.cost_limit)
        
        for step in range(self.model.episode_len):
            # Get nominal action from BC policy
            nominal_action = self.model.act(obs)
            if cbf:
                # Convert to tensor for CBF calculations
                current_state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                # print(current_state_tensor.shape)
                current_state_tensor.requires_grad_(True)
                    
                # Get CBF value
                h_x_tensor = self.combinedcbfdynamics.forward_cbf(current_state_tensor)
                h_x = h_x_tensor.item()  # Scalar CBF value
                
                # Get gradient of CBF with respect to state
                gradient_B = torch.autograd.grad(h_x_tensor, current_state_tensor)[0]
                        
                # Get dynamics
                f, g = self.combinedcbfdynamics.forward_dynamics(current_state_tensor)
                
                # Calculate LHS of CBF constraint: ∇B(x)f(x)
                cbf_lie_derivative = torch.einsum("bs,bs->b", f, gradient_B)
                
                # Add gamma(B(x)) term
                right_side = cbf_lie_derivative + h_x_tensor.squeeze(-1)###CHANGE ALPHA HERE CHANGE ALPHA HERE.  LOW ALPHA->SAFER, WORSE REWARD
                right_numpy = right_side.detach().cpu().numpy()
                
                # Calculate ∇B(x)g(x) for the control input term
                grad_b_g = torch.einsum(
                    'bs,bsa->ba', 
                    gradient_B, 
                    g.view(g.shape[0], self.combinedcbfdynamics.state_dim, self.combinedcbfdynamics.num_action)
                )
                grad_b_g_numpy = grad_b_g.detach().cpu().numpy()
                
                # Solve QP to find safe action
                u = cp.Variable(self.combinedcbfdynamics.num_action)
                objective = cp.Minimize(cp.sum_squares(u - nominal_action))
                constraints = [grad_b_g_numpy @ u >= -right_numpy,
                            # u<=1, u>=-1
                            ]
                
                prob = cp.Problem(objective, constraints)
                
                try:
                    # Solve using OSQP solver
                    prob.solve(solver=cp.OSQP)
                    
                    if prob.status in ["optimal", "optimal_inaccurate"]:
                        safe_action = u.value
                    else:
                        # print(f"Warning: QP at step {step} could not find optimal solution (status: {prob.status}), using nominal action")##I COMMENTED THIS OUT FOR NOW
                        safe_action = nominal_action
                except Exception as e:
                    print(f"Error in QP solver at step {step}: {e}")
                    safe_action = nominal_action
                
            # Store data for plotting
            states_history.append(obs.copy())
            nominal_actions_history.append(nominal_action.copy())
            if (cbf):
                safe_actions_history.append(safe_action.copy())
                cbf_values_history.append(h_x)
            
            # Apply safe action to environment
            if cbf:
                obs_next, reward, terminated, truncated, info = self.env.step(safe_action)#CHANGE HERE TO NOMINAL OR SAFE#CHANGE HERE TO NOMINAL OR SAFE#CHANGE HERE TO NOMINAL OR SAFE
            else:
                obs_next, reward, terminated, truncated, info = self.env.step(nominal_action)
            costs_history.append(info["cost"])
            
            if self.bc_mode == "multi-task":
                obs_next = np.append(obs_next, self.cost_limit)
                
            # Update state and tracking variables
            obs = obs_next
            episode_ret += reward
            episode_len += 1
            # print(info)
            episode_cost += info["cost"]
            
            if render:
                self.env.render()
                
            if terminated or truncated:
                break
        
        # Create plots if requested
        if plot:
            self._create_plots(
                states_history, 
                nominal_actions_history, 
                safe_actions_history, 
                costs_history, 
                cbf_values_history,
                episode_len
            )
            
        return episode_ret, episode_len, episode_cost

    def _create_plots(self, states, nominal_actions, safe_actions, costs, cbf_values, episode_len):
        """
        Creates and displays plots of rollout data.
        
        Args:
            states: List of states
            nominal_actions: List of nominal actions from BC policy
            safe_actions: List of safe actions after CBF modification
            costs: List of costs at each step
            cbf_values: List of CBF values at each step
            episode_len: Length of the episode
        """
        import matplotlib.pyplot as plt
        
        # Convert lists to numpy arrays for easier plotting
        states = np.array(states)
        nominal_actions = np.array(nominal_actions)
        safe_actions = np.array(safe_actions)
        costs = np.array(costs)
        cbf_values = np.array(cbf_values)
        
        # Create time steps
        time_steps = np.arange(episode_len)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(10, 8))
        
        # Plot 1: Actions (nominal vs safe)
        action_dim = nominal_actions.shape[1]
        ax1 = fig.add_subplot(3, 1, 1)
        for i in range(1):
            ax1.plot(time_steps, nominal_actions[:, i], '--', label=f'Nominal Action {i+1}')
            ax1.plot(time_steps, safe_actions[:, i], '--', label=f'Safe Action {i+1}')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Action Value')
        ax1.set_title('Nominal vs Safe Actions')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Costs
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(time_steps, costs, '-o', label='Step Cost')
        ax2.plot(time_steps, np.cumsum(costs), '-', label='Cumulative Cost')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Cost')
        ax2.set_title('Cost per Time Step and Cumulative Cost')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: CBF Values
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(time_steps, cbf_values, '-b', label='CBF Value')
        ax3.axhline(y=0, color='r', linestyle='--', label='Safety Threshold')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('CBF Value')
        ax3.set_title('Control Barrier Function Values')
        ax3.legend()
        ax3.grid(True)
        
        # Add a tight layout to prevent overlapping
        plt.tight_layout()
        
        # Save the figure if needed
        plt.savefig('/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/figs/cbf_rollout_analysis.png', dpi=300, bbox_inches='tight')
        
        # Display the figure
        plt.show()
        
        
        
#python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_bc_cbf.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180" --eval_episode 1 --device mps

#python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_bc_cbf.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180" --eval_episode 50 --device mps
