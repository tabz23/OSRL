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
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_271/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_209/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_693/combined_model.pth",
    # # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
    # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_217/combined_model.pth",
    # #   #CBF
    # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_861/combined_model.pth",
    # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_799/combined_model.pth",
    # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_159/combined_model.pth",   #CHANGE       
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_409/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_929/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_892/combined_model.pth",
    
##for ablation part 1 
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_264/combined_model.pth",#p=0.1, normal size
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_264/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_264/combined_model.pth",
    # # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_882/combined_model.pth",#p=10e-4, normal size
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_882/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_882/combined_model.pth",
    # #   #CBF
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_270/combined_model.pth",#p=10e-8, normal size
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_270/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_270/combined_model.pth",
     #########################################
#      Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.2253, std: 0.0087
#   Avg Normalized Cost: 1.1542, std: 0.2349
#   Avg Length: 323.4333, std: 11.0981

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.1908, std: 0.0053
#   Avg Normalized Cost: 0.1675, std: 0.0979
#   Avg Length: 283.7333, std: 7.5538

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.1804, std: 0.0098
#   Avg Normalized Cost: 0.4017, std: 0.0675
#   Avg Length: 266.9333, std: 12.8928
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
#      Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.1941, std: 0.0065
#   Avg Normalized Cost: 0.2100, std: 0.0449
#   Avg Length: 287.5833, std: 9.0423

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.1914, std: 0.0037
#   Avg Normalized Cost: 0.0592, std: 0.0555
#   Avg Length: 286.7167, std: 4.8815

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.1755, std: 0.0026
#   Avg Normalized Cost: 0.0100, std: 0.0054
#   Avg Length: 263.1500, std: 3.2360
     #########################################       
                  
                  
# ##for ablation part 3
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_204/combined_model.pth",#p=10, small model
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_204/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_204/combined_model.pth",
    # # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",#p=10, big model
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",
    # #   #CBF
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",#dont care
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",   
# ########################
#coptidice
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.0993, std: 0.0010
#   Avg Normalized Cost: 0.2817, std: 0.0355
#   Avg Length: 153.6667, std: 1.0617

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.2254, std: 0.0067
#   Avg Normalized Cost: 1.2675, std: 0.2554
#   Avg Length: 325.0333, std: 6.6551
  
  
# ######################## 
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_409/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_929/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_892/combined_model.pth",


#    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_848/combined_model.pth",
    
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_703/combined_model_laststep.pth",
    
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_887/combined_model.pth",


                  
##for ablation with and without detach
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_559/combined_model.pth",#this is without detach. dont care about the below
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_559/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_559/combined_model.pth",
    # # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",#p=10, big model
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_189/combined_model.pth",
    # #   #CBF
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",#dont care
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",           
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.1760, std: 0.0026
#   Avg Normalized Cost: 0.0092, std: 0.0072
#   Avg Length: 263.9667, std: 3.3532

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.2186, std: 0.0194
#   Avg Normalized Cost: 1.1300, std: 0.4714
#   Avg Length: 316.4167, std: 21.9772

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.1769, std: 0.0079
#   Avg Normalized Cost: 0.0317, std: 0.0112
#   Avg Length: 265.4500, std: 10.9728
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
##for ablation with and without detach
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_283/combined_model_laststep.pth",#this is without detach. dont care about the below
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_283/combined_model_laststep.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_283/combined_model_laststep.pth",
    # ##ccbf
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_860/combined_model_laststep.pth",#p=10, big model
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_860/combined_model_laststep.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_860/combined_model_laststep.pth",
    #   #CBF
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",#dont care
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineHopperVelocityGymnasium-v1_445/combined_model.pth",  
]
    for i, model_path in enumerate(model_paths):
        cfg, model = load_config_and_model(args.path, args.best)
        # print(f"\n[{i+1}/{len(model_paths)}] Evaluating model: {os.path.basename(os.path.dirname(model_path))}")
        
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



# #### python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed10-09ab/COptiDICE_cost20_seed10-09ab" --eval_episodes 1

# python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-11cb/COptiDICE_cost20_seed20-11cb" --eval_episodes 1
# Eval reward: 59.060403111086195, normalized reward: 0.24693355831333091; cost: 2.74, normalized cost: 0.137; length: 1000.0

# python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20-3187/COptiDICE_cost20-3187" --eval_episodes 1
# Eval reward: 83.84573596036901, normalized reward: 0.35068686165949287; cost: 58.048, normalized cost: 2.9024; length: 1000.0 






# python directory % python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3"  --eval_episodes  50 



'''
python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/hopper_random/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3"  --eval_episodes 20

hopper coptidice
Group idbf (Models 1-3):
  Avg Normalized Reward: 0.2378, std: 0.0270
  Avg Normalized Cost: 1.8067, std: 0.7346
  Avg Length: 331.5833, std: 28.6462

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.1749, std: 0.0056
  Avg Normalized Cost: 0.0283, std: 0.0349
  Avg Length: 262.9667, std: 7.9012

Group cbf (Models 7-9):
  Avg Normalized Reward: 0.1767, std: 0.0027
  Avg Normalized Cost: 0.0052, std: 0.0059
  Avg Length: 264.2167, std: 3.4328
  
None:#none means just set the use cbf param to false and check the policy without cbf how it does
  Avg Normalized Reward: 0.1822, std: 0.0072
  Avg Normalized Cost: 0.0133, std: 0.0051
  Avg Length: 271.6833, std: 10.2803
  '''
  
  
  
  #beststep new L_unsafe instead of L_c
#   Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.1764, std: 0.0043
#   Avg Normalized Cost: 0.0067, std: 0.0062
#   Avg Length: 264.0167, std: 6.0286

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.1905, std: 0.0026
#   Avg Normalized Cost: 0.3083, std: 0.2127
#   Avg Length: 278.3167, std: 1.3456

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.1815, std: 0.0035
#   Avg Normalized Cost: 0.0067, std: 0.0077
#   Avg Length: 272.0000, std: 5.6570

  # last checkpoint new L_unsafe instead of L_c 
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.1755, std: 0.0088
#   Avg Normalized Cost: 0.0767, std: 0.0304
#   Avg Length: 263.0667, std: 11.4691

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.1793, std: 0.0088
#   Avg Normalized Cost: 0.1567, std: 0.0493
#   Avg Length: 264.6333, std: 11.0903

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.1861, std: 0.0050
#   Avg Normalized Cost: 0.0067, std: 0.0062
#   Avg Length: 278.9500, std: 6.6016