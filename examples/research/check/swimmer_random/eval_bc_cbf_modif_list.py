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
        
        
        # #idbf
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_269/combined_model.pth",#"best_safe acc": 0.9791599869728088,"best_unsafe_acc": 0.8777299463748932
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_481/combined_model.pth",#"best_safe acc": 0.9478048920631409,"best_unsafe_acc": 0.84089315533638
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_576/combined_model.pth",#"best_safe acc": 0.9485815405845642,"best_unsafe_acc": 0.8504167705774307
        

        # # #CCBF
        # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##    "best_safe acc": 0.764591982960701, "best_unsafe_acc": 0.974375969171524
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_932/combined_model.pth",##    "best_safe acc": 0.7573345065116882,"best_unsafe_acc": 0.9764593034982681
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_709/combined_model.pth",##     "best_safe acc": 0.758169686794281, "best_unsafe_acc": 0.974375969171524
        # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##gr8 bc all
        
        # # ##CBF
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_933/combined_model.pth",    #"best_safe acc": 0.9614395439624787, "best_unsafe_acc": 0.9082919180393219
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_278/combined_model.pth",    #"best_safe acc": 0.9791599869728088, "best_unsafe_acc": 0.8777299463748932
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_619/combined_model.pth",    #"best_safe acc": 0.974046328663826,"best_unsafe_acc": 0.8911491602659225
        
##for ablation part 1 
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_817/combined_model.pth",#p=0.1, normal size
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_817/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_817/combined_model.pth",
    # # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model.pth",#p=10e-4, normal size
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model.pth",
    # #   #CBF
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_362/combined_model.pth",#p=10e-8, normal size
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_362/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_362/combined_model.pth",
     #########################################
#      bc safe
#      Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4247, std: 0.0092
#   Avg Normalized Cost: 0.1925, std: 0.0341
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4692, std: 0.0206
#   Avg Normalized Cost: 0.2208, std: 0.0326
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4965, std: 0.0128
#   Avg Normalized Cost: 0.2733, std: 0.0318
#   Avg Length: 1000.0000, std: 0.0000
  
#   bc
#   Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3088, std: 0.0454
#   Avg Normalized Cost: 3.2442, std: 1.4091
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4106, std: 0.0444
#   Avg Normalized Cost: 5.8167, std: 1.1286
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4452, std: 0.0306
#   Avg Normalized Cost: 5.4542, std: 1.5430
#   Avg Length: 1000.0000, std: 0.0000
     #########################################
# ##for ablation part 2 
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_187/combined_model.pth",#p=0.1, big size
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_187/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_187/combined_model.pth",
#     # ##ccbf
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_771/combined_model.pth",#p=10e-4, big size
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_771/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_771/combined_model.pth",
#     #   #CBF
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",#p=10e-8, big size
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
#     "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",  
         #########################################
#      bc safe
     
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4280, std: 0.0153
#   Avg Normalized Cost: 0.1867, std: 0.0077
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4965, std: 0.0046
#   Avg Normalized Cost: 0.3158, std: 0.0254
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4902, std: 0.0019
#   Avg Normalized Cost: 0.4658, std: 0.0879
#   Avg Length: 1000.0000, std: 0.0000
  
#   bc 
  
#   Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4168, std: 0.0272
#   Avg Normalized Cost: 6.7883, std: 1.4796
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4423, std: 0.0534
#   Avg Normalized Cost: 8.4050, std: 0.7037
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5401, std: 0.0358
#   Avg Normalized Cost: 3.9975, std: 1.2237
#   Avg Length: 1000.0000, std: 0.0000
     #########################################

    # ##for ablation part 3
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_970/combined_model.pth",#smaller model. p=10
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_970/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_970/combined_model.pth",
    # # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_395/combined_model.pth",#bigger model. p=10
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_395/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_395/combined_model.pth",
    # #   #CBF
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",#idc about this
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
    

############################
# bc safe
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4871, std: 0.0133
#   Avg Normalized Cost: 0.5400, std: 0.1170
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4722, std: 0.0173
#   Avg Normalized Cost: 0.1867, std: 0.0266
#   Avg Length: 1000.0000, std: 0.0000
  
#   bc
#   Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.5468, std: 0.0087
#   Avg Normalized Cost: 9.1817, std: 1.0526
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4598, std: 0.0518
#   Avg Normalized Cost: 5.5500, std: 1.0213
#   Avg Length: 1000.0000, std: 0.0000
############################  










#####THESE ARE NEW MODELS SAME HYPER PARAMS BUT CBF, CCBF, IDBF. LOGS ARE THERE TOO
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_362/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_362/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_362/combined_model.pth", 

        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_657/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_657/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_657/combined_model.pth",

        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_599/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_599/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_599/combined_model.pth",

###below are testing idbf with diff p 


        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_817/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_817/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_817/combined_model_laststep.pth",

        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model_laststep.pth",
        
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_362/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_362/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_362/combined_model_laststep.pth",
###bc safe diff idbf results (best chkpt) 817 463 362 in order
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4766, std: 0.0299
#   Avg Normalized Cost: 0.2925, std: 0.0337
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4789, std: 0.0289
#   Avg Normalized Cost: 0.2792, std: 0.0232
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4867, std: 0.0270
#   Avg Normalized Cost: 0.2267, std: 0.0194
#   Avg Length: 1000.0000, std: 0.0000

###bc all diff idbf results (best chkpt)  in order
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3738, std: 0.1235
#   Avg Normalized Cost: 3.1375, std: 1.6903
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4352, std: 0.0276
#   Avg Normalized Cost: 5.7275, std: 1.0680
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4491, std: 0.0554
#   Avg Normalized Cost: 7.0867, std: 1.2976
#   Avg Length: 1000.0000, std: 0.0000
        
###bc all diff idbf results (last chkpt)  in order   
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4790, std: 0.0089
#   Avg Normalized Cost: 4.3617, std: 0.7576
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5375, std: 0.0421
#   Avg Normalized Cost: 5.4900, std: 2.0951
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4803, std: 0.0108
#   Avg Normalized Cost: 3.6025, std: 0.3261
#   Avg Length: 1000.0000, std: 0.0000   
        
        
  
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_646/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_646/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_646/combined_model.pth",

        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model.pth",
        
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_180/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_180/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_180/combined_model.pth",
        
###bc all diff idbf results (best chkpt) 646 463 180 in order
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4891, std: 0.0262
#   Avg Normalized Cost: 4.0142, std: 0.4576
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5346, std: 0.0306
#   Avg Normalized Cost: 6.0550, std: 2.1229
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4823, std: 0.0323
#   Avg Normalized Cost: 4.2683, std: 0.6013
#   Avg Length: 1000.0000, std: 0.0000

###bc safe diff idbf results (best chkpt) 646 463 180 in order
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.5174, std: 0.0035
#   Avg Normalized Cost: 0.2883, std: 0.0254
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4626, std: 0.0095
#   Avg Normalized Cost: 0.3233, std: 0.0153
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4845, std: 0.0412
#   Avg Normalized Cost: 0.4183, std: 0.0433
#   Avg Length: 1000.0000, std: 0.0000

###below are testing idbf with diff p 
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_994/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_994/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_994/combined_model.pth",

        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_255/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_255/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_255/combined_model.pth",
        
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_220/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_220/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_220/combined_model.pth",
###bc all diff idbf results (best chkpt) 994 255 220 in order       
#   Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4097, std: 0.0910
#   Avg Normalized Cost: 3.7175, std: 1.2358
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4367, std: 0.0691
#   Avg Normalized Cost: 6.0442, std: 1.8609
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4330, std: 0.0668
#   Avg Normalized Cost: 5.0983, std: 1.3129
#   Avg Length: 1000.0000, std: 0.0000

###bc safe diff idbf results (best chkpt) 994 255 220 in order        
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4523, std: 0.0100
#   Avg Normalized Cost: 0.1567, std: 0.0092
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4223, std: 0.0168
#   Avg Normalized Cost: 0.0925, std: 0.0054
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4221, std: 0.0304
#   Avg Normalized Cost: 0.0867, std: 0.0286
#   Avg Length: 1000.0000, std: 0.0000
        
###these new still pending below are testing idbf with diff p 
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_410/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_410/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_410/combined_model.pth",

        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_856/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_856/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_856/combined_model.pth",
        
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model.pth",        

###bc all diff idbf results (best chkpt) 410 856 189 in order
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4762, std: 0.0269
#   Avg Normalized Cost: 6.4175, std: 0.7505
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4487, std: 0.0044
#   Avg Normalized Cost: 5.3292, std: 0.9959
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4063, std: 0.0205
#   Avg Normalized Cost: 6.0467, std: 1.0323
#   Avg Length: 1000.0000, std: 0.0000

###bc safe diff idbf results (best chkpt) 410 856 189 in order
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4959, std: 0.0120
#   Avg Normalized Cost: 0.2383, std: 0.0012
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4137, std: 0.0191
#   Avg Normalized Cost: 0.1183, std: 0.0272
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4510, std: 0.0165
#   Avg Normalized Cost: 0.0525, std: 0.0154
#   Avg Length: 1000.0000, std: 0.0000

        ###these new still pending below are testing idbf with diff p 
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_361/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_361/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_361/combined_model.pth",

        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_863/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_863/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_863/combined_model.pth",
        
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model_laststep.pth",  
###bc diff idbf results (best chkpt) 361 863 189 in order 
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.5398, std: 0.0179
#   Avg Normalized Cost: 5.4058, std: 0.5563
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5400, std: 0.0342
#   Avg Normalized Cost: 10.3625, std: 1.1386
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4495, std: 0.0370
#   Avg Normalized Cost: 6.1925, std: 1.4566
#   Avg Length: 1000.0000, std: 0.0000

###bc safe idbf results (best chkpt) 361 863 189 in order 
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4576, std: 0.0281
#   Avg Normalized Cost: 0.3108, std: 0.1918
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5145, std: 0.0112
#   Avg Normalized Cost: 0.2067, std: 0.0062
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4397, std: 0.0308
#   Avg Normalized Cost: 0.0608, std: 0.0327 ###########################
#   Avg Length: 1000.0000, std: 0.0000


# ###bc diff idbf results (last chkpt) 361 863 189 in order
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.5080, std: 0.0578
#   Avg Normalized Cost: 3.7358, std: 1.8027
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: -10.0288, std: 2.4804
#   Avg Normalized Cost: 3.1217, std: 0.7782
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4115, std: 0.0223
#   Avg Normalized Cost: 3.9667, std: 0.7462
#   Avg Length: 1000.0000, std: 0.0000
 
# ###bc safe diff idbf results (last chkpt) 361 863 189 in order 
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4653, std: 0.0205
#   Avg Normalized Cost: 0.2467, std: 0.0739
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.0559, std: 0.2330
#   Avg Normalized Cost: 0.7558, std: 0.2152
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5041, std: 0.0332
#   Avg Normalized Cost: 0.4833, std: 0.1817
#   Avg Length: 1000.0000, std: 0.0000



    # ####for ablation without detach. only the first model is important and is that CCBF without detach
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_110/combined_model.pth",#only care abt this
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_110/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_110/combined_model.pth",
    # # ##ccbf
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_395/combined_model.pth",#bigger model. p=10
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_395/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_395/combined_model.pth",
    # #   #CBF
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",#idc about this
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
    
        # ####for ablation without detach. only the first model is important and is that CCBF without detach
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_539/combined_model.pth",#only care abt this
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_539/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_539/combined_model.pth",
    # ##ccbf
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_666/combined_model.pth",#bigger model. p=10
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_666/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_666/combined_model.pth",
    #   #CBF
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",#idc about this
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
    "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
    
    
    
    
    
# bc
# #Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3110, std: 0.0276
#   Avg Normalized Cost: 0.6142, std: 0.1742
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4957, std: 0.0238
#   Avg Normalized Cost: 7.9383, std: 0.9846
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5330, std: 0.0110
#   Avg Normalized Cost: 3.6808, std: 0.2358
#bcsafe
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3963, std: 0.0095
#   Avg Normalized Cost: 0.0642, std: 0.0208
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5019, std: 0.0171
#   Avg Normalized Cost: 0.3217, std: 0.0507
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4815, std: 0.0230
#   Avg Normalized Cost: 0.7350, std: 0.0888
#   Avg Length: 1000.0000, std: 0.0000
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

python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/swimmer_random/eval_bc_cbf_modif_list.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-all_cost20-d567/BC-all_cost20-d567"  --eval_episode 20 --device cpu
swimmer BC

Group idbf (Models 1-3):
  Avg Normalized Reward: 0.3609, std: 0.0095
  Avg Normalized Cost: 3.8025, std: 0.3182
  Avg Length: 1000.0000, std: 0.0000

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.3952, std: 0.0323
  Avg Normalized Cost: 0.9442, std: 0.1807
  Avg Length: 1000.0000, std: 0.0000

Group cbf (Models 7-9):
  Avg Normalized Reward: 0.4051, std: 0.0610
  Avg Normalized Cost: 5.0517, std: 2.8448
  Avg Length: 1000.0000, std: 0.0000
  
None:
  Avg Normalized Reward: 0.4360, std: 0.033
  Avg Normalized Cost: 2.2567, std: 0.644
  Avg Length: 1000.0000, std: 0.0000
  

python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/swimmer_random/eval_bc_cbf_modif_list.py" --device="mps" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/BC-safe_bc_modesafe_cost20_seed20-2180/BC-safe_bc_modesafe_cost20_seed20-2180" --eval_episode 20 --device cpu
swimmer BC safe
Group idbf (Models 1-3):
  Avg Normalized Reward: 0.4995, std: 0.0195
  Avg Normalized Cost: 0.2825, std: 0.0871
  Avg Length: 1000.0000, std: 0.0000

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.4529, std: 0.0199
  Avg Normalized Cost: 0.0333, std: 0.0228
  Avg Length: 1000.0000, std: 0.0000

Group cbf (Models 7-9):
  Avg Normalized Reward: 0.4558, std: 0.0163
  Avg Normalized Cost: 0.1175, std: 0.0616
  Avg Length: 1000.0000, std: 0.0000
  
None:
  Avg Normalized Reward: 0.4314, std: 0.03
  Avg Normalized Cost: 0.1183, std: 0.054
  Avg Length: 1000.0000, std: 0.0000
'''



#beststep new L_unsafe instead of L_c
#bc 
# Eval reward: 120.09890644997529, normalized reward: 0.5024454124903761; cost: 60.65, normalized cost: 3.0324999999999998; length: 1000.0   
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3746, std: 0.0306
#   Avg Normalized Cost: 2.0667, std: 0.1606
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4215, std: 0.0489
#   Avg Normalized Cost: 3.1142, std: 0.7213
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5188, std: 0.0157
#   Avg Normalized Cost: 3.5683, std: 0.7697
#   Avg Length: 1000.0000, std: 0.0000

#bc safe
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3540, std: 0.0094
#   Avg Normalized Cost: 0.1325, std: 0.0820
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4315, std: 0.0224
#   Avg Normalized Cost: 0.0583, std: 0.0245
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4999, std: 0.0028
#   Avg Normalized Cost: 0.5733, std: 0.0827
#   Avg Length: 1000.0000, std: 0.0000

#laststep
#bc
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3137, std: 0.0545
#   Avg Normalized Cost: 2.7692, std: 0.7919
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.2157, std: 0.0653
#   Avg Normalized Cost: 2.0417, std: 0.8902
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5471, std: 0.0443
#   Avg Normalized Cost: 4.3800, std: 1.0605
#   Avg Length: 1000.0000, std: 0.0000
  
# #laststep
# #bcsafe
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3445, std: 0.0163
#   Avg Normalized Cost: 0.0617, std: 0.0420
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.2732, std: 0.0123
#   Avg Normalized Cost: 0.1042, std: 0.0139
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4789, std: 0.0178
#   Avg Normalized Cost: 0.7442, std: 0.0749
#   Avg Length: 1000.0000, std: 0.0000