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
    model_paths = [
        
        
        # ######final chosen ones
        # ##idbf
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_269/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_481/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_576/combined_model.pth",
        # # #CCBF
        # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_568/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##gr8 bc all
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_932/combined_model.pth",##gr8 bc all #gr8 for bc safe
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_709/combined_model.pth",##gr8 bc all #greatt for bc safe
        # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_784/combined_model.pth",##gr8 bc all
        # # ##CBF
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_933/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_278/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_619/combined_model.pth",       
        # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_626/combined_model.pth",
        # # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_758/combined_model.pth",
        
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
#      Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4393, std: 0.0196
#   Avg Normalized Cost: 8.0633, std: 0.4928
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5290, std: 0.0346
#   Avg Normalized Cost: 10.4325, std: 0.5456
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5763, std: 0.0088
#   Avg Normalized Cost: 11.7075, std: 0.8733
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
    ########################################
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.5212, std: 0.0215
#   Avg Normalized Cost: 10.0917, std: 0.8201
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4425, std: 0.0075
#   Avg Normalized Cost: 14.1267, std: 0.2303
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5443, std: 0.0005
#   Avg Normalized Cost: 5.3575, std: 0.2825
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
#coptidice
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.5927, std: 0.0132
#   Avg Normalized Cost: 13.6308, std: 0.6519
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4757, std: 0.0060
#   Avg Normalized Cost: 15.6750, std: 0.3063
#   Avg Length: 1000.0000, std: 0.0000

############################  
     
       
###below are testing idbf with diff p 
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_646/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_646/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_646/combined_model.pth",

        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_463/combined_model.pth",
        
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_180/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_180/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_180/combined_model.pth",
###coptidiff idbf results (best chkpt) 646 463 180 in order
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.6384, std: 0.0060
#   Avg Normalized Cost: 10.2025, std: 0.3963
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5426, std: 0.0230
#   Avg Normalized Cost: 10.5600, std: 0.1840
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4812, std: 0.0212
#   Avg Normalized Cost: 7.6925, std: 1.1405
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
          
###copti diff idbf results (best chkpt) 994 255 220 in order 

# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.6365, std: 0.0036
#   Avg Normalized Cost: 10.1408, std: 0.6141
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5320, std: 0.0139
#   Avg Normalized Cost: 10.3967, std: 0.2447
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4536, std: 0.0455
#   Avg Normalized Cost: 8.3733, std: 0.6685
#   Avg Length: 1000.0000, std: 0.0000

        # ###these new still pending below are testing idbf with diff p 
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_410/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_410/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_410/combined_model.pth",

        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_856/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_856/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_856/combined_model.pth",
        
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model.pth", 
###copti diff idbf results (best chkpt) 410 856 189 in order 
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.5382, std: 0.0355
#   Avg Normalized Cost: 14.1133, std: 1.8651
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.5212, std: 0.0169
#   Avg Normalized Cost: 11.1750, std: 0.8833
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5966, std: 0.0099
#   Avg Normalized Cost: 11.2317, std: 0.4156
#   Avg Length: 1000.0000, std: 0.0000

        ###these new still pending below are testing idbf with diff p 
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_580/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_580/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_580/combined_model.pth",

        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_159/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_159/combined_model.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_159/combined_model.pth",
        
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model_laststep.pth",
        # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_189/combined_model_laststep.pth", 
###bcql diff idbf results (best chkpt) 361 863 189 in order 
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.2603, std: 0.0204
#   Avg Normalized Cost: 2.6025, std: 0.3431
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.6009, std: 0.0258
#   Avg Normalized Cost: 11.3017, std: 0.8222
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5686, std: 0.0243
#   Avg Normalized Cost: 10.0917, std: 0.9865
#   Avg Length: 1000.0000, std: 0.0000

###coptidice diff idbf results (last chkpt) 361 863 189 in order 
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.2481, std: 0.0302
#   Avg Normalized Cost: 2.9742, std: 0.1309
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: -0.5338, std: 0.9402
#   Avg Normalized Cost: 3.2400, std: 0.6148
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5135, std: 0.0397
#   Avg Normalized Cost: 9.9292, std: 0.5781
#   Avg Length: 1000.0000, std: 0.0000

    #         ####for ablation without detach. only the first model is important and is that CCBF without detach
    # "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/models/OfflineSwimmerVelocityGymnasium-v1_110/combined_model.pth",#only care about this
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

# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3913, std: 0.0359
#   Avg Normalized Cost: 0.0425, std: 0.0114
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.4741, std: 0.0013
#   Avg Normalized Cost: 16.8350, std: 0.9271
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5000, std: 0.0406
#   Avg Normalized Cost: 5.2158, std: 0.5074
#   Avg Length: 1000.0000, std: 0.0000
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
# ]
    for i, model_path in enumerate(model_paths):
        # cfg, model = load_config_and_model(args.path, args.best)
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
        env=gym.make("OfflineSwimmerVelocityGymnasium"),
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



# python directory % python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineHopperVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-37f3"  --eval_episodes  50 


##### python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed10-09ab/COptiDICE_cost20_seed10-09ab" --eval_episodes 1

# python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20_seed20-11cb/COptiDICE_cost20_seed20-11cb" --eval_episodes 1
#Eval reward: 59.060403111086195, normalized reward: 0.24693355831333091; cost: 2.74, normalized cost: 0.137; length: 1000.0





# python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20-3187/COptiDICE_cost20-3187" --eval_episodes 1
# Eval reward: 83.84573596036901, normalized reward: 0.35068686165949287; cost: 58.048, normalized cost: 2.9024; length: 1000.0 



'''
python "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/Swimmer_random/eval_coptidice_modif_list.py" --path "/Users/i.k.tabbara/Documents/python directory/OSRL/logs/OfflineSwimmerVelocityGymnasium-v1-cost-20/COptiDICE_cost20-3187/COptiDICE_cost20-3187" --eval_episodes 20
Swimmer coptidice
Group idbf (Models 1-3):
  Avg Normalized Reward: 0.5661, std: 0.0265
  Avg Normalized Cost: 11.7542, std: 2.9877
  Avg Length: 1000.0000, std: 0.0000

Group ccbf (Models 4-6):
  Avg Normalized Reward: 0.2768, std: 0.0621
  Avg Normalized Cost: 0.1892, std: 0.0755
  Avg Length: 1000.0000, std: 0.0000

Group cbf (Models 7-9):
  Avg Normalized Reward: 0.0731, std: 0.0092
  Avg Normalized Cost: 1.3958, std: 0.3890
  Avg Length: 1000.0000, std: 0.0000
  
None:
  Avg Normalized Reward: 0.2974, std: 0.046
  Avg Normalized Cost: 1.6450, std: 0.33
  Avg Length: 1000.0000, std: 0.0000
'''

#beststep new L_unsafe instead of L_c
#copti 
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.4804, std: 0.0100
#   Avg Normalized Cost: 0.9633, std: 0.4220
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.3479, std: 0.0250
#   Avg Normalized Cost: 6.0108, std: 0.8182
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.5092, std: 0.0433
#   Avg Normalized Cost: 5.6092, std: 1.0484
#   Avg Length: 1000.0000, std: 0.0000


#laststep
# Group idbf (Models 1-3):
#   Avg Normalized Reward: 0.3176, std: 0.0071
#   Avg Normalized Cost: 3.0675, std: 0.5574
#   Avg Length: 1000.0000, std: 0.0000

# Group ccbf (Models 4-6):
#   Avg Normalized Reward: 0.2297, std: 0.0094
#   Avg Normalized Cost: 0.7283, std: 0.0845
#   Avg Length: 1000.0000, std: 0.0000

# Group cbf (Models 7-9):
#   Avg Normalized Reward: 0.4780, std: 0.0458
#   Avg Normalized Cost: 5.4658, std: 0.1409
#   Avg Length: 1000.0000, std: 0.0000