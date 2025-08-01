#!/bin/bash

# Activate the Conda environment
source activate osrl


# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0 --temp 1 --detach True --batch_size 256  --device="mps" --num_action_samples 10
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 1 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples 10
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 1 --temp 0.7 --detach True --batch_size 256 --device="mps" --num_action_samples 10
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 1 --temp 0.5 --detach True --batch_size 256 --device="mps" --num_action_samples 10
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 1 --temp 1 --detach False --batch_size 256  --device="mps" --num_action_samples 10
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 1 --temp 0.7 --detach False --batch_size 256  --device="mps" --num_action_samples 10
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 1 --temp 0.5 --detach False --batch_size 256  --device="mps" --num_action_samples 10
python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 1 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples 10 --seed 7
python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples 10 --seed 7

python examples/research/check/trainer_idbf.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0 --temp 1 --detach True --batch_size 256 --device="mps"  --seed 7 



# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0 --temp 1 --detach True --batch_size 256  --device="mps" --num_action_samples 10 --seed 2
python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0.1 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples 10 --seed 1
python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 1 --temp 0.7 --detach True --batch_size 256 --device="mps" --num_action_samples 10 --seed 3
python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0.1 --temp 0.5 --detach True --batch_size 256 --device="mps" --num_action_samples 10 -seed 4
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0 --temp 1 --detach True --batch_size 256  --device="mps" --num_action_samples 10
python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0.1 --temp 1 --detach True --batch_size 256 --device="mps" --num_action_samples 10 --seed 6
python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0.01 --temp 0.7 --detach True --batch_size 256 --device="mps" --num_action_samples 10 --seed 5
python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0.1 --temp 0.5 --detach True --batch_size 256 --device="mps" --num_action_samples 10 --seed 7
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0.1 --temp 1 --detach False --batch_size 256  --device="mps" --num_action_samples 10
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0.1 --temp 0.7 --detach False --batch_size 256  --device="mps" --num_action_samples 10
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1  --cql 0.1 --temp 0.5 --detach False --batch_size 256  --device="mps" --num_action_samples 10

# # Run the Python script with the specified arguments
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.0001 --temp 0.7 --num_action_samples 10
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.001 --temp 0.7 --num_action_samples 10
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.01 --temp 0.7 --num_action_samples 10
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.1 --temp 0.7 --num_action_samples 10
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 0.7 --num_action_samples 10
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 10 --temp 0.7 --num_action_samples 10

# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 0.5 --num_action_samples 10
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 0.6 --num_action_samples 10
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 0.7 --num_action_samples 10
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1  --temp 0.8 --num_action_samples 10
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1  --temp 1 --num_action_samples 10

# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.1 --temp 0.7 --num_action_samples 4
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.1 --temp 0.7 --num_action_samples 6
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.1 --temp 0.7 --num_action_samples 10



# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0 --temp 0 --num_action_samples 0 --detach False ##base case
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 1 --num_action_samples 10 --detach False
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 10 --temp 1 --num_action_samples 10 --detach False
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 0.5 --num_action_samples 10 --detach False




# # Run the Python script with the specified arguments
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.0001 --temp 0.7 --num_action_samples 10 --detach True
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.001 --temp 0.7 --num_action_samples 10 --detach True
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.01 --temp 0.7 --num_action_samples 10 --detach True
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.1 --temp 0.7 --num_action_samples 10 --detach True
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 0.7 --num_action_samples 10 --detach True
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 10 --temp 0.7 --num_action_samples 10 --detach True

# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 0.5 --num_action_samples 10 --detach True
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 0.6 --num_action_samples 10 --detach True
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 0.7 --num_action_samples 10 --detach True
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1  --temp 0.8 --num_action_samples 10 --detach True
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1  --temp 1 --num_action_samples 10 --detach True

# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.1 --temp 0.7 --num_action_samples 4 --detach True
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.1 --temp 0.7 --num_action_samples 6 --detach True
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0.1 --temp 0.7 --num_action_samples 10 --detach True



# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 0 --temp 0 --num_action_samples 0  --detach True ##base case
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 1 --num_action_samples 10  --detach True
# python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 10 --temp 1 --num_action_samples 10  --detach True
# # python examples/research/check/trainer.py --task OfflineSwimmerVelocityGymnasium-v1 --device="mps" --cql 1 --temp 0.5 --num_action_samples 10  --detach True



# chmod +x "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/training_swimmer.sh"
#caffeinate -i "/Users/i.k.tabbara/Documents/python directory/OSRL/examples/research/check/training_swimmer.sh"

