import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.normal import Normal

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Creates a multi-layer perceptron with the specified sizes and activations.

    Args:
        sizes (list): A list of integers specifying the size of each layer in the MLP.
        activation (nn.Module): The activation function to use for all layers except the output layer.
        output_activation (nn.Module): The activation function to use for the output layer. Defaults to nn.Identity.

    Returns:
        nn.Sequential: A PyTorch Sequential model representing the MLP.
    """

    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        layers += [layer, act()]##adding the () here
        '''
        if layers = [nn.Linear(10, 20), nn.ReLU()], then:
        *layers would unpack it to nn.Sequential(nn.Linear(10, 20), nn.ReLU()).
        '''
    return nn.Sequential(*layers)

class AffineDynamics(torch.nn.Module):
    def __init__(
        self,
        num_action,
        state_dim,
        hidden_dim=64,
        num_layers=3,
        dt=0.1):
        super().__init__()
        
        self.num_action=num_action
        self.state_dim=state_dim
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.dt=dt
        
        self.f=mlp( [self.state_dim] + num_layers*[self.hidden_dim] + [self.state_dim], activation=nn.ReLU)
        self.g=mlp( [self.state_dim] + num_layers*[self.hidden_dim] + [self.state_dim*self.num_action], activation=nn.ReLU)
        
    def forward(self, state):
        return self.f(state), self.g(state)
    
    def forward_x_dot(self,state, action):
        f, g = self.forward(state)
        gu=torch.einsum('bsa,ba->bs',g.view(g.shape[0], self.state_dim, self.num_action), action)
        x_dot=f+gu
        return x_dot
    
    def forward_next_state(self,state, action):
        return self.forward_x_dot(state,action)*self.dt + state

    
class CBF(torch.nn.Module):
    def __init__(
        self,        
        num_action,
        state_dim,
        hidden_dim=128,
        num_layers=3,
        dt=0.1):
        
        super().__init__()
        
        self.num_action=num_action
        self.state_dim=state_dim
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.dt=dt
        self.cbf=mlp([self.state_dim] + num_layers*[self.hidden_dim] + [1], activation=nn.ReLU, output_activation=nn.Tanh)
        
    def forward(self,state):
        return self.cbf(state)