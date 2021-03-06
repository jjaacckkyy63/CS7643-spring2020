import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from core.dqn_train import *

class LinearQNet(nn.Module):
    def __init__(self, env, config):
        """
        A state-action (Q) network with a single fully connected
        layer, takes the state as input and gives action values
        for all actions.
        """
        super().__init__()

        #####################################################################
        # TODO: Define a fully connected layer for the forward pass. Some
        # useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        
        # multiple layer !!!!!
        H, W, C = env.observation_space.shape
        self.h1_layer = nn.Linear(H * W * C * config.state_history, 128)
        self.h2_layer = nn.Linear(128, 256)
        self.q_layer = nn.Linear(256, env.action_space.n)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        """
        Returns Q values for all actions

        Args:
            state: tensor of shape (batch, H, W, C x config.state_history)

        Returns:
            q_values: tensor of shape (batch_size, num_actions)
        """
        #####################################################################
        # TODO: Implement the forward pass, 1-2 lines.
        #####################################################################
        out = F.relu(self.h1_layer(state.view(state.shape[0],-1)))
        out = F.relu(self.h2_layer(out))
#         q_table = F.relu(self.q_layer(out)) # no relu
        q_table = self.q_layer(out) # no relu
        return q_table
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
        
