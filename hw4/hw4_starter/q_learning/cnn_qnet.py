import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQNet(nn.Module):
    def __init__(self, env, config, logger=None):
        super().__init__()

        #####################################################################
        # TODO: Define a CNN for the forward pass.
        #   Use the CNN architecture described in the following DeepMind
        #   paper by Mnih et. al.:
        #       https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        #
        # Some useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        H, W, C = env.observation_space.shape
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(W)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(H)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, env.action_space.n)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        #####################################################################
        # TODO: Implement the forward pass.
        #####################################################################
        x = state.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
