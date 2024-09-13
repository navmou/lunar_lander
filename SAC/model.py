import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, # RGB channels
                out_channels=64, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU()) #nn.ReLU())
        
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, # RGB channels
                out_channels=64, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU()) #nn.ReLU())
        
        self.layer3 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, # RGB channels
                out_channels=128, # Number of kernels
                kernel_size=2, # Size of kernels, i. e. of size 2x2
                stride=1),
            nn.LeakyReLU()) #nn.ReLU())
        
        self.layer4 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128, # RGB channels
                out_channels=128, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU()) #nn.ReLU())
        
        self.layer5 = nn.Linear(in_features=128, out_features=1)
        self.apply(weights_init_)
        
    def forward(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.squeeze() # ~Keras.Flatten
        x = self.layer5(x)
        return torch.tanh(x)


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        
        # Q1 architecture
        self.Q1_layer1 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, # RGB channels
                out_channels=64, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU())
        
        self.Q1_layer2 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, # RGB channels
                out_channels=64, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU())
        
        self.Q1_layer3 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, # RGB channels
                out_channels=128, # Number of kernels
                kernel_size=2, # Size of kernels, i. e. of size 2x2
                stride=1),
            nn.LeakyReLU())
        
        self.Q1_layer4 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128, # RGB channels
                out_channels=128, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU())
        
        self.Q1_layer5 = nn.Linear(in_features=128, out_features=60)
        
                
        # Q2 architecture
        self.Q2_layer1 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, # RGB channels
                out_channels=64, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU())
        
        self.Q2_layer2 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, # RGB channels
                out_channels=64, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU())
        
        self.Q2_layer3 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, # RGB channels
                out_channels=128, # Number of kernels
                kernel_size=2, # Size of kernels, i. e. of size 2x2
                stride=1),
            nn.LeakyReLU())
        
        self.Q2_layer4 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128, # RGB channels
                out_channels=128, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU())
        
        self.Q2_layer5 = nn.Linear(in_features=128, out_features=60)       

        # self.output = nn.Sigmoid()
        
        self.apply(weights_init_)

    def forward(self, state):
        q1_x = self.Q1_layer1(state)
        q1_x = self.Q1_layer2(q1_x)
        q1_x = self.Q1_layer3(q1_x)
        q1_x = self.Q1_layer4(q1_x)
        q1_x = q1_x.squeeze() # ~Keras.Flatten
        q1_x = self.Q1_layer5(q1_x)
        
        q2_x = self.Q2_layer1(state)
        q2_x = self.Q2_layer2(q2_x)
        q2_x = self.Q2_layer3(q2_x)
        q2_x = self.Q2_layer4(q2_x)
        q2_x = q2_x.squeeze() # ~Keras.Flatten
        q2_x = self.Q2_layer5(q2_x)
        
        return q1_x, q2_x
        # return torch.tanh(q1_x), torch.tanh(q2_x)
        # return 124 * self.output(q1_x) - 62, 124 * self.output(q2_x) - 62

    
class DiscreteSACPolicy(nn.Module):
    def __init__(self):
        super(DiscreteSACPolicy, self).__init__()
            
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, # RGB channels
                out_channels=64, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU())
        
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, # RGB channels
                out_channels=64, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU())
        
        self.layer3 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, # RGB channels
                out_channels=128, # Number of kernels
                kernel_size=2, # Size of kernels, i. e. of size 2x2
                stride=1),
            nn.LeakyReLU())
        
        self.layer4 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128, # RGB channels
                out_channels=128, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.LeakyReLU())

        self.linear5 = nn.Linear(in_features=128, out_features=60)
          
        # This is or continous Action space
        # self.mean_linear = nn.Linear(in_features=128, out_features=60)
        # self.log_std_linear = nn.Linear(in_features=128, out_features=60)        
        self.apply(weights_init_)
        
    def forward(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.squeeze() # ~Keras.Flatten
        x = self.linear5(x)
               
        # mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return x
       
    # Returns one sample
    def sample(self, state, valid_actions):
        x = self.forward(state)
        return F.softmax(x[valid_actions], dim = 0), F.log_softmax(x[valid_actions], dim = 0)
    
    
class GaussianPolicy(nn.Module):
    def __init__(self):
        raise Error("Not valid")
        super(GaussianPolicy, self).__init__()
            
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, # RGB channels
                out_channels=64, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, # RGB channels
                out_channels=64, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.ReLU())
        
        self.layer3 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, # RGB channels
                out_channels=128, # Number of kernels
                kernel_size=2, # Size of kernels, i. e. of size 2x2
                stride=1),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128, # RGB channels
                out_channels=128, # Number of kernels
                kernel_size=3, # Size of kernels, i. e. of size 3x3
                stride=1),
            nn.ReLU())

        self.linear5 = nn.Linear(in_features=128, out_features=60)
        
        self.softMax = nn.Softmax()
  
        # This is or continous Action space
        # self.mean_linear = nn.Linear(in_features=128, out_features=60)
        # self.log_std_linear = nn.Linear(in_features=128, out_features=60)        
        self.apply(weights_init_)
        
    def forward(self, state, valid_actions):
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.squeeze() # ~Keras.Flatten
        x = self.linear5(x)
               
        # mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return F.softmax(x[valid_actions], dim = 0), F.log_softmax(x[valid_actions], dim = 0)

    def sample(self, state_batch, valid_actions_batch):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
"""
    def to(self, device):
        return super(GaussianPolicy, self).to(device)
"""

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
