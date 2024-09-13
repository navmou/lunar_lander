import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .utils import soft_update, hard_update
from .model import DiscreteSACPolicy, QNetwork #, DeterministicPolicy #GaussianPolicy,
import numpy as np

class SAC(object):
    def __init__(self, device = "cpu"):

        self.lr = 0.001
        
        self.gamma = 1 # 0.999
        self.tau = 0.005
        self.alpha = 0.2

        # self.policy_type = args.policy
        self.target_update_interval = 1
        
        self.device = torch.device(device)

        self.critic = QNetwork().to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork().to(self.device)
        hard_update(self.critic_target, self.critic)

        self.automatic_entropy_tuning = False
        
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(60).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
        
                
        self.policy = DiscreteSACPolicy().to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
      
        
    def select_action(self, state, valid_actions, evaluate=False):
        if len(valid_actions) <= 1:
            return valid_actions[0], 1, 0
        
        if type(state).__module__ == np.__name__:
            torch_state = torch.FloatTensor([state]).to(self.device)
        else:
            torch_state = state
        
        probs, log_probs = self.policy.sample(torch_state, valid_actions)
        probs = probs.detach().cpu().numpy()
        log_probs = log_probs.detach().cpu().numpy()
    
        if evaluate is False:
            ind = np.random.choice(len(probs), p = probs)
        else:
            ind = probs.argmax()
            
        return valid_actions[ind], probs[ind], log_probs[ind]
        
       
    def update_parameters(self, memory, batch_size, updates = 1):
        next_q_vals = torch.zeros(batch_size).to(self.device)
        
        selected_actions = []
        policy_loss = torch.Tensor([1]).to(self.device) #  requires_grad=True
        
        qf1, qf2 = torch.zeros(batch_size).to(self.device), torch.zeros(batch_size).to(self.device)
        
        ind = 0
        for state, action, reward, next_state, done, valid_actions, valid_actions_next in memory.sample(batch_size=batch_size):
            selected_actions.append(action)
            state = torch.FloatTensor([state]).to(self.device)
            next_state = torch.FloatTensor([next_state]).to(self.device)
            
            with torch.no_grad():
                if done:
                    next_q_vals[ind] = reward
                else:
                    next_state_action, _, log_prob= self.select_action(next_state, valid_actions_next)
                    qf1_next_target, qf2_next_target = self.critic_target(next_state)
                    qf1_next_target, qf2_next_target = qf1_next_target[next_state_action], qf2_next_target[next_state_action]

                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_prob

                    next_q_vals[ind] = reward + done * self.gamma * min_qf_next_target
                    
                
            q_action,_ ,_ = self.select_action(state, valid_actions)
            qf1_pi, qf2_pi = self.critic(state)
            qf1[ind], qf2[ind] = qf1_pi[action], qf2_pi[action]
            
            
            probs, log_probs = self.policy.sample(state, valid_actions)
            # policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() <- old (continous)
            policy_loss += (self.alpha * log_probs[q_action == valid_actions] - torch.min(qf1_pi[q_action], qf2_pi[q_action]).detach()) # / batch_size
            ind += 1
               
        qf1_loss = F.mse_loss(qf1, next_q_vals)
        qf2_loss = F.mse_loss(qf2, next_q_vals) 
       
        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

