import torch
import torch.nn as nn  # 添加这一行
import torch.optim as optim
import numpy as np
from models import Actor, DoubleCritic
from memory import ReplayMemory
from collections import namedtuple, deque
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class DDPGAgent:
    def __init__(self, state_size, action_size, device, hidden_size=256):
        self.device = device
        self.action_size = action_size
        
        # 初始化网络
        self.actor = Actor(state_size, action_size, hidden_size).to(device)
        self.critic = DoubleCritic(state_size, action_size, hidden_size).to(device)
        self.target_actor = Actor(state_size, action_size, hidden_size).to(device)
        self.target_critic = DoubleCritic(state_size, action_size, hidden_size).to(device)
        
        # 初始化目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器（添加学习率衰减）
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.95)
        
        # 经验回放
        self.memory = ReplayMemory(100000)
        self.batch_size = 128  # 增大batch size
        self.gamma = 0.99
        self.tau = 0.005
        
        # 噪声参数
        self.noise_scale = 0.1
        self.noise_decay = 0.995
        self.min_noise = 0.01
        
    def select_action(self, state, training=True):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
            
        if training:
            # 添加衰减的噪声
            noise = np.random.normal(0, self.noise_scale, size=self.action_size)
            self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
            action = np.clip(action + noise, -2.0, 2.0)
            
        return action
        
    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0
            
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.FloatTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            next_action_batch = self.target_actor(next_state_batch)
            target_q1, target_q2 = self.target_critic(next_state_batch, next_action_batch)
            target_q = torch.min(target_q1, target_q2)  # Double DDPG: 使用两个Critic计算最小值
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
            
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 梯度裁剪
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic(state_batch, self.actor(state_batch))[0].mean()  # 使用第一个Critic计算Actor损失
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # 梯度裁剪
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update()
        
        return actor_loss.item(), critic_loss.item()
        
    def _soft_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def state_dict(self):
        """返回智能体的状态字典"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'noise_scale': self.noise_scale
        }
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.target_actor.load_state_dict(state_dict['target_actor'])
        self.target_critic.load_state_dict(state_dict['target_critic'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.actor_scheduler.load_state_dict(state_dict['actor_scheduler'])
        self.critic_scheduler.load_state_dict(state_dict['critic_scheduler'])
        self.noise_scale = state_dict['noise_scale']