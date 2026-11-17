import gym
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super(PolicyNet, self).__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x),dim=1)
        return x #输出x: 想左推，向右推的概率

# TODO: REINFORCE算法
class Agent:
    def __init__(self):
        self.gamma = 0.99 # 折扣因子
        self.lr = 0.0002 # 学习率
        self.action_size = 2 #两个动作
        self.pi = PolicyNet(self.action_size) # 策略网络
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)#优化器

    def get_action(self,state):
        """
        根据当前状态获取动作和动作的概率
        """
        #将state转为torch张量，unsqueeze(0)增加一个维度，使state的形状为(1,4)，经过策略网络pi，得到 probs: 想左推，向右推的概率(1, 2)
        #squeeze(0)减少一个维度，使probs的形状为(2,)
        # probs: 想左推，向右推的概率
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        # 根据probs创建一个二项分布
        m = Categorical(probs)
        # 使用二项分布采样一个动作
        action = m.sample().item()
        # 返回动作和动作的概率
        return action,probs

    def collect_trajectory(self,env):
        """
        收集一条轨迹，返回状态、动作、奖励
        """
        state = env.reset()
        states,actions,rewards = [],[],[]
        done = False

        while not done:
            # 根据当前状态获取动作和动作的概率
            action,_ = self.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        return states, actions, rewards

    def update(self,trajectory: tuple):
        """
        更新策略网络
        """
        states, actions, rewards = trajectory
        G, loss = 0, 0
        for s, a, r in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = r + self.gamma * G
            probs = self.pi(torch.tensor(s).unsqueeze(0)).squeeze(0)
            log_prob = torch.log(probs)[a]
            #loss += -log_prob * G #Σlog_prob * G(t)
            loss += -log_prob * (G-5) #带基线: 可以试试199,开不到满分就挨揍
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



