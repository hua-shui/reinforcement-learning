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


        # #逆序计算G(τ)
        G = 0
        for r in rewards[::-1]:
            G = r + self.gamma * G
            loss = 0

        # for s, a in zip(states, actions):
        #     probs = self.pi(torch.tensor(s).unsqueeze(0)).squeeze(0)
        #     log_prob = torch.log(probs)[a]
        #     loss -= log_prob * G

        #并行版本
        states = torch.tensor(states)
        actions = torch.tensor(actions).view(-1,1)
        # 计算log_probs
        # pi(states): 想左推，向右推的概率
        # gather(1, actions): 按照actions的索引，从pi(states)中获取对应的概率
        log_probs = torch.log(self.pi(states).gather(1, actions))
        # 计算loss: -ΣG * log_probs
        loss = -torch.sum(log_probs) * G
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Agent()
    state = env.reset()
    action,probs = agent.get_action(state)

    print("采取的动作是:", "向左推" if action == 0 else "向右推")
    print("采取动作的概率是:", probs[action].item())
    G = 1.0
    J = -G * probs[action].log()
    J.backward()
    agent.optimizer.step()
    _,probs = agent.get_action(state)
    print("采取的动作是:", "向左推" if action == 0 else "向右推")
    # 只预测一步,能看出概率提升了
    print("采取动作的概率是:", probs[action].item())



