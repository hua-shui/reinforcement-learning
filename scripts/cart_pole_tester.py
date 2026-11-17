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

class CartPoleTester:
    def __init__(self, model_path):
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')# 创建环境，指定渲染模式
        result = self.env.reset()# 重置环境状态
        # 兼容不同 gym 版本：0.26+ 返回 (obs, info)，旧版直接返回 obs
        self.state = result[0] if isinstance(result, tuple) else result
        self.done = False# 游戏是否结束，初始值为False，也就是未结束
        self.episode_rewards: list = []# 每回合奖励
        self.total_reward = 0 # 总奖励
        self.frames = []#保存每一帧
        self.gamma = 0.95 # 折扣因子

    def get_episode(self):
        while not self.done:
            self.frames.append(self.env.render())# 渲染画面，并保存帧
            action = random.choice([0,1])# 随机选择一个动作 0: 左 1: 右
            step_result = self.env.step(action)
            # 兼容 0.26+ 的 5 元组 (obs, reward, terminated, truncated, info) 和旧版 4 元组
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            self.done = done# 关键：更新 self.done，否则会死循环
            self.episode_rewards.append(reward)

        # 计算总奖励
        for r in self.episode_rewards[::-1]:
            self.total_reward = r + self.gamma * self.total_reward

        print("total_reward:", self.total_reward)
        self.env.close()

    def show_animation(self,imgs):
        rc('animation', html='jshtml')
        fig, ax = plt.subplots(1,1,figsize=(5,3))
        frames = []
        text = ax.text(10, 20, '', fontsize=12, color='black')
        for i,img in enumerate(imgs):
            frame = [ax.imshow(img, animated=True)]
            frame.append(ax.text(10,20,f"Step: {i+1}",animated = True))
            frames.append(frame)
            ax.axis("off")
            ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)

            ani.save("cartpole.mp4", writer="ffmpeg")
            ani.save("cartpole.gif", writer="pillow")
            plt.close()
            return ani


