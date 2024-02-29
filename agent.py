import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Q_network import TransformerQNetwork
from reply_buffer import ReplayBuffer
from collections import namedtuple
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

class DQNAgent:
    def __init__(self, input_dim, output_dim, hidden_dim, env, capacity = 10000, batch_size = 32, lr=0.001, gamma=0.99, epsilon=0.1):
        self.q_network = TransformerQNetwork(input_dim, hidden_dim, output_dim)
        self.target_network = TransformerQNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.env = env
        self.capacity = capacity
        self.total_step = 0
        self.Transition_action = namedtuple('TransitionAction', ('state', 'action','next_state', 'reward', 'done', 'goal'))
        self.memory = ReplayBuffer(capacity, self.Transition_action)
        abs_path = os.path.dirname(__file__)
        logs_folder = os.path.join(abs_path, "logs")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        log_dir = os.path.join(logs_folder, "bit_string" + "_" + current_time)
        self.writer = SummaryWriter(log_dir)

    def select_action(self, state):
        self.total_step += 1
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)  # Choose random action
        else:
            state = torch.tensor(state).view(1, -1)
            state = state.unsqueeze(0)
            batched_state = state.expand(self.batch_size, -1, -1)
            goal_state = self.env.goal
            goal_state = torch.tensor(goal_state).view(1, -1)
            goal_state.unsqueeze(0)
            batched_goal = goal_state.expand(self.batch_size, -1, -1)
            probs = self.q_network(batched_state, batched_goal)
            probs = torch.sum(probs, dim=0)
            probs = torch.nn.functional.softmax(probs, dim=0)
            return torch.argmax(probs).item()

    def learn(self):
        if self.batch_size < len(self.memory):
            transitions = self.memory.sample(self.batch_size)
            batch = self.Transition_action(*zip(*transitions))
            state_batch = [torch.tensor(lst).unsqueeze(1) for lst in batch.state]
            state_batch = torch.stack(state_batch, dim = 0).permute(0, 2, 1)
            action_batch = torch.tensor(batch.action)
            reward_batch = torch.tensor(batch.reward, dtype=torch.float32)

            next_state_batch = [torch.tensor(lst).unsqueeze(1) for lst in batch.next_state]
            next_state_batch = torch.stack(next_state_batch, dim = 0).permute(0, 2, 1)

            goal_batch = [torch.tensor(lst).unsqueeze(1) for lst in batch.goal]
            goal_batch = torch.stack(goal_batch, dim = 0).permute(0, 2, 1)

            state_q_values = self.q_network(state_batch, goal_batch).gather(1, action_batch.unsqueeze(1))
            
            next_state_values = self.target_network(next_state_batch, goal_batch)
            next_state_values = next_state_values.max(1)[0].view(-1, 1)
            expected_q_values = (next_state_values * self.gamma) + reward_batch
            loss = self.loss_fn(state_q_values, expected_q_values)
            loss_q = loss
            self.optimizer.zero_grad()
            loss.backward()
            

            gradients = []
            for param in self.q_network.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))
            gradients = torch.cat(gradients)
            gradient_norm = torch.norm(gradients, 2)   
            self.optimizer.step()

            self.writer.add_scalar("loss", loss_q, self.total_step)
            self.writer.add_scalar("gradient_norm", gradient_norm, self.total_step)

    def store_exp(self, state, action, next_state, reward, done, goal):
        if len(self.memory) < self.capacity:
            self.memory.push(state, action, next_state, reward, done, goal)
    
    def update_q_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())