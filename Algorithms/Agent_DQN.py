
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Definizione della rete neurale per DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)  
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))  
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Definizione della classe agente DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size, model=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model != None:
            self.model = model
        else:
            self.model = DQN(state_dim, action_dim).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()


    def act(self, state, train=True):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        if train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim) 

        with torch.no_grad():
            q_values = self.model(state_tensor)

        return torch.argmax(q_values, dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, targets.detach())
        loss.backward()
        self.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

    def save(self, dir):
        torch.save(self.model.state_dict(), dir)