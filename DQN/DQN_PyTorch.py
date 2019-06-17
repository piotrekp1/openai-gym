import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import time


class RawDQN(nn.Module):
    def __init__(self):
        h, w = 42, 42
        num_moves = 6

        super(RawDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, num_moves)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQN_PyTorch():
    def __init__(self, learning_rate, DISCOUNT, device):
        self.device = device

        self.dqn = RawDQN().to(self.device)
        self.DISCOUNT = DISCOUNT
        self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=learning_rate)

        self.updates = 0

        self.time_start = time.time()

    def predict(self, x):
        with torch.no_grad():
            return self.dqn.forward(x)

    def num_updates(self):
        return self.updates

    def train_on_batch(self, minibatch, behaviour_network):
        states, actions, rewards, next_states = zip(*minibatch)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                next_states)), device=self.device, dtype=torch.uint8)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        next_states = torch.tensor([next_state for next_state in next_states if next_state is not None], device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)

        next_state_values = torch.zeros(len(minibatch), device=self.device)

        state_action_values = self.dqn.forward(states).gather(1, actions.unsqueeze(1))

        next_state_preds = behaviour_network.predict(next_states).max(1)[0].detach()
        next_state_values[non_final_mask] = next_state_preds

        expected_state_action_values = (next_state_values * self.DISCOUNT) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates += 1

        if self.updates % 50 == 0:
            print('updates: ', self.updates, ' time: ', time.time() - self.time_start)
