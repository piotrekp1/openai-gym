import random


class RoundBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, el):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = el
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def whole_buffer(self):
        return self.memory[:self.position] + self.memory[self.position:]

    def __len__(self):
        return len(self.memory)
