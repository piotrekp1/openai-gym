from .AbstractAgent import AbstractAgent


class RandomAgent(AbstractAgent):
    def take_action(self, state):
        return self.env.action_space.sample()
