class AbstractAgent:
    def __init__(self, env):
        self.env = env

    def take_action(self, state):
        pass

    def observe(self, state, action, reward, next_state, done):
        pass

    def step_train(self):
        pass
