from utils.RoundBuffer import RoundBuffer

import numpy as np
from skimage.measure import block_reduce


class DQN_StateProcessor:
    def __init__(self):
        self.state_cap = 4
        self.cur_processed_state = RoundBuffer(self.state_cap)

        self.obs_shape = (42, 42)

    def process_state(self, state):
        state = self.__preprocess_image(state)
        self.cur_processed_state.push(state)
        return self.cur_processed_state.whole_buffer()

    def clear_episode(self):
        self.cur_processed_state.memory = [np.zeros(self.obs_shape) for _ in range(self.state_cap)]
        self.cur_processed_state.position = 0

    @staticmethod
    def __preprocess_image(image):
        image = image[29:197, :, :]
        bw = np.pad(np.dot(image, [0.2989, 0.5870, 0.1140]), [(0, 0), (4, 4)], mode='minimum')
        res = block_reduce(bw, (4, 4), np.max)
        del image
        del bw
        return res
