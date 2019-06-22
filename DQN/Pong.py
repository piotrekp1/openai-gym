import gym
import tensorflow as tf

from agents.DQN_Agent import DQN_Agent
from utils.StateProcessors import DQN_StateProcessor


NUM_EPISODES = 100000
EP_LENGTH = 100000

env = gym.make('Pong-v0')

agent = DQN_Agent(env)
state_processor = DQN_StateProcessor()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    agent.set_session(sess)

    for i_episode in range(NUM_EPISODES):

        observation = env.reset()
        state_processor.clear_episode()
        agent.next_episode()
        observation = state_processor.process_state(observation)

        done = False
        while not done:
            if i_episode % 3 == 0:
                env.render()

            action = agent.take_action(observation)
            observation_new, reward, done, info = env.step(action)
            observation_new = state_processor.process_state(observation_new)
            agent.observe(observation, action, reward, observation_new, done)

            observation = observation_new

            agent.step_train()
