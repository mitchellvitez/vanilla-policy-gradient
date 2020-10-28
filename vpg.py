import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

from torch import nn
from torch import optim
from torch import tensor

class PolicyEstimator():
    def __init__(self, env):
        self.num_observations = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        self.network = nn.Sequential(
            nn.Linear(self.num_observations, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_actions),
            nn.Softmax(dim=-1)
        )

    def predict(self, observation):
        return self.network(torch.FloatTensor(observation))

def vanilla_policy_gradient(env, estimator, num_episodes=1500, batch_size=10, discount_factor=0.99, render=False,
                            early_exit_reward_amount=None):
    total_rewards, batch_rewards, batch_observations, batch_actions = [], [], [], []
    batch_counter = 1

    optimizer = optim.Adam(estimator.network.parameters(), lr=0.01)
    action_space = np.arange(env.action_space.n) # [0, 1] for cartpole (either left or right)

    for current_episode in range(num_episodes):
        observation = env.reset()
        rewards, actions, observations = [], [], []

        while True:
            if render:
                env.render()

            # use policy to make predictions and run an action
            action_probs = estimator.predict(observation).detach().numpy()
            action = np.random.choice(action_space, p=action_probs) # randomly select an action weighted by its probability

            # push all episodic data, move to next observation
            observations.append(observation)
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)

            if done:
                # apply discount to rewards
                r = np.full(len(rewards), discount_factor) ** np.arange(len(rewards)) * np.array(rewards)
                r = r[::-1].cumsum()[::-1]
                discounted_rewards = r - r.mean()

                # collect the per-batch rewards, observations, actions
                batch_rewards.extend(discounted_rewards)
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                if batch_counter >= batch_size:
                    # reset gradient
                    optimizer.zero_grad()

                    # tensorify things
                    batch_rewards = torch.FloatTensor(batch_rewards)
                    batch_observationss = torch.FloatTensor(batch_observations)
                    batch_actions = torch.LongTensor(batch_actions)

                    # calculate loss
                    logprob = torch.log(estimator.predict(batch_observations))
                    batch_actions = batch_actions.reshape(len(batch_actions), 1)
                    selected_logprobs = batch_rewards * torch.gather(logprob, 1, batch_actions).squeeze()
                    loss = -selected_logprobs.mean()

                    # backprop/optimize
                    loss.backward()
                    optimizer.step()

                    # reset the batch
                    batch_rewards, batch_observations, batch_actions = [], [], []
                    batch_counter = 1

                # get running average of last 100 rewards, print every 100 episodes
                average_reward = np.mean(total_rewards[-100:])
                if current_episode % 100 == 0:
                    print(f"average of last 100 rewards as of episode {current_episode}: {average_reward:.2f}")

                # quit early if average_reward is high enough
                if early_exit_reward_amount and average_reward > early_exit_reward_amount:
                    return total_rewards

                break

    return total_rewards

if __name__ == '__main__':
    # create environment
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    # actually run the algorithm
    rewards = vanilla_policy_gradient(env, PolicyEstimator(env), num_episodes=1500)

    # moving average
    moving_average_num = 100
    def moving_average(x, n=moving_average_num):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    # plotting
    plt.scatter(np.arange(len(rewards)), rewards, label='individual episodes')
    plt.plot(moving_average(rewards), label=f'moving average of last {moving_average_num} episodes')
    plt.title(f'Vanilla Policy Gradient on {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
