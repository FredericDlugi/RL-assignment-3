import numpy as np
import matplotlib.pyplot as plt
import os
import gym

def show_visitis(file):
    visit_counts = np.load(file)
    visit_counts_states = visit_counts[:, :, 0] + \
                          visit_counts[:, :, 1] + visit_counts[:, :, 2]

    vel_ticks = [
        f"{i:.2f}" for i in np.linspace(
            env.observation_space.low[1],
            env.observation_space.high[1],
            6)]
    pos_ticks = [
        f"{i:.2f}" for i in np.linspace(
            env.observation_space.low[0],
            env.observation_space.high[0],
            7)]
    plt.yticks(np.linspace(0, visit_counts.shape[1] - 1, 6), vel_ticks)
    plt.xticks(np.linspace(0, visit_counts.shape[0] - 1, 7), pos_ticks)
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Position (m)")
    plt.imshow(np.rot90(visit_counts_states.astype(float)))
    color_ticks = list(range(0, int(np.max(visit_counts_states)), 70_000)) + [np.max(visit_counts_states)]
    plt.colorbar(ticks=color_ticks)


def show_returns(files, number_of_episodes):
    returns = np.empty((len(files),number_of_episodes))
    for i,file in enumerate(files):
        returns[i,:] = np.load(file)
    return_mean = np.mean(returns, axis=0)
    return_var = np.var(returns, axis=0)

    episodes = np.arange(1, number_of_episodes+1, 1)


    plt.plot(episodes, return_mean)
    plt.yscale('symlog')
    plt.fill_between(episodes, return_mean - np.sqrt(return_var), return_mean + np.sqrt(return_var), color='gray', alpha=0.5)
    plt.legend(['MEAN(return)','STD(return)'])
    plt.xlabel('episode')
    plt.ylabel('episodic reward (return)')
    plt.ylim([-10**6, 0])
    plt.grid(True, which="both")


def show_policy(file):
    learned_policy = np.load(file)

    indicies_1 = np.argwhere(learned_policy == 1)
    plt.scatter(indicies_1[:, 0], indicies_1[:, 1],
                marker=".", label="no push")
    indicies_0 = np.argwhere(learned_policy == 0)
    plt.scatter(indicies_0[:, 0], indicies_0[:, 1],
                marker="<", label="push left")
    indicies_2 = np.argwhere(learned_policy == 2)
    plt.scatter(indicies_2[:, 0], indicies_2[:, 1],
                marker=">", label="push right")

    vel_ticks = [
        f"{i:.2f}" for i in np.linspace(
            env.observation_space.low[1],
            env.observation_space.high[1],
            6)]
    pos_ticks = [
        f"{i:.2f}" for i in np.linspace(
            env.observation_space.low[0],
            env.observation_space.high[0],
            7)]
    plt.yticks(np.linspace(0, learned_policy.shape[1] - 1, 6), vel_ticks)
    plt.xticks(np.linspace(0, learned_policy.shape[0] - 1, 7), pos_ticks)
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Position (m)")
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))


if __name__ == "__main__":
    env = gym.make('MountainCar-v0').env
    env.reset()

    show_policy('data/SARSA_POLICY_0_5000_.npy')
    plt.show()

    show_policy('data/SARSA_POLICY_0.95_2000_franek.npy')
    plt.show()
    env.reset()




    sarsa_data = ['data/VISTIS_Q_0_5000_.npy']
    show_visitis('data/SARSA_VISITS_0_5000_.npy')
    plt.show()

    show_visitis('data/SARSA_VISITS_0.95_2000_franek.npy')
    plt.show()

    show_policy()


    #show_returns(watkins_return_data, 2000)

    watkins_return_data = ['data/WATKIN_RETURN_0.95_2000_franek.npy', 'data/WATKIN_RETURN_0.95_2000_frederic.npy','data/WATKIN_RETURN_0.95_2000_robin.npy','data/WATKIN_RETURN_0.95_2000_moritz.npy']

    show_returns(watkins_return_data, 2000)
    #sarsa_return_data = ['data/SARSA_RETURN_0.95_2000_franek.npy', 'data/SARSA_RETURN_0.95_2000_moritz.npy', 'data/SARSA_RETURN_0.95_2000_frederic.npy', 'data/SARSA_RETURN_0.95_2000_robin.npy']
    #show_returns(sarsa_return_data, 2000)
    # plt.title(r'SARSA(\lambda)')
    plt.show()


