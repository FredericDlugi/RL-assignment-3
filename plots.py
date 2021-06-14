import matplotlib.figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import os
import gym
from mpl_toolkits.axes_grid1 import AxesGrid


def show_q_tables(files, titles):
    fig : matplotlib.figure.Figure = plt.figure(figsize=(3*6.4, 3*4.8))
    grid = AxesGrid(fig, 111, nrows_ncols=(len(files),3), axes_pad=0.05, cbar_mode='single', cbar_location='right', cbar_pad=0.1)


    q_tables = np.empty((len(files), 21,16,3))
    for i, file in enumerate(files):
        q_tables[i, :] = np.load(file)

    min_val = np.array(q_tables).min()
    max_val = np.array(q_tables).max()

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

    for i, ax in enumerate(grid):
        q_table = np.load(files[int(i / 3)])
        ax.set_ylabel("Velocity (m/s)")
        ax.set_xlabel("Position (m)")
        ax.set_yticks(np.linspace(0, q_table[:, :, i%3].shape[1] - 1, 6))
        ax.set_yticklabels(vel_ticks)
        ax.set_xticks(np.linspace(0, q_table[:, :, i%3].shape[0] - 1, 7))
        ax.set_xticklabels(pos_ticks)
        im = ax.imshow(np.rot90(q_table[:,:,i % 3]), norm = SymLogNorm(150,vmin = min_val, vmax = max_val), cmap='jet')

        cbar = grid.cbar_axes[0].colorbar(im)

    grid.axes_row[0][0].set_title('action "push left"')
    grid.axes_row[0][1].set_title('action "no push"')
    grid.axes_row[0][2].set_title('action "push right"')

    for i in range(len(titles)):
        grid.axes_column[0][i].set_ylabel(titles[i], rotation=0, size='large',x=-20)

    plt.suptitle('Q-Table for SARSA(\lambda)')



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


def show_returns(files, number_of_episodes : int, mean_var : bool, labels = None):
    returns = np.empty((len(files),number_of_episodes))
    for i,file in enumerate(files):
        returns[i,:] = np.load(file)

    episodes = np.arange(1, number_of_episodes + 1, 1)

    if mean_var:
        return_mean = np.mean(returns, axis=0)
        return_var = np.var(returns, axis=0)
        plt.plot(episodes, return_mean)
        plt.fill_between(episodes, return_mean - np.sqrt(return_var), return_mean + np.sqrt(return_var), color='gray',
                         alpha=0.5)
        plt.legend(['MEAN(return)', 'STD(return)'])
    else:
        plt.plot(episodes, returns.transpose(), alpha=0.6)
        plt.legend(labels)

    plt.yscale('symlog')
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


    show_q_tables(['data/SARSA_Q_0.95_2000_franek.npy','data/SARSA_Q_0_5000_.npy', 'data/SARSA_Q_0.5_5000_.npy','data/SARSA_Q_0.75_5000_.npy', 'data/SARSA_Q_0.99_5000_.npy'], ['Optimal policy', '\lambda = 0', '\lambda = 0.5', '\lambda = 0.75', '\lambda = 0.99'])

    plt.show()


    
    ##### Task 2.1



    sarsa_return_data = ['data/SARSA_RETURN_0.95_2000_franek.npy', 'data/SARSA_RETURN_0.95_2000_moritz.npy',
                         'data/SARSA_RETURN_0.95_2000_frederic.npy', 'data/SARSA_RETURN_0.95_2000_robin.npy']
    show_returns(sarsa_return_data, 2000, True)
    plt.title(r'SARSA(\lambda)')
    plt.show()

    ###### Task 2.2

    ## Returns
    sarsa_return_data = ['data/SARSA_RETURN_0_5000_.npy', 'data/SARSA_RETURN_0.5_5000_.npy', 'data/SARSA_RETURN_0.75_5000_.npy', 'data/SARSA_RETURN_0.99_5000_.npy']
    show_returns(sarsa_return_data, 5000, False, ['\lambda = 0', '\lambda = 0.5', '\lambda = 0.75', '\lambda = 0.99'])
    plt.title(r'SARSA(\lambda) on different \lambda')
    plt.show()


    ## Policy
    show_policy('data/SARSA_POLICY_0.95_2000_franek.npy')
    plt.title('SARSA(\lambda=0.95) Q_{Init}=0')
    plt.show()
    show_policy('data/SARSA_POLICY_0_5000_.npy')
    plt.title('SARSA(\lambda=0), Q_{Init}=10000')
    plt.show()
    show_policy('data/SARSA_POLICY_0.5_5000_.npy')
    plt.title('SARSA(\lambda=0.5), Q_{Init}=10000')
    plt.show()
    show_policy('data/SARSA_POLICY_0.75_5000_.npy')
    plt.title('SARSA(\lambda=0.75), Q_{Init}=10000')
    plt.show()
    show_policy('data/SARSA_POLICY_0.99_5000_.npy')
    plt.title('SARSA(\lambda=0.99), Q_{Init}=10000')
    plt.show()

    ## Visits
    show_visitis('data/SARSA_VISITS_0.95_2000_franek.npy')
    plt.title('SARSA(\lambda=0.95) Q_{Init}=0')
    plt.show()
    show_visitis('data/SARSA_VISITS_0_5000_.npy')
    plt.title('SARSA(\lambda=0), Q_{Init}=10000')
    plt.show()
    show_visitis('data/SARSA_VISITS_0.5_5000_.npy')
    plt.title('SARSA(\lambda=0.5), Q_{Init}=10000')
    plt.show()
    show_visitis('data/SARSA_VISITS_0.75_5000_.npy')
    plt.title('SARSA(\lambda=0.75), Q_{Init}=10000')
    plt.show()
    show_visitis('data/SARSA_VISITS_0.99_5000_.npy')
    plt.title('SARSA(\lambda=0.99), Q_{Init}=10000')
    plt.show()




    watkins_return_data = ['data/WATKIN_RETURN_0.95_2000_franek.npy', 'data/WATKIN_RETURN_0.95_2000_frederic.npy','data/WATKIN_RETURN_0.95_2000_robin.npy','data/WATKIN_RETURN_0.95_2000_moritz.npy']
    show_returns(watkins_return_data, 2000, True)
    plt.title('Off-policy Watkins Q (\lambda = 0.95)')

    plt.show()

    true_sarsa_return_data = ['data/TRUE_SARSA_RETURN_0.95_2000_franek.npy', 'data/TRUE_SARSA_RETURN_0.95_2000_frederic.npy','data/TRUE_SARSA_RETURN_0.95_2000_robin.npy','data/TRUE_SARSA_RETURN_0.95_2000_moritz.npy']
    show_returns(true_sarsa_return_data, 2000, True)
    plt.title('True-online Sarsa(\lambda = 0.95)')

    plt.show()

