#!/usr/bin/env python

##
#
# Quick script for making convergence plots based on saved tensorboard data.
#
##

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_mean_std_reward_per_step(log_dir):
    """Load data from a log_dir.

    Load the average reward and standard deviation at each step from the
    given tensorboard log directory.

    Args:
        log_dir (str): Path to the tensorboard log directory

    Returns:
        mean_reward (np.array): The mean reward at each timestep
        std_reward (np.array): The standard deviation of the reward at each timestep
        steps (np.array): The timestep at which each reward was computed
    """
    ea = EventAccumulator(log_dir)
    ea.Reload()

    # Get the average reward at each timestep
    rew_mean_data = np.array(ea.Scalars("eval/episode_reward"))
    rew_std_data = np.array(ea.Scalars("eval/episode_reward_std"))
    reward = []
    std = []
    steps = []
    for i in range(len(rew_mean_data)):
        reward.append(rew_mean_data[i].value)
        std.append(rew_std_data[i].value)
        steps.append(rew_mean_data[i].step)

    return np.array(reward), np.array(std), np.array(steps)


def plot_learning_curve(log_dir, label):
    """Make a plot of the learning curve for the given settings.

    Args:
        log_dir (str): Path to the tensorboard log directory
        label (str): Label for the curve
    """
    mean_reward, std_reward, steps = get_mean_std_reward_per_step(log_dir)

    # Plot the mean and standard deviation
    plt.plot(steps[:], mean_reward, label=label, linewidth=2)
    plt.fill_between(steps[:], mean_reward - std_reward, mean_reward + std_reward, alpha=0.3)


def compare_learning_curves():
    """Make a plot comparing several learning curves."""
    plot_learning_curve("/tmp/mjx_brax_logs/cart_pole_series", "Series")
    plot_learning_curve("/tmp/mjx_brax_logs/cart_pole_hierarchy", "Hierarchy")
    plot_learning_curve("/tmp/mjx_brax_logs/cart_pole_parallel", "Parallel")

    # Use a nicer formatting for large numbers
    ax = plt.gca()
    mkfunc = lambda x, pos: "%1.0fM" % (x * 1e-6) if x >= 1e6 else "%1.0fK" % (x * 1e-3) if x >= 1e3 else "%1.1f" % x
    mkformatter = ticker.FuncFormatter(mkfunc)
    ax.xaxis.set_major_formatter(mkformatter)

    plt.title("Cart-Pole Swingup")
    plt.ylabel("Reward")
    plt.xlabel("Simulated Timesteps")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plt.title("Cart-Pole Swingup: N modules with 4 hidden units each")
    module_sizes = [1, 4, 7, 10, 13, 16, 19, 22, 25, 30]

    # plt.subplot(1, 3, 1)
    composition_type = "parallel"
    series_mean = []
    series_std = []
    for num_modules in module_sizes:
        mean, std, steps = get_mean_std_reward_per_step(
            f"/tmp/mjx_brax_logs/cart_pole_{composition_type}_{num_modules}"
        )
        series_mean.append(mean[-1])
        series_std.append(std[-1])

    plt.plot(module_sizes, series_mean, label="Series", linewidth=2)
    plt.fill_between(
        module_sizes,
        np.array(series_mean) - np.array(series_std),
        np.array(series_mean) + np.array(series_std),
        alpha=0.3,
    )
    # plt.ylim(-1700, -400)

    # plt.subplot(1, 3, 2)
    composition_type = "series"
    parallel_mean = []
    parallel_std = []
    for num_modules in module_sizes:
        mean, std, steps = get_mean_std_reward_per_step(
            f"/tmp/mjx_brax_logs/cart_pole_{composition_type}_{num_modules}"
        )
        parallel_mean.append(mean[-1])
        parallel_std.append(std[-1])

    plt.plot(module_sizes, parallel_mean, label="Parallel", linewidth=2)
    plt.fill_between(
        module_sizes,
        np.array(parallel_mean) - np.array(parallel_std),
        np.array(parallel_mean) + np.array(parallel_std),
        alpha=0.3,
    )
    # plt.ylim(-1700, -400)

    # plt.subplot(1, 3, 3)
    composition_type = "hierarchy"
    hierarchy_mean = []
    hierarchy_std = []
    for num_modules in module_sizes:
        mean, std, steps = get_mean_std_reward_per_step(
            f"/tmp/mjx_brax_logs/cart_pole_{composition_type}_{num_modules}"
        )
        hierarchy_mean.append(mean[-1])
        hierarchy_std.append(std[-1])

    plt.plot(module_sizes, hierarchy_mean, label="Hierarchy", linewidth=2)
    plt.fill_between(
        module_sizes,
        np.array(hierarchy_mean) - np.array(hierarchy_std),
        np.array(hierarchy_mean) + np.array(hierarchy_std),
        alpha=0.3,
    )
    # plt.ylim(-1700, -400)

    plt.ylabel("Reward")
    plt.xlabel("Number of modules")
    plt.legend()

    plt.show()
