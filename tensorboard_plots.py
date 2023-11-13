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
    # plt.fill_between(steps[:], mean_reward - std_reward,
    #                 mean_reward + std_reward, alpha=0.3)


def make_subplot(num_blocks):
    """Make a little subplot."""
    series = f"/tmp/mjx_brax_logs/cart_pole_series_{num_blocks}"
    parallel = f"/tmp/mjx_brax_logs/cart_pole_parallel_{num_blocks}"
    hierarchy = f"/tmp/mjx_brax_logs/cart_pole_hierarchy_{num_blocks}"
    plot_learning_curve(series, "Series")
    plot_learning_curve(parallel, "Parallel")
    plot_learning_curve(hierarchy, "Hierarchy")

    # Use a nicer formatting for large numbers
    ax = plt.gca()
    mkfunc = lambda x, pos: "%1.0fM" % (x * 1e-6) if x >= 1e6 else "%1.0fK" % (x * 1e-3) if x >= 1e3 else "%1.1f" % x
    mkformatter = ticker.FuncFormatter(mkfunc)
    ax.xaxis.set_major_formatter(mkformatter)

    plt.title(f"{num_blocks} Blocks")
    plt.ylabel("Reward")
    plt.xlabel("Simulated Timesteps")


if __name__ == "__main__":
    plt.subplot(1, 3, 1)
    make_subplot(3)

    plt.subplot(1, 3, 2)
    make_subplot(9)

    plt.subplot(1, 3, 3)
    make_subplot(27)

    plt.legend()
    plt.show()
