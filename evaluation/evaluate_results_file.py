import yaml
# from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np 

if __name__ == "__main__":
    batch1_fixed = "../runs/snn_cmaes15072021_083523/results.txt"
    batch10_fixed = "../runs/snn_cmaes13072021_102418/results.txt"
    batch10_random = "../runs/snn_cmaes13072021_150755/results.txt"

    with open(batch1_fixed, "r") as f:
        results_batch1_fixed = yaml.load(f, Loader=yaml.FullLoader)

    with open(batch10_fixed, "r") as f:
        results_batch10_fixed = yaml.load(f, Loader=yaml.FullLoader)

    with open(batch10_random, "r") as f:
        results_batch10_random = yaml.load(f, Loader=yaml.FullLoader)

    plt.subplot(1, 2, 1)
    plt.plot(np.sqrt(results_batch1_fixed['fitness_hist'][:1500]) * 180 / np.pi, label='fixed sequence')
    plt.plot(np.sqrt(results_batch10_fixed['fitness_hist'][:1500]) * 180 / np.pi, label='fixed batch')
    plt.plot(np.sqrt(results_batch10_random['fitness_hist'][:1500]) * 180 / np.pi, label='random batch')
    plt.ylim([0, 12])
    plt.title('Training error')
    plt.ylabel('average absolute error [deg]')
    plt.xlabel('generation')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(np.sqrt(results_batch1_fixed['validation_hist'][:1500]) * 180 / np.pi, label='fixed sequence')
    plt.plot(np.sqrt(results_batch10_fixed['validation_hist'][:1500]) * 180 / np.pi, label='fixed batch')
    plt.plot(np.sqrt(results_batch10_random['validation_hist'][:1500]) * 180 / np.pi, label='random batch')
    plt.ylim([0, 12])
    plt.title('Validation error')
    plt.ylabel('average absolute error [deg]')
    plt.xlabel('generation')
    plt.legend()
    plt.show()
    # print(results)
    