import numpy as np
import matplotlib.pyplot as plt


def estimate_gaussian_params(sample):
    ### START CODE HERE ###
    mu = np.mean(sample)
    sigma = np.std(sample)
    ### END CODE HERE ###

    return mu, sigma


def estimate_binomial_params(sample):
    ### START CODE HERE ###
    n = 30
    p = (sample / n).mean()
    ### END CODE HERE ###

    return n, p


def estimate_uniform_params(sample):
    ### START CODE HERE ###
    a = sample.min()
    b = sample.max()
    ### END CODE HERE ###

    return a, b


def plot_gaussian_distributions(gaussian_0, gaussian_1, gaussian_2):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(gaussian_0, alpha=0.5, label="gaussian_0", bins=32)
    ax.hist(gaussian_1, alpha=0.5, label="gaussian_1", bins=32)
    ax.hist(gaussian_2, alpha=0.5, label="gaussian_2", bins=32)
    ax.set_title("Histograms of Gaussian distributions")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequencies")
    ax.legend()
    plt.show()
    
    
def plot_binomial_distributions(binomial_0, binomial_1, binomial_2):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(binomial_0, alpha=0.5, label="binomial_0")
    ax.hist(binomial_1, alpha=0.5, label="binomial_1")
    ax.hist(binomial_2, alpha=0.5, label="binomial_2")
    ax.set_title("Histograms of Binomial distributions")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequencies")
    ax.legend()
    plt.show()