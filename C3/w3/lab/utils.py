import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import ipywidgets as widgets
from ipywidgets import interact, interact_manual


def sample_means(data, sample_size):
    means = []

    for _ in range(10_000):
        sample = np.random.choice(data, size=sample_size)
        means.append(np.mean(sample))

    return np.array(means)


def gaussian_clt():
    def _plot(mu, sigma, sample_size):
        #         mu = 10
        #         sigma = 5

        gaussian_population = np.random.normal(mu, sigma, 100_000)
        gaussiam_sample_means = sample_means(gaussian_population, sample_size)
        x_range = np.linspace(
            min(gaussiam_sample_means), max(gaussiam_sample_means), 100
        )

        sample_means_mean = np.mean(gaussiam_sample_means)
        sample_means_std = np.std(gaussiam_sample_means)
        clt_std = sigma / np.sqrt(sample_size)

        estimated_pop_sigma = sample_means_std * np.sqrt(sample_size)

        std_err = abs(clt_std - sample_means_std) / clt_std

        clt_holds = True if std_err < 0.1 else False

        #         print(f"Mean of sample means: {sample_means_mean:.2f}\n")
        #         print(f"Std of sample means: {sample_means_std:.2f}\n")
        #         print(f"Theoretical sigma: {clt_std:.2f}\n")
        #         print(f"Estimated population sigma: {estimated_pop_sigma:.2f}\n")

        #         print(f"Error: {std_err:.2f}\n")
        #         print(f"CLT holds?: {clt_holds}\n")

        mu2 = mu
        sigma2 = sigma / np.sqrt(sample_size)
        #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))
        fig, axes = plt.subplot_mosaic(
            [["top row", "top row"], ["bottom left", "bottom right"]], figsize=(10, 5)
        )

        ax1 = axes["top row"]
        ax2 = axes["bottom left"]
        ax3 = axes["bottom right"]
        sns.histplot(gaussian_population, stat="density", ax=ax1)
        ax1.set_title("Population Distribution")
        ax2.set_title("Sample Means Distribution")
        ax3.set_title("QQ Plot of Sample Means")

        sns.histplot(gaussiam_sample_means, stat="density", ax=ax2, label="hist")
        sns.kdeplot(
            data=gaussiam_sample_means,
            color="crimson",
            ax=ax2,
            label="kde",
            linestyle="dashed",
            fill=True,
        )
        ax2.plot(
            x_range,
            norm.pdf(x_range, loc=mu2, scale=sigma2),
            color="black",
            label="gaussian",
            linestyle="solid",
        )
        ax2.legend()

        stats.probplot(gaussiam_sample_means, plot=ax3, fit=True)
        plt.tight_layout()
        plt.show()

    mu_selection = widgets.FloatSlider(
        value=10.0,
        min=0.01,
        max=50.0,
        step=1.0,
        description="mu",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )

    sigma_selection = widgets.FloatSlider(
        value=5.0,
        min=0.01,
        max=20.0,
        step=0.1,
        description="sigma",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )

    sample_size_selection = widgets.IntSlider(
        value=2,
        min=2,
        max=100,
        step=1,
        description="sample_size",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    interact_manual(
        _plot, sample_size=sample_size_selection, mu=mu_selection, sigma=sigma_selection
    )


def binomial_clt():
    def _plot(n, p, sample_size):
        mu = n * p
        sigma = np.sqrt(n * p * (1 - p)) / np.sqrt(sample_size)
        N = n * sample_size
#         sigma = np.sqrt(n * p * (1 - p)) / np.sqrt(N)

        binomial_population = np.random.binomial(n, p, 100_000)

        binomial_sample_means = sample_means(binomial_population, sample_size)

        x_range = np.linspace(
            min(binomial_sample_means), max(binomial_sample_means), 100
        )

        condition_val = np.min([N * p, N * (1 - p)])

        condition = True if condition_val >= 5 else False

        sample_means_mean = np.mean(binomial_sample_means)
        sample_means_std = np.std(binomial_sample_means)
        clt_std = np.std(binomial_population) / np.sqrt(sample_size)

        estimated_pop_sigma = sample_means_std * np.sqrt(sample_size)

        std_err = abs(clt_std - sample_means_std) / clt_std

        clt_holds = True if std_err < 0.1 else False

        #         print(f"Value of N: {N}\n")
        print(f"Condition value: {condition_val:.1f}")

        #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig, axes = plt.subplot_mosaic(
            [["top row", "top row"], ["bottom left", "bottom right"]], figsize=(10, 5)
        )

        ax1 = axes["top row"]
        ax2 = axes["bottom left"]
        ax3 = axes["bottom right"]
        ax1.set_title("Population Distribution")
        ax2.set_title("Sample Means Distribution")
        ax3.set_title("QQ Plot of Sample Means")
        sns.histplot(binomial_population, stat="density", ax=ax1)

        sns.histplot(binomial_sample_means, stat="density", ax=ax2, label="hist")
        sns.kdeplot(
            data=binomial_sample_means,
            color="crimson",
            ax=ax2,
            label="kde",
            linestyle="dashed",
            fill=True,
        )
        ax2.plot(
            x_range,
            norm.pdf(x_range, loc=mu, scale=sigma),
            color="black",
            label="gaussian",
            linestyle="solid",
        )
        ax2.legend()
        stats.probplot(binomial_sample_means, plot=ax3, fit=True)
        plt.tight_layout()
        plt.show()

    #         print(f"Condition holds?: {condition} with value of {condition_val:.2f}\n")

    #         print(f"Mean of sample means: {sample_means_mean:.2f}\n")
    #         print(f"Std of sample means: {sample_means_std:.2f}\n")
    #         print(f"Theoretical sigma: {clt_std:.2f}\n")
    #         print(f"Estimated population sigma: {estimated_pop_sigma:.2f}\n")

    sample_size_selection = widgets.IntSlider(
        value=2,
        min=2,
        max=50,
        step=1,
        description="sample_size",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    n_selection = widgets.IntSlider(
        value=2,
        min=2,
        max=50,
        step=1,
        description="n",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    prob_success_selection = widgets.FloatSlider(
        value=0.5,
        min=0.01,
        max=0.99,
        step=0.1,
        description="p",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )

    interact_manual(
        _plot,
        sample_size=sample_size_selection,
        p=prob_success_selection,
        n=n_selection,
    )


def poisson_clt():
    def _plot(mu, sample_size):
        sigma = np.sqrt(mu) / np.sqrt(sample_size)

        poisson_population = np.random.poisson(mu, 100_000)

        poisson_sample_means = sample_means(poisson_population, sample_size)

        x_range = np.linspace(min(poisson_sample_means), max(poisson_sample_means), 100)

        fig, axes = plt.subplot_mosaic(
            [["top row", "top row"], ["bottom left", "bottom right"]], figsize=(10, 5)
        )

        ax1 = axes["top row"]
        ax2 = axes["bottom left"]
        ax3 = axes["bottom right"]
        ax1.set_title("Population Distribution")
        ax2.set_title("Sample Means Distribution")
        ax3.set_title("QQ Plot of Sample Means")
        sns.histplot(poisson_population, stat="density", ax=ax1)

        sns.histplot(poisson_sample_means, stat="density", ax=ax2, label="hist")
        sns.kdeplot(
            data=poisson_sample_means,
            color="crimson",
            ax=ax2,
            label="kde",
            linestyle="dashed",
            fill=True,
        )
        ax2.plot(
            x_range,
            norm.pdf(x_range, loc=mu, scale=sigma),
            color="black",
            label="gaussian",
            linestyle="solid",
        )
        ax2.legend()
        stats.probplot(poisson_sample_means, plot=ax3, fit=True)
        plt.tight_layout()
        plt.show()

    sample_size_selection = widgets.IntSlider(
        value=2,
        min=2,
        max=50,
        step=1,
        description="sample_size",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    mu_selection = widgets.FloatSlider(
        value=1.5,
        min=0.01,
        max=5.0,
        #         step=1.0,
        description="mu",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )

    interact_manual(_plot, sample_size=sample_size_selection, mu=mu_selection)


def plot_kde_and_qq(sample_means_data, mu_sample_means, sigma_sample_means):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Define the x-range for the Gaussian curve (this is just for plotting purposes)
    x_range = np.linspace(min(sample_means_data), max(sample_means_data), 100)

    # Histogram of sample means (blue)
    sns.histplot(sample_means_data, stat="density", label="hist", ax=ax1)

    # Estimated PDF of sample means (red)
    sns.kdeplot(
        data=sample_means_data,
        color="crimson",
        label="kde",
        linestyle="dashed",
        fill=True,
        ax=ax1,
    )

    # Gaussian curve with estimated mu and sigma (black)
    ax1.plot(
        x_range,
        norm.pdf(x_range, loc=mu_sample_means, scale=sigma_sample_means),
        color="black",
        label="gaussian",
    )

    res = stats.probplot(sample_means_data, plot=ax2, fit=True)

    ax1.legend()
    plt.show()
