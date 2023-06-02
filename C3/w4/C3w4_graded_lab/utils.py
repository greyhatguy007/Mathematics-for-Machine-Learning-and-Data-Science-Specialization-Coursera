import string
import random
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import lognorm
import ipywidgets as widgets
from ipywidgets import interact_manual
from dataclasses import dataclass


def sample_size_diff_means(mu1, mu2, sigma, alpha=0.05, beta=0.20, two_sided=True):
    delta = abs(mu2 - mu1)

    if two_sided:
        alpha = alpha / 2

    n = (
        (np.square(sigma) + np.square(sigma))
        * np.square(stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - beta))
    ) / np.square(delta)

    return math.ceil(n)


def sample_size_diff_proportions(p1, p2, alpha=0.05, beta=0.20, two_sided=True):
    k = 1

    q1, q2 = (1 - p1), (1 - p2)
    p_bar = (p1 + k * p2) / (1 + k)
    q_bar = 1 - p_bar
    delta = abs(p2 - p1)

    if two_sided:
        alpha = alpha / 2

    n = np.square(
        np.sqrt(p_bar * q_bar * (1 + (1 / k))) * stats.norm.ppf(1 - (alpha))
        + np.sqrt((p1 * q1) + (p2 * q2 / k)) * stats.norm.ppf(1 - beta)
    ) / np.square(delta)

    return math.ceil(n)


def generate_user_ids(num_users):

    user_ids = []
    
    while len(user_ids) < num_users:
        new_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        
        if new_id not in user_ids:
            user_ids.append(new_id)
    
    return user_ids


def run_ab_test_background_color(n_days):
    
    np.random.seed(42)
    
    daily_users = 104
    n_control = int(daily_users*n_days*np.random.uniform(0.98, 1.02))
    n_variation = int(daily_users*n_days*np.random.uniform(0.98, 1.02))
    data_control = lognorm.rvs(0.5, loc=0, scale=np.exp(1)*10.5, size=n_control)
    data_variation = lognorm.rvs(0.5, loc=0, scale=np.exp(1)*11.01, size=n_variation)    
    
    user_ids = generate_user_ids(n_control+n_variation)
    
    control_dict = {"user_id": user_ids[:n_control], "user_type": "control", "session_duration": data_control}
    variation_dict = {"user_id": user_ids[n_control:], "user_type": "variation", "session_duration": data_variation}
    
    control_df = pd.DataFrame(control_dict)
    variation_df = pd.DataFrame(variation_dict)
    
    df_ab_test = pd.concat([control_df, variation_df])

    df_ab_test = df_ab_test.sample(frac=1).reset_index(drop=True)
    
    return df_ab_test
    
    
    
def run_ab_test_personalized_feed(n_days):
    
    np.random.seed(69)
    
    daily_users = 519
    n_control = int(daily_users*n_days*np.random.uniform(0.98, 1.02))
    n_variation = int(daily_users*n_days*np.random.uniform(0.98, 1.02))
    data_control = np.random.choice([0, 1], size=n_control, p=[1-0.12, 0.12])
    data_variation = np.random.choice([0, 1], size=n_variation, p=[1-0.15, 0.15])
    
    user_ids = generate_user_ids(n_control+n_variation)
    
    control_dict = {"user_id": user_ids[:n_control], "user_type": "control", "converted": data_control}
    variation_dict = {"user_id": user_ids[n_control:], "user_type": "variation", "converted": data_variation}
    
    control_df = pd.DataFrame(control_dict)
    variation_df = pd.DataFrame(variation_dict)
    
    df_ab_test = pd.concat([control_df, variation_df])

    df_ab_test = df_ab_test.sample(frac=1).reset_index(drop=True)
    
    return df_ab_test


@dataclass
class estimation_metrics_prop:
    n: int
    x: int
    p: float
        
    def __repr__(self):
        return f"sample_params(n={self.n}, x={self.x}, p={self.p:.3f})"
    
    
def AB_test_dashboard(z_statistic_diff_proportions, reject_nh_z_statistic):
    def _AB(n1, x1, n2, x2, alpha):
        
        m1 = estimation_metrics_prop(n=n1, x=x1, p=x1/n1)
        m2 = estimation_metrics_prop(n=n2, x=x2, p=x2/n2)
        z = z_statistic_diff_proportions(m1, m2)
        reject_nh = reject_nh_z_statistic(z, alpha=alpha)
        print(f"The null hypothesis can be rejected at the {alpha:.5f} level of significance: {reject_nh}\n")

        msg = "" if reject_nh else " not"
        print(f"There is{msg} enough statistical evidence against H0.\nThus it can be concluded that there is{msg} a statistically significant difference between the two proportions.")

    n1_selection = widgets.IntText(
        value=4632,
        description='Users A:',
        disabled=False
    )

    n2_selection = widgets.IntText(
        value=4728,
        description='Users B:',
        disabled=False
    )
    
    x1_selection = widgets.IntText(
        value=576,
        description='Conversions A:',
        disabled=False,
        style = {'description_width': 'initial'}
    )
    
    x2_selection = widgets.IntText(
        value=718,
        description='Conversions B:',
        disabled=False,
        style = {'description_width': 'initial'}
    )
    
    alpha_selection = widgets.FloatSlider(
        value=0.05,
        min=0,
        max=1,
        step=0.001,
        description='Alpha:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )
    

    interact_manual(_AB, n1=n1_selection, x1=x1_selection, n2=n2_selection, x2=x2_selection, alpha=alpha_selection)
    