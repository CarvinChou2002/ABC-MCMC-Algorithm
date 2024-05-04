import numpy as np
from scipy.stats import gamma, norm, uniform
import pandas as pd


def likelihood(alpha, data, beta):
    likelihood_val = 0
    delta_data = np.copy(data)
    column = delta_data.shape[1]
    for i in range(column - 1):
        delta_data[:, column - 1 - i] -= delta_data[:, column - i - 2]
    for component in delta_data:
        for measurement in component:
            # We use log_likelihood to avoid Numerical underflow.
            likelihood_val += np.log(gamma.pdf(measurement,
                                     a=alpha, scale=beta))
    return likelihood_val


def acceptance_probability(current_param, proposed_param, proposal_mean, proposal_std, beta, data):
    # depend on distribution type
    prior_distribution = uniform(0, 10)
    proposal_distribution = norm(
        loc=proposal_mean, scale=np.sqrt(proposal_std))

    prior_current = prior_distribution.pdf(current_param)
    prior_proposed = prior_distribution.pdf(proposed_param)

    proposal_current_to_proposed = proposal_distribution.pdf(
        current_param) / proposal_distribution.pdf(proposed_param)
    proposal_proposed_to_current = proposal_distribution.pdf(
        proposed_param) / proposal_distribution.pdf(current_param)

    acceptance_ratio = min(1, likelihood(proposed_param, data, beta)*prior_proposed * proposal_current_to_proposed /
                           (likelihood(current_param, data, beta)*prior_current * proposal_proposed_to_current))

    return acceptance_ratio


def metropolis_hastings(observed_datas, iterations, current_alpha, burn_in, proposal_cv, beta):
    accepted_alphas = [current_alpha]
    alpha_current = np.copy(current_alpha)
    # Burn in time
    for _ in range(burn_in):
        proposal_mean = sum(accepted_alphas) / len(accepted_alphas)
        proposal_std = proposal_cv * proposal_mean
        proposed_param = norm(loc=proposal_mean, scale=proposal_std).rvs()
        accept_prob = acceptance_probability(
            alpha_current, proposed_param, proposal_mean, proposal_std, beta, observed_datas)
        if np.random.rand() < accept_prob:
            alpha_current = proposed_param

    # Sampling phase
    for _ in range(iterations-burn_in):
        proposed_param = norm(loc=proposal_mean, scale=proposal_std).rvs()
        accept_prob = acceptance_probability(
            alpha_current, proposed_param, proposal_mean, proposal_std, beta, observed_datas)
        if np.random.rand() < accept_prob:
            alpha_current = proposed_param
            accepted_alphas.append(alpha_current)
    return accepted_alphas


# Given parameters
beta = 0.015
eta = 1
inspection_times = [5, 10, 15]      # Inspection times (years)
num_components = 5      # Number of components
proposal_cv = 0.1
observed_datas = np.array([
    [0.28858961, 0.67501878, 0.97772541],
    [0.14211732, 0.63894478, 0.81678105],
    [0.12099909, 0.55307049, 1.07481258],
    [0.23745562, 0.64363251, 1.04951346],
    [0.14019347, 0.71834158, 0.85304153]
])
# Running Metropolis-Hastings
alpha = [4, 4, 4]
iterations = 1000
burn_in = 100
num_runs = 3
results = []
for i in range(num_runs):
    accepted_samples = metropolis_hastings(
        observed_datas, iterations, alpha[i], burn_in, proposal_cv, beta)
    results.append(accepted_samples)

# Storage as one_par_MH.csv
df = pd.DataFrame(results)  # 将结果转为DataFrame，每一列代表一次运行的结果
df.to_csv('one_par_MH.csv', header=False, index=False)
