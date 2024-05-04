import pandas as pd
import numpy as np
from scipy.stats import norm, uniform


def generate_data(alpha, beta, num_components, inspection_times):
    data = np.empty((num_components, len(inspection_times)))
    for i in range(num_components):
        for j, t in enumerate(inspection_times):
            # Simulate degradation increment from gamma distribution for time t
            degradation_measurement = np.random.gamma(alpha * t, beta)
            data[i, j] = degradation_measurement
    return np.array(data)


def calculate_distance(observed_data, proposed_data):
    # Create deep copies of the input arrays to avoid modifying them
    delta_data1 = np.copy(observed_data)
    delta_data2 = np.copy(proposed_data)
    column = delta_data1.shape[1]
    for i in range(column - 1):
        delta_data1[:, column - 1 - i] -= delta_data1[:, column - i - 2]
        delta_data2[:, column - 1 - i] -= delta_data2[:, column - i - 2]

    distance = np.sqrt(np.sum(np.square(delta_data2 - delta_data1)))
    return distance


def acceptance_probability(k_alpha, proposed_alpha, proposal_mean, proposal_std):
    # depend on distribution type
    prior_distribution = uniform(0, 10)
    proposal_distribution = norm(
        loc=proposal_mean, scale=np.sqrt(proposal_std))

    prior_current = prior_distribution.pdf(k_alpha)
    prior_proposed = prior_distribution.pdf(proposed_alpha)

    proposal_current_to_proposed = proposal_distribution.pdf(
        k_alpha) / proposal_distribution.pdf(proposed_alpha)
    proposal_proposed_to_current = proposal_distribution.pdf(
        proposed_alpha) / proposal_distribution.pdf(k_alpha)

    acceptance_ratio = min(1, prior_proposed * proposal_current_to_proposed /
                           (prior_current * proposal_proposed_to_current))

    return acceptance_ratio


def abc_mcmc(observed_datas, chain_length, current_alpha, proposal_cv, beta, num_components, inspection_times, epsilon):
    accepted_alphas = [current_alpha]

    for _ in range(chain_length):
        proposal_mean = sum(accepted_alphas) / len(accepted_alphas)
        proposal_std = proposal_cv * proposal_mean
        # Generate proposal parameter
        proposed_alpha = norm(loc=proposal_mean, scale=proposal_std).rvs()
        # Generate data* for proposal parameter
        synthetic_datas = generate_data(
            proposed_alpha, beta, num_components, inspection_times)
        # Calculate the distance
        distance = calculate_distance(observed_datas, synthetic_datas)
        if distance <= epsilon:
            # Calculate the acceptance_prob by Parameters
            acceptance_prob = acceptance_probability(
                accepted_alphas[-1], proposed_alpha, proposal_mean, proposal_std)
            # Generate u and Judge
            u = np.random.uniform()
            if u <= acceptance_prob:
                current_alpha = proposed_alpha
                accepted_alphas.append(current_alpha)
                if len(accepted_alphas) % 500 == 0:
                    print(len(accepted_alphas))
    return np.array(accepted_alphas)


def thinning(thinning_interval, accepted_samples):
    start_index = 1
    max_index = min(len(accepted_samples), start_index + (len(accepted_samples) -
                    start_index) // thinning_interval * thinning_interval)
    alpha_chain = [accepted_samples[i]
                   for i in range(start_index, max_index, thinning_interval)]
    return alpha_chain


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
# Running ABC-MCMC results
num_runs = 1
chain_length = int(1e7)
thinning_interval = 100
epsilon = 0.48

results = []
alpha = 4
for _ in range(num_runs):
    accepted_samples = []  # Initialize accepted_samples
    accepted_samples = abc_mcmc(observed_datas, chain_length, alpha,
                                proposal_cv, beta, num_components, inspection_times, epsilon)
    accepted_chains = []
    accepted_chains = thinning(thinning_interval, accepted_samples)
    results.append(accepted_chains)
# Storage as one_par_MH.csv
df = pd.DataFrame(results)  # 将结果转为DataFrame，每一列代表一次运行的结果
df.to_csv('one_par_045.csv', header=False, index=False)
