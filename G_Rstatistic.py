import numpy as np


def gelman_rubin(chains):
    J = len(chains)  # 链的个数
    n = len(chains[0])  # 每个链的样本数量

    # 计算每个参数的每个链的平均值
    mean_chain_values = []
    for chain in chains:
        mean_chain_values.append(np.mean(chain))

    # 计算每个参数的平均值
    overall_mean = np.mean(mean_chain_values)
    # 计算每个参数的间隔方差
    between_chain_variance = n * np.sum(
        (mean_chain_values - overall_mean)**2) / (J-1)
    # 计算每个参数的内部方差
    new_chains = []
    for j in range(len(mean_chain_values)):
        new_chain = []
        for i in range(len(chains[0])):
            new_chain.append(chains[j][i] - mean_chain_values[j])
        new_chains.append(new_chain)

    within_chain_variance = np.sum(np.square(new_chains))
    # 计算 Gelman-Rubin 统计
    r_hat = ((n-1)*within_chain_variance + between_chain_variance) / \
        (n * within_chain_variance)

    return r_hat
