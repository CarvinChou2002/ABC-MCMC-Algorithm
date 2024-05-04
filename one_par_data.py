import matplotlib.pyplot as plt
import numpy as np
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
# Gamma process parameters
alpha_true = 4
beta = 0.015

# Sizing error parameters
sizing_error_mean = 0
sizing_error_std = 0.1

# Number of components and inspections
num_components = 5
num_inspections = 3
Inspection_times = [5, 10, 15]  # Inspection times in years

# Initialize degradation data array with zeros (initial degradation is 0)
degradation_data = np.zeros((num_components, num_inspections))

# Simulate degradation increments for each component
for i in range(num_components):
    for j, time in enumerate(Inspection_times):
        # Simulate degradation increment from gamma process
        degradation_increment = np.random.gamma(alpha_true * time, beta)
        # Add sizing error to the degradation increment
        sizing_error = np.random.normal(sizing_error_mean, sizing_error_std)

        # Total degradation increment with sizing error
        degradation_data[i, j] = degradation_increment + sizing_error

# Print the simulated degradation increments
print("模拟退化增量数据:")
print(degradation_data)

# Optionally, you can save the data to a file or perform further analysis

# Plotting the degradation increments over time (if desired)
markers = ['o', '*', '^', 'p', 's']
plt.figure(figsize=(10, 6))
for i in range(num_components):
    plt.plot(Inspection_times,
             degradation_data[i, :], marker=markers[i], color='black', label=f'元器件 {i+1}')

plt.xlabel('时间 (年)')
plt.ylabel('退化增量(mm)')
plt.legend()
plt.grid(True)
plt.show()
