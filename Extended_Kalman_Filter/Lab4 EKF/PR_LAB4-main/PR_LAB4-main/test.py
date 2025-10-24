import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# Degrees of freedom
df = 2

# List of alpha values
alpha_values = np.linspace(0.01, 0.99, 100)

# Calculate chi-squared critical values for each alpha
critical_values = [scipy.stats.chi2.ppf(alpha, df) for alpha in alpha_values]

# Plot the results
plt.plot(alpha_values, critical_values)
plt.title('Chi-Squared Critical Values vs. alpha')
plt.xlabel('1 - Alpha')
plt.ylabel('Chi-Squared Critical Value')
plt.show()

