import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
# generate data 
np.random.seed(123) 
N = 20 
beta0 = -4 
beta1 = 2 
x = np.random.randn(N) 
e = np.random.randn(N) 
y = beta0 + beta1*x + e 
true_x = np.linspace(min(x), max(x), 100) 
true_y = beta0 + beta1*true_x 
 
# estimate model  
beta1_hat = sum((x - np.mean(x))*(y - np.mean(y)))/sum((x - np.mean(x))**2) 
beta0_hat = np.mean(y) - beta1_hat*np.mean(x) 
fit_y = beta0_hat + beta1_hat*true_x 
 
# plot 
fig, ax = plt.subplots() 
sns.scatterplot(x, y, s = 40, label = 'Data') 
sns.lineplot(true_x, true_y, color = 'red', label = 'True Model') 
sns.lineplot(true_x, fit_y, color = 'purple', label = 'Estimated Model') 
ax.set_xlabel('x', fontsize = 14) 
ax.set_title(fr"Linear Regression for $y = {beta0} + ${beta1}$x + \epsilon$", fontsize 
= 16) 
ax.set_ylabel('y', fontsize=14, rotation=0, labelpad=10) 
ax.legend(loc = 4) 
sns.despine()
plt.show()