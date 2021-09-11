import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 
boston = datasets.load_boston() 
X = boston['data'] 
y = boston['target']

class BayesianRegression: 
     
    def fit(self, X, y, sigma_squared, tau, add_intercept = True): 
         
        # record info 
        if add_intercept: 
            ones = np.ones(len(X)).reshape((len(X),1)) 
            X = np.append(ones, np.array(X), axis = 1) 
        self.X = X 
        self.y = y 
         
        # fit 
        XtX = np.dot(X.T, X)/sigma_squared 
        I = np.eye(X.shape[1])/tau 
        inverse = np.linalg.inv(XtX + I) 
        Xty = np.dot(X.T, y)/sigma_squared 
        self.beta_hats = np.dot(inverse , Xty) 
         
        # fitted values 
        self.y_hat = np.dot(X, self.beta_hats)
        
sigma_squared = 11.8 
tau = 10 
model = BayesianRegression() 
model.fit(X, y, sigma_squared, tau) 

Xs = ['X'+str(i + 1) for i in range(X.shape[1])] 
taus = [100, 10, 1] 
 
fig, ax = plt.subplots(ncols = len(taus), figsize = (20, 4.5), sharey = True) 
for i, tau in enumerate(taus): 
    model = BayesianRegression() 
    model.fit(X, y, sigma_squared, tau)  
    betas = model.beta_hats[1:] 
    sns.barplot(Xs, betas, ax = ax[i], palette = 'PuBu') 
    ax[i].set(xlabel = 'Regressor', title = fr'Regression Coefficients with $\tau = $ {tau}') 
    ax[i].set(xticks = np.arange(0, len(Xs), 2), xticklabels = Xs[::2]) 
 
ax[0].set(ylabel = 'Coefficient') 
sns.set_context("talk") 
sns.despine()
plt.show()

