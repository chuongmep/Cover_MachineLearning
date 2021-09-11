import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 
boston = datasets.load_boston() 
X = boston['data'] 
y = boston['target']

def standard_scaler(X): 
    means = X.mean(0) 
    stds = X.std(0) 
    return (X - means)/stds

class PoissonRegression: 
     
    def fit(self, X, y, n_iter = 1000, lr = 0.00001, add_intercept = True, standardize 
= True): 
         
        # record stuff 
        if standardize: 
            X = standard_scaler(X) 
        if add_intercept: 
            ones = np.ones(len(X)).reshape((len(X), 1)) 
            X = np.append(ones, X, axis = 1) 
        self.X = X 
        self.y = y 
         
        # get coefficients 
        beta_hats = np.zeros(X.shape[1]) 
        for i in range(n_iter): 
            y_hat = np.exp(np.dot(X, beta_hats)) 
            dLdbeta = np.dot(X.T, y_hat - y) 
            beta_hats -= lr*dLdbeta 
 
        # save coefficients and fitted values 
        self.beta_hats = beta_hats 
        self.y_hat = y_hat
        
model = PoissonRegression() 
model.fit(X, y)
fig, ax = plt.subplots() 
sns.scatterplot(model.y, model.y_hat) 
ax.set_xlabel(r'$y$', size = 16) 
ax.set_ylabel(r'$\hat{y}$', rotation = 0, size = 16, labelpad = 15) 
ax.set_title(r'$y$ vs. $\hat{y}$', size = 20, pad = 10) 
sns.despine() 
plt.show()


import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 
boston = datasets.load_boston() 
X_train = boston['data'] 
y_train = boston['target'] 
from sklearn.linear_model import Ridge, Lasso 
alpha = 1 
 
# Ridge 
ridge_model = Ridge(alpha = alpha) 
ridge_model.fit(X_train, y_train) 
 
 
# Lasso 
lasso_model = Lasso(alpha = alpha) 
lasso_model.fit(X_train, y_train)
from sklearn.linear_model import RidgeCV, LassoCV 
alphas = [0.01, 1, 100] 
 
# Ridge 
ridgeCV_model = RidgeCV(alphas = alphas) 
ridgeCV_model.fit(X_train, y_train) 
 
# Lasso 
lassoCV_model = LassoCV(alphas = alphas) 
lassoCV_model.fit(X_train, y_train)
print('Ridge alpha:', lassoCV_model.alpha_) 
print('Lasso alpha:', lassoCV_model.alpha_)

#Bayesian Regression
from sklearn.linear_model import BayesianRidge 
bayes_model = BayesianRidge() 
bayes_model.fit(X_train, y_train)
big_number = 10**5 
 
# alpha 
alpha = 1/11.8 
alpha_1 = big_number*alpha 
alpha_2 = big_number 
 
# lambda  
lam = 1/10 
lambda_1 = big_number*lam 
lambda_2 = big_number 
 
# fit  
bayes_model = BayesianRidge(alpha_1 = alpha_1, alpha_2 = alpha_2, alpha_init = alpha, 
                     lambda_1 = lambda_1, lambda_2 = lambda_2, lambda_init = lam) 
bayes_model.fit(X_train, y_train)'


#Poisson Regression
import statsmodels.api as sm 
X_train_with_constant = sm.add_constant(X_train) 
 
poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()) 
poisson_model.fit()