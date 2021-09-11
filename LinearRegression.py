import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import datasets
class LinearRegression: 
 
    def fit(self, X, y, intercept = False): 
 
        # record data and dimensions 
        if intercept == False: # add intercept (if not already included) 
            ones = np.ones(len(X)).reshape(len(X), 1) # column of ones  
            X = np.concatenate((ones, X), axis = 1) 
        self.X = np.array(X) 
        self.y = np.array(y) 
        self.N, self.D = self.X.shape 
         
        # estimate parameters 
        XtX = np.dot(self.X.T, self.X) 
        XtX_inverse = np.linalg.inv(XtX) 
        Xty = np.dot(self.X.T, self.y) 
        self.beta_hats = np.dot(XtX_inverse, Xty) 
         
        # make in-sample predictions 
        self.y_hat = np.dot(self.X, self.beta_hats) 
         
        # calculate loss 
        self.L = .5*np.sum((self.y - self.y_hat)**2) 
         
    def predict(self, X_test, intercept = True): 
         
        # form predictions 
        self.y_test_hat = np.dot(X_test, self.beta_hats) 


boston = datasets.load_boston() 
X = boston['data'] 
y = boston['target']
model = LinearRegression() # instantiate model 
model.fit(X, y, intercept = False) # fit model 
fig, ax = plt.subplots() 
sns.scatterplot(model.y, model.y_hat) 
ax.set_xlabel(r'$y$', size = 16) 
ax.set_ylabel(r'$\hat{y}$', rotation = 0, size = 16, labelpad = 15) 
ax.set_title(r'$y$ vs. $\hat{y}$', size = 20, pad = 10) 
sns.despine()
plt.show()