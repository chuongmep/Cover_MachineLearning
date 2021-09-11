import numpy as np  
np.set_printoptions(suppress=True) 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 

# import data 
cancer = datasets.load_breast_cancer() 
X = cancer['data'] 
y = cancer['target']

def logistic(z): 
    return (1 + np.exp(-z))**(-1) 
 
def standard_scaler(X): 
    mean = X.mean(0) 
    sd = X.std(0) 
    return (X - mean)/sd

class BinaryLogisticRegression: 
     
    def fit(self, X, y, n_iter, lr, standardize = True, has_intercept = False): 
         
        ### Record Info ### 
        if standardize: 
            X = standard_scaler(X)  
        if not has_intercept: 
            ones = np.ones(X.shape[0]).reshape(-1, 1) 
            X = np.concatenate((ones, X), axis = 1) 
        self.X = X 
        self.N, self.D = X.shape 
        self.y = y 
        self.n_iter = n_iter 
        self.lr = lr 
 
        ### Calculate Beta ### 
        beta = np.random.randn(self.D)  
        for i in range(n_iter): 
            p = logistic(np.dot(self.X, beta)) # vector of probabilities  
            gradient = -np.dot(self.X.T, (self.y-p)) # gradient 
            beta -= self.lr*gradient  
             
        ### Return Values ### 
        self.beta = beta 
        self.p = logistic(np.dot(self.X, self.beta))  
        self.yhat = self.p.round()

binary_model = BinaryLogisticRegression() 
binary_model.fit(X, y, n_iter = 10**4, lr = 0.0001) 
print('In-sample accuracy: '  + str(np.mean(binary_model.yhat == binary_model.y)))

fig, ax = plt.subplots() 
sns.distplot(binary_model.p[binary_model.yhat == 0], kde = False, bins = 8, label = 'Class 0', color = 'cornflowerblue') 
sns.distplot(binary_model.p[binary_model.yhat == 1], kde = False, bins = 8, label = 'Class 1', color = 'darkblue') 
ax.legend(loc = 9, bbox_to_anchor = (0,0,1.59,.9)) 
ax.set_xlabel(r'Estimated $P(Y_n = 1)$', size = 14) 
ax.set_title(r'Estimated $P(Y_n = 1)$ by True Class', size = 16) 
sns.despine() 
plt.show()

#Multiclass Logistic Regression

# import data 
wine = datasets.load_wine() 
X = wine['data'] 
y = wine['target']
def softmax(z): 
    return np.exp(z)/(np.exp(z).sum()) 
 
def softmax_byrow(Z): 
    return (np.exp(Z)/(np.exp(Z).sum(1)[:,None])) 
 
def make_I_matrix(y): 
    I = np.zeros(shape = (len(y), len(np.unique(y))), dtype = int) 
    for j, target in enumerate(np.unique(y)): 
        I[:,j] = (y == target) 
    return I 
 
 
Z_test = np.array([[1, 1], 
              [0,1]]) 
print('Softmax for Z:\n', softmax_byrow(Z_test).round(2)) 
 
y_test = np.array([0,0,1,1,2]) 
print('I matrix of [0,0,1,1,2]:\n', make_I_matrix(y_test), end = '\n\n')

class MulticlassLogisticRegression: 
     
    def fit(self, X, y, n_iter, lr, standardize = True, has_intercept = False): 
         
        ### Record Info ### 
        if standardize: 
            X = standard_scaler(X)  
        if not has_intercept: 
            ones = np.ones(X.shape[0]).reshape(-1, 1) 
            X = np.concatenate((ones, X), axis = 1) 
        self.X = X 
        self.N, self.D = X.shape 
        self.y = y 
        self.K = len(np.unique(y)) 
        self.n_iter = n_iter 
        self.lr = lr 
         
        ### Fit B ### 
        B = np.random.randn(self.D*self.K).reshape((self.D, self.K)) 
        self.I = make_I_matrix(self.y) 
        for i in range(n_iter): 
            Z = np.dot(self.X, B) 
            P = softmax_byrow(Z) 
            gradient = np.dot(self.X.T, self.I - P) 
            B += lr*gradient 
         
        ### Return Values ### 
        self.B = B 
        self.Z = np.dot(self.X, B) 
        self.P = softmax_byrow(self.Z) 
        self.yhat = self.P.argmax(1) 
        
multiclass_model = MulticlassLogisticRegression() 
multiclass_model.fit(X, y, 10**4, 0.0001) 
print('In-sample accuracy: '  + str(np.mean(multiclass_model.yhat == y)))

fig, ax = plt.subplots(1, 3, figsize = (17, 5)) 
for i, y in enumerate(np.unique(y)): 
    sns.distplot(multiclass_model.P[multiclass_model.y == y, i], 
                 hist_kws=dict(edgecolor="darkblue"),  
                 color = 'cornflowerblue', 
                 bins = 15,  
                 kde = False, 
                 ax = ax[i]); 
    ax[i].set_xlabel(xlabel = fr'$P(y = {y})$', size = 14) 
    ax[i].set_title('Histogram for Observations in Class '+ str(y), size = 16) 
sns.despine()