import numpy as np  
np.set_printoptions(suppress=True) 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 
 
# import data 
cancer = datasets.load_breast_cancer() 
X = cancer['data'] 
y = cancer['target']
 
def sign(a): 
    return (-1)**(a < 0) 
 
def to_binary(y): 
        return y > 0   
def standard_scaler(X): 
    mean = X.mean(0) 
    sd = X.std(0) 
    return (X - mean)/sd
class Perceptron: 
 
    def fit(self, X, y, n_iter = 10**3, lr = 0.001, add_intercept = True, standardize = True): 
         
        # Add Info # 
        if standardize: 
            X = standard_scaler(X) 
        if add_intercept: 
            ones = np.ones(len(X)).reshape(-1, 1) 
        self.X = X 
        self.N, self.D = self.X.shape 
        self.y = y 
        self.n_iter = n_iter 
        self.lr = lr 
        self.converged = False 
         
        # Fit # 
        beta = np.random.randn(self.D)/5 
        for i in range(int(self.n_iter)): 
             
            # Form predictions 
            yhat = to_binary(sign(np.dot(self.X, beta))) 
             
            # Check for convergence 
            if np.all(yhat == sign(self.y)): 
                self.converged = True 
                self.iterations_until_convergence = i 
                break 
                 
            # Otherwise, adjust 
            for n in range(self.N): 
                yhat_n = sign(np.dot(beta, self.X[n])) 
                if (self.y[n]*yhat_n == -1): 
                    beta += self.lr * self.y[n]*self.X[n] 
 
        # Return Values # 
        self.beta = beta 
        self.yhat = to_binary(sign(np.dot(self.X, self.beta)))
        
perceptron = Perceptron() 
perceptron.fit(X, y, n_iter = 1e3, lr = 0.01) 
if perceptron.converged: 
    print(f"Converged after {perceptron.iterations_until_convergence} iterations") 
else: 
    print("Not converged") 
    
np.mean(perceptron.yhat == perceptron.y)


