import numpy as np  
np.set_printoptions(suppress=True) 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 

# import data 
cancer = datasets.load_breast_cancer() 
X = cancer['data'] 
y = cancer['target']

class FisherLinearDiscriminant: 
     
    def fit(self, X, y): 
        ## Save stuff 
        self.X = X 
        self.y = y 
        self.N, self.D = self.X.shape 
         
        ## Calculate class means 
        X0 = X[y == 0] 
        X1 = X[y == 1] 
        mu0 = X0.mean(0) 
        mu1 = X1.mean(0) 
         
        ## Sigma_w 
        Sigma_w = np.empty((self.D, self.D)) 
        for x0 in X0: 
            x0_minus_mu0 = (x0 - mu0).reshape(-1, 1) 
            Sigma_w += np.dot(x0_minus_mu0, x0_minus_mu0.T) 
        for x1 in X1: 
            x1_minus_mu1 = (x1 - mu1).reshape(-1, 1) 
            Sigma_w += np.dot(x1_minus_mu1, x1_minus_mu1.T)             
        Sigma_w_inverse = np.linalg.inv(Sigma_w) 
         
        ## Beta 
        self.beta = np.dot(Sigma_w_inverse, mu1 - mu0) 
        self.f = np.dot(X, self.beta)
        
model = FisherLinearDiscriminant() 
model.fit(X, y); 

fig, ax = plt.subplots(figsize = (7,5)) 
sns.distplot(model.f[model.y == 0], bins = 25, kde = False,  
             color = 'cornflowerblue', label = 'Class 0') 
sns.distplot(model.f[model.y == 1], bins = 25, kde = False,  
             color = 'darkblue', label = 'Class 1') 
ax.set_xlabel(r"$f\hspace{.25}(x_n)$", size = 14) 
ax.set_title(r"Histogram of $f\hspace{.25}(x_n)$ by Class", size = 16) 
ax.legend() 
sns.despine() 
plt.show()


#Implementation

import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 
 
# import data 
cancer = datasets.load_breast_cancer() 
X_cancer = cancer['data'] 
y_cancer = cancer['target'] 
wine = datasets.load_wine() 
X_wine = wine['data'] 
y_wine = wine['target']
#Logistic Regression
## Binary Logistic Regression
from sklearn.linear_model import LogisticRegression 
binary_model = LogisticRegression(C = 10**5, max_iter = 1e5) 
binary_model.fit(X_cancer, y_cancer)
y_hats = binary_model.predict(X_cancer) 
p_hats = binary_model.predict_proba(X_cancer) 
print(f'Training accuracy: {binary_model.score(X_cancer, y_cancer)}') 
## Multiclass Logistic Regression
from sklearn.linear_model import LogisticRegression 
multiclass_model = LogisticRegression(multi_class = 'multinomial', C = 10**5, max_iter 
= 10**4) 
multiclass_model.fit(X_wine, y_wine)
y_hats = multiclass_model.predict(X_wine) 
p_hats = multiclass_model.predict_proba(X_wine) 
print(f'Training accuracy: {multiclass_model.score(X_wine, y_wine)}')

# The Perceptron Algorithm
from sklearn.linear_model import Perceptron 
perceptron = Perceptron() 
perceptron.fit(X_cancer, y_cancer)

# Fisher’s Linear Discriminant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
lda = LinearDiscriminantAnalysis(n_components = 1) 
lda.fit(X_cancer, y_cancer); 
 
f0 = np.dot(X_cancer, lda.coef_[0])[y_cancer == 0] 
f1 = np.dot(X_cancer, lda.coef_[0])[y_cancer == 1] 
print('Separated:', (min(f0) > max(f1)) | (max(f0) < min(f1))) 
fig, ax = plt.subplots(figsize = (7,5)) 
sns.distplot(f0, bins = 25, kde = False,  
             color = 'cornflowerblue', label = 'Class 0') 
sns.distplot(f1, bins = 25, kde = False,  
             color = 'darkblue', label = 'Class 1') 
ax.set_xlabel(r"$f\hspace{.25}(x_n)$", size = 14) 
ax.set_title(r"Histogram of $f\hspace{.25}(x_n)$ by Class", size = 16) 
ax.legend() 
sns.despine()