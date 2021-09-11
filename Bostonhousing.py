import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 
boston = datasets.load_boston() 
X_train = boston['data'] 
y_train = boston['target']
#Scikit Learn
from sklearn.linear_model import LinearRegression 
sklearn_model = LinearRegression() 
sklearn_model.fit(X_train, y_train) 
sklearn_predictions = sklearn_model.predict(X_train) 
fig, ax = plt.subplots() 
sns.scatterplot(y_train, sklearn_predictions) 
ax.set_xlabel(r'$y$', size = 16) 
ax.set_ylabel(r'$\hat{y}$', rotation = 0, size = 16, labelpad = 15) 
ax.set_title(r'$y$ vs. $\hat{y}$', size = 20, pad = 10) 
sns.despine() 

predictors = boston.feature_names 
beta_hats = sklearn_model.coef_ 
print('\n'.join([f'{predictors[i]}: {round(beta_hats[i], 3)}' for i in range(3)])) 
plt.show()

#Statsmodels
import statsmodels.api as sm 
 
X_train_with_constant = sm.add_constant(X_train) 
sm_model1 = sm.OLS(y_train, X_train_with_constant) 
sm_fit1 = sm_model1.fit() 
sm_predictions1 = sm_fit1.predict(X_train_with_constant)
import pandas as pd 
df = pd.DataFrame(X_train, columns = boston['feature_names']) 
df['target'] = y_train 
print(df.head())
 
formula = 'target ~ ' + ' + '.join(boston['feature_names']) 
print('formula:', formula) 
import statsmodels.formula.api as smf 
 
sm_model2 = smf.ols(formula, data = df) 
sm_fit2 = sm_model2.fit() 
sm_predictions2 = sm_fit2.predict(df)