## Import decision trees 
import import_ipynb 
import RegressionTrees as ct; 
 
## Import numpy and visualization packages 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets

#Classification with AdaBoost

## Load data 
penguins = sns.load_dataset('penguins') 
penguins.dropna(inplace = True) 
X = np.array(penguins.drop(columns = ['species', 'island'])) 
y = 1*np.array(penguins['species'] == 'Adelie') 
y[y == 0] = -1 
 
## Train-test split 
np.random.seed(123) 
test_frac = 0.25 
test_size = int(len(y)*test_frac) 
test_idxs = np.random.choice(np.arange(len(y)), test_size, replace = False) 
X_train = np.delete(X, test_idxs, 0) 
y_train = np.delete(y, test_idxs, 0) 
X_test = X[test_idxs] 
y_test = y[test_idxs] 

## Loss Functions 
def get_weighted_pmk(y, weights): 
    ks = np.unique(y) 
    weighted_pmk = [sum(weights[y == k]) for k in ks]       
    return(np.array(weighted_pmk)/sum(weights)) 
 
def gini_index(y, weights): 
    weighted_pmk = get_weighted_pmk(y, weights) 
    return np.sum( weighted_pmk*(1-weighted_pmk) ) 
 
def cross_entropy(y, weights): 
    weighted_pmk = get_weighted_pmk(y, weights)     
    return -np.sum(weighted_pmk*np.log2(weighted_pmk)) 
 
def split_loss(child1, child2, weights1, weights2, loss = cross_entropy): 
    return (len(child1)*loss(child1, weights1) + len(child2)*loss(child2, weights2))/(len(child1) + len(child2))

## Helper Classes 
class Node: 
     
    def __init__(self, Xsub, ysub, observations, ID, depth = 0, parent_ID = None, leaf 
= True): 
        self.Xsub = Xsub 
        self.ysub = ysub 
        self.observations = observations 
        self.ID = ID 
        self.size = len(ysub) 
        self.depth = depth 
        self.parent_ID = parent_ID 
        self.leaf = leaf 
         
 
class Splitter: 
     
    def __init__(self): 
        self.loss = np.inf 
        self.no_split = True 
         
    def _replace_split(self, loss, d, dtype = 'quant', t = None, L_values = None): 
        self.loss = loss 
        self.d = d 
        self.dtype = dtype 
        self.t = t 
        self.L_values = L_values   
        self.no_split = False 
 
         
## Main Class 
class DecisionTreeClassifier: 
     
    ############################# 
    ######## 1. TRAINING ######## 
    ############################# 
     
    ######### FIT ########## 
    def fit(self, X, y, weights, loss_func = cross_entropy, max_depth = 100, min_size =
 2, C = None): 
         
        ## Add data 
        self.X = X 
        self.y = y 
        self.N, self.D = self.X.shape 
        dtypes = [np.array(list(self.X[:,d])).dtype for d in range(self.D)] 
        self.dtypes = ['quant' if (dtype == float or dtype == int) else 'cat' for 
dtype in dtypes] 
        self.weights = weights 
         
        ## Add model parameters 
        self.loss_func = loss_func 
        self.max_depth = max_depth 
        self.min_size = min_size 
        self.C = C 
         
        ## Initialize nodes 
        self.nodes_dict = {} 
        self.current_ID = 0 
        initial_node = Node(Xsub = X, ysub = y, observations = np.arange(self.N), ID = 
self.current_ID, parent_ID = None) 
        self.nodes_dict[self.current_ID] = initial_node 
        self.current_ID += 1 
         
        # Build 
        self._build() 
 
    ###### BUILD TREE ###### 
    def _build(self): 
         
        eligible_buds = self.nodes_dict  
        for layer in range(self.max_depth): 
             
            ## Find eligible nodes for layer iteration 
            eligible_buds = {ID:node for (ID, node) in self.nodes_dict.items() if  
                                (node.leaf == True) & 
                                (node.size >= self.min_size) &  
                                (~ct.all_rows_equal(node.Xsub)) & 
                                (len(np.unique(node.ysub)) > 1)} 
            if len(eligible_buds) == 0: 
                break 
             
            ## split each eligible parent 
            for ID, bud in eligible_buds.items(): 
                                 
                ## Find split 
                self._find_split(bud) 
                                 
                ## Make split 
                if not self.splitter.no_split: 
                    self._make_split() 
                 
    ###### FIND SPLIT ###### 
    def _find_split(self, bud): 
         
        ## Instantiate splitter 
        splitter = Splitter() 
        splitter.bud_ID = bud.ID 
         
        ## For each (eligible) predictor... 
        if self.C is None: 
            eligible_predictors = np.arange(self.D) 
        else: 
            eligible_predictors = np.random.choice(np.arange(self.D), self.C, replace = False) 
        for d in sorted(eligible_predictors): 
            Xsub_d = bud.Xsub[:,d] 
            dtype = self.dtypes[d] 
            if len(np.unique(Xsub_d)) == 1: 
                continue 
 
            ## For each value... 
            if dtype == 'quant': 
                for t in np.unique(Xsub_d)[:-1]: 
                    L_condition = Xsub_d <= t 
                    ysub_L = bud.ysub[L_condition] 
                    ysub_R = bud.ysub[~L_condition] 
                    weights_L = self.weights[bud.observations][L_condition] 
                    weights_R = self.weights[bud.observations][~L_condition] 
                    loss = split_loss(ysub_L, ysub_R, 
                                      weights_L, weights_R, 
                                      loss = self.loss_func) 
                    if loss < splitter.loss: 
                        splitter._replace_split(loss, d, 'quant', t = t) 
            else: 
                for L_values in ct.possible_splits(np.unique(Xsub_d)): 
                    L_condition = np.isin(Xsub_d, L_values) 
                    ysub_L = bud.ysub[L_condition] 
                    ysub_R = bud.ysub[~L_condition] 
                    weights_L = self.weights[bud.observations][L_condition] 
                    weights_R = self.weights[bud.observations][~L_condition] 
                    loss = split_loss(ysub_L, ysub_R, 
                                      weights_L, weights_R, 
                                      loss = self.loss_func) 
                    if loss < splitter.loss:  
                        splitter._replace_split(loss, d, 'cat', L_values = L_values) 
                         
        ## Save splitter 
        self.splitter = splitter 
     
    ###### MAKE SPLIT ###### 
    def _make_split(self): 
         
        ## Update parent node 
        parent_node = self.nodes_dict[self.splitter.bud_ID] 
        parent_node.leaf = False 
        parent_node.child_L = self.current_ID 
        parent_node.child_R = self.current_ID + 1 
        parent_node.d = self.splitter.d 
        parent_node.dtype = self.splitter.dtype 
        parent_node.t = self.splitter.t         
        parent_node.L_values = self.splitter.L_values 
         
        ## Get X and y data for children 
        if parent_node.dtype == 'quant': 
            L_condition = parent_node.Xsub[:,parent_node.d] <= parent_node.t 
        else: 
            L_condition = np.isin(parent_node.Xsub[:,parent_node.d], parent_node.L_values) 
        Xchild_L = parent_node.Xsub[L_condition] 
        ychild_L = parent_node.ysub[L_condition] 
        child_observations_L = parent_node.observations[L_condition] 
        Xchild_R = parent_node.Xsub[~L_condition] 
        ychild_R = parent_node.ysub[~L_condition] 
        child_observations_R = parent_node.observations[~L_condition] 
         
        ## Create child nodes 
        child_node_L = Node(Xchild_L, ychild_L, child_observations_L, 
                            ID = self.current_ID, depth = parent_node.depth + 1, 
                            parent_ID = parent_node.ID) 
        child_node_R = Node(Xchild_R, ychild_R, child_observations_R, 
                            ID = self.current_ID + 1, depth = parent_node.depth + 1, 
                            parent_ID = parent_node.ID) 
        self.nodes_dict[self.current_ID] = child_node_L 
        self.nodes_dict[self.current_ID + 1] = child_node_R 
        self.current_ID += 2 
                 
             
    ############################# 
    ####### 2. PREDICTING ####### 
    ############################# 
     
    ###### LEAF MODES ###### 
    def _get_leaf_modes(self): 
        self.leaf_modes = {} 
        for node_ID, node in self.nodes_dict.items(): 
            if node.leaf: 
                values, counts = np.unique(node.ysub, return_counts=True) 
                self.leaf_modes[node_ID] = values[np.argmax(counts)] 
     
    ####### PREDICT ######## 
    def predict(self, X_test): 
         
        # Calculate leaf modes 
        self._get_leaf_modes() 
         
        yhat = [] 
        for x in X_test: 
            node = self.nodes_dict[0]  
            while not node.leaf: 
                if node.dtype == 'quant': 
                    if x[node.d] <= node.t: 
                        node = self.nodes_dict[node.child_L] 
                    else: 
                        node = self.nodes_dict[node.child_R] 
                else: 
                    if x[node.d] in node.L_values: 
                        node = self.nodes_dict[node.child_L] 
                    else: 
                        node = self.nodes_dict[node.child_R] 
            yhat.append(self.leaf_modes[node.ID]) 
        return np.array(yhat) 

class AdaBoost: 
     
    def fit(self, X_train, y_train, T, stub_depth = 1): 
        self.y_train = y_train 
        self.X_train = X_train 
        self.N, self.D = X_train.shape 
        self.T = T 
        self.stub_depth = stub_depth 
         
        ## Instantiate stuff 
        self.weights = np.repeat(1/self.N, self.N) 
        self.trees = [] 
        self.alphas = [] 
        self.yhats = np.empty((self.N, self.T)) 
         
        for t in range(self.T): 
             
            ## Calculate stuff 
            self.T_t = DecisionTreeClassifier() 
            self.T_t.fit(self.X_train, self.y_train, self.weights, max_depth = self.stub_depth) 
            self.yhat_t = self.T_t.predict(self.X_train) 
            self.epsilon_t = sum(self.weights*(self.yhat_t != self.y_train))/sum(self.weights) 
            self.alpha_t = np.log( (1-self.epsilon_t)/self.epsilon_t ) 
            self.weights = np.array([w*(1-self.epsilon_t)/self.epsilon_t if self.yhat_t[i] != self.y_train[i] 
                                    else w for i, w in enumerate(self.weights)]) 
            ## Append stuff 
            self.trees.append(self.T_t) 
            self.alphas.append(self.alpha_t) 
            self.yhats[:,t] = self.yhat_t  
             
        self.yhat = np.sign(np.dot(self.yhats, self.alphas)) 
         
    def predict(self, X_test): 
        yhats = np.zeros(len(X_test)) 
        for t, tree in enumerate(self.trees): 
            yhats_tree = tree.predict(X_test) 
            yhats += yhats_tree*self.alphas[t] 
        return np.sign(yhats)
booster = AdaBoost() 
booster.fit(X_train, y_train, T = 30, stub_depth = 3) 
yhat = booster.predict(X_test) 
np.mean(yhat == y_test)

#AdaBoost.R2
## Import packages 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 
 
## Load data 
tips = sns.load_dataset('tips') 
X = np.array(tips.drop(columns = 'tip')) 
y = np.array(tips['tip']) 
 
## Train-test split 
np.random.seed(1) 
test_frac = 0.25 
test_size = int(len(y)*test_frac) 
test_idxs = np.random.choice(np.arange(len(y)), test_size, replace = False) 
X_train = np.delete(X, test_idxs, 0) 
y_train = np.delete(y, test_idxs, 0) 
X_test = X[test_idxs] 
y_test = y[test_idxs]
## Import decision trees 
import import_ipynb 
import RegressionTrees as rt

def weighted_median(values, weights): 
     
    sorted_indices = values.argsort() 
    values = values[sorted_indices] 
    weights = weights[sorted_indices] 
    weights_cumulative_sum = weights.cumsum() 
    median_weight = np.argmax(weights_cumulative_sum >= sum(weights)/2) 
    return values[median_weight] 
class AdaBoostR2: 
     
    def fit(self, X_train, y_train, T = 100, stub_depth = 1, random_state = None): 
         
        self.y_train = y_train 
        self.X_train = X_train 
        self.T = T 
        self.stub_depth = stub_depth 
        self.N, self.D = X_train.shape 
        self.weights = np.repeat(1/self.N, self.N) 
        np.random.seed(random_state) 
         
        self.trees = []     
        self.fitted_values = np.empty((self.N, self.T)) 
        self.betas = [] 
        for t in range(self.T): 
             
            ## Draw sample, fit tree, get predictions 
            bootstrap_indices = np.random.choice(np.arange(self.N), size = self.N, replace = True, p = self.weights) 
            bootstrap_X = self.X_train[bootstrap_indices] 
            bootstrap_y = self.y_train[bootstrap_indices] 
            tree = rt.DecisionTreeRegressor() 
            tree.fit(bootstrap_X, bootstrap_y, max_depth = stub_depth) 
            self.trees.append(tree) 
            yhat = tree.predict(X_train) 
            self.fitted_values[:,t] = yhat 
             
            ## Calculate observation errors 
            abs_errors_t = np.abs(self.y_train - yhat) 
            D_t = np.max(abs_errors_t) 
            L_ts = abs_errors_t/D_t 
             
            ## Calculate model error (and possibly break) 
            Lbar_t = np.sum(self.weights*L_ts) 
            if Lbar_t >= 0.5: 
                self.T = t - 1 
                self.fitted_values = self.fitted_values[:,:t-1] 
                self.trees = self.trees[:t-1] 
                break 
             
            ## Calculate and record beta  
            beta_t = Lbar_t/(1 - Lbar_t) 
            self.betas.append(beta_t) 
             
            ## Reweight 
            Z_t = np.sum(self.weights*beta_t**(1-L_ts)) 
            self.weights *= beta_t**(1-L_ts)/Z_t 
             
        ## Get median  
        self.model_weights = np.log(1/np.array(self.betas)) 
        self.y_train_hat = np.array([weighted_median(self.fitted_values[n], 
self.model_weights) for n in range(self.N)]) 
         
    def predict(self, X_test): 
        N_test = len(X_test) 
        fitted_values = np.empty((N_test, self.T)) 
        for t, tree in enumerate(self.trees): 
            fitted_values[:,t] = tree.predict(X_test) 
        return np.array([weighted_median(fitted_values[n], self.model_weights) for n in range(N_test)])
    
booster = AdaBoostR2() 
booster.fit(X_train, y_train, T = 50, stub_depth = 4, random_state = 123) 
 
fig, ax = plt.subplots(figsize = (7,5)) 
sns.scatterplot(y_test, booster.predict(X_test)); 
ax.set(xlabel = r'$y$', ylabel = r'$\hat{y}$', title = 'Fitted vs. Observed Values for AdaBoostR2') 
sns.despine() 


#----------Skearn----------
## Import packages 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
#Bagging and Random Forests
## Load penguins data 
penguins = sns.load_dataset('penguins') 
penguins = penguins.dropna().reset_index(drop = True) 
X = penguins.drop(columns = 'species') 
y = penguins['species'] 
 
## Train-test split 
np.random.seed(1) 
test_frac = 0.25 
test_size = int(len(y)*test_frac) 
test_idxs = np.random.choice(np.arange(len(y)), test_size, replace = False) 
X_train = X.drop(test_idxs) 
y_train = y.drop(test_idxs) 
X_test = X.loc[test_idxs] 
y_test = y.loc[test_idxs] 
 
## Get dummies 
X_train = pd.get_dummies(X_train, drop_first = True) 
X_test = pd.get_dummies(X_test, drop_first = True) 
#Bagging
from sklearn.ensemble import BaggingClassifier 
from sklearn.naive_bayes import GaussianNB 
 
## Decision Tree bagger 
bagger1 = BaggingClassifier(n_estimators = 50, random_state = 123) 
bagger1.fit(X_train, y_train) 
 
## Naive Bayes bagger 
bagger2 = BaggingClassifier(base_estimator = GaussianNB(), random_state = 123) 
bagger2.fit(X_train, y_train) 
 
## Evaluate 
print(np.mean(bagger1.predict(X_test) == y_test)) 
print(np.mean(bagger2.predict(X_test) == y_test))
#Random Forests
from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier(n_estimators = 100, max_features = 
int(np.sqrt(X_test.shape[1])), random_state = 123) 
rf.fit(X_train, y_train) 
print(np.mean(rf.predict(X_test) == y_test)) 

#Boosting 
## AdaBoost Classification
## Make binary 
y_train = (y_train == 'Adelie') 
y_test = (y_test == 'Adelie')
from sklearn.ensemble import AdaBoostClassifier 
 
## Get dummies 
X_train = pd.get_dummies(X_train, drop_first = True) 
X_test = pd.get_dummies(X_test, drop_first = True) 
 
## Build model 
abc = AdaBoostClassifier(n_estimators = 50) 
abc.fit(X_train, y_train) 
y_test_hat = abc.predict(X_test) 
 
## Evaluate  
np.mean(y_test_hat == y_test)
from sklearn.linear_model import LogisticRegression 
abc = AdaBoostClassifier(base_estimator = LogisticRegression(max_iter = 1000)) 
abc.fit(X_train, y_train)
## AdaBoost Regression
## Load penguins data 
tips = sns.load_dataset('tips') 
tips = tips.dropna().reset_index(drop = True) 
X = tips.drop(columns = 'tip') 
y = tips['tip'] 
 
## Train-test split 
np.random.seed(1) 
test_frac = 0.25 
test_size = int(len(y)*test_frac) 
test_idxs = np.random.choice(np.arange(len(y)), test_size, replace = False) 
X_train = X.drop(test_idxs) 
y_train = y.drop(test_idxs) 
X_test = X.loc[test_idxs] 
y_test = y.loc[test_idxs] 
from sklearn.ensemble import AdaBoostRegressor 
 
## Get dummies 
X_train = pd.get_dummies(X_train, drop_first = True) 
X_test = pd.get_dummies(X_test, drop_first = True) 
 
## Build model 
abr = AdaBoostRegressor(n_estimators = 50) 
abr.fit(X_train, y_train) 
y_test_hat = abr.predict(X_test) 
 
## Visualize predictions 
fig, ax = plt.subplots(figsize = (7, 5)) 
sns.scatterplot(y_test, y_test_hat) 
ax.set(xlabel = r'$y$', ylabel = r'$\hat{y}$', title = r'Test Sample $y$ vs. $\hat{y}$') 
sns.despine() 