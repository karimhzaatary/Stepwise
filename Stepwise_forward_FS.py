
# coding: utf-8

# In[101]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[6]:


# read dataset with dummified ordinal and categorical variables
train = pd.read_csv('/Users/monazaatari/Downloads/Kaggle_ML_Project-master/Sex.csv',index_col=0)
train1=pd.read_csv('/Users/monazaatari/Downloads/Kaggle_ML_Project-master/Data.csv',index_col=0)


# In[7]:


train.head()


# In[8]:


train['SalePrice']=train1['SalePrice']


# In[33]:


from sklearn.linear_model import LinearRegression
np.random.seed(0)
lm = LinearRegression()
regr = OLS(y, add_constant(X)).fit()
print(regr.aic)


# In[35]:


#Fit=lm.fit(X, Y)


# In[88]:


from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
# Creating a list of X variables
explanatory=train.drop(['SalePrice'],axis=1)
X_list=list(explanatory.columns) 
Y=train['SalePrice']
# Creating an AIC list 

AIC_track=[]
p=0

AIC_initial=10000000
# list of selected features
features_selected=[]
Output_features=[]
while p<1:
    AIC=[]
    featuresIndicies =[]
    for i in X_list:
        #x_zed = features_selected.append(i)
        model=OLS(Y,X[features_selected+[i]]).fit()
        AIC.append(model.aic)
        featuresIndicies.append(i)
        #print('when adding', i, 'the AIC is', model.aic)
    temp = pd.DataFrame({'Feature':featuresIndicies,'AIC':AIC})
    bestFeature = list(temp.sort_values(by = 'AIC',ascending=True)['Feature'])[0]
    selected=AIC.index(min(AIC))
    #features_selected.append(X_list[selected])
    AIC_updated=min(AIC)
    print('updated aic',AIC_updated)
    print('initial aic',AIC_initial)
    if AIC_updated>AIC_initial:
        p=1
    else:
        print('new feature added',bestFeature)
        AIC_initial=AIC_updated
        #AIC_track.append(AIC_updated)
        features_selected.append(bestFeature)
        X_list.remove(bestFeature)
        
print('final',features_selected)

    
    
    


# In[18]:


print(regr.aic)


# In[103]:


Final_X=pd.DataFrame(X[features_selected])
Final_X['SalePrice']=train['SalePrice']
Final_X.to_csv("stepwise_forward.csv", sep=',', encoding='utf-8')

