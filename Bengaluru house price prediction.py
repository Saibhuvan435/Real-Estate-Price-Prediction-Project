#!/usr/bin/env python
# coding: utf-8

# # Data Science Regression Project: Predicting Home Prices in Banglore

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)
import seaborn as sns


# # Data Load: Loading banglore home prices into a dataframe

# In[14]:


df1=pd.read_csv("C:\\Users\\SAI BHUVAN\\Documents\\ACG_files\\Bengaluru_House_Data.csv")
df1.head()


# In[15]:


df1.shape


# In[16]:


df1.columns


# In[17]:


df1['area_type'].value_counts()


# In[18]:


# Drop features that are not required to build our model

df2=df1.drop(['area_type','availability','society','balcony'],axis=1)
df2.head()


# # Data Cleaning: Handling with Null values

# In[19]:


df2.isnull().sum()


# In[20]:


df3=df2.dropna()
df3.isnull().sum()


# In[21]:


df3.head()


# In[22]:


df3['size'].unique()


# # Feature Engineering

# In[23]:


# Adding new feature(integer) for bhk (Bedrooms Hall Kitchen)

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# In[24]:


df3.head()


# In[25]:


# Explore total_sqft feature

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[26]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# Above shows that total_sqft can be a range (e.g. 2100-2850). For such case we can just take average of min and max value in the range. There are other cases such as 34.46Sq. Meter which one can convert to square ft using unit conversion.

# In[27]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   


# In[28]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head()


# # Adding new feature called price per square feet

# In[29]:


df5=df4.copy()
df5['price_per_sqft']=(df5['price']*100000)/df5['total_sqft']


# In[30]:


df5.head()


# In[31]:


df5_stats = df5['price_per_sqft'].describe()
df5_stats


# # Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations

# In[32]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# In[33]:


location_stats.values.sum()


# In[34]:


len(location_stats)


# In[35]:


len(location_stats[location_stats>10])


# In[36]:


len(location_stats[location_stats<=10])


# # Dimensionality Reduction
# Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, it will help us with having fewer dummy columns

# In[37]:


location_stats_less_than_10=location_stats[location_stats<=10]


# In[38]:


location_stats_less_than_10


# In[39]:


len(df5.location.unique())


# In[40]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[41]:


df5.head(10)


# In[42]:


sns.boxplot('bhk','price_per_sqft',data=df5)


# # Outlier Removal
# We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft

# In[43]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[44]:


df6=df5[~(df5.total_sqft/df5.bhk<300)]


# In[45]:


df6.head()


# In[46]:


df6.shape


# Outlier Removal Using Standard Deviation and Mean

# In[47]:


df6.price_per_sqft.describe()


# Here we find that min price per sqft is 267 rs/sqft whereas max is 176470 rs/sqft, this shows a wide variation in property prices. We should remove outliers per location using mean and one standard deviation

# In[48]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out


# In[49]:


df7 = remove_pps_outliers(df6)
df7.shape


# In[50]:


# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()


# In[51]:


plot_scatter_chart(df7,"Rajaji Nagar")


# In[52]:


plot_scatter_chart(df7,"Yeshwanthpur")


# We should also remove properties where for same location. for excample, if the price of  3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). 

# In[53]:


# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')


# In[54]:


df8 = remove_bhk_outliers(df7)
df8.shape


# In[55]:


# Plotting same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties

plot_scatter_chart(df8,"Rajaji Nagar")


# In[56]:


plot_scatter_chart(df8,"Yeshwanthpur")


# In[57]:


plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel('Price Per Square Feet')
plt.ylabel('count')
plt.show()


# # Outlier Removal Using Bathrooms Feature

# In[58]:


df8.bath.unique()


# In[59]:


plt.hist(df8['bath'],rwidth=0.8)
plt.xlabel('No of bathrooms')
plt.ylabel('count')
plt.show()


# In[60]:


df8[df8.bath>10]


# It is unusual to have 2 more bathrooms than number of bedrooms in a home

# In[61]:


df8[df8.bath>df8.bhk+2]


#  If you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed

# In[62]:


df9=df8[df8.bath<df8.bhk+2]
df9.head()


# In[63]:


df10 = df9.drop(['size','price_per_sqft'],axis=1)
df10.head()


# # Using One Hot Encoding For Location

# In[64]:


dummies=pd.get_dummies(df10.location)
dummies.head()


# In[65]:


df11 = pd.concat([df10,dummies.drop('other',axis=1)],axis=1)
df11.head()


# In[66]:


df12=df11.drop('location',axis=1)


# # Building the model

# In[67]:


X=df12.drop('price',axis=1)
X.head()


# In[68]:


X.shape


# In[69]:


y=df12.price
y.head()


# In[70]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)


# In[71]:


X_train.shape


# In[72]:


X_test.shape


# In[73]:


from sklearn.linear_model import LinearRegression
lr_reg=LinearRegression()
lr_reg.fit(X_train,y_train)


# In[74]:


y_pred=lr_reg.predict(X_test)
y_pred


# In[75]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# # Using K Fold cross validation to measure accuracy of our LinearRegression model

# In[76]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=21)
cross_val_score(lr_reg,X,y,cv=cv)


# We can see that in 5 iterations we get a score above 80% all the time. This is pretty good but we want to test few other algorithms for regression to see if we can get even better score. We will use GridSearchCV for this purpose

# # Find best model using GridSearchCV

# from sklearn.model_selection import GridSearchCV
# 
# from sklearn.linear_model import Lasso
# from sklearn.tree import DecisionTreeRegressor
# 
# def find_best_model_using_gridsearchcv(X,y):
#     algos = {
#         'linear_regression' : {
#             'model': LinearRegression(),
#             'params': {
#                 'normalize': [True, False]
#             }
#         },
#         'lasso': {
#             'model': Lasso(),
#             'params': {
#                 'alpha': [1,2],
#                 'selection': ['random', 'cyclic']
#             }
#         },
#         'decision_tree': {
#             'model': DecisionTreeRegressor(),
#             'params': {
#                 'criterion' : ['mse','friedman_mse'],
#                 'splitter': ['best','random']
#             }
#         }
#     }
#      scores = []
#     cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#     for algo_name, config in algos.items():
#         gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
#         gs.fit(X,y)
#         scores.append({
#             'model': algo_name,
#             'best_score': gs.best_score_,
#             'best_params': gs.best_params_
#         })
# 
#     return pd.DataFrame(scores,columns=['model','best_score','best_params'])
# 
# find_best_model_using_gridsearchcv(X,y)

# Based on above results we can say that LinearRegression gives the best score. Hence we will use that.

# # Test the model for few properties

# In[77]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_reg.predict([x])[0]


# In[78]:


predict_price('Yeshwanthpur',1000,2,2)


# In[79]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[80]:


predict_price('Indira Nagar',1000, 2, 4)


# In[ ]:




