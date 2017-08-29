
# coding: utf-8

# Based on Paulo Vasconselhos notebooks:
# https://paulovasconcellos.com.br/competicao-kaggle-titanic-tutorial-5b11993774f7

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[2]:


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[3]:


train.info()


# In[4]:


test.info()


# In[5]:


train.head()


# In[6]:


train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[7]:


train.head()


# In[8]:


new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)


# In[9]:


new_data_train.head()


# In[10]:


new_data_train.isnull().sum().sort_values(ascending=False).head(10)


# In[11]:


# input mean age on null ages
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)


# In[13]:


new_data_test.isnull().sum().sort_values(ascending=False).head(10)


# In[14]:


new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)


# In[15]:


X = new_data_train.drop('Survived', axis=1)
y = new_data_train['Survived']


# In[16]:


tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X, y)


# In[18]:


tree.score(X, y)


# In[19]:


submission = pd.DataFrame()
submission['PassengerId'] = new_data_test['PassengerId']
submission['Survived'] = tree.predict(new_data_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




