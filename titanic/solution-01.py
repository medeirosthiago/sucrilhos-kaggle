
# coding: utf-8

# Based on Ahmed BESBES notebooks:
# http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html

# In[1]:


from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


# In[2]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


# In[3]:


pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
matplotlib.style.use('ggplot')


# In[4]:


train = pd.read_csv('data/train.csv')
train.shape


# In[5]:


train.head()


# In[6]:


train.describe()


# In[7]:


train['Age'].fillna(train['Age'].median(), inplace=True)


# In[8]:


train.describe()


# In[9]:


survived_sex = train[train['Survived'] == 1]['Sex'].value_counts()
dead_sex = train[train['Survived'] == 0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex, dead_sex])
df.index = ['Survived', 'Dead']


# In[10]:


df.plot(kind='bar', stacked=True, figsize=(15, 8))


# In[14]:


figure = plt.figure(figsize=(15, 8))
plt.hist([train[train['Survived'] == 1]['Age'],
          train[train['Survived'] == 0]['Age']],
          stacked=True,
          bins=30, label=['Survived', 'Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# In[22]:


figure = plt.figure(figsize=(15, 8))
plt.hist([train[train['Survived'] == 1]['Fare'],
          train[train['Survived'] == 0]['Fare']],
          stacked=True,
          bins=30, label=['Survived', 'Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# In[33]:


plt.figure(figsize=(15, 8))
ax = plt.subplot()
ax.scatter(train[train['Survived'] == 1]['Age'], train[train['Survived'] == 1]['Fare'], c='green', s=40)
ax.scatter(train[train['Survived'] == 0]['Age'], train[train['Survived'] == 0]['Fare'], c='red', s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived', 'dead'), scatterpoints=1, loc='upper_right')


# In[38]:



ax = plt.subplot()
ax.set_ylabel('Average fare')
train.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(15, 8), ax=ax)


# In[43]:


survived_embark = train[train['Survived'] == 1]['Embarked'].value_counts()
dead_embark = train[train['Survived'] == 0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark, dead_embark])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(15, 8))

