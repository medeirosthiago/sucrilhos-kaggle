
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

get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


# In[3]:


pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
matplotlib.style.use('ggplot')


# In[5]:


data = pd.read_csv('data/train.csv')
data.shape


# In[6]:


data.head()


# In[7]:


data.describe()


# In[ ]:


data['Age'].fillna(data['Age'].m)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




