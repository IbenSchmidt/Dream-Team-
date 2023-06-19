#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install numpy scipy matplotlib seaborn pandas altair vega_datasets sklearn bokeh datashader holoviews wordcloud spacy')


# In[11]:


import numpy
import scipy
import matplotlib
import seaborn
import pandas
import altair
import vega_datasets
import sklearn
import bokeh
import datashader
import holoviews
import wordcloud
import spacy


# In[14]:


import pandas as pd
pd.__version__


# In[16]:


import matplotlib.pyplot as plt


# In[18]:


import matplotlib
matplotlib.__version__


# In[ ]:





# In[ ]:





# In[20]:


pump_df = pd.read_csv('https://raw.githubusercontent.com/yy/dviz-course/master/data/pumps.csv')


# In[47]:


pump_df.head() 


# In[50]:


pump_df.head(3)


# In[51]:


len(pump_df)


# In[52]:


pump_df.size


# In[53]:


pump_df.shape  # 13 rows and 2 columns


# In[54]:


pump_df.columns


# In[55]:


pump_df.describe()


# In[56]:


pump_df[:2]


# In[57]:


pump_df[-2:]


# In[58]:


pump_df[1:5]


# In[59]:


pump_df[pump_df.X > 13]


# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


# TODO: Remove below dummy dataframe and write your code here. You probably want to create multiple cells.
death_df = pd.DataFrame({"X": [2., 3.], "Y": [1., 2.]})


# In[62]:


death_df.head(2)


# In[63]:


len(death_df)


# In[ ]:





# In[36]:


death_df.plot()


# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


death_df.plot(x='X', y='Y', kind='scatter', label='Deaths')


# In[39]:


death_df.plot(x='X', y='Y', kind='scatter', label='Deaths', s=2, c='black')


# In[99]:


death_df.plot(x='X', y='Y', s=2, c='black', kind='scatter', label='Deaths')
pump_df.plot(x='X', y='Y', kind='scatter', c='red', s=8, label='Pumps')


# In[41]:


ax = death_df.plot(x='X', y='Y', s=2, c='black', kind='scatter', label='Deaths')


# In[42]:


ax


# In[43]:


ax = death_df.plot(x='X', y='Y', s=2, c='black', alpha=0.5, kind='scatter', label='Deaths')
pump_df.plot(x='X', y='Y', kind='scatter', c='red', s=8, label='Pumps', ax=ax)


# In[78]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()



# In[45]:


from scipy.spatial import Voronoi, voronoi_plot_2d


# In[46]:


# you'll need this
points = pump_df.values
points


# In[82]:


from scipy.spatial import Voronoi, voronoi_plot_2d


# In[83]:


vor = Voronoi(points)


# In[84]:


import matplotlib.pyplot as plt


# In[85]:


voronoi_plot_2d(vor)


# In[93]:


plt.show()


# In[98]:


death_df.plot(x='X', y='Y', kind='scatter', label='Deaths')
ax = death_df.plot(x='X', y='Y', s=2, c='black', alpha=0.5, kind='scatter', label='Deaths')
pump_df.plot(x='X', y='Y', kind='scatter', c='red', s=8, label='Pumps', ax=ax)
voronoi_plot_2d(vor)
plt.savefig('Kudsk.png')


# In[87]:


import matplotlib.pyplot as plt
plt.plot([1,2,3], [4,2,3])
plt.savefig('foo.png')


# In[ ]:





# In[ ]:




