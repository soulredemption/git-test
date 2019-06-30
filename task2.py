#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


from pandas import Series, DataFrame


# In[4]:


obj = pd.Series([4,7,-5,3])


# In[5]:


obj


# In[6]:


obj.values


# In[7]:


obj.index 


# In[10]:


obj2 = pd.Series([4,7,-5,3],index=['d','b','a','c'])


# In[11]:


obj2


# In[12]:


obj2.index


# In[13]:


obj2['a']


# In[14]:


obj2['d']=6


# In[15]:


obj2[['c', 'a', 'd']]


# In[16]:


obj2[obj2>0]


# In[17]:


obj2 * 2


# In[18]:


np.exp(obj2)


# In[19]:


'b' in obj2


# In[20]:


sdata = {'0hio':35000, 'Texas':71000, '0regon':16000, 'Utah':5000}


# In[21]:


obj3 = pd.Series(sdata)


# In[22]:


obj3


# In[23]:


states = ['California','0hio','Oregon','Texas']


# In[24]:


obj4 = pd.Series(sdata, index = states)


# In[25]:


obj4


# In[26]:


pd.isnull(obj4)


# In[27]:


obj4.isnull


# In[28]:


obj4.isnull()


# In[29]:


obj3


# In[30]:


obj4


# In[31]:


obj3 + obj4


# In[32]:


obj4.name = 'population'


# In[33]:


obj4.index.name = 'state'


# In[34]:


obj4


# In[35]:


obj


# In[36]:


obj.index = ['bob','steve','jeff','ryan']


# In[37]:


o


# In[38]:


data = {'state':['a', 'a','a','b','b','b'], 'year':[2000,2001,2002,2001,2002,2003],'pop':[1.5,1.7,3.6,2.4,2.9,3.2]}
frame = pd.DataFrame(data)


# In[39]:


frame


# In[40]:


frame.head()


# In[42]:


pd.DataFrame(data,columns = ['year','pop','state'])


# In[43]:


frame2 = pd.DataFrame(data,columns = ['year','state','pop','debt'])


# In[44]:


frame2


# In[45]:


frame2.columns


# In[46]:


frame2['state']


# In[47]:


frame2.year


# In[49]:


frame2.loc['3']


# In[50]:


frame2['debt'] = 16.5


# In[51]:


frame2


# In[52]:


frame2['debt'] = np.arange(6.)


# In[53]:


import numpy as np


# In[54]:


frame2['debt'] = np.arange(6.)


# In[55]:


frame2


# In[57]:


val = pd.Series([-1.2,-1.5,-.1.7],index = ['2','4','5'])


# In[58]:


frame2['debt'] = val


# In[59]:


frame2['eastren'] =frame2.state == 'Ohino'


# In[60]:


frame2


# In[61]:


del frame2['eastren']


# In[63]:


frame2.columns


# In[64]:


pd.DataFrame(pop, index = [2001,2002,2003])


# In[65]:


frame3.index.name = 'year'; frame3.columns.name = 'state'


# In[66]:


frame3 = pd.DataFrame(pop)


# In[67]:


pop = {'b':{2001:2.4, 2002:2.9},'a': {2000:1.5, 2001:1.7, 2002:3.6}}


# In[68]:


frame3 = pd.DataFrame(pop)


# In[69]:


frame3


# In[70]:


frame3.T


# In[71]:


pd.DataFrame(pop, index = [2001,2002,2003])


# In[72]:


pdata = {'a':frame3['a'][:-1],'b':frame3['b'][:2]}


# In[73]:


pd.DataFrame(pdata)


# In[74]:


frame3.index.name = 'year';frame3.columns.name = 'state'


# In[75]:


frame3


# In[76]:


frame3.values


# In[77]:


frame.values


# In[78]:


obj = pd.Series(range(3), index = ['a','b','c'])


# In[79]:


index = obj.index


# In[80]:


index


# In[81]:


index[1:]


# In[82]:


labels = pd.Index(np.arange(3))


# In[83]:


labels


# In[84]:


obj2 = pd.Series([1.5,-2.5,0],index = labels)


# In[85]:


obj2


# In[86]:


obj2.index is labels


# In[87]:


frame3


# In[88]:


frame3.columns


# In[89]:


2003 in frame3.index


# In[90]:


dup_labels = pd.Index(['foo','foo','bar','bar'])


# In[91]:


dup_labels


# In[92]:


obj = pd.Series([4.5, 7.2, -5.3, 3.6], index = ['d', 'b', 'a', 'c'])


# In[93]:


obj


# In[94]:


obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])


# In[95]:


obj


# In[97]:


obj3 = pd.Series(['blue','purple','yellow'],index = [0,2,4])


# In[98]:


obj3


# In[99]:


obj3.reindex(range(6),method='ffill')


# In[102]:


frame = pd.DataFrame(np.arange(9).reshape((3,3)), index = ['a', 'c', 'd'], columns = ['Ohio', 'Texas','California'])


# In[103]:


frame


# In[104]:


frame2


# In[105]:


frame2 = frame.reindex(['a','b','c','d'])


# In[106]:


frame2


# In[107]:


states = ['Texas', 'Utah', 'Califonia']


# In[108]:


frame.reindex(columns = states)


# In[ ]:




