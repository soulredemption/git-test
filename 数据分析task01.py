#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
a = arange(15).reshape(3, 5)
a 


# In[2]:


a.shape


# In[3]:


import numpy as np


# In[5]:


print(np.__version__)
np.show_config()


# In[9]:


z = np.zeros(10)
print(z)


# In[11]:


z = np.zeros((10,10))
print('%d bytes'%(z.size*z.itemsize))


# In[15]:


np.info(np.add)


# In[16]:


z = np.zeros(10)
z[4] = 1
print(z)


# In[18]:


z = np.arange(10,50)
print(z)


# In[21]:


z = np.arange(50)
z = z[::-1]
print(z)


# In[23]:


z = np.arange(9).reshape(3,3)
print(z)


# In[24]:


nz = np.nonzero([1,2,0,0,4,0])
print(nz)


# In[26]:


z = np.eye(3)
print(z)


# In[27]:


z = np.random.random((3,3,3))
print(z)


# In[28]:


z = np.random.random((10,10))
zmin, zmax = z.min(),z.max()
print(zmin,zmax)


# In[29]:


z = np.random.random(30)
m = z.mean()
print(m)


# In[30]:


z = np.ones((10,10))
z[1:-1,1:-1]=0
print(z)


# In[32]:


z = np.ones((5,5))
z = np.pad(z, pad_width = 1, mode = 'constant',constant_values=0)
print(z)


# In[33]:


z = np.diag(1+np.arange(4), k=-1)
print(z)


# In[35]:


z = np.zeros((8,8),dtype =int)
z[1::2,::2] = 1
z[::2,1::2] = 1
print(z)


# In[36]:


print(np.unravel_index(100,(6,7,8)))


# In[37]:


z = np.tile(np.array([[0,1],[1,0]]),(4,4))
print(z)


# In[39]:


z = np.random.random((5,5))
zmax, zmin = z.max(), z.min()
z = (z - zmin)/(zmax - zmin)
print(z)


# In[40]:


color = np.dtype([("r", np.ubyte, 1),
                   ("g", np.ubyte, 1),
                   ("b", np.ubyte, 1),
                   ("a", np.ubyte, 1)])
color


# In[41]:


z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(z)


# In[43]:


z = np.arange(11)
z[(3 < z) & (z <= 8)] *= -1
print(z)


# In[45]:


print(sum(range(5),-1))


# In[46]:


from numpy inport *
print(sum(range(5),-1))


# In[47]:


z = np.random.uniform(-10,+10,10)
print(np.copysign(np.ceil(np.abs(z)), z))


# In[48]:


z1 = np.random.randint(0,10,10)
z2 = np.random.randint(0,10,10)
print(np.intersect1d(z1,z2))


# In[49]:


np.sqrt(-1) == np.emath.sqrt(-1)


# In[50]:


yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print ("Yesterday is " + str(yesterday))
print ("Today is " + str(today))
print ("Tomorrow is "+ str(tomorrow))


# In[51]:


z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(z)


# In[52]:


z = np.random.uniform(0,10,10)
print(z-z%1)


# In[54]:


z = np.zeros((5,5))
z += np.arange(5)
print(z)


# In[55]:


def generate():
    for x in range(10):
        yield x
z = np.fromiter(generate(),dtype = float, count = -1)
print(z)


# In[56]:


z = np.linspace(0,1,11,endpoint = False)[1:]
print(z)


# In[57]:


z = np.random.random(10)
z.sort()
print(z)


# In[58]:


z = np.arange(10)
np.add.reduce(z)


# In[59]:


A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A,B)
print(equal)


# In[60]:


z = np.random.random((10,2))
x, y = z[:,0], z[:,1]
r = np.sqrt(x**2+y**2)
t = np.arctan2(y,x)
print(r)
print(t)


# In[61]:


z = np.random.random(10)
z[z.argmax()] = 0
print(z)


# In[62]:


Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)


# In[63]:


x = np.arange(8)
y = x + 0.5
c = 1.0/np.subtract.outer(x, y)
print(np.linalg.det(c))


# In[64]:


for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)

for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)


# In[65]:


np.set_printoptions(threshold=np.nan)
z = np.zeros((16,16))
print(z)


# In[68]:


z = np.arange(100)
v = np.random.uniform(0,100)
index = (np,abs(z-v)).min()
print(z[index])


# In[69]:


Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print (D)


# In[70]:


z = np.arange(10, dtype = np.int32)
z = z.astype(np.float32, copy = False)
print(z)


# In[72]:


Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print (index, value)
for index in np.ndindex(Z.shape):
print (index, Z[index])


# In[73]:


X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print (G)


# In[74]:


n = 10
p = 3
z = np.zeros((n,n))
np.put(z, np.random.choice(range(n*n), p, replace = False),1)
print(z)


# In[78]:


x = np.random.rand(5,10)
Recent versions of np
y = x - x.mean(axis = 1,keepdims = True)
print(y)


# In[79]:


z = np.random.randint(0,10,(3,3))
print(z)
print(z[z[:,1].argsort()])


# In[80]:


z = np.random.randint(0,3,(3,10))
print((~z.any(axis=0)).any())


# In[82]:


Z = np.random.uniform(0,1,10)
z = 0.5
m = z.float[np.abs(Z - z).argmin()]
print(m)


# In[83]:


A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it:
    z[...] = x+y
print(it.operands[2])


# In[84]:


class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)


# In[85]:


x = [1,2,3,4,5,6]
i = [1,3,9,3,4,1]
f = np.bincount(i,x)
print(f)


# In[87]:


d = np.random.uniform(0,1,100)
s = np.random.randint(0,10,100)
d_sums = np.bincount(s, weights=d)
d_counts = np.bincount(s)
d_means = d_sums / d_counts
print(d_means)


# In[88]:


a = np.ones((5,5,3))
b = 2*np.ones((5,5))
print(a * b[:,:,None])


# In[89]:


a = np.arange(25).reshape(5,5)
a[[0,1]] = a [[1,0]]
print(a)


# In[ ]:




