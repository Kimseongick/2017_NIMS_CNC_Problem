#!/usr/bin/env python
# coding: utf-8

# # Data 소개
# ## Data : 1 ~ 5까지 있음 각기 다른 제품
# ## 문제 : 불량의 피쳐를 찾는데 각 공정마다 피쳐가 다르다.

# In[1]:


import numpy as np
import pandas as pd
import time
import random
import os
import math
import datetime


# from scipy.cluster.hierarchy import dendrogram,linkage
from matplotlib import pyplot as plt
from collections import Counter


# In[39]:


i = 4 # 1 to 5
data= pd.read_csv('./data_'+str(i)+'.csv'
                      ,header=None
                      ,index_col=None
#                 ,encoding='cp1252'
                     )
data

for i in range(len(data)):
    y = data.loc[i][1:]
    x = np.arange(len(y))
    plt.plot(x,y)
plt.show()


# In[57]:


for i in range(1,6):
    data= pd.read_csv('./data_'+str(i)+'.csv'
                          ,header=None
                          ,index_col=None
    #                 ,encoding='cp1252'
                         )
    plt.close()
    plt.figure(figsize=(10,6))
    plt.title(str(i)+"th Data, length is "+str(len(data))
             ,fontsize = 20
             )

    for j in range(len(data)):
        y = data.loc[j][1:]
        x = np.arange(len(y))
        plt.plot(x,y)
    plt.show()
    plt.savefig('./Results/Fig_01_data_dist'+str(j)+'.pdf', format='pdf', bbox_inches='tight')


# In[56]:


i = 2
data= pd.read_csv('./data_'+str(i)+'.csv'
                      ,header=None
                      ,index_col=None
#                 ,encoding='cp1252'
                     )
plt.close()
plt.figure(figsize=(10,6))
plt.title(str(i)+"th Data, length is "+str(len(data))
         ,fontsize = 20
         )

for i in range(len(data)):
    y = data.loc[i][1:]
    x = np.arange(len(y))
    plt.plot(x,y)
plt.show()
plt.savefig('./Results/Fig_01_data_dist'+str(i)+'.pdf', format='pdf', bbox_inches='tight')


# In[81]:


i = 2 
data= pd.read_csv('./data_'+str(i)+'.csv'
                      ,header=None
                      ,index_col=None
                     )
y = []
for idx in range(1,len(data.columns)):
    y.append(np.var(data[idx]))
x = np.arange(len(y))
plt.plot(x,y)
plt.show()


# In[82]:


i =  2# 1 to 5
data= pd.read_csv('./data_'+str(i)+'.csv'
                      ,header=None
                      ,index_col=None
#                 ,encoding='cp1252'
                     )
data

for i in range(len(data)):
    y = data.loc[i][1:]
    x = np.arange(len(y))
    plt.plot(x,y)
plt.show()


# In[86]:


temp = data.loc[0][1:]


# # Moving average

# In[107]:


total_mov_ave = []
for i in range(len(data)):
    temp = data.loc[i][1:]
    mov_ave = []
    for ii in range(1,len(data.columns)):
        mov_ave.append(np.average(temp[ii-1:ii+2]))
    y = mov_ave
    x = np.arange(len(y))
    total_mov_ave.append(y)
    plt.plot(x,y)
plt.show()


# In[102]:


xx = np.ndarray((len(data),len(data.columns)),dtype = "double")
xx


# In[111]:


pd.DataFrame([total_mov_ave[0]])


# In[156]:


df = pd.DataFrame(columns=np.arange(len(total_mov_ave[0])))
for idx in range(len(data)):
    df = df.append(pd.DataFrame([total_mov_ave[idx]]), ignore_index=True)


# In[159]:


df_normalized = pd.DataFrame(columns=np.arange(len(total_mov_ave[0])))
for idx in range(len(df.columns)):
     np.average(df[0])


# In[210]:


df_normalized = pd.DataFrame(columns=np.arange(len(total_mov_ave[0])))

for idx in range(len(df.columns)):
    df_normalized[idx] = df[idx]/np.average(df[idx])
x = list(df_normalized.columns)

for idx in range(len(df)):
    plt.plot(x, df_normalized.loc[idx])
# plt.xlim([100,200])
plt.show()


# In[ ]:





# In[209]:


df_var = []
for idx in range(len(df.columns)):
    df_var.append(np.var(df_normalized[idx]))
plt.plot(x,df_var)


# In[124]:


data_ave = []
for idx in range(1,len(data.columns)):
    data_ave.append(np.average(data[idx]))


# In[ ]:


total_mov_ave = []
for i in range(len(data)):
    temp = data.loc[i][1:]
    mov_ave = []
    for ii in range(1,len(data.columns)):
        mov_ave.append(np.average(temp[ii-1:ii+2]))
    y = mov_ave
    x = np.arange(len(y))
    total_mov_ave.append(y)
    plt.plot(x,y)
plt.show()


# In[ ]:




