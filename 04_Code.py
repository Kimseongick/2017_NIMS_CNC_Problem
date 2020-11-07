#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data_1 = pd.read_csv('data/data_1.csv', header=None, index_col=None)
data_2 = pd.read_csv('data/data_2.csv', header=None, index_col=None)
data_3 = pd.read_csv('data/data_3.csv', header=None, index_col=None)
data_4 = pd.read_csv('data/data_4.csv', header=None, index_col=None)
data_5 = pd.read_csv('data/data_5.csv', header=None, index_col=None)


# In[3]:


data_frame1 = np.array(data_1[list(range(2,np.shape(data_1)[1]))])
data_frame2 = np.array(data_2[list(range(2,np.shape(data_2)[1]))])
data_frame3 = np.array(data_3[list(range(2,np.shape(data_3)[1]))])
data_frame4 = np.array(data_4[list(range(2,np.shape(data_4)[1]))])
data_frame5 = np.array(data_5[list(range(2,np.shape(data_5)[1]))])


# In[4]:


error_list1 = list(data_1[0])
error_list2 = list(data_2[0])
error_list3 = list(data_3[0])
error_list4 = list(data_4[0])
error_list5 = list(data_5[0])


# In[6]:


plt.figure(figsize=(16,9))

plt.subplot(331)
for i in range(len(data_frame1)):
    plt.plot(data_frame1[i])
    
plt.subplot(332)
for i in range(len(data_frame2)):
    plt.plot(data_frame2[i])
    
plt.subplot(333)
for i in range(len(data_frame3)):
    plt.plot(data_frame3[i])
    
plt.subplot(334)
for i in range(len(data_frame4)):
    plt.plot(data_frame4[i])
    
plt.subplot(335)
for i in range(len(data_frame5)):
    plt.plot(data_frame5[i])


# -------------------------

# In[18]:


plt.figure(figsize=(16,9))

for i in range(len(data_frame1)):
    if error_list1[i] == 0:
        plt.plot(data_frame1[i],"b")
    else:
        plt.plot(data_frame1[i],"r")
        print("error index : ", i)


# In[19]:


plt.figure(figsize=(16,9))

for i in range(len(data_frame2)):
    if error_list2[i] == 0:
        plt.plot(data_frame2[i],"b")
    else:
        plt.plot(data_frame2[i],"r")
        print("error index : ", i)


# In[8]:


plt.figure(figsize=(16,9))

for i in range(len(data_frame3)):
    if error_list3[i] == 0:
        plt.plot(data_frame3[i],"b")
    else:
        plt.plot(data_frame3[i],"r")
        print("error index : ", i)


# In[9]:


plt.figure(figsize=(16,9))

for i in range(len(data_frame4)):
    if error_list4[i] == 0:
        plt.plot(data_frame4[i],"b")
    else:
        plt.plot(data_frame4[i],"r")
        print("error index : ", i)


# In[24]:


plt.figure(figsize=(16,9))

for i in range(len(data_frame5)):
    if error_list5[i] == 0:
        plt.plot(data_frame5[i],"b")
    else:
        plt.plot(data_frame5[i],"r")
        print("error index : ", i)


# -----------------------------

# In[22]:


feature = np.mean(np.diff(data_frame2, axis=0), axis=1)
#feature2 = np.mean(data_frame1, axis=1)


# In[23]:


plt.plot(feature, ".")


# In[25]:


plt.figure(figsize=(12,9))
for i in range(len(data_frame2)):
    plt.plot(data_frame2[i][30:150])


# In[14]:


for i in range(len(data_frame2)):
    plt.plot(np.diff(data_frame2[i][50:150], axis=0))


# ## 공지사항 12월 26일
# - 저녁 식사 후 그룹을 나눌 예정입니다.
# - 수료증(참가 확인증)을 위해서는 저녁모임에도 참석하셔야 합니다.
# - 금요일 결과발표 지원자 받습니다.

# - 베이즈 정리 및 기타 통계적 접근
# - 다중 회귀분석
# - derivative

# ## 공지사항 12월 27일
# - 반갑습니다.
# - 금일 저녁은 만찬입니다.

# In[26]:


plt.figure(figsize=(16,9))

err_dt = []
for i in range(len(data_frame3)):
    if error_list3[i] == 0:
        plt.plot(data_frame3[i],"grey")
    else:
        err_dt.append(data_frame3[i])
#         plt.plot(data_frame3[i])
        print("error index : ", i)
for i in err_dt:
    plt.plot(i)
plt.show()


# In[30]:


data_frame = data_frame5
error_list = error_list5

plt.figure(figsize=(16,9))

err_dt = []
for i in range(len(data_frame)):
    if error_list[i] == 0:
        plt.plot(data_frame[i],"grey")
    else:
        err_dt.append(data_frame[i])
#         plt.plot(data_frame3[i])
        print("error index : ", i)
for i in err_dt:
    plt.plot(i)
plt.show()


# In[31]:


data_frame = data_frame4
error_list = error_list4

plt.figure(figsize=(16,9))

err_dt = []
for i in range(len(data_frame)):
    if error_list[i] == 0:
        plt.plot(data_frame[i],"#ECDADA")
    else:
        err_dt.append(data_frame[i])
        print("error index : ", i)
for i in err_dt:
    plt.plot(i)
plt.show()


# In[27]:


ever_list1 = []
for i in range(len(data_frame1)):
def func_chk(_data, _mean, _var, _Z_val):
    _cnt = 0;
    for _i in range(len(_data)):
        if _data[_i] > (_mean[_i] + _var[_i]*_Z_val) or _data[_i] < (_mean[_i] - _var[_i]*_Z_val):
            _cnt = _cnt + 1
    return _cnt/len(_data)

def model_01(data_frame, diff_ratio, Z_val, train_size):
    # initialize
    n = 1

    data_sum = data_frame[0]
    data_square_sum = data_frame[0]*data_frame[0]
    data_mean = data_sum/n
    data_var = data_square_sum/n - data_mean*data_mean
    data_sqrt = np.sqrt(data_var)
    
    over_cnt = [0]
    eval_anomaly = [0]

    
    for i in range(1,len(data_frame)):
        data, n = data_frame[i], i+1
        if i > train_size:
            over_cnt.append(func_chk(data,data_mean,data_sqrt,Z_val))
            if over_cnt[i] > diff_ratio :
                eval_anomaly.append(1)
            else:
                eval_anomaly.append(0)
        else :
            over_cnt.append(0)
            eval_anomaly.append(0)

        data_sum = data_sum + data
        data_square_sum = data_square_sum + data*data
        data_mean = data_sum/n
        data_var = data_square_sum/n - data_mean*data_mean
        data_sqrt = np.sqrt(data_var)

    err_dt = []
    plt.figure(figsize=(16,9))
    plt.title("Total Data")
    for i in range(len(data_frame)):
        if eval_anomaly[i] == 1: # anomaly!
            print("anomaly : ",i)
            err_dt.append(data_frame[i])
            break
        else:
            plt.plot(data_frame[i],"#ECDADA")
    for i in err_dt:
        plt.plot(i)
    plt.show()

    plt.figure(figsize=(6,4))
    plt.title("Over count ratio")
    plt.plot(over_cnt)
    plt.show()

    plt.figure(figsize=(6,4))
    plt.title("Detect anomaly")
    plt.plot(eval_anomaly)
    plt.show()   if error_list1[i] == 0:
        ever_list1.append(data_frame1[i])
ever1 = np.mean(ever_list1, 0)


# In[28]:


def angle(v, u):
    return np.arccos(np.dot(v, u)/(np.sqrt(np.dot(v,v))*np.sqrt(np.dot(u,u))))


# In[30]:


for i in range(len(data_frame1)):
    if error_list1[i] == 0:
        print 'good', angle(ever, data_frame1[i])


# In[31]:


for i in range(len(data_frame1)):
    if error_list1[i] == 1:
        print 'bad', angle(ever, data_frame1[i])


# In[32]:


ever_list2 = []
for i in range(len(data_frame2)):
    if error_list2[i] == 0:
        ever_list2.append(data_frame2[i])
ever2 = np.mean(ever_list2, 0)
for i in range(len(data_frame2)):
    if error_list2[i] == 0:
        print 'good', angle(ever2, data_frame2[i])
for i in range(len(data_frame2)):
    if error_list2[i] == 1:
        print 'bad', angle(ever2, data_frame2[i])


# In[34]:


def func_chk(_data, _mean, _var, _Z_val):
    _cnt = 0;
    for _i in range(len(_data)):
        if _data[_i] > (_mean[_i] + _var[_i]*_Z_val) or _data[_i] < (_mean[_i] - _var[_i]*_Z_val):
            _cnt = _cnt + 1
    return _cnt/len(_data)

def model_01(data_frame, diff_ratio, Z_val, train_size):
    # initialize
    n = 1

    data_sum = data_frame[0]
    data_square_sum = data_frame[0]*data_frame[0]
    data_mean = data_sum/n
    data_var = data_square_sum/n - data_mean*data_mean
    data_sqrt = np.sqrt(data_var)
    
    over_cnt = [0]
    eval_anomaly = [0]

    
    for i in range(1,len(data_frame)):
        data, n = data_frame[i], i+1
        if i > train_size:
            over_cnt.append(func_chk(data,data_mean,data_sqrt,Z_val))
            if over_cnt[i] > diff_ratio :
                eval_anomaly.append(1)
            else:
                eval_anomaly.append(0)
        else :
            over_cnt.append(0)
            eval_anomaly.append(0)

        data_sum = data_sum + data
        data_square_sum = data_square_sum + data*data
        data_mean = data_sum/n
        data_var = data_square_sum/n - data_mean*data_mean
        data_sqrt = np.sqrt(data_var)

    err_dt = []
    plt.figure(figsize=(16,9))
    plt.title("Total Data")
    for i in range(len(data_frame)):
        if eval_anomaly[i] == 1: # anomaly!
            print("anomaly : ",i)
            err_dt.append(data_frame[i])
            break
        else:
            plt.plot(data_frame[i],"#ECDADA")
    for i in err_dt:
        plt.plot(i)
    plt.show()

    plt.figure(figsize=(6,4))
    plt.title("Over count ratio")
    plt.plot(over_cnt)
    plt.show()

    plt.figure(figsize=(6,4))
    plt.title("Detect anomaly")
    plt.plot(eval_anomaly)
    plt.show()


# In[ ]:




