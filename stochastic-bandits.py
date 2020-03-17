#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:32:12 2020

@author: yimingxu
"""

import numpy as np
import random as rand
import pandas as pd
import multiprocessing as mp

def ETC(X, m):
    result = []
    ex_est = []
    n = np.shape(X)[0]
    k = np.shape(X)[1]
    
    for _ in range(k):
        a = X[range(_*m, (_+1)*m), _]
        ex_est.append(np.mean(a))
        result.extend(a)
    
    result.extend(X[range(m*k, n), 
                    np.argmax(ex_est)])
    return(result)
    

def UCB(X, delta):
    n = np.shape(X)[0]
    k = np.shape(X)[1]
    result = []
    policy = []
    result.extend([X[i, i] for i in range(k)])
    policy.extend(range(k))
    
    c = []
    for _ in range(k):
        l = np.where(np.array(policy) == _ )[0]
        c.append(np.mean(np.array(result)[l])+np.sqrt(2*np.log(1/delta[len(policy)])/len(l)))
    
    while len(result)< n:
        r = np.argmax(c)
        policy.append(r)
        result.append(X[(len(result)), r])
        l = np.where(np.array(policy)==r)[0]
        c[r] = np.mean(np.array(result)[l])+np.sqrt(2*np.log(1/delta[len(policy)-1])/len(l))
    return(result)
    

def MOSS(X):
    n = np.shape(X)[0]
    k = np.shape(X)[1]
    result = []
    policy = []
    result.extend([X[i, i] for i in range(k)])
    policy.extend(range(k))
    
    c = []
    for _ in range(k):
        l = np.where(np.array(policy)==_)[0]
        c.append(np.mean(np.array(result)[l])+np.sqrt(4*np.log(max(1, n/(k*len(l))))/len(l)))
    
    while len(result)< n:
        r = np.argmax(c)
        policy.append(r)
        result.append(X[(len(result)), r])
        l = np.where(np.array(policy)==r)[0]
        c[r] = np.mean(np.array(result)[l])+np.sqrt(4*np.log(max(1, n/(k*(len(l)-1))))/(len(l)-1))
    return(result)
    

def e_Greedy(X, epsilon):
    n = np.shape(X)[0]
    k = np.shape(X)[1]
    result = []
    policy = []
    c = []
    result.extend([X[i, i] for i in range(k)])
    c.extend([X[i, i] for i in range(k)])
    policy.extend(range(k))
    
    while len(result)< n:
        dice = np.random.binomial(1, epsilon[len(result)])
        if dice == 1:
            z = rand.sample(range(k), 1)
            policy.extend(z)
            result.append(X[(len(result)), z[0]])
            l = np.where(np.array(policy)==z[0])[0]
            c[z[0]] = np.mean(np.array(result)[l])
        else:
            policy.append(np.argmax(c))
            result.append(X[(len(result)), np.argmax(c)])
            z = policy[len(policy)-1]
            l = np.where(np.array(policy)==z)[0]
            c[z] = np.mean(np.array(result)[l])
    return(result)


 
mu = [0.1*k for k in range(1, 11)]
delta_UCB = [10**(-6) for t in range(1, 1001)]
delta_UCBhf = [1/(1+t*np.log(t)*np.log(t)) for t in range(1, 1001)]
epsilon_eGreedy = [min(1, 0.2*2/(t*0.2*0.2)) for t in range(1, 1001)]
c_ETC = []
c_UCB = []
c_UCBhf = []
c_MOSS = []
c_eGreedy = []
c_ETC20 = []
c_ETC50 = []
c_ETC80 = []

for _ in mu:
    print(_)
    d_ETC = 0
    d_UCB = 0
    d_UCBhf = 0
    d_MOSS = 0
    d_eGreedy = 0
    d_ETC20 = 0
    d_ETC50 = 0
    d_ETC80 = 0
    for __ in range(1000):
        X = np.zeros(shape=(1000,2))
        for i in range(2):
            X[:,i] = np.random.normal(i*_, 1, 1000)
        d_ETC = d_ETC + sum(X[:, 1]) - sum(ETC(X, int(1+max(1, np.log(1000*_*_/4)*4/(_**2)))))
        d_UCB = d_UCB + sum(X[:, 1]) - sum(UCB(X, delta_UCB))
        d_UCBhf = d_UCBhf + sum(X[:, 1]) - sum(UCB(X, delta_UCBhf))
        d_MOSS = d_MOSS + sum(X[:, 1]) - sum(MOSS(X))
        d_eGreedy = d_eGreedy + sum(X[:, 1]) - sum(e_Greedy(X, epsilon_eGreedy))
        d_ETC20 = d_ETC20 + sum(X[:, 1]) - sum(ETC(X, 20))
        d_ETC50 = d_ETC50 + sum(X[:, 1]) - sum(ETC(X, 50))
        d_ETC80 = d_ETC80 + sum(X[:, 1]) - sum(ETC(X, 80))
    c_ETC.append(d_ETC/1000)
    c_UCB.append(d_UCB/1000)
    c_UCBhf.append(d_UCBhf/1000)
    c_MOSS.append(d_MOSS/1000)
    c_eGreedy.append(d_eGreedy/1000)
    c_ETC20.append(d_ETC20/1000)
    c_ETC50.append(d_ETC50/1000)
    c_ETC80.append(d_ETC80/1000)
     
        
mydata = {'ETC': c_ETC, 'ETC(m=20)': c_ETC20, 'ETC(m=50)': c_ETC50,
          'ETC(m=80)': c_ETC80, 'UCB': c_UCB, 'UCB (horizon-free)': c_UCBhf, 'MOSS': c_MOSS, 'e-Greedy': c_eGreedy}

mydata = pd.DataFrame(data = mydata)
mydata.to_csv('mydata.csv', header=False, index=False)






