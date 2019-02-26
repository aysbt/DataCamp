#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[10]:


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n
    return x, y


# In[11]:


def perform_bernoulli_trials(n, p):
    """Bernoulli trial as a flip of a possibly biased coin. Specifically, each coin flip has a probability p 
    of landing heads (success) and probability 1−p of landing tails (failure). We will write a function to 
    perform n Bernoulli trials, perform_bernoulli_trials(n, p), which returns the number of successes out of n 
    Bernoulli trials, each of which has probability p of success."""
    n_success = 0
    
    for i in range(n):
        random_number = np.random.random()
        if random_number < p :
            n_success += 1
            
    return n_success


# In[12]:


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)
    # Return entry [0,1]
    return corr_mat[0,1]


# In[13]:


def bootstrap_replicate_1d(data, func):
    """Generate the bootstrap replicate of 1D data
       We pass the data and also a function that compute the statisitc of interest such as np.mean, np.median"""
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)


# In[14]:


def draw_bs_reps(data, func, size=1):
    """Genrate bootstrap replicate over and over again taking the size argument and loop"""
    #initialize an empty array
    bs_replicates = np.empty(size)
    #Write a for loop that ranges over size and computes a replicate using bootstrap_replicate_1d()
    for i in range(size):
       bs_replicates[i] = bootstrap_replicate_1d(data, func)
    #Return the array of replicates bs_replicates
    return bs_replicates


# In[9]:


def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    """Compute the slope and intercept from resampled data to get bootstrap replicated """
    #Use np.arange() to set up an array of indices going from 0 to len(x). 
    #These are what you will resample and use them to pick values out of the x and y arrays
    inds = np.arange(len(x))
    #initialize an empty slope and an empty intercept replicate arrays to be of size
    bs_slope_rep = np.empty(size)
    bs_intercept_rep =np.empty(size)
    
    for i in range(size):
        #Resample the indices inds
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        #Use np.polyfit() on the new x and y arrays and store the computed slope and intercept
        bs_slope_rep[i], bs_intercept_rep[i] = np.polyfit(bs_x, bs_y,1)
    #Return the pair bootstrap replicates of the slope and intercept
    return bs_slope_rep, bs_intercept_rep


# In[15]:


def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
    #Return the array of replicates
    return perm_replicates


# In[16]:


def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff


# In[ ]:




