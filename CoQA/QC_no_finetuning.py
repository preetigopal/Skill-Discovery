#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm
import pickle
import logging
logger = logging.getLogger(__name__)
import gc
import random
from random import randint
import os
import shutil


# In[2]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:

def sigmoid(x):
    ex = np.exp(x)
    return ex / (1 + ex)


# In[ ]:

class myBanditClass:
    def __init__(self,nchoices,cost,algo_list,batch_size,numQueries,dim,explore_prob=0.2, decay=0.9999,random_seed = 1, SW_size = 500):   
        self.nchoices = nchoices   
        self.algo_list = algo_list
        self.cost = cost
        self.t = batch_size
        self.explore_prob = explore_prob
        self.decay = decay
        self.random_seed = random_seed
        self.arms_EG = [Ridge(alpha=1.0) for _ in range(self.nchoices)]
        self.arms_SWUCB = [Ridge(alpha=1.0) for _ in range(self.nchoices)]        
        self.alpha = 1.0
        self.dim = dim;
        self.SW_size = SW_size
        self.A_SW = [np.identity(dim) for _ in range(self.nchoices)]    

            
    def predict(self,context,batch_size):
        self.t += batch_size
        for algo in self.algo_list:
            if algo=='Random':
                print('Using Random selection for bandit')
                np.random.seed(self.random_seed)
                best_arms_RS = np.random.choice(self.nchoices, size = context.size(0))     
            if algo=='EpsilonGreedy':
                print('Using Epsilon Greedy for bandit')
                cost_sensitive_score = np.zeros([context.size(0),self.nchoices])
                for arm in range(self.nchoices):
                    score = sigmoid(self.arms_EG[arm].predict(context))
                    cost_sensitive_score[:,arm] = score/self.cost[arm]
                best_arms_EG = np.argmax(cost_sensitive_score,axis=1)
                np.random.seed(self.random_seed)
                toss = np.random.random(size = context.shape[0])# get random floats between 0 and 1 
                toss_indices = (toss <= self.explore_prob) # bools
                nTosses = toss_indices.sum()
                best_arms_EG[toss_indices] = np.random.choice(self.nchoices, size = nTosses)# select a random arm
                self.explore_prob = self.explore_prob*self.decay            
            if algo=='SWUCB':
                print('Using SW-UCB for bandit')
                cost_sensitive_score = np.zeros([context.size(0),self.nchoices]) 
                for arm in range(self.nchoices):
                    A_SW_inv = np.linalg.inv(self.A_SW[arm])
                    uncertainty = np.zeros(batch_size)
                    for i in range(batch_size):                           
                        temp = np.matmul(context[i,:],A_SW_inv);
                        uncertainty[i] = self.alpha * np.sqrt(np.matmul(temp,np.transpose(context[i,:])))
                    mean_score = sigmoid(self.arms_SWUCB[arm].predict(context))
                    cost_sensitive_score[:,arm] = (mean_score/self.cost[arm]) + uncertainty
                best_arms_SWUCB = np.argmax(cost_sensitive_score,axis=1)
        best_arms = [best_arms_RS, best_arms_EG, best_arms_SWUCB]
        return [best_arms]
    
    def fit(self,context,arm_memberships):                                            
        # Fit ridge regression for Epsilon Greedy
        for i in range(self.nchoices):
            soft_label = arm_memberships[:,i]
            soft_label = np.clip(soft_label, 1e-8, 1 - 1e-8)   # numerical stability
            inv_sig_y = np.log(soft_label / (1 - soft_label))  # transform to log-odds-ratio space
            self.arms_EG[i].fit(context,inv_sig_y)
        # Fit ridge regression for each arm for SW-UCB
        for arm in range(self.nchoices):
            startPt = max(0,self.t - self.SW_size)
            endPt = self.t
            context_within_window = context[startPt:endPt,:]
            arm_memberships_within_window = arm_memberships[startPt:endPt,:]
            soft_label = arm_memberships_within_window[:,arm]
            soft_label = np.clip(soft_label, 1e-8, 1 - 1e-8)   # numerical stability
            inv_sig_y = np.log(soft_label / (1 - soft_label))  # transform to log-odds-ratio space
            self.arms_SWUCB[arm].fit(context_within_window,inv_sig_y)
            
            self.A_SW[arm] = np.identity(self.dim)
            for i in range(context_within_window.shape[0]):                    
                self.A_SW[arm] = np.add(self.A_SW[arm],np.matmul(np.transpose(context_within_window[i,:]),context_within_window[i,:]))
        return self  
    
 

 # Set the cost
nArms = 4
cost = np.zeros(nArms)
cost[0] = 1
cost[1] = 1
cost[2] = 1
cost[3] = 2 # cost of oracle arm is twice the other arms
print(cost)

# In[ ]:


# lst_seeds = [i for i in range(5)]  # averagin over 10 runs
lst_seeds = [4,5,6,7,8,9]
print(lst_seeds)

# In[3]:


# Set the rewards
def set_rewards(eval_examples, ideal_policy,nArms):
    
    y = np.zeros([len(ideal_policy),nArms])
    print(y.shape)
    rewards_what = np.array(ideal_policy==0)
    rewards_who_where_when = np.array(ideal_policy==1)
    rewards_others = np.array(ideal_policy==2)
    rewards_oracle = [1 for _ in range(len(eval_examples))]
    y[:,0] = rewards_what
    y[:,1] = rewards_who_where_when
    y[:,2] = rewards_others
    y[:,3] = rewards_oracle      
    
    for i in range(len(eval_examples)):
        if i<1000:
            y[i,0] = rewards_what[i] *0
            y[i,1] = rewards_who_where_when[i] *0
            y[i,2] = rewards_others[i] *0
        elif i<2000:
            y[i,0] = rewards_what[i] *0.1
            y[i,1] = rewards_who_where_when[i] *0.1
            y[i,2] = rewards_others[i] *0.1
        elif i<3000:
            y[i,0] = rewards_what[i] *0.2
            y[i,1] = rewards_who_where_when[i] *0.2
            y[i,2] = rewards_others[i] *0.2
        elif i<4000:
            y[i,0] = rewards_what[i] *0.3
            y[i,1] = rewards_who_where_when[i] *0.3
            y[i,2] = rewards_others[i] *0.3
        elif i<5000:#1500:
            y[i,0] = rewards_what[i] *0.4
            y[i,1] = rewards_who_where_when[i] *0.4
            y[i,2] = rewards_others[i] *0.4
        elif i<6000:#1800:
            y[i,0] = rewards_what[i] *0.5
            y[i,1] = rewards_who_where_when[i] *0.5
            y[i,2] = rewards_others[i] *0.5
        elif i<7000:
            y[i,0] = rewards_what[i] *0.6
            y[i,1] = rewards_who_where_when[i] *0.6
            y[i,2] = rewards_others[i] *0.6
        elif i<8000:
            y[i,0] = rewards_what[i] *0.7
            y[i,1] = rewards_who_where_when[i] *0.7
            y[i,2] = rewards_others[i] *0.7
        elif i<9000:
            y[i,0] = rewards_what[i] *0.8
            y[i,1] = rewards_who_where_when[i] *0.8
            y[i,2] = rewards_others[i] *0.8
        elif i<10000:
            y[i,0] = rewards_what[i] *0.9
            y[i,1] = rewards_who_where_when[i] *0.9
            y[i,2] = rewards_others[i] *0.9
        else:
            y[i,0] = rewards_what[i]
            y[i,1] = rewards_who_where_when[i] 
            y[i,2] = rewards_others[i]

    return y


# In[ ]:


def set_ideal_policies(eval_examples,ideal_policy,nArms):
    new_ideal_policy1 = ideal_policy.copy()
    new_ideal_policy2 = ideal_policy.copy()
    for i in range(len(eval_examples)):
        if i<5000:#1500:
            new_ideal_policy1[i] = nArms-1
            new_ideal_policy2[i] = nArms-1
        elif i<6000:#1800:
            new_ideal_policy2[i] = nArms-1
    return [new_ideal_policy1,new_ideal_policy2]

# In[ ]:


def print_and_check(new_ideal_policy1,new_ideal_policy2,y):
    print('printing policies\n')


# In[ ]:


def run_skill_discovery(eval_examples,new_ideal_policy1,new_ideal_policy2,y,banditModel,batch_size,num_algos):
    all_rewards = np.zeros([num_algos,len(eval_examples)])
    all_actions = np.zeros([num_algos,len(eval_examples)])
    new_data_for_bandit = np.zeros([len(eval_examples),768])
    initial_history_actions_counter_UCB = np.zeros([1,nArms])
    initial_history_actions_SWUCB = np.zeros([batch_size,nArms])

    # initial seed - all policies start with the same small random selection of actions/rewards
    cached_train_examples_file = 'train_QC_no_finetuning_examples.pk'
    with open(cached_train_examples_file, "wb") as writer:
        pickle.dump(eval_examples[:batch_size], writer)

    first_batch_flag = True
    bert_model = 'bert-base-uncased'
    dir_path = 'no_finetuning_QC_models/batch_'
#     dir_path = 'UCB/intermediate_QLen_models/batch_'
    batch_num = 1
    output_dir = f'{dir_path}{batch_num}'
    old_output_dir = output_dir

    new_context = run_mycoqa.getFineTunedContext(first_batch_flag,bert_model,
                                             output_dir,output_dir,
                                             cached_train_examples_file)
    first_batch = new_context
    np.random.seed(1)
    actions_this_batch = np.random.randint(nArms, size=batch_size)
    
    for algo_number in range(num_algos):            
        rewards1_this_batch = (new_ideal_policy1[:batch_size] == actions_this_batch)
        rewards2_this_batch = (new_ideal_policy2[:batch_size] == actions_this_batch)
        rewards_this_batch = (np.logical_or(rewards1_this_batch,rewards2_this_batch)).astype(int)
        all_actions[algo_number,:batch_size] = actions_this_batch
        all_rewards[algo_number,:batch_size] = rewards_this_batch

    new_data_for_bandit[:batch_size] = first_batch    

    banditModel.fit(first_batch,y[:batch_size,:])

#     run_mycoqa.fineTune(first_batch_flag,bert_model,output_dir,
#                         output_dir,cached_train_examples_file)


    # In[ ]:


    for i in tqdm(range(int(np.floor(len(eval_examples) / batch_size)))):
        #print('batch',i+2)
        batch_st = (i + 1) * batch_size
        batch_end = (i + 2) * batch_size
        batch_end = np.min([batch_end, len(eval_examples)])

        if batch_end==len(eval_examples):
            break

        with open(cached_train_examples_file, "wb") as writer:
            pickle.dump(eval_examples[batch_st:batch_end], writer) 
#         first_batch_flag = False
        new_context = run_mycoqa.getFineTunedContext(first_batch_flag,bert_model,
                                                     old_output_dir,output_dir,
                                                     cached_train_examples_file)

        new_data_for_bandit[batch_st:batch_end] = new_context    
        actions_this_batch_list = banditModel.predict(new_context,batch_size)
        

        for algo_number in range(num_algos):            
            rewards1_this_batch = (new_ideal_policy1[batch_st:batch_end] == actions_this_batch_list[0][algo_number])
            rewards2_this_batch = (new_ideal_policy2[batch_st:batch_end] == actions_this_batch_list[0][algo_number])
            rewards_this_batch = (np.logical_or(rewards1_this_batch,rewards2_this_batch)).astype(int)
            all_actions[algo_number,batch_st:batch_end] = actions_this_batch
            all_rewards[algo_number,batch_st:batch_end] = rewards_this_batch

        old_output_dir = output_dir
        batch_num+=1
        output_dir = f'{dir_path}{batch_num}'
        
#         run_mycoqa.fineTune(first_batch_flag, bert_model, old_output_dir,
#                             output_dir, cached_train_examples_file)
#         shutil.rmtree(old_output_dir)
#         os.rmdir(old_output_dir)

        banditModel.fit(new_data_for_bandit[:batch_end],y[:batch_end,:])
        
    return [all_rewards, all_actions]
# In[ ]:


def store_cumulative_rewards(all_rewards,all_actions,seed,new_ideal_policy1,new_ideal_policy2,ideal_policy):

    filename = 'outputs/QC/QC_with_cost_no_finetuning_train_data_all_rewards_{}.npy'.format(seed)
    np.save(filename,all_rewards)
    filename = 'outputs/QC/QC_with_cost_no_finetuning_train_data_all_actions_{}.npy'.format(seed)
    np.save(filename,all_actions) 
   
    
    filename = 'outputs/QC/QC_with_cost_no_finetuning_train_data_new_ideal_policy1_{}.npy'.format(seed)
    np.save(filename,new_ideal_policy1)
    
    filename = 'outputs/QC/QC_with_cost_no_finetuning_train_data_new_ideal_policy2_{}.npy'.format(seed)
    np.save(filename,new_ideal_policy2)
    
    filename = 'outputs/QC/QC_with_cost_no_finetuning_train_data_ideal_policy_{}.npy'.format(seed)
    np.save(filename,ideal_policy)

# In[ ]:


import run_mycoqa
# import importlib
# importlib.reload(run_mysquad)
os.environ['CUDA_VISIBLE_DEVICES']='3,4'

# In[ ]:


bandit_algo_list = ['Random','EpsilonGreedy','SWUCB']
num_algos = len(bandit_algo_list);

for seed_num,seed in tqdm(enumerate(lst_seeds)):

    print(seed_num)
    print(seed)
    numQueries = 20000
    dim_context = 768;
    ideal_policy = np.load('outputs/labels_of_coqa_question_category_train_data.npy')
    eval_examples = torch.load('outputs/coqa_examples_train_data.pt')
    ideal_policy = ideal_policy[:numQueries]
    eval_examples = eval_examples[:numQueries]

    random.seed(seed)
    random.shuffle(eval_examples)
    random.seed(seed)
    random.shuffle(ideal_policy)

    y = set_rewards(eval_examples,ideal_policy,nArms)    
    [new_ideal_policy1,new_ideal_policy2] = set_ideal_policies(eval_examples, ideal_policy,nArms)    
    print_and_check(new_ideal_policy1,new_ideal_policy2,y)   
    batch_size = 20

    banditModel = myBanditClass(nchoices = nArms,cost = cost,algo_list = bandit_algo_list,batch_size = batch_size,numQueries=numQueries,dim=dim_context)

    [all_rewards, all_actions] = run_skill_discovery(eval_examples,new_ideal_policy1,new_ideal_policy2,y
                                                    ,banditModel,batch_size,num_algos)
    store_cumulative_rewards(all_rewards,all_actions,seed,new_ideal_policy1,new_ideal_policy2,ideal_policy)
    