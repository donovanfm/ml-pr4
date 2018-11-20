#!/usr/bin/env python
# coding: utf-8

# In[251]:


import pandas as pd
import numpy as np
from matplotlib import pyplot
from gym.envs.toy_text import frozen_lake
from gym.envs.toy_text import taxi
import time


# In[2]:


# https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    V = np.zeros(env.nS)
    iterationCount = 0
    deltas = []
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s] = best_action_value        
            iterationCount += 1
            deltas.append(delta)
            #print "%-4d: %f" %(iterationCount, delta)
        # Check if we can stop 
        if delta < theta:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    
    return policy, V, iterationCount


# In[3]:


# https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    iterationCount = 0
    while True:
        iterationCount += 1
        #print iterationCount
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V, iterationCount


# In[4]:


frozen_lake.FrozenLakeEnv().render()


# In[5]:


taxi.TaxiEnv().render()


# In[6]:


print frozen_lake.FrozenLakeEnv().nS
print taxi.TaxiEnv().nS


# In[7]:


get_ipython().magic(u'timeit print value_iteration(frozen_lake.FrozenLakeEnv())')


# In[8]:


get_ipython().magic(u'timeit print policy_improvement(frozen_lake.FrozenLakeEnv())')


# In[10]:


get_ipython().magic(u'timeit print value_iteration(taxi.TaxiEnv(), theta=0.0001, discount_factor=0.99)')


# In[337]:


taxi_policy, taxi_values, taxi_iters = value_iteration(taxi.TaxiEnv(), theta=0.0001, discount_factor=0.99)


# In[11]:


get_ipython().magic(u'timeit print policy_improvement(taxi.TaxiEnv(), discount_factor=0.99)')


# In[339]:


taxi_policy_pi, taxi_values_pi, taxi_iters_pi = policy_improvement(taxi.TaxiEnv(), discount_factor=0.99)


# In[345]:


def test_policy(env, policy, episodes=1000, max_iters=100):
    rewards = []
    for _ in range(episodes):
        env.reset()
        r = 0
        done = False
        iters = 0
        while not done:
            if iters > max_iters: break
            iters += 1
            a = np.argmax(policy[env.s])
            ob, reward, done, prob = env.step(a)
            r += reward
        rewards.append(r)
    return np.mean(rewards)


# In[326]:


policy = [[1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 1.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 0., 1., 0.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.]]
env = frozen_lake.FrozenLakeEnv()
test_policy(env, policy)


# In[347]:


test_policy(taxi.TaxiEnv(), taxi_policy, episodes=1000000)


# In[348]:


test_policy(taxi.TaxiEnv(), taxi_policy_pi, episodes=1000000)


# In[302]:


class QAgent(object):
    def __init__(self, env, gamma=0.99, alpha=0.2, alpha_decay=1e-3, epsilon=1, epsilon_decay=1e-3):
        self.env = env
        self.Q = {s : {a : 0.5 for a in range(env.action_space.n)} for s in range(env.observation_space.n)}
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.max_change = 0
        
    def train(self, convergence_threshold=1e-5, max_iters=5000):
        while True:
            ob = self.env.reset()
            agent.max_change = 0
            ob_seq = [ob]
            previous_ob = None
            action = None
            reward = 0
            iters = 0
            while True:
                if iters > max_iters: break
                iters += 1
                action = agent.act(previous_ob, action, ob, reward)
                previous_ob = ob
                ob, reward, done, prob = self.env.step(action)
                ob_seq.append(ob)
                if done:
                    agent.act(previous_ob, action, ob, reward, terminal=True) #final update
                    break
            agent.decay()
            if agent.max_change < convergence_threshold:
                break

    def act(self, previous_observation, previous_action, observation, reward, terminal=False):
        maxA, maxV = max([(a,v) for a,v in self.Q[observation].iteritems()], key=lambda x: x[1])
        if terminal: maxV = 0
        if previous_observation is not None and previous_action is not None:
            prev = self.Q[previous_observation][previous_action]
            self.Q[previous_observation][previous_action] =                 (1 - self.alpha) * self.Q[previous_observation][previous_action] +                 self.alpha * (reward + self.gamma * maxV)
            self.max_change = max(self.max_change, abs(prev - self.Q[previous_observation][previous_action]))
        if np.random.random() < self.epsilon:
            chosenA = self.env.action_space.sample()
        else:
            chosenA = maxA
        return chosenA
    
    def decay(self, parameter='both'):
        if parameter=='alpha' or parameter=='both':
            self.alpha *= (1-self.alpha_decay)
        if parameter=='epsilon' or parameter=='both':
            self.epsilon *= (1-self.epsilon_decay)
            
    def run(self, n_trials=1, max_iters=100):
        rewards = []
        for _ in range(n_trials):
            self.env.reset()
            done = False
            iters = 0
            r = 0
            while not done:
                if iters > max_iters: break
                iters += 1
                a, _ = max([(a,v) for a,v in self.Q[env.s].iteritems()], key=lambda x: x[1])
                #print self.env.s
                ob, reward, done, prob = self.env.step(a)
                r += reward
            #print self.env.s, reward
            rewards.append(r)
        return rewards


# In[ ]:





# In[260]:


lake_data = []
env = frozen_lake.FrozenLakeEnv()
for ep in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#for ep in [0]:
    for al in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
        print "e: %f   a: %f" % (ep, al)
        agent = QAgent(env, epsilon=ep, alpha=al)
        start = time.time()
        agent.train()
        trainTime = time.time() - start
        trainScore = np.mean(agent.run(1000))
        lake_data.append({"alpha": al, "epsilon": ep, "trainTime": trainTime, "trainScore": trainScore})
env.close()


# In[298]:


for ep in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    data = [d for d in lake_data if d['epsilon']==ep]
    pyplot.plot([d['alpha'] for d in data], [d['trainScore'] for d in data])
pyplot.legend([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], loc=(1.03, 0.02), title="epsilon")
pyplot.title("Q-Learning Performance on Lake Environment", fontsize=15)
pyplot.xlabel("alpha", fontsize=14)
pyplot.ylabel("average reward", fontsize=14)
#pyplot.plot([d['alpha'] for d in lake_data], [d['trainTime'] for d in lake_data])


# In[334]:


for ep in [0.8, 0.9, 1]:
    data = [d for d in lake_data if d['epsilon']==ep]
    pyplot.plot([d['alpha'] for d in data][2:16], [d['trainScore'] for d in data][2:16])
pyplot.legend([0.8, 0.9, 1], loc=(1.03, 0.02), title="epsilon")
pyplot.title("Q-Learning Performance on Lake Environment", fontsize=15)
pyplot.xlabel("alpha", fontsize=14)
pyplot.ylabel("average reward", fontsize=14)


# In[300]:


for ep in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    data = [d for d in lake_data if d['epsilon']==ep]
    pyplot.plot([d['alpha'] for d in data], [d['trainTime'] for d in data])
pyplot.legend([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], loc=(1.03, 0.02), title="epsilon")
pyplot.title("Q-Learning Run-Time on Lake Environment", fontsize=15)
pyplot.xlabel("alpha", fontsize=14)
pyplot.ylabel("seconds", fontsize=14)


# In[269]:


env = frozen_lake.FrozenLakeEnv()
agent = QAgent(env, epsilon=ep, alpha=al)
start = time.time()
agent.train()
trainTime = time.time() - start
trainScore = np.mean(agent.run(1000))
print trainScore, trainTime


# In[303]:


taxi_data_bak = taxi_data[:]


# In[304]:


taxi_data = []
env = taxi.TaxiEnv()
for ep in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#for ep in [0]:
    for al in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
        print "e: %f   a: %f" % (ep, al)
        agent = QAgent(env, epsilon=ep, alpha=al)
        start = time.time()
        agent.train()
        trainTime = time.time() - start
        trainScore = np.mean(agent.run(1000))
        taxi_data.append({"alpha": al, "epsilon": ep, "trainTime": trainTime, "trainScore": trainScore})
env.close()


# In[305]:


for ep in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    data = [d for d in taxi_data if d['epsilon']==ep]
    pyplot.plot([d['alpha'] for d in data], [d['trainScore'] for d in data])
pyplot.legend([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], loc=(1.03, 0.02), title="epsilon")
pyplot.title("Q-Learning Performance on Taxi Environment", fontsize=15)
pyplot.xlabel("alpha", fontsize=14)
pyplot.ylabel("average reward", fontsize=14)


# In[311]:


for ep in [0.8, 0.9, 1.0]:
    data = [d for d in taxi_data if d['epsilon']==ep]
    pyplot.plot([d['alpha'] for d in data][2:16], [d['trainScore'] for d in data][2:16])
pyplot.legend([0.8, 0.9, 1.0], loc=(1.03, 0.02), title="epsilon")
pyplot.title("Q-Learning Performance on Taxi Environment (Zoomed)", fontsize=15)
pyplot.xlabel("alpha", fontsize=14)
pyplot.ylabel("average reward", fontsize=14)


# In[349]:


data = [d for d in taxi_data if d['epsilon']==1]
data[8:16]


# In[306]:


for ep in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    data = [d for d in taxi_data if d['epsilon']==ep]
    pyplot.plot([d['alpha'] for d in data], [d['trainTime'] for d in data])
pyplot.legend([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], loc=(1.03, 0.02), title="epsilon")
pyplot.title("Q-Learning Run-Time on Taxi Environment", fontsize=15)
pyplot.xlabel("alpha", fontsize=14)
pyplot.ylabel("seconds", fontsize=14)


# In[ ]:




