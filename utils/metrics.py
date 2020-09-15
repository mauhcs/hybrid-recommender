"""
This file computes the normilized discounted cumulative gains at level k 
to measure the performance of a recommendation system.

A common way to use is:
`DCG_at_k([1 for _ in range(72)], 72), nDCG_at_k([1 for _ in range(72)], 72)`

But you often uses these functions implicitly via `evaluate.py` 

See a brief introduction to nDCG at https://en.wikipedia.org/wiki/Discounted_cumulative_gain

"""

import numpy as np

def DCG_at_k(recommendation_relevance_for_user_i, k):
  """
  Binary ideal Discounted Cummulative Gains @ k
  """
  r_ui = recommendation_relevance_for_user_i[:k]
  if len(r_ui) < k: 
    print(f"WARN: Recommendation less than k:{k}")
  return sum( r_ui[i]/np.log2( (i+1)+1 ) for i in range(k) )

def iDCG_at_k(recommendation_relevance_for_user_i, k):
  """
  Binary ideal Discounted Cummulative Gains @ k
  """
  r_ui = recommendation_relevance_for_user_i[:k]
  if len(r_ui) < k: 
    print(f"WARN: Recommendation less than k:{k}")
  return DCG_at_k([ 1 for _ in r_ui], k)
  
def nDCG_at_k(recommendation_relevance_for_user_i, k=72):
  r_ui = recommendation_relevance_for_user_i[:k]
  if len(r_ui) < k: 
    print(f"WARN: Recommendation less than k:{k}")
  return DCG_at_k(r_ui, k)/iDCG_at_k(r_ui,k)

