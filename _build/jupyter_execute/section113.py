#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## En Garde

# *En garde* is a French phrase used in chess to warn the opponent that their queen is threatened. Announcing it is not required under the rules of chess, but it can be used to ensure that capturing the enemy queen is not just the result of an oversight from our opponent. Below an example of a very elegant and rare queen sacrifice ([Donald Byrne vs Bobby Fisher, 1956](https://lichess.org/analysis/r3r1k1/pp3pbp/1qp1b1p1/2B5/2BP4/Q1n2N2/P4PPP/3R1K1R_w_-_-_0_1)).

# In[2]:


board = fenBoard('r4rk1/pp2Bpbp/1qp3p1/8/2BP2b1/Q1n2N2/P4PPP/3RK2R w K - 0 1')
board.push(chess.Move.from_uci('e7c5'))
display(displayBoard(board))
getTurn(board)


# **Question:** After sacrificing the queen, what move would you suggest for black?

# In[3]:


display(displayBoard(board))
getTurn(board)


# In[ ]:




