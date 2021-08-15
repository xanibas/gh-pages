#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## Interactive play

# ````{margin}
# ```{note} Note
# :class: tip
# For each problem we provide the link to **lichess** interactive board.
# ```
# ````

# Below an example of a very elegant and rarely seen queen sacrifice from a game between *Donald Byrne* (White) and Bobby Fisher (Black) in New York, 1956 (<a href="https://lichess.org/analysis/r3r1k1/pp3pbp/1qp1b1p1/2B5/2BP4/Q1n2N2/P4PPP/3R1K1R_w_-_-_0_1" target="_blank">Donald Byrne vs Bobby Fisher, 1956</a>). It is pratical to explore the position in an interactive board, as provided by *lichess* with the provided link.

# In[2]:


board = newBoard()
display(displayBoard(board))
getTurn(board)


# ````{margin}
# ```{note} Note
# :class: tip
# You can either check the answer or explore solutions using the link provided above.
# ```
# ````

# **Question:** What move should black do?

# In[3]:


board = newBoard()
display(displayBoard(board))
getTurn(board)


# In[ ]:




