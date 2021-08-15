#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## Setting up the Chess Board

# Chess is a two-player strategy board game played on a chessboard, a checkered gameboard with 64 squares arranged in an 8×8 grid consisting of eight vertical rows called files (*a-h*) and eight horizontal rows called ranks (*1-8*). Below you can see an example of an empty *chessboard*, the gameboard on which the chess pieces are placed.

# ````{margin}
# ```{note} Note
# :class: tip
# At the beginning of the game the queens are placed on the square of the same color.
# ```
# ````

# In[2]:


board = fenBoard('8/8/8/8/8/8/8/8 w - - 0 1')
display(displayBoard(board))


# **Question:** How do we arrange the chess pieces on the chessboard?

# In[3]:


board = newBoard()
display(displayBoard(board))


# In[ ]:




