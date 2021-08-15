#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## The King

# The white and black kings are the most important pieces, as loosing them signifies losing the game. The king can only move one square in any direction - up, down, to the sides, and diagonally. It can capture other pieces by moving into their positions, but can never move himself into a *check position* (a positions where he could be captured by the opponent in the next move.

# In[2]:


board = fenBoard('8/8/3R4/3Pk3/8/8/8/2Q5 b - - 0 1')
display(displayBoard(board))
getTurn(board)


# **Question:** What are possible moves for the black king?

# In[3]:


display(displayLegalBoard(board))
getTurn(board)


# In[ ]:




