#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## The Bishops

# The bishop is a piece in the game of chess whose movement are limited two diagonals. As a consequence of its diagonal movement, each bishop always remains on either the white or black squares, and so it is also common to refer to them as light-squared or dark-squared bishops. Bishops, like all other pieces except the knight, cannot jump over other pieces.

# In[2]:


board = fenBoard('8/8/8/8/1Q1P4/1b2R3/8/6b1 b - - 0 1')
display(displayBoard(board))
getTurn(board)


# **Question:** What are possible moves for the black bishops?

# In[3]:


display(displayLegalBoard(board))
getTurn(board)


# In[ ]:




