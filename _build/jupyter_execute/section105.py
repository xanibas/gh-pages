#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## The Rooks

# The rook moves horizontally or vertically, through any number of unoccupied squares. As with captures by other pieces, the rook captures by occupying the square on which the enemy piece sits.

# In[2]:


board = fenBoard('8/3r4/8/3R4/8/3n1q2/3k4/7R w - - 0 1')
display(displayBoard(board))
getTurn(board)


# **Question:** What are possible moves for the white rooks?

# In[3]:


display(displayLegalBoard(board))
getTurn(board)


# In[ ]:




