#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## The Queen

# The queen, by far the most powerfull piece, is able to move any number of squares vertically, horizontally or diagonally. She is often used in conjunction with another piece, such as teamed with a bishop or rook, where the pieces can guard each other while threatening the opponent pieces.

# In[2]:


board = fenBoard('8/8/8/8/QR2q3/8/5PP1/7K b - - 0 1')
display(displayBoard(board))
getTurn(board)


# **Question:** What are possible moves for the black queen?

# In[3]:


display(displayLegalBoard(board))
getTurn(board)


# In[ ]:




