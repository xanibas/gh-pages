#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## The Black Pawns

# Black pawns move in the opposite direction of the white pawns, and follow the same rules.

# In[2]:


board = fenBoard('8/2p5/1P1P4/1R6/pK3pp1/5P1P/8/8 b - - 0 1')
display(displayBoard(board))
getTurn(board)


# **Question:** What are possible moves for the black pawns?

# In[3]:


display(displayLegalBoard(board))
getTurn(board)


# In[ ]:




