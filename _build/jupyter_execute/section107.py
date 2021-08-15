#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## The Knights

# When the knight moves, it jumps to a square that is two squares away horizontally and one square vertically, or two squares vertically and one square horizontally. The complete move therefore looks like the letter L. Unlike all other standard chess pieces, the knight can jump over all other pieces (of either color) to its destination square.

# In[2]:


board = fenBoard('8/8/8/6q1/2N1rbk1/5N2/8/8 w - - 0 1')
display(displayBoard(board))
getTurn(board)


# **Question:** What are possible moves for the white knights?

# In[3]:


display(displayLegalBoard(board))
getTurn(board)


# In[ ]:




