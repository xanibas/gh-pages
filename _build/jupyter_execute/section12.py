#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# In[2]:


get_ipython().run_cell_magic('html', '', '<style>\n@media print {\n    .bd-toc {\n        visibility: hidden;\n    }\n}\n</style>')


# ## The Chess Board

# Chess is a two-player strategy board game played on a chessboard, a checkered gameboard with 64 squares arranged in an 8×8 grid, where each square is labaled by its row(a-h) and column position (1-8). Each player begins with 16 pieces, either black or white, arranged at the top and bottom of the chessboard.

# In[3]:


board = fenBoard('8/8/8/8/8/8/8/8 w - - 0 1')
display(displayBoard(board))


# **Question:** Can you locate the inital position of each piece for white and black?

# In[4]:


board = newBoard()
display(displayBoard(board))


# In[ ]:




