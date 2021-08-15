#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# In[2]:


get_ipython().run_cell_magic('html', '', '<style>\n@media print {\n    .bd-toc {\n        visibility: hidden;\n    }\n}\n</style>')


# ## The White Pawn 

# The pawn is the most numerous piece in the game of chess, and in most circumstances, also the weakest. Normally white pawns move by advancing a single square upward. However, the first time a pawn is moved, it has the option of advancing two squares. Unlike other pieces, the pawn does not capture in the same direction as it moves, instead it captures diagonally forward one square to the left or right.

# In[3]:


board = fenBoard('8/1p2P3/2p1rp2/3P1P2/7p/2P4P/1P6/8 w - - 0 1')
display(displayBoard(board))


# **Question:** Can you trace all the possible moves that white can take?

# In[4]:


board = fenBoard('8/1p2P3/2p1rp2/3P1P2/7p/2P4P/1P6/8 w - - 0 1')
display(displayLegalBoard(board))


# In[ ]:




