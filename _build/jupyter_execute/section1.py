#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## The White Pawn 

# The pawn is the most numerous piece in the game of chess, and in most circumstances, also the weakest. Historically it represents the army infantry, in particular armed peasants or pikemens. Normally white pawns move by advancing a single square upward. However, the first time a pawn is moved, it has the option of advancing two squares. Unlike other pieces, the pawn does not capture in the same direction as it moves, instead it captures diagonally forward one square to the left or right. A pawn that advances all the way to the opposite side of the board (the opposing player's first rank) is promoted to another piece of that player's choice: a *queen*, *rook*, *bishop*, or *knight* of the same color.

# In[2]:


board = fenBoard('8/1p2P3/2p1rp2/3P1P2/7p/2P4P/1P6/8 w - - 0 1')
display(displayBoard(board))


# **Question:** Can you trace all the possible moves that white can take? And which piece can be promoted?

# In[3]:


board = fenBoard('8/1p2P3/2p1rp2/3P1P2/7p/2P4P/1P6/8 w - - 0 1')
display(displayLegalBoard(board))


# In[ ]:




