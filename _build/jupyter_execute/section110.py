#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## Check

# Chess pieces capture as they move. The rule is simple: if an opponent's piece is on a square on which you are able to move one of your pieces, then on the next move, your piece can go to the square occupied by the opponent's piece and remove it from the chessboard.

# In[2]:


board = fenBoard('8/R7/3R4/4k3/8/8/8/2QK4 b - - 0 1')
board.push(chess.Move.from_uci('e5d6'))
display(displayBoard(board))
getTurn(board)


# **Question:** What happens after the king moves to the rook's square?

# In[3]:


board.push(chess.Move.from_uci('b1b6'))
display(displayBoard(board))
getTurn(board)


# In[ ]:




