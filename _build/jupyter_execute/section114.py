#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## Piece Sacrifice

# In chess, a sacrifice is a deliberate move to exchange a chess piece of higher value for an opponent's piece of lower value, with the objective of gaining tactical or positional compensation. Below an example of a very elegant and rarely seen queen sacrifice ([Donald Byrne vs Bobby Fisher, 1956](https://lichess.org/analysis/r3r1k1/pp3pbp/1qp1b1p1/2B5/2BP4/Q1n2N2/P4PPP/3R1K1R_w_-_-_0_1)).

# In[2]:


board = fenBoard('r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3R1K1R b - - 0 1')
board.push(chess.Move.from_uci('g4e6'))
display(displayBoard(board))
getTurn(board)


# **Question:** After sacrificing the queen, what move would you suggest for black?

# In[3]:


board.push(chess.Move.from_uci('c5b6'))
display(displayBoard(board))
getTurn(board)


# In[ ]:




