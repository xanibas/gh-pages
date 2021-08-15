#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## Playing the game

# The game of chess starts with the player with white pieces (*White*) moving first, after which the player with black pieces (*Black*) will respond with a move of his own. From than on players alternate moves (*Turns*) until the end of the game. In the position below *White* has moved it's pawn upward two squares.

# ````{margin}
# ```{note} Note
# :class: tip
# The last move is represented with dark gray squares.
# ```
# ````

# In[2]:


board = newBoard()
board.push(chess.Move.from_uci('e2e4'))
display(displayBoard(board))


# **Question:** Who's turn to play now?

# In[3]:


display(displayBoard(board))
getTurn(board)


# In[ ]:




