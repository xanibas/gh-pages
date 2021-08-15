#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# ## Capturing Pieces

# Chess pieces can move across the board according to very specific rules, and by doing so they can *capture* enemy pieces. In the example below *Black* has just moved the black pawn donward two squares. The white pawn can now move diagonally and capture it, removing the opponent piece from the chessboard.

# ````{margin}
# ```{note} Note
# :class: tip
# We represent possible interesting moves using a green arrow.
# ```
# ````

# In[2]:


board = newBoard()
board.push(chess.Move.from_uci('e2e4'))
board.push(chess.Move.from_uci('d7d5'))
display(displayBoard(board))
getTurn(board)


# **Question:** How will the chessboard look after the white pawn captures the black pawn?

# In[3]:


board.push(chess.Move.from_uci('e4d5'))
display(displayBoard(board))
getTurn(board)


# In[ ]:




