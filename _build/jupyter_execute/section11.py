#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *


# In[2]:


get_ipython().run_cell_magic('html', '', '<style>\n@media print {\n    .bd-toc {\n        visibility: hidden;\n    }\n}\n</style>')


# ## The Chess Pieces

# The chess pieces are what you move on a chessboard when playing a game of chess. There are six different types of chess pieces. Each side starts with 16 pieces: eight pawns, two bishops, two knights, two rooks, one queen, and one king. Kasparov vs Deep blue.

# In[3]:


board = fenBoard('rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
display(displayBoard(board))


# **Question:** Can you idenitify the name of the different chess pieces?

# In[4]:


getChessTable()


# In[ ]:




