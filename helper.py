from IPython.core.interactiveshell import InteractiveShell

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.image as pli
from IPython.display import SVG
from IPython.display import display_svg
from IPython.display import Image
from IPython.display import display
from IPython.display import HTML
from PIL import Image
from glob import glob

import sys
import re
import inspect
import time
import copy
import io
import glob
import cv2
import string

import chess
import chess.pgn
import chess.engine
import chess.svg

import math
import random
import numpy as np
import pandas as pd
import scipy as sc
import pandas as pd
import networkx as nx

# Some useful generic functions
def writeMat(mat, colNames, filename):
    np.savetxt(filename, mat, fmt='%1.4f', header=" ".join(['\''+str(x)+'\'' for x in colNames]), comments='', delimiter='\t')

def readMat(mat, colNames, filename):
    return np.loadtxt(filename, fmt='%1.4f', skiprows=1, comments='', delimiter='\t')

def saveImage(m, directory, filename, vmin, vmax):
    pli.imsave(directory + '/' + filename + '.png', m, cmap="gray", vmin=vmin, vmax=vmax)
    
def plotMatrix(m,matmin,matmax):
    plt.matshow(m, cmap="gray", vmin=matmin, vmax=matmax)
        
# Some useful chess functions
piecevalues = { 1 : 1, 2 : 3, 3 : 3, 4 : 5, 5 : 9, 6 : 15}

def getChessTable():
    df = pd.DataFrame({
        'White': [
             displayPiece(chess.PAWN,chess.WHITE),
             displayPiece(chess.KNIGHT,chess.WHITE),
             displayPiece(chess.BISHOP,chess.WHITE),
             displayPiece(chess.ROOK,chess.WHITE),
             displayPiece(chess.QUEEN,chess.WHITE),
             displayPiece(chess.KING,chess.WHITE)],
        'Black': [
             displayPiece(chess.PAWN,chess.BLACK),
             displayPiece(chess.KNIGHT,chess.BLACK),
             displayPiece(chess.BISHOP,chess.BLACK),
             displayPiece(chess.ROOK,chess.BLACK),
             displayPiece(chess.QUEEN,chess.BLACK),
             displayPiece(chess.KING,chess.BLACK)],
        'Role': ['Infantry', 'Cavalry', 'Elephantry', 'Chariotry', 'Counselor', 'Counselor'],
        'English': ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King'],
        'Hindi': ['Pyada', 'Ghora', 'Umt', 'Hathi', 'Vazir', 'Raja'],
        'Spanish': ['Peon', 'Caballo', 'Alfill', 'Rukhkh', 'Dama', 'Rey'],
        'Arabic': ['Baidaq', 'Hisan', 'Fil', 'Rook', 'Wazir', 'Malik'],
        'Russian': ['Peshka', 'Kon', 'Slon', 'Ladya', 'Ferz', 'Korol'],
        'Bengali': ['Bora', 'Ghora', 'Goj', 'Nouka', 'Montri', 'Raja'],
    })
    return HTML(df.to_html(escape=False))

def getPieceValue(p):
    if p is None:
        return 0
    elif p.color:
        return piecevalues[p.piece_type]
    else:
        return -piecevalues[p.piece_type]

def getPieceType(p):
    if p is None:
        return 0
    elif p.color:
        return p.piece_type
    else:
        return -p.piece_type
    
def getPieceColor(p):
    if p.color:
        return 1
    else:
        return -1
    
def getBoardColor(board):
    if board.turn:
        return 1
    else:
        return -1

def getBoardType(board):
    if board.turn:
        return 1
    else:
        return 0
    
def getSquareNames(k):
    return map(lambda x:chess.SQUARE_NAMES[x], k)

def getMove(string):
    return chess.Move.from_uci(string)

# Get moves
def getColor(board):
    if board.turn: return 'White'
    else: return 'Black'

def getTurn(board):
    if board.turn: print('Is White Turn' + '  ')
    else: print('Is Black Turn' + '  ')

def getSign(board):
    if board.turn: return 1
    else: return -1

def getCheckMate(board):
    if board.is_checkmate():
        print(getColor(board) + ' is checkmate!' + '  ')
    
def getLegalMoves(board):
    legal = list(board.legal_moves)
    for i in range(len(legal)):
        legal[i] = str(board.san(legal[i]))+' ('+str(board.uci(legal[i]))+')'
    return ', '.join(legal)

def getMovesList(board):
    return chess.Board().variation_san(list(board.move_stack))

def getGameMoves(game):
    moves = game.main_line()
    return third_game.board().variation_san(moves)

def getGameMovesN(game, n):
    moves = list(game.main_line())[-(2*n-1):]
    return getEndBoard(game, n).variation_san(moves)

def getGameMateN(pgn, engine, handler, maxTime, n):
    game = chess.pgn.read_game(pgn)
    board = getEndBoard(game, n)
    while (game is not None and n != getNToMate(board, engine, handler, maxTime)):
        game = chess.pgn.read_game(pgn)
        board = getEndBoard(game, n)
    return game

# Get move scores 
# Socres are given from -1 to 1, with advantage to us if >0.0
def getScores(board, engine, handler, maxTime):
    scores = dict()
    mates = dict()
    for el in board.legal_moves:
        move = engine.go(searchmoves=[el],movetime=maxTime)
        print(move)
        scores[str(board.uci(el))] = handler.info["score"][1].cp
        mates[str(board.uci(el))] = handler.info["score"][1].mate
    for key in scores.keys():
        if isinstance(mates[key], int):
            scores[key] = np.sign(float(mates[key]))
        else:
            scores[key] = 2*round(0.5 + 0.5*(2/(1+np.exp(-0.004*scores[key]))-1),2)-1
    return scores

def getLegalScores(board):
    scores = {}
    legal = list(board.legal_moves)
    for el in legal:
        scores[str(board.uci(el))] = 1
    return scores

def getNextScores(board, engine, handler, move, maxTime):
    board.push(move)
    engine.position(board)
    scores = getScores(board, engine, handler, maxTime)
    board.pop()
    return scores

def getScores2Matrix(scores):
    m = np.zeros((64, 64))
    for key in scores.keys():
        m[getMove(key).from_square,getMove(key).to_square] = scores[key]
    return m

# Filter scores
def getMaxScore(board, engine, handler, maxTime):
    select = getFastBestMove(board, engine, maxTime)
    if (select is not None):
        move = engine.go(searchmoves=[getFastBestMove(board, engine, maxTime)],movetime=maxTime)
        score = handler.info["score"][1].cp
        mate = handler.info["score"][1].mate
        if isinstance(mate, int):
            score = np.sign(float(mate))
        else:
            score = 2*round(0.5 + 0.5*(2/(1+np.exp(-0.004*score))-1),2)-1
        return score
    else:
        return 0

def getBestScores(scores):
    if (len(scores)>1):
        if np.max(scores.values()) == 1.0: scores = {k: v for k, v in scores.items() if v == 1.0 }
        elif np.max(scores.values()) > 0.0: scores = {k: v for k, v in scores.items() if v > 0.0 }
        else: scores = {k: v for k, v in scores.items() if v >= np.percentile(scores.values(),75) }
    return scores
    
def getWorstScores(scores):
    if (len(scores)>1):
        if np.min(scores.values()) < 0.0: return {k: v for k, v in scores.items() if v < 0.0 }
        else: return {k: v for k, v in scores.items() if v <= np.percentile(scores.values(),25) }
    else:
        return scores
    
def getNormScores(scores):
    if (len(scores)>1):
        temp = {k: (v+1)/2 for k, v in scores.items() }
        return {k: (v+0.01)/(np.max(list(temp.values()))+0.01) for k, v in temp.items() }
    elif (len(scores)==1):
        scores[scores.keys()[0]] = 1
        return scores
    else:
        return scores
    
# Get board scores
def getBoardValue(scores):
    return max(scores.values())

# Get move selection (do not call directly from code!!)
def getBestMove(board, engine, handler, maxTime):
    scores = getScores(board, engine, handler, maxTime)
    return chess.Move.from_uci(scores.keys()[scores.values().index(max(scores.values()))])

def getWorstMove(board, engine, handler, maxTime):
    scores = getScores(board, engine, handler, maxTime)
    return chess.Move.from_uci(scores.keys()[scores.values().index(min(scores.values()))])

def getFastBestMove(board, engine, maxTime):
    bestmove = engine.go(searchmoves=list(board.legal_moves),movetime=maxTime)
    return bestmove[0]

def getRandomMove(board):
    return random.choice(list(board.legal_moves))

# Get board
def newBoard():
    return chess.Board()

def fenBoard(fenCode):
    return chess.Board(fen=fenCode, chess960=False)

def getSwitchBoard(board):
    nextboard = copy.deepcopy(board)
    nextboard.push(chess.Move.null())
    nextboard.clear_stack()
    return nextboard

def getEndGame(game, n):
    for i in range(0, int(game.headers['PlyCount'])-max(2*n-1,0)):
        next_variation = game.variation(0)
        game = next_variation
    return game

def getEndBoard(game, n):
    for i in range(0, int(game.headers['PlyCount'])-max(2*n-1,0)):
        next_variation = game.variation(0)
        game = next_variation
    return game.board()

def getEndMoves(game, n):
    for i in range(0, int(game.headers['PlyCount'])-max(2*n-1,0)):
        next_variation = game.variation(0)
        game = next_variation
    return list(game.main_line())

def getMateDepth(game, engine, maxdepth, maxTime):
    for nmoves in range(maxdepth,0,-1):
        board = getEndBoard(game, nmoves)
        startBoard(board, engine, handler)
        turn = board.turn
        for i in range(0,2*nmoves):
            if not board.is_checkmate():
                board.push(getFastBestMove(board, engine, maxTime))
                engine.position(board)
            elif turn != board.turn:
                return nmoves
            else:
                return float('Inf')
    return float('Inf')

def getNToMate(board, engine, handler, maxTime):
    startBoard(board, engine, handler)
    move = engine.go(searchmoves=[getFastBestMove(board, engine, maxTime)],movetime=maxTime)
    mate = handler.info["score"][1].mate
    if isinstance(mate, int):
        if mate>0: return float(mate)
        else: return float('inf')
    else:
        return float('inf')

# Initialize boards
def startBoard(board, engine, handler):
    engine.ucinewgame()
    engine.position(board)

def nextBoard(board, engine, handler, move, maxTime):
    board.push(move)
    engine.position(board)
    return getScores(board, engine, handler, maxTime)

# Play boards
def playGame(game, nmate):
    game = getEndGame(game, nmate)
    board = game.board()
    for mv in list(game.main_line()):
        board.push(mv)
        display(displayBoard(board))

def playMove(board, engine, handler, move, maxTime):
    startBoard(board, engine, handler)
    start_scores = getScores(board, engine, handler, maxTime)
    next_scores = nextBoard(board, engine, handler, chess.Move.from_uci(move), maxTime)
    display(SVG(getScoreBoard(board, getBestScores(next_scores))))

def playBoard(board, engine, handler, nmoves, maxTime):
    scoreList = {}
    startBoard(board, engine, handler)
    scores = getScores(board, engine, handler, maxTime)
    getTurn(board)
    print(getBestScores(scores))
    display(SVG(getScoreBoard(board, getBestScores(scores))))
    for i in range(0,2*nmoves):
        if scores:
            move = getFastBestMove(board, engine, maxTime)
            scoreList[str(board.uci(move))] = scores[str(board.uci(move))]
            scores = nextBoard(board, engine, handler, move, maxTime)
            getTurn(board)
            print(getBestScores(scores))
            display(SVG(getScoreBoard(board, getBestScores(scores))))
        else:
            getCheckMate(board)
            return
    getCheckMate(board)

def playWorstBoard(board, engine, handler, nmoves, maxTime):
    scoreList = {}
    startBoard(board, engine, handler)
    scores = getScores(board, engine, handler, maxTime)
    getTurn(board)
    display(SVG(getScoreBoard(board, getBestScores(scores))))
    for i in range(0,2*nmoves):
        if scores:
            move = getWorstMove(board, engine, handler, maxTime)
            scoreList[str(board.uci(move))] = scores[str(board.uci(move))]
            scores = nextBoard(board, engine, handler, move)
            getTurn(board)
            display(SVG(getScoreBoard(board, getBestScores(scores))))
        else:
            getCheckMate(board)
            return
    getCheckMate(board)

def playRandomBoard(board, engine, handler, nmoves, maxTime):
    scoreList = {}
    startBoard(board, engine, handler)
    scores = getScores(board, engine, handler, maxTime)
    getTurn(board)
    display(SVG(getScoreBoard(board, getBestScores(scores))))
    for i in range(0,2*nmoves):
        if scores:
            move = getRandomMove(board)
            scoreList[str(board.uci(move))] = scores[str(board.uci(move))]
            scores = nextBoard(board, engine, handler, move, maxTime)
            getTurn(board)
            display(SVG(getScoreBoard(board, getBestScores(scores))))
        else:
            getCheckMate(board)
            return
    getCheckMate(board)
    
# Display pieces
def displayPiece(string):
    return chess.svg.piece(chess.piece_name(chess.Piece.from_symbol(string)))

# Display boards
def getScoreBoard(board, scores):
    arrow = []
    lastmove = None
    check = None
    normscores = getNormScores(scores)
    for key in normscores.keys():
        move = chess.Move.from_uci(key)
        arrow.append([move.from_square, move.to_square, normscores[key]])
    if not arrow:
        arrow = ()
    if board.move_stack:
        lastmove = board.peek()
    if board.is_checkmate():
        check = list(board.pieces(chess.KING, board.turn))[0]
    return resize_svg(chess.svg.newboard(chess.svg, board=board, arrows=arrow, acolup="#808000", acoldown="#808000", lastmove=lastmove, check=check))

def getRectBoard(board, rect):
    lastmove = None
    check = None
    rect = {k: v/np.max(np.absolute(rect.values())) for k, v in rect.items() }
    if board.move_stack:
        lastmove = board.peek()
    if board.is_checkmate():
        check = list(board.pieces(chess.KING, board.turn))[0]
    return resize_svg(chess.svg.newboard(chess.svg, board=board, rects=rect, lastmove=lastmove, check=check))

def getArrowBoard(board, arrow):
    lastmove = None
    check = None
    if board.move_stack:
        lastmove = board.peek()
    if board.is_checkmate():
        check = list(board.pieces(chess.KING, board.turn))[0]
    return resize_svg(chess.svg.newboard(chess.svg, board=board, arrows=arrow, acolup="#008080", acoldown="#AB1323", lastmove=lastmove, check=check))

def displayBoard(board):
    return SVG(getScoreBoard(board, {}))

def displayCustomBoard(board, scores):
    getTurn(board)
    return SVG(getScoreBoard(board, scores))

def displayArrowBoard(board, arrow):
    getTurn(board)
    return SVG(getArrowBoard(board, arrow))

def displayLegalBoard(board):
    scores = getLegalScores(board)
    return SVG(getScoreBoard(board, scores))

def displayBestBoard(board, engine, handler, maxTime):
    getTurn(board)
    startBoard(board, engine, handler)
    scores = getScores(board, engine, handler, maxTime)
    return SVG(getScoreBoard(board, scores))

def displayTopBestBoard(board, engine, handler, maxTime):
    getTurn(board)
    startBoard(board, engine, handler)
    scores = getScores(board, engine, handler, maxTime)
    return SVG(getScoreBoard(board, getBestScores(scores)))
    
def displayRectBoard(board, rect):
    getTurn(board)
    return SVG(getRectBoard(board, rect))

def displaySidebySide(svg1,svg2):
    no_wrap_div = '<div style="white-space: nowrap">{}{}</div>'
    return HTML(no_wrap_div.format(svg1, svg2))

def displayPiece(index, color):
    if color==True:
        return chess.svg.piece(chess.Piece.from_symbol(chess.piece_symbol(index).upper()))
    else:
        return chess.svg.piece(chess.Piece.from_symbol(chess.piece_symbol(index)))
    
# Move graphs
def showScoreBarChart(scores):
    plt.clf()
    normscores = getNormScores(scores)
    plt.bar(range(len(normscores)), normscores.values())
    plt.xticks(range(len(normscores)), normscores.keys(), rotation='vertical')

def showPerturbationMatrix(plt, mat, k):
    f, ax = plt.subplots(figsize=(7, 7))
    ax.matshow(mat, cmap="gray", vmin=-2, vmax=2);
    plt.xticks(range(len(mat[0])), getSquareNames(k), rotation='vertical');
    
def showPerturbationCorrMatrix(plt, mat, k):
    d = pd.DataFrame(data=mat,columns=list(k))
    f, ax = plt.subplots(figsize=(7, 7))
    ax.matshow(d.corr(), cmap="gray", vmin=-2, vmax=2);
    plt.xticks(range(len(mat[0])), getSquareNames(k), rotation='vertical');
    plt.yticks(range(len(mat[0])), getSquareNames(k), rotation='horizontal');
    
def getGraphScores(G):
    return dict.fromkeys(G.nodes(), 1)
    
def plotGraph(G):
    pos=nx.spectral_layout(G)
    x = [row[0] for row in pos.values()]
    y = [row[1] for row in pos.values()]
    nx.draw(G,pos,node_color='#808000')
    for i in range(len(G.nodes())):
        plt.text(x[i]+0.05,y[i]+0.05,s=G.nodes()[i], bbox=dict(facecolor='white', alpha=0.5),horizontalalignment='center')

# Piece movement fields   
def getWhitePionField():
    field = {}
    for i in range(8):
        for j in range(8):
            field.setdefault(i, {})[j] = 0
            field.setdefault(-i, {})[j] = 0
            field.setdefault(i, {})[-j] = 0
            field.setdefault(-i, {})[-j] = 0
            if j==0 and i>=0:
                field.setdefault(i, {})[j] = round(1.0/(i+1),2)
            elif j==1 and i>0:
                field.setdefault(i, {})[j] = round(1.0/(i+1+1),2)
                field.setdefault(i, {})[-j] = round(1.0/(i+1+1),2)
    return field
                
def plotWhitePionField():
    squares = np.zeros((2*8-1, 2*8-1))
    for i in range(8):
        for j in range(8):
            if j==0 and i>=0:
                squares[i+7,j+7] = round(1.0/(i+1),2)
            elif j==1 and i>0:
                squares[i+7,j+7] = round(1.0/(i+1+1),2)
                squares[i+7,-j+7] = round(1.0/(i+1+1),2)
    plt.matshow(np.flip(squares, 0), cmap="gray", vmin=0, vmax=1);
    plt.axis('off');
    
def getBlackPionField():
    field = {}
    for i in range(8):
        for j in range(8):
            field.setdefault(i, {})[j] = 0
            field.setdefault(-i, {})[j] = 0
            field.setdefault(i, {})[-j] = 0
            field.setdefault(-i, {})[-j] = 0
            if j==0 and i>=0:
                field.setdefault(-i, {})[j] = round(1.0/(i+1),2)
            elif j==1 and i>0:
                field.setdefault(-i, {})[j] = round(1.0/(i+1+1),2)
                field.setdefault(-i, {})[-j] = round(1.0/(i+1+1),2)
    return field
                
def plotBlackPionField():
    squares = np.zeros((2*8-1, 2*8-1))
    for i in range(8):
        for j in range(8):
            if j==0 and i>=0:
                squares[-i+7,j+7] = round(1.0/(i+1),2)
            elif j==1 and i>0:
                squares[-i+7,j+7] = round(1.0/(i+1+1),2)
                squares[-i+7,-j+7] = round(1.0/(i+1+1),2)
    plt.matshow(np.flip(squares, 0), cmap="gray", vmin=0, vmax=1);
    plt.axis('off');
    
def getKnightField():
    field = {}
    squares = np.zeros((2*8-1, 2*8-1))
    for i in range(8):
        for j in range(8):
            field.setdefault(i, {})[j] = 0
            field.setdefault(-i, {})[j] = 0
            field.setdefault(i, {})[-j] = 0
            field.setdefault(-i, {})[-j] = 0
    moves = [[0,0]]
    for i in range(10):
        nextmoves = []
        for m in moves:
            if abs(m[0])<8 and abs(m[1])<8:
                if field[m[0]][m[1]]==0:
                    field.setdefault(m[0], {})[m[1]] = round(1.0/(i+1),2)
                    nextmoves.append([m[0]+2,m[1]+1])
                    nextmoves.append([m[0]+2,m[1]-1])
                    nextmoves.append([m[0]+1,m[1]+2])
                    nextmoves.append([m[0]-1,m[1]+2])
                    nextmoves.append([m[0]-2,m[1]+1])
                    nextmoves.append([m[0]-2,m[1]-1])
                    nextmoves.append([m[0]+1,m[1]-2])
                    nextmoves.append([m[0]-1,m[1]-2])
        moves = nextmoves
    return field
                
def plotKnightField():
    squares = np.zeros((2*8-1, 2*8-1))
    moves = [[0,0]]
    for i in range(10):
        nextmoves = []
        for m in moves:
            if m[0]+7<2*8-1 and m[1]+7<2*8-1:
                if squares[m[0]+7,m[1]+7]==0:
                    squares[m[0]+7,m[1]+7] = round(1.0/(i+1),2)
                    nextmoves.append([m[0]+2,m[1]+1])
                    nextmoves.append([m[0]+2,m[1]-1])
                    nextmoves.append([m[0]+1,m[1]+2])
                    nextmoves.append([m[0]-1,m[1]+2])
                    nextmoves.append([m[0]-2,m[1]+1])
                    nextmoves.append([m[0]-2,m[1]-1])
                    nextmoves.append([m[0]+1,m[1]-2])
                    nextmoves.append([m[0]-1,m[1]-2])
        moves = nextmoves
    plt.matshow(np.flip(squares, 0), cmap="gray", vmin=0, vmax=1);
    plt.axis('off');
    
def getBishopField():
    field = {}
    for i in range(8):
        for j in range(8):
            field.setdefault(i, {})[j] = 0
            field.setdefault(-i, {})[j] = 0
            field.setdefault(i, {})[-j] = 0
            field.setdefault(-i, {})[-j] = 0
            if i==j:
                field.setdefault(i, {})[j] = round(1.0/2,2)
                field.setdefault(-i, {})[j] = round(1.0/2,2)
                field.setdefault(i, {})[-j] = round(1.0/2,2)
                field.setdefault(-i, {})[-j] = round(1.0/2,2)
    field.setdefault(0, {})[0] = 1
    return field
                
def plotBishopField():
    squares = np.zeros((2*8-1, 2*8-1))
    for i in range(8):
        for j in range(8):
            if i==j:
                squares[i+7,j+7] = round(1.0/2,2)
                squares[-i+7,j+7] = round(1.0/2,2)
                squares[i+7,-j+7] = round(1.0/2,2)
                squares[-i+7,-j+7] = round(1.0/2,2)
    squares[0+7,0+7] = 1
    plt.matshow(np.flip(squares, 0), cmap="gray", vmin=0, vmax=1);
    plt.axis('off');

def getTowerField():
    field = {}
    for i in range(8):
        for j in range(8):
            field.setdefault(i, {})[j] = 0
            field.setdefault(-i, {})[j] = 0
            field.setdefault(i, {})[-j] = 0
            field.setdefault(-i, {})[-j] = 0
            if i==0:
                field.setdefault(i, {})[j] = round(1.0/2,2)
                field.setdefault(i, {})[-j] = round(1.0/2,2)
            if j==0:
                field.setdefault(i, {})[j] = round(1.0/2,2)
                field.setdefault(-i, {})[j] = round(1.0/2,2)
    field.setdefault(0, {})[0] = 1
    return field
                
def plotTowerField():
    squares = np.zeros((2*8-1, 2*8-1))
    for i in range(8):
        for j in range(8):
            if i==0:
                squares[i+7,j+7] = round(1.0/2,2)
                squares[i+7,-j+7] = round(1.0/2,2)
            if j==0:
                squares[i+7,j+7] = round(1.0/2,2)
                squares[-i+7,j+7] = round(1.0/2,2)
    squares[0+7,0+7] = 1
    plt.matshow(np.flip(squares, 0), cmap="gray", vmin=0, vmax=1);
    plt.axis('off');
    
def getQueenField():
    field = {}
    for i in range(8):
        for j in range(8):
            field.setdefault(i, {})[j] = 0
            field.setdefault(-i, {})[j] = 0
            field.setdefault(i, {})[-j] = 0
            field.setdefault(-i, {})[-j] = 0
            if i==j:
                field.setdefault(i, {})[j] = round(1.0/2,2)
                field.setdefault(-i, {})[j] = round(1.0/2,2)
                field.setdefault(i, {})[-j] = round(1.0/2,2)
                field.setdefault(-i, {})[-j] = round(1.0/2,2)
            if i==0:
                field.setdefault(i, {})[j] = round(1.0/2,2)
                field.setdefault(i, {})[-j] = round(1.0/2,2)
            if j==0:
                field.setdefault(i, {})[j] = round(1.0/2,2)
                field.setdefault(-i, {})[j] = round(1.0/2,2)
    field.setdefault(0, {})[0] = 1
    return field
                
def plotQueenField():
    squares = np.zeros((2*8-1, 2*8-1))
    for i in range(8):
        for j in range(8):
            squares[i+7,j+7] = 0
            squares[-i+7,j+7] = 0
            squares[i+7,-j+7] = 0
            squares[-i+7,-j+7] = 0
            if i==j:
                squares[i+7,j+7] = round(1.0/2,2)
                squares[-i+7,j+7] = round(1.0/2,2)
                squares[i+7,-j+7] = round(1.0/2,2)
                squares[-i+7,-j+7] = round(1.0/2,2)
            if i==0:
                squares[i+7,j+7] = round(1.0/2,2)
                squares[i+7,-j+7] = round(1.0/2,2)
            if j==0:
                squares[i+7,j+7] = round(1.0/2,2)
                squares[-i+7,j+7] = round(1.0/2,2)
    squares[0+7,0+7] = 1
    plt.matshow(np.flip(squares, 0), cmap="gray", vmin=0, vmax=1);
    plt.axis('off');
    
def getKingField():
    field = {}
    for i in range(8):
        for j in range(8):
            field.setdefault(i, {})[j] = 0
            field.setdefault(-i, {})[j] = 0
            field.setdefault(i, {})[-j] = 0
            field.setdefault(-i, {})[-j] = 0
            if i==j:
                field.setdefault(i, {})[j] = round(1.0/(i+1),2)
                field.setdefault(-i, {})[j] = round(1.0/(i+1),2)
                field.setdefault(i, {})[-j] = round(1.0/(i+1),2)
                field.setdefault(-i, {})[-j] = round(1.0/(i+1),2)
            if i==0:
                field.setdefault(i, {})[j] = round(1.0/(j+1),2)
                field.setdefault(i, {})[-j] = round(1.0/(j+1),2)
            if j==0:
                field.setdefault(i, {})[j] = round(1.0/(i+1),2)
                field.setdefault(-i, {})[j] = round(1.0/(i+1),2)
    return field
                
def plotKingField():
    squares = np.zeros((2*8-1, 2*8-1))
    for i in range(8):
        for j in range(8):
            if i==j:
                squares[i+7,j+7] = round(1.0/(i+1),2)
                squares[-i+7,j+7] = round(1.0/(i+1),2)
                squares[i+7,-j+7] = round(1.0/(i+1),2)
                squares[-i+7,-j+7] = round(1.0/(i+1),2)
            if i==0:
                squares[i+7,j+7] = round(1.0/(j+1),2)
                squares[i+7,-j+7] = round(1.0/(j+1),2)
            if j==0:
                squares[i+7,j+7] = round(1.0/(i+1),2)
                squares[-i+7,j+7] = round(1.0/(i+1),2)
    plt.matshow(np.flip(squares, 0), cmap="gray", vmin=0, vmax=1);
    plt.axis('off');

# Board visualization functions
def resize_svg(string):
    string = string.replace('viewBox="0 0 45 45"', 'viewBox="0 0 45 45"  width="45" height="45"')
    string = string.replace('viewBox="0 0 400 400"', 'viewBox="0 0 400 400"  width="400" height="400"')
    return string

def newboard(this, board=None, squares=None, flipped=False, coordinates=True, lastmove=None, check=None, arrows=(), acolup=None, acoldown=None, rects=None, size=None, style=None):
    
    _svg = this._svg
    #_text = this._text
    math = this.math
    ET = this.ET
    SQUARE_SIZE = this.SQUARE_SIZE
    PIECES = this.PIECES
    DEFAULT_COLORS = this.DEFAULT_COLORS
    XX = this.XX
    CHECK_GRADIENT = this.CHECK_GRADIENT
    
    margin = 20 if coordinates else 0
    svg = _svg(8 * SQUARE_SIZE + 2 * margin, size)

    if style:
        ET.SubElement(svg, "style").text = style

    defs = ET.SubElement(svg, "defs")
    if board:
        for color in chess.COLORS:
            for piece_type in chess.PIECE_TYPES:
                if board.pieces_mask(piece_type, color):
                    defs.append(ET.fromstring(PIECES[chess.Piece(piece_type, color).symbol()]))
    if squares:
        defs.append(ET.fromstring(XX))
    if check is not None:
        defs.append(ET.fromstring(CHECK_GRADIENT))

    for square, bb in enumerate(chess.BB_SQUARES):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        x = (file_index if not flipped else 7 - file_index) * SQUARE_SIZE + margin
        y = (7 - rank_index if not flipped else rank_index) * SQUARE_SIZE + margin

        cls = ["square", "light" if chess.BB_LIGHT_SQUARES & bb else "dark"]
        if lastmove and square in [lastmove.from_square, lastmove.to_square]:
            cls.append("lastmove")
        fill_color = DEFAULT_COLORS[" ".join(cls)]
        cls.append(chess.SQUARE_NAMES[square])

        ET.SubElement(svg, "rect", {
            "x": str(x),
            "y": str(y),
            "width": str(SQUARE_SIZE),
            "height": str(SQUARE_SIZE),
            "class": " ".join(cls),
            "stroke": "none",
            "fill": fill_color,
        })

        if square == check:
            ET.SubElement(svg, "rect", {
                "x": str(x),
                "y": str(y),
                "width": str(SQUARE_SIZE),
                "height": str(SQUARE_SIZE),
                "class": "check",
                "fill": "url(#check_gradient)",
            })

#     if coordinates:
#         for file_index, file_name in enumerate(chess.FILE_NAMES):
#             x = (file_index if not flipped else 7 - file_index) * SQUARE_SIZE + margin
#             svg.append(_text(file_name, x, 0, SQUARE_SIZE, margin))
#             svg.append(_text(file_name, x, margin + 8 * SQUARE_SIZE, SQUARE_SIZE, margin))
#         for rank_index, rank_name in enumerate(chess.RANK_NAMES):
#             y = (7 - rank_index if not flipped else rank_index) * SQUARE_SIZE + margin
#             svg.append(_text(rank_name, 0, y, margin, SQUARE_SIZE))
#             svg.append(_text(rank_name, margin + 8 * SQUARE_SIZE, y, margin, SQUARE_SIZE))

    for arrow in arrows:
        head_file = chess.square_file(arrow[1])
        head_rank = chess.square_rank(arrow[1])
        tail_file = chess.square_file(arrow[0])
        tail_rank = chess.square_rank(arrow[0])
        asize = round(abs(arrow[2]))
        acolor = acolup if arrow[2] >= 0 else acoldown

        xhead = margin + (head_file + 0.5 if not flipped else 7.5 - head_file) * SQUARE_SIZE
        yhead = margin + (7.5 - head_rank if not flipped else head_rank + 0.5) * SQUARE_SIZE
        xtail = margin + (tail_file + 0.5 if not flipped else 7.5 - tail_file) * SQUARE_SIZE
        ytail = margin + (7.5 - tail_rank if not flipped else tail_rank + 0.5) * SQUARE_SIZE

        if (head_file, head_rank) == (tail_file, tail_rank):
            ET.SubElement(svg, "circle", {
                "cx": str(xhead),
                "cy": str(yhead),
                "r": str(SQUARE_SIZE * ((asize/2)+0.5) / 2),
                "stroke-width": str(SQUARE_SIZE * 0.1),
                "stroke": acolor,
                "fill": acolor,
                "opacity": "1.0",
            })
        else:
            marker_size = 0.5 * SQUARE_SIZE * (asize + 0.5)
            marker_margin = 0.1 * SQUARE_SIZE

            dx, dy = xhead - xtail, yhead - ytail
            hypot = math.hypot(dx, dy)

            shaft_x = xhead - dx * (marker_size + marker_margin) / hypot
            shaft_y = yhead - dy * (marker_size + marker_margin) / hypot

            xtip = xhead - dx * marker_margin / hypot
            ytip = yhead - dy * marker_margin / hypot

            ET.SubElement(svg, "line", {
                "x1": str(xtail),
                "y1": str(ytail),
                "x2": str(shaft_x),
                "y2": str(shaft_y),
                "stroke": acolor,
                "stroke-width": str(5 * asize * SQUARE_SIZE * 0.1 + 5),
                "opacity": "0.3",
                "stroke-linecap": "butt",
                "class": "arrow",
            })

            marker = []
            marker.append((xtip, ytip))
            marker.append((shaft_x + dy * 0.5 * marker_size / hypot,
                           shaft_y - dx * 0.5 * marker_size / hypot))
            marker.append((shaft_x - dy * 0.5 * marker_size / hypot,
                           shaft_y + dx * 0.5 * marker_size / hypot))

            ET.SubElement(svg, "polygon", {
                "points": " ".join(str(x) + "," + str(y) for x, y in marker),
                "fill": acolor,
                "opacity": "0.3",
                "class": "arrow",
            })
            
    for square, bb in enumerate(chess.BB_SQUARES):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        x = (file_index if not flipped else 7 - file_index) * SQUARE_SIZE + margin
        y = (7 - rank_index if not flipped else rank_index) * SQUARE_SIZE + margin

        cls = ["square", "light" if chess.BB_LIGHT_SQUARES & bb else "dark"]
        if lastmove and square in [lastmove.from_square, lastmove.to_square]:
            cls.append("lastmove")
        fill_color = DEFAULT_COLORS[" ".join(cls)]
        cls.append(chess.SQUARE_NAMES[square])
            
        # Render pieces.
        if board is not None:
            piece = board.piece_at(square)
            if piece:
                ET.SubElement(svg, "use", {
                    "xlink:href": "#%s-%s" % (chess.COLOR_NAMES[piece.color], chess.PIECE_NAMES[piece.piece_type]),
                    "transform": "translate(%d, %d)" % (x, y),
                })

        # Render selected squares.
        if squares is not None and squares & bb:
            ET.SubElement(svg, "use", {
                "xlink:href": "#xx",
                "x": str(x),
                "y": str(y),
            })
            
        if rects is not None:
            if square in rects:
                if rects[square]!=0:
                    offset = SQUARE_SIZE/2-SQUARE_SIZE*abs(rects[square])/2
                    ET.SubElement(svg, "rect", {
                        "x": str(x+offset),
                        "y": str(y+offset),
                        "width": str(SQUARE_SIZE*abs(rects[square])),
                        "height": str(SQUARE_SIZE*abs(rects[square])),
                        "class": " ".join(cls),
                        "stroke": '#AB1323' if rects[square]>0 else '#008080',
                        "stroke-width": str(3),
                        "fill": "none",
                        "opacity": "0.8",
                    }) 

    return ET.tostring(svg).decode("utf-8")

setattr(chess.svg, 'newboard', newboard)
    
board = chess.Board()

chess.svg.DEFAULT_COLORS['square dark'] = '#DDDDDD'
chess.svg.DEFAULT_COLORS['square dark lastmove'] = '#808080'
chess.svg.DEFAULT_COLORS['square light'] = '#FFFFFF'
chess.svg.DEFAULT_COLORS['square light lastmove'] = '#808080'

chess.svg.PIECES['b'] = """<g id="black-bishop" class="black bishop" fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2zm6-4c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2zM25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z" fill="#000" stroke-linecap="butt"/><path d="M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5" stroke="#fff" stroke-linejoin="miter"/></g>"""
chess.svg.PIECES['k'] = """<g id="black-king" class="black king" fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6" stroke-linejoin="miter"/><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#000" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z" fill="#000"/><path d="M20 8h5" stroke-linejoin="miter"/><path d="M32 29.5s8.5-4 6.03-9.65C34.15 14 25 18 22.5 24.5l.01 2.1-.01-2.1C20 18 9.906 14 6.997 19.85c-2.497 5.65 4.853 9 4.853 9M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0" stroke="#fff"/></g>"""
chess.svg.PIECES['n'] = """<g id="black-knight" class="black knight" fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18" style="fill:#000000; stroke:#000000;"/><path d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10" style="fill:#000000; stroke:#000000;"/><path d="M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z" style="fill:#ececec; stroke:#ececec;"/><path d="M 15 15.5 A 0.5 1.5 0 1 1 14,15.5 A 0.5 1.5 0 1 1 15 15.5 z" transform="matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)" style="fill:#ececec; stroke:#ececec;"/><path d="M 24.55,10.4 L 24.1,11.85 L 24.6,12 C 27.75,13 30.25,14.49 32.5,18.75 C 34.75,23.01 35.75,29.06 35.25,39 L 35.2,39.5 L 37.45,39.5 L 37.5,39 C 38,28.94 36.62,22.15 34.25,17.66 C 31.88,13.17 28.46,11.02 25.06,10.5 L 24.55,10.4 z " style="fill:#ececec; stroke:none;"/></g>"""
chess.svg.PIECES['p'] = """<g id="black-pawn" class="black pawn"><path d="M22 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38-1.95 1.12-3.28 3.21-3.28 5.62 0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" stroke="#000" stroke-width="1.5" stroke-linecap="round"/></g>"""
chess.svg.PIECES['q'] = """<g id="black-queen" class="black queen" fill="#000" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><g fill="#000" stroke="none"><circle cx="6" cy="12" r="2.75"/><circle cx="14" cy="9" r="2.75"/><circle cx="22.5" cy="8" r="2.75"/><circle cx="31" cy="9" r="2.75"/><circle cx="39" cy="12" r="2.75"/></g><path d="M9 26c8.5-1.5 21-1.5 27 0l2.5-12.5L31 25l-.3-14.1-5.2 13.6-3-14.5-3 14.5-5.2-13.6L14 25 6.5 13.5 9 26zM9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z" stroke-linecap="butt"/><path d="M11 38.5a35 35 1 0 0 23 0" fill="none" stroke-linecap="butt"/><path d="M11 29a35 35 1 0 1 23 0M12.5 31.5h20M11.5 34.5a35 35 1 0 0 22 0M10.5 37.5a35 35 1 0 0 24 0" fill="none" stroke="#fff"/></g>"""
chess.svg.PIECES['r'] = """<g id="black-rook" class="black rook" fill="#000" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zM12.5 32l1.5-2.5h17l1.5 2.5h-20zM12 36v-4h21v4H12z" stroke-linecap="butt"/><path d="M14 29.5v-13h17v13H14z" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M14 16.5L11 14h23l-3 2.5H14zM11 14V9h4v2h5V9h5v2h5V9h4v5H11z" stroke-linecap="butt"/><path d="M12 35.5h21M13 31.5h19M14 29.5h17M14 16.5h17M11 14h23" fill="none" stroke="#fff" stroke-width="1" stroke-linejoin="miter"/></g>"""
chess.svg.PIECES['B'] = """<g id="white-bishop" class="white bishop" fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><g fill="#fff" stroke-linecap="butt"><path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2zM15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2zM25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"/></g><path d="M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5" stroke-linejoin="miter"/></g>"""
chess.svg.PIECES['K'] = """<g id="white-king" class="white king" fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6M20 8h5" stroke-linejoin="miter"/><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#fff" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z" fill="#fff"/><path d="M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0"/></g>"""
chess.svg.PIECES['N'] = """<g id="white-knight" class="white knight" fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18" style="fill:#ffffff; stroke:#000000;"/><path d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10" style="fill:#ffffff; stroke:#000000;"/><path d="M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z" style="fill:#000000; stroke:#000000;"/><path d="M 15 15.5 A 0.5 1.5 0 1 1 14,15.5 A 0.5 1.5 0 1 1 15 15.5 z" transform="matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)" style="fill:#000000; stroke:#000000;"/></g>"""
chess.svg.PIECES['P'] = """<g id="white-pawn" class="white pawn"><path d="M22 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38-1.95 1.12-3.28 3.21-3.28 5.62 0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#fff" stroke="#000" stroke-width="1.5" stroke-linecap="round"/></g>"""
chess.svg.PIECES['Q'] = """<g id="white-queen" class="white queen" fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM24.5 7.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM41 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM16 8.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM33 9a2 2 0 1 1-4 0 2 2 0 1 1 4 0z"/><path d="M9 26c8.5-1.5 21-1.5 27 0l2-12-7 11V11l-5.5 13.5-3-15-3 15-5.5-14V25L7 14l2 12zM9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z" stroke-linecap="butt"/><path d="M11.5 30c3.5-1 18.5-1 22 0M12 33.5c6-1 15-1 21 0" fill="#fff"/></g>"""
chess.svg.PIECES['R'] = """<g id="white-rook" class="white rook" fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zM12 36v-4h21v4H12zM11 14V9h4v2h5V9h5v2h5V9h4v5" stroke-linecap="butt"/><path d="M34 14l-3 3H14l-3-3"/><path d="M31 17v12.5H14V17" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M31 29.5l1.5 2.5h-20l1.5-2.5"/><path d="M11 14h23" fill="#fff"/></g>"""

# Some useful data structures
# white_pion_dict = getWhitePionField()
# black_pion_dict = getBlackPionField()
# knight_dict = getKnightField()
# bishop_dict = getBishopField()
# tower_dict = getTowerField()
# queen_dict = getQueenField()
# king_dict = getKingField()