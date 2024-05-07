# two classes useful in solving project euler #96
# https://projecteuler.net/problem=96

import itertools
import numpy as np
import random

class Board():
    ''' defines a Sudoku board (basically an 81-len np.array of digits, with 0 representing empty)
    '''

    DIGITS = {1,2,3,4,5,6,7,8,9}

    def __init__(self,board_ndarray):
        self.board = board_ndarray.flatten()
        assert(len(self.board)==81)

    def __repr__(self):
        slist = repr(self.board.reshape(9,9)).replace('0',' ').split('\n')
        s = ''
        for n, line in enumerate(slist):
            s+= line[:15] + ' | ' + line[17:24] + ' | ' + line[26:] + '\n'
            if n in {2,5,8}:
                s+='\n'
        return s

    def unassigned_cells(self):
        ''' returns set of unassigned cells (delivered as r,c tuples) '''
        return set((divmod(i,9) for i,n in enumerate(self.board) if n==0))

    def row(self,i):
        ''' returns a row as an array '''
        return self.board.reshape(9,9)[i]

    def column(self,j):
        ''' returns column as an array '''
        return self.board.reshape(9,9)[:,j]

    def block(self,i,j=None,flatten = True):
        ''' returns block (0-8) or the block associated with (r,c) as array or 2d array'''
        if j is None:
            i,j = divmod(i,3)
        else:
            i,j = i//3, j//3
        sq = self.board.reshape(9,9)[i*3:i*3+3, j*3:j*3+3]
        return sq.flatten() if flatten else sq

    def count_filled(self):
        return np.count_nonzero(self.board)

    def calc_domain_at(self, i,j=None):
        ''' calculate domain at i,j based on its row, column and block neighbors
        '''
        if j is None:
            (i,j)=i
        if (d:=self.board[i*9+j]):
            return {d}
        blockset = set(self.block(i,j)) - {0}
        rowset = set(self.row(i)) - {0}
        colset = set(self.column(j)) - {0}
        neighborset = blockset | colset | rowset  #union of all already assigned numbers
        return self.DIGITS - neighborset

    def cell(self,r,c):
        return self.board.reshape(9,9)[r,c]

    def unsolvable(self):
        ''' return True if board contains empty assignment domains
        '''
        valid_domains = all(map(len,
                       np.array([{n} if n else self.calc_domain_at(*divmod(i,9)) for i,n in enumerate(self.board)])
                       ))
        return not valid_domains

    def violates_constraint(self):
        ''' check if any duplicates in row, col or block - count nonzeroes vs set of items
        '''
        for i in range(9):
            ra = self.row(i)
            ca = self.column(i)
            ba = self.block(i)
            if (
                np.count_nonzero(ra) - len(set(ra)-{0})
                ) or (
                np.count_nonzero(ca) - len(set(ca)-{0})
                ) or (
                np.count_nonzero(ca) - len(set(ca)-{0})):
                return True
        # all good
        return False

    def assign(self,cell,digit):
        ''' assigns cell (r,c) tuple to item
        '''
        i,j = cell
        #assert(self.board[i*9+j]==0)
        self.board[i*9+j]=digit

    def copy(self):
        '''
        returns a new board copy
        '''
        return Board(np.copy(self.board))

    def neighbors_of(self, i, j):
        ''' returns set of neighbors (connected to cell via row, column or block)'''
        n = set(itertools.chain(
            ((r,j) for r in range(9)),
            ((i,c) for c in range(9)),
            (((i//3)*3+r, (j//3)*3+c) for r in range(3) for c in range(3))
            )
        )
        return n-{(i,j)}

    def sort_cells_by_rc(self,cells):
        ''' takes set of cells or iterable collection and sorts by row, column '''
        return sorted(cells,key=lambda x:x[0]*9+x[1])

    def sort_cells_by_domain(self,cells):
        ''' takes set of cells or iterable collection and sorts by domain size '''
        return sorted(cells,key=lambda x:len(self.calc_domain_at(x[0],x[1])))

    def unassigned_domains(self,sort_domain=True):
        ''' returns list of tuples of unassigned cells with their corresponding domains'''
        cells = self.unassigned_cells()
        if sort_domain:
            return [((r,c),self.calc_domain_at(r,c)) for r,c in self.sort_cells_by_domain(cells)]
        return [((r,c),self.calc_domain_at(r,c)) for r,c in self.sort_cells_by_rc(cells)]

    def unassigned_neighbors(self,r,c,sort_domain=True):
        ''' returns list of tuples of unassigned neighbors with their corresponding domains'''
        cells = self.unassigned_cells() & self.neighbors_of(r,c)
        if sort_domain:
            return [((r,c),self.calc_domain_at(r,c)) for r,c in self.sort_cells_by_domain(cells)]
        return [((r,c),self.calc_domain_at(r,c)) for r,c in self.sort_cells_by_rc(cells)]


class Solver():
    ''' defines an ai solver for a sudoku board.  All methods here are static...
        no real reason these are encapsulated in the class other than to keep it organized
    '''

    @staticmethod
    def best_unassigned(board,randomize=False):
        ''' returns most restricted (cell, {domain}).  We use a list to capture all cells with smallest domains in case
            can randomize our selections; otherwise we provide first item in the list
        '''
        sorted_domains = board.unassigned_domains(sort_domain=True)
        first = sorted_domains[0]
        if not randomize:
            return first
        smallest_domains = list(filter(lambda x:len(x[1])==len(first[1]),sorted_domains))
        return random.choice(smallest_domains)

    @staticmethod
    def df_search(board, rand=False):
        ''' stack implementation of backtracking depth first search for board solutions
            stack contains a 3-tuple (cell, assignment, board state prior to assignment)
            think of cell and assignment as edge between board states
        '''

        history = [(board.copy(),board.count_filled())]

        if board.violates_constraint() or board.unsolvable():
            # unsolved state; bail out with history
            return history

        # create empty stack of edges; access via append and pop
        edges_to_explore = []

        while True:
            # return board if completed
            if board.count_filled()==81:
                return history

            # find new edges to explore:
            # get the cell with the smallest domains and put those edges on the exploration stack
            cell, domain = Solver.best_unassigned(board, randomize=rand)
            for digit in domain:
                # we store edges as transitions from one state to another
                edge = (cell, digit, board.copy())
                edges_to_explore.append(edge)

            # explore our graph pulling candidates from our stack
            try_another_edge = True
            while try_another_edge:
                if not edges_to_explore:
                    # nothing to explore; couldnt find solution
                    return history

                cell, value, board = edges_to_explore.pop()
                # update board state based on the transition
                board.assign(cell,value)
                history.append((board, board.count_filled()))
                # if our assignment results in unsolvable board, we try another edge
                try_another_edge = board.unsolvable()


    @staticmethod
    def solve(board,history=False,randomize=False):
        ''' convenience method to call df_search.  Uses history flag to get final solution
            or return history of intermediate steps
        '''

        solution = Solver.df_search(board,randomize)
        if history:
            return solution
        else:
            return solution[-1][0] #just the board
