"""
RL Framework
Mineworld Environment
"""

import numpy as np
from Environment import *
import functools 
import re

class Mineworld():
    """
    Mineworld Environment
    Expects size of area, mean and covariance to be given
    """

    MINE        = 1 

    MOVE_UP     = 0
    MOVE_DOWN   = 1
    MOVE_LEFT   = 2
    MOVE_RIGHT  = 3

    ACCURACY = 0.80

    REWARD_BIAS = -1
    REWARD_FAILURE = -20 - REWARD_BIAS
    REWARD_SUCCESS = 50 - REWARD_BIAS
    REWARD_CHECKPOINT = 0 # - REWARD_BIAS

    @staticmethod
    def state_idx( size, y, x ):
        """Compute the index of the state"""

        st, offset = x, size[1]
        st, offset = st + offset * y, offset * size[0]

        return st

    @staticmethod
    def idx_state( size, state ):
        """Compute the state for the index"""
        x, state = state % size[1], state / size[1]
        y, state = state % size[0], state / size[0]

        return y, x

    @staticmethod
    def get_random_goal( grid ):
        size = grid.shape
        loc = np.random.randint( 0, size[0] ), np.random.randint( 0, size[1] ) 
        while grid[loc]==Mineworld.MINE:
            loc = np.random.randint( 0, size[0] ), np.random.randint( 0, size[1] ) 
        return loc

    @staticmethod
    def make_mdp( grid ):
        size = grid.shape
        state_idx = functools.partial( Mineworld.state_idx, size )
        goal = Mineworld.get_random_goal( grid )

        S = size[ 0 ] * size[ 1 ]
        A = 4 # up down left right
        P = [ [ [] for i in xrange( S ) ] for j in xrange( A ) ]
        R = {}
        R_bias = Mineworld.REWARD_BIAS


        # Populate the P table
        ACCURACY = Mineworld.ACCURACY
        RESIDUE = (1.0 - ACCURACY)/3
        for y in xrange( size[ 0 ] ):
            for x in xrange( size[ 1 ] ):
                s = state_idx( y, x )
                if y > 0:
                    up_state = y-1, x
                else:
                    up_state = y, x
                if y + 1 < size[ 0 ]:
                    down_state = y+1, x
                else:
                    down_state = y, x
                if x > 0:
                    left_state = y, x-1
                else:
                    left_state = y, x
                if x + 1 < size[ 1 ]:
                    right_state = y, x+1
                else:
                    right_state = y, x

                P[ Mineworld.MOVE_UP ][ s ] = [
                        ( state_idx( *up_state ), ACCURACY ),
                        ( state_idx( *down_state ), RESIDUE ),
                        ( state_idx( *left_state ), RESIDUE ),
                        ( state_idx( *right_state ), RESIDUE ), ]
                P[ Mineworld.MOVE_DOWN ][ s ] = [
                        ( state_idx( *up_state ), RESIDUE ),
                        ( state_idx( *down_state ), ACCURACY ),
                        ( state_idx( *left_state ), RESIDUE ),
                        ( state_idx( *right_state ), RESIDUE ), ]
                P[ Mineworld.MOVE_LEFT ][ s ] = [
                        ( state_idx( *up_state ), RESIDUE ),
                        ( state_idx( *down_state ), RESIDUE ),
                        ( state_idx( *left_state ), ACCURACY ),
                        ( state_idx( *right_state ), RESIDUE ), ]
                P[ Mineworld.MOVE_RIGHT ][ s ] = [
                        ( state_idx( *up_state ), RESIDUE ),
                        ( state_idx( *down_state ), RESIDUE ),
                        ( state_idx( *left_state ), RESIDUE ),
                        ( state_idx( *right_state ), ACCURACY ), ]

        s = state_idx( *goal )
        start_set = None
        end_set = [ s ]
        # Add rewards to all states that transit into the goal state
        for s_ in xrange( S ):
            R[ (s_,s) ] = Mineworld.REWARD_SUCCESS - Mineworld.REWARD_BIAS
        
        for y in range(size[0]):
            for x in range(size[1]):
                mine = (y,x)
                if grid[mine]==Mineworld.MINE:
                    s = state_idx(*mine)
                    # Add rewards to all states that transit into the mine state
                    for s_ in xrange(S):
                        R[ (s_,s) ] = Mineworld.REWARD_FAILURE - Mineworld.REWARD_BIAS
                        
        return S, A, P, R, R_bias, start_set, end_set

    @staticmethod
    def convert_to_list(string):
        retlist = []
        outer = re.compile('\[(.+)\]')
        replace = re.compile(',(?=[^\]]*(?:\[|$))')

        tempstring = outer.search(string)
        if tempstring is not None:
            tempstring = tempstring.group(1)
            newlist = replace.sub(r'|', tempstring).split('|')
            for string in newlist:
                retlist.append(Mineworld.convert_to_list(string))
        else:
            retlist = float(string)

        return retlist

    @staticmethod
    def create( height, width, mean = (0,0), cov = [[1,0],[0,1]], num_mines=15 ):
        """Create a place from @spec"""
        mean = Mineworld.convert_to_list(mean)
        cov = Mineworld.convert_to_list(cov)
        grid = np.zeros((height, width))
        for i in range(num_mines):
            x, y = width, height
            while x>=width or x<0 or y>=height or y<0 or grid[y,x]==Mineworld.MINE:
                y,x = map(np.floor, np.random.multivariate_normal(mean,cov))
                y += np.floor(height/2)
                x += np.floor(width/2)
            grid[y,x] = Mineworld.MINE
        return Environment( Mineworld, *Mineworld.make_mdp( grid ) )

    @staticmethod
    def reset_rewards( env, height, width, mean = (0,0), cov = [[1,0],[0,1]], num_mines=15 ):
        mean = Mineworld.convert_to_list(mean)
        cov = Mineworld.convert_to_list(cov)
        grid = np.zeros((height, width))
        for i in range(num_mines):
            x, y = width, height
            while x>=width or x<0 or y>=height or y<0 or grid[y,x]==Mineworld.MINE:
                y,x = map(np.floor, np.random.multivariate_normal(mean,cov))
                y += np.floor(height/2)
                x += np.floor(width/2)
            grid[y,x] = Mineworld.MINE

        state_idx = functools.partial( Mineworld.state_idx, (height, width) )
        goal = Mineworld.get_random_goal( grid )
        # Reset the rewards
        R = {}
        s = state_idx( *goal )
        start_set = None
        end_set = [ s ]
        # Add rewards to all states that transit into the goal state
        for s_ in xrange( env.S ):
            R[ (s_,s) ] = Mineworld.REWARD_SUCCESS - Mineworld.REWARD_BIAS

        for y in range(height):
            for x in range(width):
                mine = (y,x)
                if grid[mine]==Mineworld.MINE:
                    s = state_idx(*mine)
                    # Add rewards to all states that transit into the mine state
                    for s_ in xrange(env.S):
                        R[ (s_,s) ] = Mineworld.REWARD_FAILURE - Mineworld.REWARD_BIAS

        return Environment( Mineworld, env.S, env.A, env.P, R, env.R_bias, start_set, end_set )
