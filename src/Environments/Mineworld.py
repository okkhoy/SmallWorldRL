"""
RL Framework
Mineworld Environment
"""

import numpy as np
from Environment import *
import functools 
import pdb

class Mineworld():
    """
    Mineworld Environment
    Expects size of area, mean and covariance to be given
    """

    #TODO Figure out what these vars are
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
    def idx_state( size, st ):
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
    def make_mdp( size, mean, cov ):
        state_idx = functools.partial( Mineworld.state_idx, size )

        goal = Mineworld.get_random_goal( size )

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

        # Add rewards to all states that transit into the goal state
        s = state_idx( *goal )
        for s_ in xrange( S ):
            R[ (s_,s) ] = Mineworld.REWARD_SUCCESS - Mineworld.REWARD_BIAS
        
        start_set = None
        end_set = [ s ]

        return S, A, P, R, R_bias, start_set, end_set

    @staticmethod
    def create( height, width, mean, cov ):
        """Create a place from @spec"""
        return Environment( Mineworld, *Mineworld.make_mdp( (height, width) , mean, cov) )

    @staticmethod
    def reset_rewards( env, height, width, mean, cov ):
        size = (height, width)
        state_idx = functools.partial( Mineworld.state_idx, size )
        goal = Mineworld.get_random_goal( size )

        # Reset the rewards
        R = {}
        # Add rewards to all states that transit into the goal state
        s = state_idx( *goal )
        for s_ in xrange( env.S ):
            R[ (s_,s) ] = Mineworld.REWARD_SUCCESS - Mineworld.REWARD_BIAS
        
        start_set = None
        end_set = [ s ]

        return Environment( Mineworld, env.S, env.A, env.P, R, env.R_bias, start_set, end_set )


