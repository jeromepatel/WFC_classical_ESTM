# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 22:37:02 2021

Wave function collapse algorithm classical implementation

@author: jyotm
"""

# this code produces random generated time and shows it as a block 
import random
import pygame as pg
import numpy as np
import colorama

input_matrix = np.array([
    ['L','L','L','L'],
    ['L','L','L','L'],
    ['L','L','L','L'],
    ['L','C','C','L'],
    ['C','S','S','C'],
    ['S','S','S','S'],
    ['S','S','S','S'],
])


#define directions for moving 

UP = (-1,0)
DOWN = (1,0)
LEFT = (0,-1)
RIGHT = (0,1)

DIR = [UP,DOWN,LEFT,RIGHT]


def render_colors(matrix):
    """Render the fully collapsed `matrix` using the given `colors.
    Arguments:
    matrix -- 2-D matrix of tiles
    """
    colors = {
    'L': colorama.Fore.GREEN,
    'S': colorama.Fore.BLUE,
    'C': colorama.Fore.WHITE,
    'A': colorama.Fore.CYAN,
    'B': colorama.Fore.MAGENTA,
    }
    for row in matrix:
        output_row = []
        for val in row:
            color = colors[val]
            output_row.append(color + val + colorama.Style.RESET_ALL)

        print("".join(output_row))

#defines valid direction to see all the possible direction from a coordinate
def valid_dirs(coordinates,size):
    x,y = coordinates
    Height,Width = size
    valid_values = []
    if y > 0: 
        valid_values.append(LEFT)
    if y < Width-1:
        valid_values.append(RIGHT)
    if x > 0:
        valid_values.append(UP)
    if x < Height-1:
        valid_values.append(DOWN)
    return valid_values


#start with pre processing the input matrix 
#extract compatibility matrix info
def find_compatibility(matrix):
    '''
    first function so wrote all this 
    Parameters
    ----------
    matrix : (x,y dim metrix )
     input pattern matrix to extract details.

    Returns
    -------
    compatility_list : list of (coordinates, direction, result) pair
        gives complete list of available directions which are compatible.
        
    weights: set of all possible values of a state 
    '''
    #necessary to define rules from the input
    
    Width = matrix.shape[1]
    Height = matrix.shape[0]
    compatibility_list = []
    for x in range(Height):
        row = []
        for y in range(Width):
            coord = (x,y)
            for val in valid_dirs(coord,matrix.shape):
                #print(f'{val} at {x} , {y}') 
                x2,y2 = ( coord[0] + val[0], coord[1] + val[1] )
                compatibility_list.append((matrix[x][y],val,matrix[x2][y2]))
                
    #get weights or all possible values for a tile
    weights = {}
    for x in range(Height):
        for y in range(Width):
            val = input_matrix[x][y]
            if val not in weights.keys():
                weights[val] = 0
            weights[val] += 1
    return (weights,compatibility_list)


#print(find_compatibility(input_matrix))

def create_output_superposition_matrix(states, output_size):
    '''
    initialize the output matrix as superposition of all states in it with size output size

    Parameters
    ----------
    states : array of all possible states
    output_size : tuple 
        numpy array.shape .

    Returns
    -------
    numpy array 
        output matrix initialized.

    '''
    Height, Width = output_size
    output_matrix = []
    for x in range(Height):
        row = []
        for y in range(Width):
            row.append(states.copy())
        output_matrix.append(row)
    return np.array(output_matrix)
   
def check_compatibility_tiles(tile, direction, neighbour, comp_list):
    '''
    check the constraint by finding the input in list 

    Parameters
    ----------
    tile : char, input state(eg. 's') 

    direction : tuple from DIR
        given direction from tile state
    neighbour : char
        neighbour tile in direction dir from tile state.
    comp_list : array 
        consists of already defined constraints.

    Returns
    -------
    true false based on availability of tuple

    '''
    #checks if the tile neighbour pair is valid and is in compatibility list
    return (tile, direction, neighbour) in comp_list

def find_min_entropy(weights,matrix):
    '''
    returns min entropy state coordinates accord. to formula of shanon entropy 
    Parameters
    ----------
    weights : simple array of weights of states in overall matrix
    matrix : output matrix constisting of superposition states

    Returns
    -------
    min_ent_state : tuple (x, y) coordinates 
    calculated from formula ENT = log(sigma(w)) - log(w)/sigma(w) (sigma over all possible states in coordinate)

    '''
    Height,Width = matrix.shape
    min_ent_state = None
    lowest_entropy = np.inf
    

    for x in range(Height):
        for y in range(Width):
            states = matrix[x][y]
            if len(states) == 1:
                continue
            
            shannon_entropy_for_square = 0
            sum_weights = 0
            log_sum_weights = 0
            for state in states:
                weight = weights[state]
                sum_weights += weight
                log_sum_weights += weight * np.log(weight)
            shannon_entropy_for_square = np.log(sum_weights) - log_sum_weights / sum_weights
             
            if min_ent_state is None or lowest_entropy > shannon_entropy_for_square:
                min_ent_state = (x, y)
                lowest_entropy = shannon_entropy_for_square
    return min_ent_state

def collapse_state(weights, min_ent_state, matrix):
    '''
    collapses the state at given coordinates 
    Parameters
    ----------
    min_ent_state : given coord where collapse happens accord. to shanon entropy 
     (u know other parameters) 
    Returns
    -------
    selected_state : state after the collapse at given coord.

    '''
    #get proportions in which we wanna collapse the state accord. to weights
    x,y = min_ent_state
    assert(len(matrix[x][y]) != 1)
    state_probs = []
    #print(state_probs)
    for state_val in matrix[x][y]:
        state_probs.append(weights[state_val])
    probabilities =  np.array(state_probs) /np.sum(state_probs)
    #print(probabilities, matrix[x][y])
    selected_state = np.random.choice(list(matrix[x][y]), p = probabilities)
    
    #actually collapse the state at that particular coordinates
    matrix[x][y] = set(selected_state).copy()
    return selected_state

def propogate_collapse(coords, matrix, compatibility_list):
    '''
    propogates the collapse of a coordinate till it reaches the end or empties the queue
    we propogates the collapse and removes the tile from neighbours if its not compatible with any of the 
    combination of coord states, then we propogate these changes to their neighbours 

    Parameters
    ----------
    coords : coordinate to propogate from (tuple)
    u know others things

    '''
    #check for valid directions 
    #get all neighbours 
    #delete neighbours which contradicts with the new collapsed state
    queue = [coords]
    
    while len(queue) > 0:
        
        current = queue.pop(0)
        #check for all possible states
        x,y = (current[0],current[1])
        current_possible_states = matrix[x][y]
        #iterate all neighbours
        #get all valid directions
        for dirs in valid_dirs(current,matrix.shape):
            x2,y2 = ( current[0] + dirs[0], current[1] + dirs[1] )
            neighbour = (x2, y2)
            possible_tiles_neighbours = matrix[x2][y2].copy()
            for tile in possible_tiles_neighbours:
                tile_is_valid = any([check_compatibility_tiles(current, dirs, tile,compatibility_list) for current in current_possible_states])
                if not tile_is_valid:
                    #remove the tile from neighbour's state
                    queue.append(neighbour)
                    matrix[x2][y2].remove(tile)
        
        
    

    pass
def check_collapsed(matrix):
    '''
    simple checking if all states are collapsed in output array 
    '''
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if len(matrix[x][y]) != 1:
                return False
    return True


def collapse_wavefunction_superposition(weights,output_matrix,compatibility_list):
    '''
    iteratively collapse wave function of output matrix until its fully collapsed 
    Parameters
    ----------
    weights : dictionary of weights of all possible states input state
    output_matrix : np array of superposition of states of size output size
    compatibility_list : list of tuples defining the constrains accord. to input 

    '''
    #it consists of multiple steps
    collapsed = False
    
    while not collapsed:
        #starting with find min entropy function perform multiple iterations
        min_entropy_state = find_min_entropy(weights, output_matrix)
        
        #collapse that state 
        selected_state = collapse_state(weights,min_entropy_state,output_matrix)  
    
        #popogate the changes throughout the output matrix
        propogate_collapse(min_entropy_state, output_matrix,compatibility_list)
        
        
        #check whether the matrix is fully collapsed 
        collapsed = check_collapsed(output_matrix)

def get_final_output(matrix):
    #convert sets into final elements as only last 1 state is remaining in output matrix
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            matrix[x][y] = matrix[x][y].pop()
            
    
def waveFunctionCollapseAlgorithm(input_matrix, Height, Width):
    '''
    order of wave function collapse algorithm, simple calling each functions 
    Parameters
    ----------
    input_matrix : no need to define (simple input matrix )
    dimentions of output matrix
    Returns
    -------
    output_matrix : collapsed output matrix 

    '''
    weights, comp_lst = find_compatibility(input_matrix)
    #create output matrix from superposition of input weights states
    output_matrix = create_output_superposition_matrix(set(weights.keys()),(Height,Width))
    #run exp, get example min entropy
    
    #output_matrix[1][0].remove('L')
    
    collapse_wavefunction_superposition(weights,output_matrix,comp_lst)
    get_final_output(output_matrix)
    #print(output_matrix)
    return output_matrix
    
    


def Display_matrix(matrix):
    '''
    simple function adapted from reddit to print the output in tiled manner usign pygame
    Parameters
    ----------
    matrix : matrix to print (input or output any )
    '''
    pg.init()
    blocksize = 30
    
    Width = matrix.shape[1]
    Height = matrix.shape[0]
    
    
    window = pg.display.set_mode((blocksize*Width, blocksize*Height))
    screen = pg.display.get_surface()
    
    # colors = {
    #     'green': (50, 180, 100),
    #     'blue':  (50, 50, 150),
    #     'red':   (150, 0, 0)
    # }
    
    colorsRefList = {
        'L': (50, 200, 20),
        'S': (20, 40, 180),
        'C': (200,200,90),
        'A': (173,255,47),
        'B': (255,255,255),
    }
    colList = []
    for i in range(Width):
        row = []
        for j in range(Height):
            val = matrix[j][i]
            row.append(colorsRefList[val])
        colList.append(row)
    
    def blockolize():
        for x in range(Width):
            for y in range(Height):
                screen.fill(colList[x][y],
                            pg.Rect( (x*blocksize, y*blocksize),
                                     (blocksize, blocksize) ))
    
    blockolize()
    pg.display.update()
    
    playing = True
    while playing:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                playing = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                blockolize()
                pg.display.update()
                
      
                
    
#use wave function collapse algorithm for getting output 
output = waveFunctionCollapseAlgorithm(input_matrix, 5, 10)


render_colors(output)
Display_matrix(output)    
#Display_matrix(input_matrix)