########################################################################################################################
# Imports
# Stuff for boot from cmd
import os
import sys
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, r'C:\Users\Robotika\Desktop\Robotika')
import numpy as np
from Lib import colorsys
from controller import Robot, Motor, DistanceSensor, InertialUnit, GPS, Emitter, Receiver, Camera, Lidar
import math
import copy
from typing import Tuple, Literal, List, Dict, Set
import struct
# noinspection PyUnresolvedReferences
import cv2
# noinspection PyUnresolvedReferences
from Victim import Victim

from datetime import datetime
left_camera_t = False
right_camera_t = False
# Stuff for nice prints
# noinspection PyUnresolvedReferences
np.set_printoptions(threshold=sys.maxsize)

starting_time = datetime.now()
#print (starting_time)

########################################################################################################################
# Constants

# Frames/Updates in a second
timeStep = 32

# Max speed allowed by the engine
_MAX_VELOCITY = 6.28

# Prepare the message for lack of progress
_LoP = struct.pack('c', 'L'.encode())

# If a sensor detects an obstacle within this distance, it will act as there is a wall there
_DANGER_ZONE_CLOSENESS = 0.04

# The layer of the lidar to take(top to bottom starting at 0)
_LIDAR_LAYER = 2

# Lidar point detection min-max and angles
_LIDAR_LIMIT = {
    'R1': (0.05, 0.065),
    'R1.5': (0.09, 0.0975),
    'R2': (0.1125, 0.125),
    'R2.5': (0.15, 0.1575),
    'R3': (0.1725, 0.185),
    'K1': (0.05, 0.08),
    'K2': (0.1175, 0.145),
    'K3': (0.16, 0.1775),
    'C1.5': (0.1, 0.125),
    'C2': (0.125, 0.175),
    'C2.5': (0.18, 0.21),
    'C3': (0.215, 0.2375),
    'B1': (0.0725, 0.09),
    'B2': (0.155, 0.1725),
    'B3': (0.235, 0.265),
    'CK2': (0.13, 0.135),
    'CK2.5': (0.1425, 0.17),
    'CK3': (0.17, 0.19),
    'CB': (0.2025, 0.215),
    'CR': (0.1875, 0.21)
}
_LIDAR_ANGLES = {
    # https://www.desmos.com/calculator/flzivwznih
    'B': 45,
    'R': 90,
    'K1': 29,
    'K2': 19,
    'K3': 15,
    'C1.5': 31,
    'C2': 38,
    'C2.5': 38,
    'C3': 41,
    'CK2': 26,
    'CK2.5': 20,
    'CK3': 18,
    'CB': 33,
    'CR': 27
}

# Area 1-2-3 possible angles of movement
_MVMT_ANGLES = {0, 45, 90, 135, 180, 225, 270, 315}



victim_found = False

class Dot:
    def __init__(self, value: int, color_range: Dict[Literal['min', 'max'], Tuple[int, float, float]]):
        self.val = value
        self.color_range = color_range
        
    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self.val)

    def __eq__(self, other):
        if isinstance(other, Dot) or issubclass(type(other), Dot):
            return self.val == other.val
        elif type(other) == int:
            return self.val == other
        else:
            print('Equality with these types not supported:', type(self), type(other))
            return False


class Tunnel(Dot):
    def __init__(self, value: int, color_range: Dict[Literal['min', 'max'], Tuple[int, float, float]]):
        super().__init__(value, color_range)
        self.coord = None
        self.found = False

_UNDEFINED = -1
_EMPTY = Dot(value=0, color_range={'min': (0, 0, 0.85), 'max': (0, 0, 1.00)})
_WALL = 1
_HOLE = Dot(value=2, color_range={'min': (0, 0, 0.15), 'max': (0, 0, 0.20)})
_SWAMP = Dot(value=3, color_range={'min': (37, 0.40, 0.75), 'max': (47, 0.55, 0.95)})
_CHECKPOINT = Dot(value=4, color_range={'min': (115, 0.00, 0.55), 'max': (245, 0.03, 0.95)})
_START = 5
_BLUE = Tunnel(value=6, color_range={'min': (237, 0.65, 0.95), 'max': (243, 0.80, 1.05)}) 
_PURPLE = Tunnel(value=7, color_range={'min': (267, 0.60, 0.85), 'max': (277, 0.75, 1.00)})
_RED = Tunnel(value=8, color_range={'min': (0, 0.65, 0.95), 'max': (3, 0.80, 1.00)})
_GREEN = Tunnel(value=9, color_range={'min': (117, 0.80, 0.95), 'max': (123, 0.90, 1.00)})
_YELLOW = Tunnel(value=13, color_range={'min': (58, 0.70, 0.996), 'max': (60, 0.71, 1.0)})
_ORANGE = Tunnel(value=14, color_range={'min': (55, 0.686, 0.99), 'max': (57, 0.72, 1.00)})
_BLUE2 = Tunnel(value=15, color_range={'min': (237, 0.65, 0.95), 'max': (243, 0.80, 1.05)})
_PURPLE2 = Tunnel(value=16, color_range={'min': (267, 0.60, 0.85), 'max': (277, 0.75, 1.00)})
_RED2 = Tunnel(value=17, color_range={'min': (0, 0.65, 0.95), 'max': (3, 0.80, 1.00)})
_GREEN2 = Tunnel(value=18, color_range={'min': (117, 0.80, 0.95), 'max': (123, 0.90, 1.00)})
_YELLOW2 = Tunnel(value=19, color_range={'min': (58, 0.70, 0.996), 'max': (60, 0.71, 1.0)})
_ORANGE2 = Tunnel(value=20, color_range={'min': (55, 0.686, 0.99), 'max': (57, 0.72, 1.00)}) 
_STRONG_EMPTY = 10
_TEMP_WALL = 11
_VOID = 12
_ROOM_4 = 21
_VICTIMS = {
    'H': 30,
    'S': 31,
    'U': 32,
    'C': 33,
    'F': 34,
    'O': 35,
    'P': 36,
    30: 'H',
    31: 'S',
    32: 'U',
    33: 'C',
    34: 'F',
    35: 'O',
    36: 'P'
}

_TUNNELS = dict()


########################################################################################################################
# Main Classes


class Cell:
    # noinspection PyShadowingNames
    def __init__(self, explored=False, c_t=_EMPTY.val, rotation=-1, center=_UNDEFINED,
                 cc=_UNDEFINED, cf=_UNDEFINED, cr=_UNDEFINED, cb=_UNDEFINED, cl=_UNDEFINED,
                 dsf=_UNDEFINED, dsr=_UNDEFINED, dsb=_UNDEFINED, dsl=_UNDEFINED,
                 ds0=_UNDEFINED, ds1=_UNDEFINED, ds2=_UNDEFINED, ds3=_UNDEFINED,
                 ds4=_UNDEFINED, ds5=_UNDEFINED, ds6=_UNDEFINED, ds7=_UNDEFINED,
                 fr=_UNDEFINED, br=_UNDEFINED, fl=_UNDEFINED, bl=_UNDEFINED):
        """
        The new Cell's attributes are its actual cell, which is a 5x5 matrix with the composition shown below,
        and two booleans that determine proprieties of that particular cell

        @param explored: Boolean to tell if the cell is explored or not
        @param c_t: Cell type used to determine if it's a: swamp, hole, checkpoint, the start or a tunnel to a new area
        @param cc: The center of the cell
        @param cf: The front dot referenced to the center of the cell
        @param cr: The right dot referenced to the center of the cell
        @param cb: The back dot referenced to the center of the cell
        @param cl: The left dot referenced to the center of the cell
        @param dsf: The front distance sensor
        @param dsr: The right distance sensor
        @param dsb: The back distance sensor
        @param dsl: The left distance sensor
        @param fr: The front-right corner
        @param br: The back-left corner
        @param fl: The front-left corner
        @param bl: The back-left corner
        @param ds0: The distance sensor n°0
        @param ds1: The distance sensor n°1
        @param ds2: The distance sensor n°2
        @param ds3: The distance sensor n°3
        @param ds4: The distance sensor n°4
        @param ds5: The distance sensor n°5
        @param ds6: The distance sensor n°6
        @param ds7: The distance sensor n°7
        """
        if explored and center == _UNDEFINED:
            center = _STRONG_EMPTY

        if _EMPTY.val == cf == cr == cb == cl:
            cc = _EMPTY.val
        else:
            cf = center if cf == _UNDEFINED else cf
            cr = center if cr == _UNDEFINED else cr
            cb = center if cb == _UNDEFINED else cb
            cl = center if cl == _UNDEFINED else cl
            cc = center if cc == _UNDEFINED else cc

        # Spaced weirdly to allow more readability
        # @formatter:off
        self.cell = [
            [fl, ds0, dsf, ds1, fr],
            [ds7, c_t, cf, c_t, ds2],
            [dsl, cl, cc, cr, dsr],
            [ds6, c_t, cb, c_t, ds3],
            [bl, ds5, dsb, ds4, br]
        ]
        
        # @formatter:on

        self.explored = explored
        self.room = None

        # Rotates the matrix, so it points always to the same directions
        rotation = getRotation() // 2 if rotation == -1 else rotation
        for _ in range(rotation):
            self.cell = [list(row)[::-1] for row in zip(*self.cell)]

    def __str__(self):
        """
        @return: Returns the cell in an understandable format
        """
        s = ''
        for row in self.cell:
            s += self.formatDot(str(row)) + '\n'
        return s

    @property
    def available(self) -> bool:
        """
        Checks if the center of the cell is a wall or if at least one of the cell types is a hole
        @return: True if the cell is determined to be available, otherwise False
        """

        for c in {(2, 2), (2, 1), (2, 3), (1, 2), (3, 2)}:
            if self.cell[c[0]][c[1]] == _WALL or self.cell[c[0]][c[1]] == _TEMP_WALL:
                return False

        return not self.isNearTunnel and not self.nearHole

    @property
    def nearHole(self) -> bool:
        """
        Checks if at least one of the cell types is a hole
        @return: True if there's a hole, otherwise False
        """
        for row, col in [(1, 1), (3, 1), (1, 3), (3, 3)]:
            if self.cell[row][col] == _HOLE.val or self.cell[row][col] == _VOID:
                return True
        return False

    @property
    def check34(self) -> bool:
        for row in [(0), (1), (2), (3), (4)]:
            if self.cell[row] == 34:
                print('there is a 34 in cell')
                return True
        return False
    @property
    def canBeAvailable(self) -> bool:
        """
        Checks if the center of the cell is a wall or if at least one of the cell types is a hole
        @return: True if the cell is determined to be available, otherwise False
        """
        if self.cell[2][2] == 1:
            return False

        return not self.nearHole

    @property
    def isNearTunnel(self) -> bool:
        """
        Checks if at least one of the cell types is a tunnel
        @return: True if there's a tunnel, otherwise False
        """
        for row, col in [(1, 1), (3, 1), (1, 3), (3, 3)]:
            if (self.cell[row][col] == _BLUE.val and _BLUE.found and _BLUE.coord) or \
                    (self.cell[row][col] == _BLUE2.val and _BLUE2.found and _BLUE2.coord) or \
                    (self.cell[row][col] == _PURPLE.val and _PURPLE.found and _PURPLE.coord) or \
                    (self.cell[row][col] == _PURPLE2.val and _PURPLE2.found and _PURPLE2.coord) or \
                    (self.cell[row][col] == _RED.val and _RED.found and _RED.coord) or \
                    (self.cell[row][col] == _RED2.val and _RED2.found and _RED2.coord) or \
                    (self.cell[row][col] == _GREEN.val and _GREEN.found and _GREEN.coord) or \
                    (self.cell[row][col] == _GREEN2.val and _GREEN2.found and _GREEN2.coord) or \
                    (self.cell[row][col] == _YELLOW.val and _YELLOW.found and _YELLOW.coord) or \
                    (self.cell[row][col] == _YELLOW2.val and _YELLOW2.found and _YELLOW2.coord) or \
                    (self.cell[row][col] == _ORANGE.val and _ORANGE.found and _ORANGE.coord) or \
                    (self.cell[row][col] == _ORANGE2.val and _ORANGE2.found and _ORANGE2.coord):
                return True

        return False

    def stop_at_purple(self):
        for row, col in [(1, 1), (3, 1), (1, 3), (3, 3)]:
           if (self.cell[row][col] == _PURPLE.val and _PURPLE.found and _PURPLE.coord):
               wheel_left.setVelocity(0)
               wheel_right.setVelocity(0)
    
    def addVictimsToMap(self, camera: Literal['left', 'right']):
        a = maze.map[getCoords()]
        print(a)
        if victim_found == True:
            current_coords = getCoords()  # Get the current coordinates of the robot
            
            if camera == 'left':

                    if compas() == "north":
                        if self.cell[2][0] == _WALL  and _VICTIMS['C'] == prediction:
                            for self.cell[2][0] in current_coords:  # Check if the left wall is a wall
                                self.cell[2][0] = int(_VICTIMS[prediction])  # Replace left wall with victim
                        
                    elif compas() == "east":
                        if self.cell[0][2]  == _WALL and _VICTIMS['C'] == prediction:  # Check if the front wall is self.cell wall
                            for self.cell[0][2] in current_coords:
                                self.cell[0][2]= int(_VICTIMS[prediction])  # Replace front wall with victim
                        
                        
                    elif compas() == "south":
                        if self.cell[2][4] == _WALL  and _VICTIMS['C'] == prediction:
                            for self.cell[2][4] in current_coords:  # Check if the right wall is self.cell wall
                                self.cell[2][4] = int(_VICTIMS[prediction])  # Replace right wall with victim
                        
                    elif compas() == "west":
                        if self.cell[4][2] == _WALL  and _VICTIMS['C'] == prediction:
                            for self.cell[4][2] in current_coords:  # Check if the back wall is self.cell wall
                                self.cell[4][2] = int(_VICTIMS[prediction])  # Replace back wall with victim   
                        
                        
                
            elif camera == 'right':
                
                    if compas() == "north":
                        if self.cell[2][4] == _WALL and _VICTIMS['C'] == prediction:
                            for self.cell[2][4] in current_coords:  # Check if the right wall is self.cell wall
                                self.cell[2][4] = int(_VICTIMS[prediction])  # Replace right wall with victim
                        
                    elif compas() == "east":
                        if self.cell[4][2] == _WALL and _VICTIMS['C'] == prediction:
                            for self.cell[4][2] in current_coords:  # Check if the front wall is self.cell wall
                                self.cell[4][2] = int(_VICTIMS[prediction])  # Replace front wall with victim
                        
                    elif compas() == "south":
                        if self.cell[2][0] == _WALL and _VICTIMS['C'] == prediction:
                            for self.cell[2][0] in current_coords:  # Check if the left wall is self.cell wall
                                self.cell[2][0] = int(_VICTIMS[prediction])  # Replace left wall with victim
                        
                    elif compas() == "west":
                        if self.cell[0][2] == _WALL  and _VICTIMS['C'] == prediction:
                            for self.cell[0][2] in current_coords:  # Check if the back wall is self.cell wall
                                self.cell[0][2] = int(_VICTIMS[prediction])  # Replace back wall with victim
                        
                
        else:
            print('Error, try something to fix it')
        
    
                            
    @classmethod
    def formatDot(cls, dot) -> str:
        """
        Converts the dot or series of dots in a more visually understandable string

        @param dot: Either an int or a string
        @return: The formatted dot
        """
        placeholder = copy.copy(dot)
        if type(placeholder) == int:
            placeholder = str(placeholder)

        if type(placeholder) == str:
            # General formatting
            placeholder = placeholder.replace(',', '').replace('[', '').replace(']', '') \
                .replace(' [', '').replace('] ', '').replace('  ', ' ')

            # Used for visualize pathing
            placeholder = placeholder.replace('99', 'R').replace('91', 'ø').replace('90', 'o')

                
            # Replace victims
            placeholder = placeholder.replace(str(_VICTIMS['P']), 'P').replace(str(_VICTIMS['O']), 'o') \
                .replace(str(_VICTIMS['F']), 'F').replace(str(_VICTIMS['C']), 'C').replace(str(_VICTIMS['U']), 'U') \
                .replace(str(_VICTIMS['S']), 'S').replace(str(_VICTIMS['H']), 'H')

            #print(f"Current placeholder\n{placeholder}")
            # Format the actual dots
            placeholder = placeholder.replace(str(_ROOM_4), '*').replace(str(_YELLOW), 'Y').replace(str(_ORANGE), 'O') \
                .replace(str(_BLUE2), 'B').replace(str(_PURPLE2), 'P').replace(str(_RED2), 'R') \
                .replace(str(_GREEN2), 'G').replace(str(_YELLOW2), 'Y').replace(str(_ORANGE2), 'O') \
                .replace(str(_STRONG_EMPTY), '░').replace(str(_TEMP_WALL), '▓') \
                .replace(str(_VOID), 'U').replace(str(_EMPTY), ' ').replace(str(_UNDEFINED), '?') \
                .replace(str(_WALL), '█').replace(str(_HOLE), 'u').replace(str(_SWAMP), '≈') \
                .replace(str(_CHECKPOINT), '©').replace(str(_START), 'S') \
                .replace(str(_BLUE), 'B').replace(str(_PURPLE), 'P').replace(str(_RED), 'R') \
                .replace(str(_GREEN), 'G')

            #print(f"Placeholder after changing tile types\n{placeholder}")
            return placeholder
        else:
            print('formatDot() requires an int or a str')
            return ''

    def merge(self, other: 'Cell') -> 'Cell':
        """
        Merges two cell by combining their matrices by taking for each dot the greater one

        @param other: The other cell to be merged
        @return: The new, merged cell
        """
        if type(other) == type(self):
            for row in range(5):
                for col in range(5):
                    if not (self.cell[row][col] <= 0 and other.cell[row][col] >= 30 or
                            other.cell[row][col] <= 0 and self.cell[row][col] >= 30):
                        self.cell[row][col] = max(self.cell[row][col], other.cell[row][col])
        else:
            print(f'Tried to merge a Cell and {type(other)}')
        return self

    def unmerge(self, other: 'Cell') -> 'Cell':
        """
        Merges two cell by combining their matrices by taking for each dot the greater one

        @param other: The other cell to be merged
        @return: The new, merged cell
        """
        if type(other) == type(self):
            for row in range(5):
                for col in range(5):
                    if self.cell[row][col] > other.cell[row][col]:
                        self.cell[row][col] = other.cell[row][col]
        else:
            print(f'Tried to unmerge a Cell and {type(other)}')
        return self


class Net:
    def __init__(self, starting_coord: Tuple[int, int], starting_cell: Cell):
        """
        It creates a new net/map which has the {starting_coord: starting_cell} as its start.

        @param starting_coord: The starting coordinate
        @param starting_cell: The starting cell
        """
        self.current_room = 1
        self.map: Dict[Tuple[int, int]: Cell] = dict()
        self.start: Tuple[int, int] = starting_coord
        self.to_explore: Set[Tuple[int, int]] = {self.start}
        self.__add__({starting_coord: starting_cell})
        #print("---------------\n", self.start)

    def __add__(self, other):
        """
        Adds every specified cell to the map and its 24 neighbours
        For example, this is needed when the robot wants to check if a cell is available

        @param other: A dictionary of the cells to add and their coordinates
        @return: The final Net
        """

        if type(other) is dict:
            for coord, cell in other.items():
                if coord in list(self.map.keys()):
                    if cell.explored:
                        self.map[coord] = cell.merge(self.map[coord])
                    else:
                        self.map[coord].merge(cell)
                else:
                    self.map[coord] = cell

                # A wall in the main cell will influence ALL 24 neighbouring nodes(nodes: 'n', wall + node: 'N')
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
                #       ?  ?  n  ?  n  ?  n  ?  n  ?  n  ?  ?
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
                #       ?  ?  n  ?  N     n     n  ?  n  ?  ?
                #       ?  ?  ?  ?  █              ?  ?  ?  ?
                #       ?  ?  n  ?  N           n  ?  n  ?  ?
                #       ?  ?  ?  ?  █              ?  ?  ?  ?
                #       ?  ?  n  ?  N     n     n  ?  n  ?  ?
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
                #       ?  ?  n  ?  n  ?  n  ?  n  ?  n  ?  ?
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?

                neighbourhood = []

                for row in range(13):
                    neighbourhood.append([])
                    for col in range(13):
                        neighbourhood[row].append(-1)

                for row in range(5):
                    for col in range(5):
                        neighbourhood[row + 4][col + 4] = self.map[coord].cell[row][col]
                neighbourhood = np.array(neighbourhood)

                np.rot90(neighbourhood)
                np.rot90(neighbourhood)
                np.flip(neighbourhood)

                # I created it 30 minutes ago and I already find it difficult to understand, good luck to the reader,
                # but basically it's the part that adds all the neighbours to the map
                for neighbour_x in range(-2, 3):
                    for neighbour_z in range(-2, 3):
                        neighbour_coord = (neighbour_x + coord[0], neighbour_z + coord[1])

                        # Create not existing cells
                        if neighbour_coord not in list(self.map.keys()):
                            self.map[neighbour_coord] = Cell()

                        first_el = [neighbour_x * 2 + 4, neighbour_z * 2 + 4]
                        last_el = [neighbour_x * 2 + 8, neighbour_z * 2 + 8]
                        cell_section = copy.copy(neighbourhood[first_el[1]:last_el[1] + 1, first_el[0]:last_el[0] + 1])

                        placeholder = Cell()
                        placeholder.cell = cell_section

                        self.map[neighbour_coord].merge(placeholder)

        return self

    def __iadd__(self, other) -> 'Net':
        return self.__add__(other)

    def __sub__(self, other):
        """
        Removes every specified cell from the map and its references in the 24 neighbours
        This is needed when you're teleported back to a checkpoint and you need to remove the previous position's cell

        @param other: A set of coordinates of the cells to remove
        @return: The final Net
        """  
        if type(other) is set:

            add_to_explore = copy.copy(set())
            try:
                other.remove(None)
            except (Exception,):
                pass

            for coord in other:
                self.map[coord] = Cell()

                # A wall in the main cell will influence ALL 24 neighbouring nodes(nodes: 'n', wall + node: 'N')
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
                #       ?  ?  n  ?  n  ?  n  ?  n  ?  n  ?  ?
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
                #       ?  ?  n  ?  N     n     n  ?  n  ?  ?
                #       ?  ?  ?  ?  █              ?  ?  ?  ?
                #       ?  ?  n  ?  N           n  ?  n  ?  ?
                #       ?  ?  ?  ?  █              ?  ?  ?  ?
                #       ?  ?  n  ?  N     n     n  ?  n  ?  ?
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
                #       ?  ?  n  ?  n  ?  n  ?  n  ?  n  ?  ?
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
                #       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?

                neighbourhood = []

                for row in range(13):
                    neighbourhood.append([])
                    for col in range(13):
                        neighbourhood[row].append(100)

                for row in range(5):
                    for col in range(5):
                        if not (row % 2 == 1 and col % 2 == 1):
                            neighbourhood[row + 4][col + 4] = -1

                neighbourhood = np.array(neighbourhood)

                # I created it 30 minutes ago and I already find it difficult to understand, good luck to the reader,
                # but basically it's the part that adds all the neighbours to the map
                for neighbour_x in range(-2, 3):
                    for neighbour_z in range(-2, 3):
                        neighbour_coord = (neighbour_x + coord[0], neighbour_z + coord[1])

                        # Create not existing cells
                        if neighbour_coord not in list(self.map.keys()):
                            self.map[neighbour_coord] = Cell()

                        first_el = [neighbour_x * 2 + 4, neighbour_z * 2 + 4]
                        last_el = [neighbour_x * 2 + 8, neighbour_z * 2 + 8]
                        cell_section = copy.copy(neighbourhood[first_el[1]:last_el[1] + 1, first_el[0]:last_el[0] + 1])

                        placeholder = Cell()
                        placeholder.cell = cell_section

                        self.map[neighbour_coord].unmerge(placeholder)

                for neighbour_x in range(-1, 2):
                    for neighbour_z in range(-1, 2):
                        neighbour_coord = (neighbour_x + coord[0], neighbour_z + coord[1])
                        add_to_explore.add(neighbour_coord)
                        self.map[neighbour_coord].explored = False

            self.updateToExplore(last_pos=None, curr_pos=None, next_to_explore=list(add_to_explore.union(other)))
            self.checkToExplore()

        return self

    def __isub__(self, other) -> 'Net':
        return self.__sub__(other)

    def __str__(self):
        return maze.getMap(nice_view=True)

    def updateToExplore(self, last_pos: Tuple[int, int], curr_pos: Tuple[int, int],
                        next_to_explore: List[Tuple[int, int]]):
        """
        Updates the current set of all the cells that need to be explored

        @param last_pos: The last position of the robot
        @param curr_pos: The current position of the robot
        @param next_to_explore: A list of all the cells that have the potential to be still unexplored
        """
        try:
            next_to_explore.remove(None)
        except (Exception,):
            pass

        next_to_explore = {next_pos for next_pos in next_to_explore if not self.map[next_pos].explored}

        self.to_explore = self.to_explore.union(next_to_explore)

        if last_pos in self.to_explore:
            self.to_explore.remove(last_pos)
        if curr_pos in self.to_explore:
            self.to_explore.remove(curr_pos)

    def removeToExploreArea(self, center_coord: Tuple[int, int], radius: int = 1):
        """
        Sets all the cell ina a square area(radius*2 +1) around the specified center to unexplored/ not to be explored

        @param center_coord: The center coordinate of the square
        @param radius: The radius of the area
        """
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                try:
                    self.to_explore.remove((x + center_coord[0], y + center_coord[1]))
                except (Exception,):
                    pass

    def removeToExplore(self, *to_remove: Tuple[int, int]):
        """
        Sets all the specified cells to unexplored/ not to be explored

        @param to_remove: List/Coordinates to remove
        """
        for coord in to_remove:
            try:
                self.to_explore.remove(coord)
            except (Exception,):
                pass

    def checkToExplore(self):
        """
        Generally speaking, it checks if the cells that will be explored CAN actually be explored or if it's useful.
        The steps are:
        1. Checks if all the cells are actually available
        2. Removes the starting cell unless it's the only one that needs to be explored
        3. If a cell that needs to be explored has at least 3 explored cells around it, it doesn't need to be explored
        """
        to_remove = []
        for cell in self.to_explore:
            if not maze.map[cell].available:
                to_remove.append(cell)

        for cell in to_remove:
            self.to_explore.remove(cell)

        if len(self.to_explore) != 1 and self.start in self.to_explore:
            self.to_explore.remove(self.start)

        cells_are_removed = True
        while cells_are_removed:
            cells_are_removed = False

            for cell in self.to_explore:
                count_explored_cells = 0
                for angle in {0, 90, 180, 270}:
                    neighbour_coord = getNearCoords(angle=angle + getRotation() * 45, from_coord=cell)

                    if maze.map[neighbour_coord].explored and maze.map[neighbour_coord].available:
                        count_explored_cells += 1

                if count_explored_cells >= 3:
                    to_remove.append(cell)

            for cell in to_remove:
                try:
                    self.to_explore.remove(cell)
                    self.map[cell].explored = True
                    cells_are_removed = True
                except (Exception,):
                    pass

    def getCorners(self):
        """
        Get the two opposite corners of the map

        @return: A tuple consisting of the bottom corner and the top corner like this: (CornerBot(x,y), CornerTop(x,y))
        """
        corner_bot = [None, None]
        corner_top = [None, None]

        for coord, cell in self.map.items():
            if corner_bot[0] is None:
                corner_bot[0] = coord[0]
            elif corner_bot[0] > coord[0]:
                corner_bot[0] = coord[0]

            if corner_top[0] is None:
                corner_top[0] = coord[0]
            elif corner_top[0] < coord[0]:
                corner_top[0] = coord[0]

            if corner_bot[1] is None:
                corner_bot[1] = coord[1]
            elif corner_bot[1] > coord[1]:
                corner_bot[1] = coord[1]

            if corner_top[1] is None:
                corner_top[1] = coord[1]
            elif corner_top[1] < coord[1]:
                corner_top[1] = coord[1]

        corner_bot = tuple(corner_bot)
        corner_top = tuple(corner_top)

        return corner_bot, corner_top

    def getMap(self, nice_view=True):
        room_4_matrix = [[_ROOM_4, _ROOM_4, _ROOM_4, _ROOM_4, _ROOM_4],
                        [_ROOM_4, _ROOM_4, _ROOM_4, _ROOM_4, _ROOM_4],
                        [_ROOM_4, _ROOM_4, _ROOM_4, _ROOM_4, _ROOM_4],
                        [_ROOM_4, _ROOM_4, _ROOM_4, _ROOM_4, _ROOM_4],
                        [_ROOM_4, _ROOM_4, _ROOM_4, _ROOM_4, _ROOM_4]]
        """
        Get the actual map by ordering all the cells in the map in a nice 3D matrix, then unpack it and make it 2D
        and finally compact horizontally and vertically.
        If there are any undefined dots, they will be converted into _EMPTY dots. Note that this last part of the code
        is probably commented off during testing

        @param nice_view: If set to True, it will return a string,
            and if printed is way more understandable than the whole matrix
        @return: Either a string printable to the console or a 2D matrix with all the dots
        """

        corner_bot, corner_top = self.getCorners()

        # Get width and length of the maze
        # --- The +1 is to get every cell in the rectangle, otherwise it would have skipped the last row and column
        width = abs((corner_top[0]) - (corner_bot[0])) + 1
        height = abs((corner_top[1]) - (corner_bot[1])) + 1

        offset_w = -corner_bot[0]
        offset_h = -corner_bot[1]

        # Prepare the matrix to reorder the cells
        segmented_map = []

        for row in range(width):
            segmented_map.append([])
            for col in range(height):
                if (row - offset_w, col - offset_h) in list(self.map.keys()):
                    #print((row - offset_w, col - offset_h), self.map[(row - offset_w, col - offset_h)].room)
                    if self.map[(row - offset_w, col - offset_h)].room == 4:
                        #print(room_4_matrix)
                        segmented_map[row].append(room_4_matrix)
                    else:
                        segmented_map[row].append(self.map[(row - offset_w, col - offset_h)].cell)
                else:
                    segmented_map[row].append(Cell().cell)
        for i in range(10):

            print(segmented_map)
        complete_map = []

        for col in range(height):
            for line in range(5):
                complete_map.append([])
                for row in range(width):
                    complete_map[col * 5 + line].extend(segmented_map[row][col][line])

        # This merges all the cell to get a final grid
        # Starts by merging vertically
        #print(f"Complete map before merging:\n{complete_map}")
        if height > 1:
            complete_map = shiftMatrix(repeat=2, direction='up', old_list=complete_map)

            for row in range(height - 1):
                for line in range(3):
                    for dot in range(len(complete_map[line])):
                        complete_map[line][dot] = \
                            max(complete_map[line][dot],
                                complete_map[line + 3][dot])
                complete_map = shiftMatrix(repeat=2, direction='up', old_list=complete_map)
                complete_map.pop(1)
                complete_map.pop(1)
                complete_map.pop(1)

            complete_map = shiftMatrix(repeat=3, direction='up', old_list=complete_map)

        # And then horizontally
        if width > 1:
            complete_map = shiftMatrix(repeat=2, direction='left', old_list=complete_map)

            for col in range(width - 1):
                for line in range(len(complete_map)):
                    for dot in range(3):
                        complete_map[line][dot] = max(complete_map[line][dot], complete_map[line][dot + 3])
                    complete_map[line].pop(3)
                    complete_map[line].pop(3)
                    complete_map[line].pop(3)
                complete_map = shiftMatrix(repeat=2, direction='left', old_list=complete_map)

            complete_map = shiftMatrix(repeat=3, direction='left', old_list=complete_map)

        margin_distance = 2
        
        complete_map = np.array(complete_map)[margin_distance:-margin_distance, margin_distance:-margin_distance]
        #print(f"Complete map:\n{complete_map}")
        
        if nice_view:
            nice_map = ''

            for row in complete_map:
                nice_map += Cell.formatDot(str(row).replace('\n', '').replace('[ ', '[')) + '\n'

            return nice_map

        # --- Correct some stuff
        # Handle strong empties and temp walls
        for row in range(len(complete_map)):
            for col in range(len(complete_map[row])):
                if complete_map[row, col] == _TEMP_WALL or complete_map[row, col] == _STRONG_EMPTY:
                    complete_map[row][col] = _EMPTY.val

        # Every undefined tile is set to empty
        for row in range(1, len(complete_map), 2):
            for col in range(1, len(complete_map[row]), 2):
                if complete_map[row, col] == _UNDEFINED:
                    complete_map[row, col] = _EMPTY.val

        # Reduce the map to a decent size
        corner_top = [float('inf'), float('inf')]
        corner_bot = [0, 0]
        for row_i, row in enumerate(complete_map):
            for col_i, col in enumerate(row):
                if col > 0:
                    corner_top = [min(corner_top[0], row_i), min(corner_top[1], col_i)]
                    corner_bot = [max(corner_bot[0], row_i), max(corner_bot[1], col_i)]
        complete_map = np.array(complete_map)[corner_top[0]:corner_bot[0] + 1, corner_top[1]:corner_bot[1] + 1]

        if True:
            engine_map = np.empty(shape=complete_map.shape, dtype=str)
            # Finally if an _UNDEFINED type dot is present, it will convert it into an _EMPTY.val str and, also, it will
            # convert to str every dot
            #print("-----------------")
            for row in range(len(complete_map)):
                for col in range(len(complete_map[row])):
                    if complete_map[row][col] == _UNDEFINED:
                        engine_map[row][col] = str(_EMPTY.val)
                    elif complete_map[row][col] == _ROOM_4:
                        engine_map[row][col] = '*'
                    elif complete_map[row][col] == _STRONG_EMPTY:
                        engine_map[row][col] = str(_EMPTY.val)
                   
                    elif complete_map[row][col] == _TEMP_WALL:
                        engine_map[row][col] = str(_WALL)
                    elif complete_map[row][col] == _VOID:
                        engine_map[row][col] = str(_EMPTY.val)
                    elif complete_map[row][col] in [_BLUE.val, _BLUE2.val] :
                        engine_map[row][col] = 'b'
                    elif complete_map[row][col] in [_GREEN.val, _GREEN2.val]:
                        engine_map[row][col] = 'g'
                    elif complete_map[row][col] in [_RED.val, _RED2.val]:
                        engine_map[row][col] = 'r'
                    elif complete_map[row][col] in [_PURPLE.val, _PURPLE2.val]:
                        engine_map[row][col] = 'p'
                    elif complete_map[row][col] in [_YELLOW.val, _YELLOW2.val]:
                        engine_map[row][col] = 'y'
                    elif complete_map[row][col] in [_ORANGE.val, _ORANGE2.val]:
                        engine_map[row][col] = 'o'
                     
                    else:
                        engine_map[row][col] = str(complete_map[row][col])

                print(engine_map[row])
            return engine_map
    """
        elif complete_map[row][col] == _VICTIMS:
        engine_map[row][col] = str(_VICTIMS)
    """

    def getBestPath(self, origin: Tuple[int, int] = None, view_map=False):
        """
        The first part is used to avoid searching the path to all the possible destinations by looking for the ones that
        are next to the origin, the order it searches for is right, forward and left. If this first scanning fails it
        doesn't matter and will proceed as intended, since it's just used to cut some time.

        The second part creates a custom map that is easier to navigate for the A* algorithm

        The third part applies a slightly modified A* algorithm(based on the one that can be seen in the video below)
        to every possible destination and, when all the distances and paths are found, it takes the shortest path as the
        return value. Note that during the general search if it finds a path of length 2 it will take it, since is the
        shortest distance possible after 1.

        A* algorithm video: https://youtu.be/A60q6dcoCjw

        @param origin: Starting point, if None, it will set it to the current coordinates
        @param view_map: Use to view a representation of the map used to find the path
        @return:
        """

        if origin is None:
            origin = getCoords()

        maze.checkToExplore()
        possible_destinations = maze.to_explore

        ################################################################################################################

        corner_bot, corner_top = self.getCorners()

        # Get width and length of the maze
        # --- The +1 is to get every cell in the rectangle, otherwise it would have skipped the last row and column
        width = abs((corner_top[0]) - (corner_bot[0])) + 1
        height = abs((corner_top[1]) - (corner_bot[1])) + 1

        offset_w = -corner_bot[0]
        offset_h = -corner_bot[1]

        # Prepare the custom matrix
        segmented_map = []

        for row in range(width):
            segmented_map.append([])
            for col in range(height):
                if (row - offset_w, col - offset_h) in list(self.map.keys()):
                    segmented_map[row].append(
                        _EMPTY.val if self.map[(row - offset_w, col - offset_h)].available else  _WALL)
                else:
                    segmented_map[row].append(_WALL)
                    
        segmented_map = np.array(segmented_map)
        segmented_map = np.rot90(segmented_map)
        segmented_map = np.flip(m=segmented_map, axis=0)

        segmented_map = np.pad(segmented_map, (1, 1), 'constant', constant_values=1)
        offset_w += 1
        offset_h += 1

        if view_map:
            segmented_map[origin[1] + offset_h, origin[0] + offset_w] = 99

            for destination in possible_destinations:
                if segmented_map[destination[1] + offset_h, destination[0] + offset_w] == 0:
                    segmented_map[destination[1] + offset_h, destination[0] + offset_w] = 90

            nice_map = ''
            for row in segmented_map:
                nice_map += Cell.formatDot(str(row).replace('\n', '').replace('[ ', '[')) + '\n'
            print(nice_map)

        ################################################################################################################

        # noinspection PyShadowingNames
        def pathAStar(curr_map, offset: Tuple[int, int], origin: Tuple[int, int], destination: Tuple[int, int]):
            # noinspection PyShadowingNames
            def potential(node: Tuple[int, int]):
                return math.sqrt((origin[0] - node[0]) ** 2 + (origin[1] - node[1]) ** 2)

            def distance(current: Tuple[int, int], following: Tuple[int, int], rotation: int):
                euclidian_distance = math.sqrt((current[0] - following[0]) ** 2 + (current[1] - following[1]) ** 2)

                weight = 0

                # Calculate the angle between the current node and the following
                angle = (math.atan2(current[0] - following[0], current[1] - following[1])) % (2 * math.pi)
                if angle > math.pi:
                    angle = 2 * math.pi - angle
                else:
                    angle = -angle
                # Convert the angle to degrees
                angle = (angle * 180 / math.pi)
                # Round the angle to 8th of a circle
                angle = (round(angle / 45) * 45) % 360
                # Get the actual angle considering the cell forward as 0°
                angle = angle - rotation

                if angle == 0:
                    weight += -.2

                elif angle == 45:
                    weight += 0.199

                elif angle == 315:
                    weight += .2

                elif angle == 90:
                    weight += .149

                elif angle == 270:
                    weight += .15

                elif angle == 135:
                    weight += .249

                elif angle == 225:
                    weight += .25

                else:
                    weight += .3

                return euclidian_distance + weight

            origin = origin[0] + offset[0], origin[1] + offset[1]
            destination = destination[0] + offset[0], destination[1] + offset[1]

            boundary_nodes = {origin}
            distances = {
                origin: {
                    'distance': 0,
                    'previous': None,
                    'angle': getRotation() * 45
                }
            }

            while len(boundary_nodes) > 0:
                lowest_f_cost = float('inf')
                curr_node = None
                for node in boundary_nodes:
                    if distances[node]['distance'] + potential(node) < lowest_f_cost:
                        lowest_f_cost = distances[node]['distance'] + potential(node)
                        curr_node = node

                boundary_nodes.remove(curr_node)

                ########################################################################################################
                if curr_node == destination:
                    returned_path = copy.copy(set())
                    while distances[curr_node]['previous'] is not None:
                        returned_path.add((curr_node[0] - offset[0], curr_node[1] - offset[1]))
                        curr_node = distances[curr_node]['previous']

                    if distances[destination]['distance'] == 1:
                        destination_potential = 0

                    # Check if the path leads to a hallway, which is preferred by the algorithm
                    elif getRotation() % 2 == 0 and \
                            ((curr_map[destination[0] - 1, destination[1]] == _WALL and
                              curr_map[destination[0] + 1, destination[1]] == _WALL)
                             or
                             (curr_map[destination[0], destination[1] - 1] == _WALL and
                              curr_map[destination[0], destination[1] + 1] == _WALL)):
                        destination_potential = -0.5
                    else:
                        destination_potential = 0

                    return returned_path, distances[destination]['distance'] + destination_potential
                ########################################################################################################

                for shift in {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)}:
                    neighbour = (curr_node[0] + shift[0], curr_node[1] + shift[1])

                    if curr_map[neighbour[0], neighbour[1]] != _WALL:
                        proposed_distance = distances[curr_node]['distance'] + \
                                            distance(current=curr_node, following=neighbour,
                                                     rotation=distances[curr_node]['angle'])
                        if neighbour not in distances.keys() or distances[neighbour]['distance'] > proposed_distance:
                            distances[neighbour] = {
                                'distance': proposed_distance,
                                'previous': curr_node,
                                'angle': getRotation() * 45
                            }
                            boundary_nodes.add(neighbour)

            print(origin, '->', destination, 'or ', end='')

            origin = origin[0] - offset[0], origin[1] - offset[1]
            destination = destination[0] - offset[0], destination[1] - offset[1]

            print(origin, '->', destination)

            raise ValueError(destination)

        segmented_map = np.swapaxes(segmented_map, 0, 1)
        paths = dict()
        to_remove = list()

        for destination in possible_destinations:
            try:
                paths[destination] = pathAStar(curr_map=segmented_map, offset=(offset_w, offset_h), origin=origin,
                                               destination=destination)
            except Exception as exc:
                print(exc, 'didn\'t have a path when executing pathAStar()')
                to_remove.append(exc.args[0])

        if len(to_remove) == len(self.to_explore) > 0:
            emitter.send(_LoP)
            motor_left.setVelocity(0)
            motor_right.setVelocity(0)
            robot.step(timeStep)

            for unexplorable_cell in to_remove:
                try:
                    self.to_explore.remove(unexplorable_cell)
                except (Exception,):
                    pass

            print('Call LoP, remove all cells to explore and go back to the start now')
        else:
            for unexplorable_cell in to_remove:
                try:
                    self.to_explore.remove(unexplorable_cell)
                except Exception as exc:
                    print(exc, 'was not in the set of cells to explore, maybe due to an actual error in pathAStar()')

        best_path = None
        shortest_found = float('inf')
        for destination, path_dist in paths.items():
            if path_dist[1] < shortest_found:
                best_path = path_dist[0]
                shortest_found = path_dist[1]

        return best_path

    def changeCurrentRoom(self, tunnel_colour):
        change_from_1 = {"BLUE" : 2, "YELLOW" : 3, "GREEN" : 4}
        change_from_2 = {"BLUE" : 1, "PURPLE" : 3, "ORANGE" : 4}
        change_from_3 = {"YELLOW" : 1, "PURPLE" : 2, "RED" : 4}
        change_from_4 = {"GREEN" : 1, "ORANGE" : 2, "RED" : 3}
        if self.current_room == 1:
            if tunnel_colour in change_from_1:
                self.current_room = change_from_1[tunnel_colour]
            else:
                print("ERROR: Colour cant appear in room 1.")

        elif self.current_room == 2:
            if tunnel_colour in change_from_2:
                self.current_room = change_from_2[tunnel_colour]
            else:
                print("ERROR: Colour cant appear in room 2.")

        elif self.current_room == 3:
            if tunnel_colour in change_from_3:
                self.current_room = change_from_3[tunnel_colour]
            else:
                print("ERROR: Colour cant appear in room 3.")

        elif self.current_room == 4:
            if tunnel_colour in change_from_4:
                self.current_room = change_from_4[tunnel_colour]
            else:
                print("ERROR: Colour cant appear in room 4.")

        else:
            print("ERROR: Room with invalid number.")
            

########################################################################################################################
# Methods initialization

def getAngle() -> float:
    """
    Gets the current angle in radiants and converts it to normal degrees(0 is north and goes clockwise)

    @return: The direction you're looking at in degrees
    """
    angle = (inertial_unit.getRollPitchYaw()[2] * 180 / math.pi + 360) % 360
    if angle > 180:
        angle = 360 - angle
    else:
        angle = -angle
    return angle % 360


def getLidarValue(angle: int) -> float:
    global curr_cloud

    if curr_cloud is None:
        return float('inf')

    index_central_ray = round((angle % 360) * 1.422) - 1  # Multiply by the ratio 512 / 360
    # Get the 3 rays around that angle
    rays = [
        curr_cloud[(index_central_ray - 1) % 512],
        curr_cloud[index_central_ray % 512],
        curr_cloud[(index_central_ray + 1) % 512]
    ]
    average = sum(rays) / 3
    standard_deviation = math.sqrt(((rays[0] - average) ** 2 + (rays[1] - average) ** 2 + (rays[2] - average) ** 2) / 3)

    if standard_deviation >= 0.02:
        avg_left = (rays[0] + rays[1]) / 2
        sd_left = math.sqrt(((rays[0] - avg_left) ** 2 + (rays[1] - avg_left) ** 2) / 2)

        avg_right = (rays[1] + rays[2]) / 2
        sd_right = math.sqrt(((rays[1] - avg_right) ** 2 + (rays[2] - avg_right) ** 2) / 2)

        if sd_left >= 0.02 and sd_right >= 0.02:
            return min(rays)
        elif sd_left < 0.02 and sd_right < 0.02:
            return average
        else:
            if sd_left < 0.02:
                return min(avg_left, rays[2])

            if sd_right < 0.02:
                return min(rays[0], avg_right)

    return average


def getMinLidarValue(angle, fov_indexs):
    true_lidar_values = lidar.range_image[512 * _LIDAR_LAYER:512 * (_LIDAR_LAYER + 1):]
    index_c_ray = round((angle % 360) * 1.422) - 1

    return min(true_lidar_values[ray % 512] for ray in range(-fov_indexs + index_c_ray, fov_indexs + 1 + index_c_ray))


def ckWallLidar(angle: int, limit: Literal[
    'R1', 'R1.5', 'R2', 'R2.5', 'R3',
    'B1', 'B2', 'B3',
    'K1', 'K2', 'K3',
    'C1.5', 'C2', 'C2.5', 'C3',
    'CK2', 'CK2.5', 'CK3',
    'CB', 'CR'
]):
    global _LIDAR_LIMIT
    lidar_val = getLidarValue(angle)

    if _LIDAR_LIMIT[limit][0] <= lidar_val <= _LIDAR_LIMIT[limit][1]:
        return 1
    elif lidar_val < _LIDAR_LIMIT[limit][0]:
        return -1
    else:
        return 0


def getCoords(divisor: float = 6, round_to_int: bool = True) -> Tuple:
    """
    Gets the X and Z coordinates from the gps, since Y is basically useless. Gets the current coords in a grid like
    pattern if the params are set to default, you can do some nice stuff if you modify them.
    For clarity, the * 100 is needed to convert from meters to centimeters

    @param divisor: Is 6 by default, needed to have a more precise coordinate system based on half-tiles
    @param round_to_int: Is True by default, Needed to have a precise reading
    @return: Gets the current (X, Z) coordinates, in cm's
    """
    if round_to_int:
        return tuple(round(coord * 100 / divisor) for coord in gps.getValues()[::2])
    else:
        return tuple(round(coord * 100 / divisor, 3) for coord in gps.getValues()[::2])


def getRotation() -> int:
    """
    Gets the current angle in degrees and converts it to the number of 90°/45° rotations from north clockwise

    @return: The number of 90°/45° rotations from north clockwise
    """
    
    return round(getAngle() / 45) % 8

def compas():

    """
    Based on the current angle, it determines his rotation
    """

    

    if round(getAngle() / 45) % 8 == 0:
        compas = 'north'
    elif round(getAngle() / 45) % 8 == 1:
        compas = 'northeast'
    elif round(getAngle() / 45) % 8 == 2:
        compas = 'east'
    elif round(getAngle() / 45) % 8 == 3:
        compas = 'southeast'
    elif round(getAngle() / 45) % 8 == 4:
        compas = 'south'
    elif round(getAngle() / 45) % 8 == 5:
        compas = 'southwest'
    elif round(getAngle() / 45) % 8 == 6:
        compas = 'west'
    elif round(getAngle() / 45) % 8 == 7:
        compas = 'northwest'
    else:
        print('unable to determine rotation')
    
    return compas


def getNearCoords(angle: int, distance=1, from_coord: Tuple[int, int] = None) \
        -> Tuple[int, int]:
    """
    Gets the coordinate of a half-tile in the direction specified, with 0 meaning in front of it and going on clockwise

    @param from_coord: The current coordinates by default, the coordinates of the origin of motion
    @param angle: The turn the robot would need to do for it to get the coordinate in front
    @param distance: How many half-tiles to go in that direction
    @return: The coordinate of a half-tile in the direction specified
    """
    angle = abs(round(angle / 45) + getRotation() + 8) % 8

    if from_coord is None:
        from_coord = getCoords()

    def shiftCoord(coord: Tuple[int, int], direction: Literal[0, 1, 2, 3]):
        nonlocal distance
        direction = direction % 4
        if direction == 0:
            return coord[0], coord[1] - distance
        if direction == 1:
            return coord[0] + distance, coord[1]
        if direction == 2:
            return coord[0], coord[1] + distance
        if direction == 3:
            return coord[0] - distance, coord[1]

    returned_coord = shiftCoord(from_coord, angle // 2)

    if angle % 2 == 1:
        returned_coord = shiftCoord(returned_coord, (angle + 1) // 2)

    return returned_coord


def addCells(central_coords):
    #print("Current coords: ", central_coords)
        maze.__add__({
            central_coords:
                Cell(explored=False,
                    dsf=ckWallLidar(0, 'R1'), dsr=ckWallLidar(90, 'R1'),
                    dsb=ckWallLidar(180, 'R1'), dsl=ckWallLidar(270, 'R1'),
                    fr=ckWallLidar(45, 'B1'), br=ckWallLidar(135, 'B1'),
                    bl=ckWallLidar(225, 'B1'), fl=ckWallLidar(315, 'B1'),
                    ds0=ckWallLidar(360 - _LIDAR_ANGLES['K1'], 'K1'), ds1=ckWallLidar(_LIDAR_ANGLES['K1'], 'K1'),
                    ds2=ckWallLidar(90 - _LIDAR_ANGLES['K1'], 'K1'), ds3=ckWallLidar(90 + _LIDAR_ANGLES['K1'], 'K1'),
                    ds4=ckWallLidar(180 - _LIDAR_ANGLES['K1'], 'K1'), ds5=ckWallLidar(180 + _LIDAR_ANGLES['K1'], 'K1'),
                    ds6=ckWallLidar(270 - _LIDAR_ANGLES['K1'], 'K1'), ds7=ckWallLidar(270 + _LIDAR_ANGLES['K1'], 'K1'),
                    ),
            getNearCoords(angle=0, distance=2, from_coord=central_coords):
                Cell(
                    cc=max(ckWallLidar(0, 'R2'), ckWallLidar(0, 'R1.5')), cb=ckWallLidar(0, 'R1.5'),
                    dsf=max(ckWallLidar(0, 'R3'), ckWallLidar(0, 'R2.5')), cf=ckWallLidar(0, 'R2.5'),
                    cr=ckWallLidar(_LIDAR_ANGLES['K2'], 'K2'), cl=ckWallLidar(360 - _LIDAR_ANGLES['K2'], 'K2'),
                    ds0=ckWallLidar(360 - _LIDAR_ANGLES['K3'], 'K3'), ds1=ckWallLidar(_LIDAR_ANGLES['K3'], 'K3'),
                    dsl=ckWallLidar(360 - _LIDAR_ANGLES['CK2'], 'CK2'), dsr=ckWallLidar(_LIDAR_ANGLES['CK2'], 'CK2'),
                    ds7=ckWallLidar(360 - _LIDAR_ANGLES['CK2.5'], 'CK2.5'),
                    ds2=ckWallLidar(_LIDAR_ANGLES['CK2.5'], 'CK2.5'),
                    fl=ckWallLidar(360 - _LIDAR_ANGLES['CK3'], 'CK3'), fr=ckWallLidar(_LIDAR_ANGLES['CK3'], 'CK3')
                ),
            getNearCoords(angle=45, distance=2, from_coord=central_coords):
                Cell(
                    cc=ckWallLidar(45, 'B2'), fr=ckWallLidar(45, 'B3'),
                    ds6=ckWallLidar(_LIDAR_ANGLES['C1.5'], 'C1.5'), ds5=ckWallLidar(90 - _LIDAR_ANGLES['C1.5'], 'C1.5'),
                    cl=ckWallLidar(_LIDAR_ANGLES['C2'], 'C2'), cb=ckWallLidar(90 - _LIDAR_ANGLES['C2'], 'C2'),
                    cf=ckWallLidar(_LIDAR_ANGLES['C2.5'], 'C2.5'), cr=ckWallLidar(90 - _LIDAR_ANGLES['C2.5'], 'C2.5'),
                    ds1=ckWallLidar(_LIDAR_ANGLES['C3'], 'C3'), ds2=ckWallLidar(90 - _LIDAR_ANGLES['C3'], 'C3'),
                    dsf=ckWallLidar(_LIDAR_ANGLES['CB'], 'CB'), dsr=ckWallLidar(90 - _LIDAR_ANGLES['CB'], 'CB'),
                    ds0=ckWallLidar(_LIDAR_ANGLES['CR'], 'CR'), ds3=ckWallLidar(90 - _LIDAR_ANGLES['CR'], 'CR')
                ),
            getNearCoords(angle=90, distance=2, from_coord=central_coords):
                Cell(
                    cc=max(ckWallLidar(90, 'R2'), ckWallLidar(90, 'R1.5')), cl=ckWallLidar(90, 'R1.5'),
                    dsr=max(ckWallLidar(90, 'R3'), ckWallLidar(90, 'R2.5')), cr=ckWallLidar(90, 'R2.5'),
                    cf=ckWallLidar(90 - _LIDAR_ANGLES['K2'], 'K2'), cb=ckWallLidar(90 + _LIDAR_ANGLES['K2'], 'K2'),
                    ds2=ckWallLidar(90 - _LIDAR_ANGLES['K3'], 'K3'), ds3=ckWallLidar(90 + _LIDAR_ANGLES['K3'], 'K3'),
                    dsf=ckWallLidar(90 - _LIDAR_ANGLES['CK2'], 'CK2'), dsb=ckWallLidar(90 + _LIDAR_ANGLES['CK2'], 'CK2'),
                    ds1=ckWallLidar(90 - _LIDAR_ANGLES['CK2.5'], 'CK2.5'),
                    ds4=ckWallLidar(90 + _LIDAR_ANGLES['CK2.5'], 'CK2.5'),
                    fr=ckWallLidar(90 - _LIDAR_ANGLES['CK3'], 'CK3'), br=ckWallLidar(90 + _LIDAR_ANGLES['CK3'], 'CK3')
                ),
            getNearCoords(angle=135, distance=2, from_coord=central_coords):
                Cell(
                    cc=ckWallLidar(135, 'B2'), br=ckWallLidar(135, 'B3'),
                    ds0=ckWallLidar(90 + _LIDAR_ANGLES['C1.5'], 'C1.5'),
                    ds7=ckWallLidar(180 - _LIDAR_ANGLES['C1.5'], 'C1.5'),
                    cl=ckWallLidar(180 - _LIDAR_ANGLES['C2'], 'C2'), cf=ckWallLidar(90 + _LIDAR_ANGLES['C2'], 'C2'),
                    cb=ckWallLidar(180 - _LIDAR_ANGLES['C2.5'], 'C2.5'),
                    cr=ckWallLidar(90 + _LIDAR_ANGLES['C2.5'], 'C2.5'),
                    ds4=ckWallLidar(180 - _LIDAR_ANGLES['C3'], 'C3'), ds3=ckWallLidar(90 + _LIDAR_ANGLES['C3'], 'C3'),
                    dsr=ckWallLidar(90 + _LIDAR_ANGLES['CB'], 'CB'), dsb=ckWallLidar(180 - _LIDAR_ANGLES['CB'], 'CB'),
                    ds2=ckWallLidar(90 + _LIDAR_ANGLES['CR'], 'CR'), ds5=ckWallLidar(180 - _LIDAR_ANGLES['CR'], 'CR')
                ),
            getNearCoords(angle=180, distance=2, from_coord=central_coords):
                Cell(
                    cc=max(ckWallLidar(180, 'R2'), ckWallLidar(180, 'R1.5')), cf=ckWallLidar(180, 'R1.5'),
                    dsb=max(ckWallLidar(180, 'R3'), ckWallLidar(180, 'R2.5')), cb=ckWallLidar(180, 'R2.5'),
                    cr=ckWallLidar(180 - _LIDAR_ANGLES['K2'], 'K2'), cl=ckWallLidar(180 + _LIDAR_ANGLES['K2'], 'K2'),
                    ds4=ckWallLidar(180 - _LIDAR_ANGLES['K3'], 'K3'), ds5=ckWallLidar(180 + _LIDAR_ANGLES['K3'], 'K3'),
                    dsr=ckWallLidar(180 - _LIDAR_ANGLES['CK2'], 'CK2'), dsl=ckWallLidar(180 + _LIDAR_ANGLES['CK2'], 'CK2'),
                    ds3=ckWallLidar(180 - _LIDAR_ANGLES['CK2.5'], 'CK2.5'),
                    ds6=ckWallLidar(180 + _LIDAR_ANGLES['CK2.5'], 'CK2.5'),
                    br=ckWallLidar(180 - _LIDAR_ANGLES['CK3'], 'CK3'), bl=ckWallLidar(180 + _LIDAR_ANGLES['CK3'], 'CK3')
                ),
            getNearCoords(angle=225, distance=2, from_coord=central_coords):
                Cell(
                    cc=ckWallLidar(225, 'B2'), bl=ckWallLidar(225, 'B3'),
                    ds2=ckWallLidar(180 + _LIDAR_ANGLES['C1.5'], 'C1.5'),
                    ds1=ckWallLidar(270 - _LIDAR_ANGLES['C1.5'], 'C1.5'),
                    cr=ckWallLidar(180 + _LIDAR_ANGLES['C2'], 'C2'), cf=ckWallLidar(270 - _LIDAR_ANGLES['C2'], 'C2'),
                    cb=ckWallLidar(180 + _LIDAR_ANGLES['C2.5'], 'C2.5'),
                    cl=ckWallLidar(270 - _LIDAR_ANGLES['C2.5'], 'C2.5'),
                    ds5=ckWallLidar(180 + _LIDAR_ANGLES['C3'], 'C3'), ds6=ckWallLidar(270 - _LIDAR_ANGLES['C3'], 'C3'),
                    dsb=ckWallLidar(180 + _LIDAR_ANGLES['CB'], 'CB'), dsl=ckWallLidar(270 - _LIDAR_ANGLES['CB'], 'CB'),
                    ds4=ckWallLidar(180 + _LIDAR_ANGLES['CR'], 'CR'), ds7=ckWallLidar(270 - _LIDAR_ANGLES['CR'], 'CR')
                ),
            getNearCoords(angle=270, distance=2, from_coord=central_coords):
                Cell(
                    cc=max(ckWallLidar(270, 'R2'), ckWallLidar(270, 'R1.5')), cr=ckWallLidar(270, 'R1.5'),
                    dsl=max(ckWallLidar(270, 'R3'), ckWallLidar(270, 'R2.5')), cl=ckWallLidar(270, 'R2.5'),
                    cf=ckWallLidar(270 + _LIDAR_ANGLES['K2'], 'K2'), cb=ckWallLidar(270 - _LIDAR_ANGLES['K2'], 'K2'),
                    ds6=ckWallLidar(270 - _LIDAR_ANGLES['K3'], 'K3'), ds7=ckWallLidar(270 + _LIDAR_ANGLES['K3'], 'K3'),
                    dsb=ckWallLidar(270 - _LIDAR_ANGLES['CK2'], 'CK2'), dsf=ckWallLidar(270 + _LIDAR_ANGLES['CK2'], 'CK2'),
                    ds5=ckWallLidar(270 - _LIDAR_ANGLES['CK2.5'], 'CK2.5'),
                    ds0=ckWallLidar(270 + _LIDAR_ANGLES['CK2.5'], 'CK2.5'),
                    bl=ckWallLidar(270 - _LIDAR_ANGLES['CK3'], 'CK3'), fl=ckWallLidar(270 + _LIDAR_ANGLES['CK3'], 'CK3')
                ),
            getNearCoords(angle=315, distance=2, from_coord=central_coords):
                Cell(
                    cc=ckWallLidar(315, 'B2'), fl=ckWallLidar(315, 'B3'),
                    ds3=ckWallLidar(360 - _LIDAR_ANGLES['C1.5'], 'C1.5'),
                    ds4=ckWallLidar(270 + _LIDAR_ANGLES['C1.5'], 'C1.5'),
                    cr=ckWallLidar(360 - _LIDAR_ANGLES['C2'], 'C2'), cb=ckWallLidar(270 + _LIDAR_ANGLES['C2'], 'C2'),
                    cf=ckWallLidar(360 - _LIDAR_ANGLES['C2.5'], 'C2.5'),
                    cl=ckWallLidar(270 + _LIDAR_ANGLES['C2.5'], 'C2.5'),
                    ds0=ckWallLidar(360 - _LIDAR_ANGLES['C3'], 'C3'), ds7=ckWallLidar(270 + _LIDAR_ANGLES['C3'], 'C3'),
                    dsl=ckWallLidar(270 + _LIDAR_ANGLES['CB'], 'CB'), dsf=ckWallLidar(360 - _LIDAR_ANGLES['CB'], 'CB'),
                    ds6=ckWallLidar(270 + _LIDAR_ANGLES['CR'], 'CR'), ds1=ckWallLidar(360 - _LIDAR_ANGLES['CR'], 'CR')
                )
        })
    
        


def bgra_to_hsv(bgra_color):
    # Normalize the RGB values to be in the range [0, 1]
    r = bgra_color[2] / 255.
    g = bgra_color[1] / 255.
    b = bgra_color[0] / 255.

    # Convert the RGB color to an HSV color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Scale the hue value to be in the range [0, 360]
    h = int(h * 360)

    # Scale the saturation and value values to be in the range [0, 1]
    s, v = round(s, 3), round(v, 3)

    # Return the HSV color as a tuple
    return h, s, v


def getColor() -> Tuple[int, float, float]:
    """
    Gets the color from the color sensor, which points to a single pixel on the floor in front of the robot, and
    transform it to an BGRA tuple. Also note that the distance of the point is not following any type of grid

    @return: A tuple with HSV values
    """
    # noinspection PyUnresolvedReferences
    color = color_sensor.getImage()
    
    
    color = struct.unpack('BBBB', color)
    #print(color)
    return bgra_to_hsv(color)


def checkColor(color: Tuple[int, float, float], color_range: Dict[Literal['min', 'max'], Tuple[int, float, float]]) \
        -> bool:
    """
    Check if the color given is within the range given

    @param color: The color checked in HSV format
    @param color_range: A dictionary that follows this format: {'min': hsv(0, 0, 0), 'max': (360, 1, 1)}
    @return: True if every color is within the range, otherwise False
    """
    return all(color_range['min'][i] <= color[i] <= color_range['max'][i] for i in range(3))


def coordInGrid(coord: Tuple[int, int]) -> bool:
    """
    Check if the given coordinate is following the tile pattern and not the half-tile's

    @param coord: The coordinate to check
    @return: True if it does follow the tile pattern, otherwise False
    """
    return abs(coord[0]) % 2 == abs(maze.start[0]) % 2 and abs(coord[1]) % 2 == abs(maze.start[1]) % 2


def placeAndHandleHole(hole_coordinate: Tuple[int, int], possible_offset: int, is_void=False):
    # TODO: Add doc
    if not coordInGrid(hole_coordinate):
        hole_coordinate = getNearCoords(from_coord=hole_coordinate, angle=possible_offset)
    if not coordInGrid(hole_coordinate):
        hole_coordinate = getNearCoords(from_coord=hole_coordinate, angle=(possible_offset * 3) % 360)

    if is_void:
        maze.__add__({hole_coordinate: Cell(c_t=_VOID)})
    else:
        maze.__add__({hole_coordinate: Cell(c_t=_HOLE.val)})

    if maze.map[hole_coordinate].nearHole:
        maze.removeToExploreArea(hole_coordinate)
        maze.map[hole_coordinate].explored = True
        return True
    else:
        return False


def checkAndPlaceTile(tile: Dot, color: Tuple[int, float, float]):
    # TODO: Add doc
    if getRotation() % 2 == 1 or not checkColor(color, tile.color_range):
        return False

    #print(f"Current coordinates: {getCoords()}, Coordinates in front: {getNearCoords(0)}")
    if coordInGrid(getNearCoords(0)):
        maze.__add__({getNearCoords(0): Cell(c_t=tile.val)})
        return True
    return False


def checkAndPlaceTunnel(tunnel: Tunnel, color: Tuple[int, float, float], color_name: str):
    # TODO: Add doc
    global obstruction_found, _BLUE, _RED, _YELLOW, _GREEN, _ORANGE, _PURPLE, _TUNNELS

    if tunnel.val == 15:
        if not tunnel.found and checkColor(color, tunnel.color_range) and coordInGrid(getNearCoords(0)) and _BLUE.found and \
        getNearCoords(0) not in _TUNNELS:
            print(f"Adding second blue tunnel. {_BLUE.coord} {getNearCoords(0)} {getNearCoords(45)} {getNearCoords(315)}")
            tunnel.coord = getNearCoords(0)

            maze.__add__({tunnel.coord: Cell(c_t=tunnel.val, explored=True)})

            maze.removeToExploreArea(tunnel.coord)
            _TUNNELS[tunnel.coord] = color_name

            tunnel.found = True
            obstruction_found = True
            return True

    elif tunnel.val == 16:
        if not tunnel.found and checkColor(color, tunnel.color_range) and coordInGrid(getNearCoords(0)) and _PURPLE.found and \
        getNearCoords(0) not in _TUNNELS:
            print("Adding second purple tunnel.")
            tunnel.coord = getNearCoords(0)

            maze.__add__({tunnel.coord: Cell(c_t=tunnel.val, explored=True)})
            
            maze.removeToExploreArea(tunnel.coord)
            _TUNNELS[tunnel.coord] = color_name

            tunnel.found = True
            obstruction_found = True
            return True

    elif tunnel.val == 17:
        if not tunnel.found and checkColor(color, tunnel.color_range) and coordInGrid(getNearCoords(0)) and _RED.found and \
        getNearCoords(0) not in _TUNNELS:
            print("Adding second red tunnel.")
            tunnel.coord = getNearCoords(0)

            maze.__add__({tunnel.coord: Cell(c_t=tunnel.val, explored=True)})
            
            maze.removeToExploreArea(tunnel.coord)
            _TUNNELS[tunnel.coord] = color_name

            tunnel.found = True
            obstruction_found = True
            return True

    elif tunnel.val == 18:
        if not tunnel.found and checkColor(color, tunnel.color_range) and coordInGrid(getNearCoords(0)) and _GREEN.found and \
        getNearCoords(0) not in _TUNNELS:
            print("Adding second green tunnel.")
            tunnel.coord = getNearCoords(0)

            maze.__add__({tunnel.coord: Cell(c_t=tunnel.val, explored=True)})
            
            maze.removeToExploreArea(tunnel.coord)
            _TUNNELS[tunnel.coord] = color_name

            tunnel.found = True
            obstruction_found = True
            return True

    elif tunnel.val == 19:
        if not tunnel.found and checkColor(color, tunnel.color_range) and coordInGrid(getNearCoords(0)) and _YELLOW.found and \
        getNearCoords(0) not in _TUNNELS:
            print("Adding second yellow tunnel.")
            tunnel.coord = getNearCoords(0)

            maze.__add__({tunnel.coord: Cell(c_t=tunnel.val, explored=True)})
            
            maze.removeToExploreArea(tunnel.coord)
            _TUNNELS[tunnel.coord] = color_name

            tunnel.found = True
            obstruction_found = True
            return True

    elif tunnel.val == 20:
        if not tunnel.found and checkColor(color, tunnel.color_range) and coordInGrid(getNearCoords(0)) and _ORANGE.found and \
        getNearCoords(0) not in _TUNNELS:
            print("Adding second orange tunnel.")
            tunnel.coord = getNearCoords(0)

            maze.__add__({tunnel.coord: Cell(c_t=tunnel.val, explored=True)})
            
            maze.removeToExploreArea(tunnel.coord)
            _TUNNELS[tunnel.coord] = color_name

            tunnel.found = True
            obstruction_found = True
            return True

    else:
        if not tunnel.found and checkColor(color, tunnel.color_range) and coordInGrid(getNearCoords(0)):
            print(f"Adding first tunnel: {tunnel.val} {getNearCoords(0)}")
            tunnel.coord = getNearCoords(0)

            maze.__add__({tunnel.coord: Cell(c_t=tunnel.val, explored=True)})

            maze.removeToExploreArea(tunnel.coord)
            _TUNNELS[tunnel.coord] = color_name

            tunnel.found = True
            obstruction_found = True
            return True
    return False


def insideSwamp() -> bool:
    """
    Checks if the robot is inside a cell within a swamp(it reduces speed by 32%)

    @return: True if the robot is in a swamp, otherwise False
    """
    return any(_SWAMP.val in row for row in maze.map[getCoords()].cell)


def shiftMatrix(direction: Literal['up', 'down', 'left', 'right'], repeat: int, old_list: List[List]):
    """
    Cool function that shifts a matrix around its axis found at:
    https://stackoverflow.com/questions/19878280/efficient-way-to-shift-2d-matrices-in-both-directions
    The function is modified to match the needs of the code

    @param direction: The direction of the operation
    @param repeat: How many times the operation is needed
    @param old_list: The list to shift
    @return: The shifted list
    """
    height = len(old_list)
    width = len(old_list[0])
    if direction == "up":
        return [old_list[i % height] for i in range(repeat, repeat + height)]
    elif direction == "down":
        return [old_list[-i] for i in range(repeat, repeat - height, -1)]
    elif direction == "left":
        tlist = list(zip(*old_list))
        return list(map(list, zip(*[tlist[i % width] for i in range(repeat, repeat + width)])))
    elif direction == "right":
        tlist = list(zip(*old_list))
        return list(map(list, zip(*[tlist[-i] for i in range(repeat, repeat - width, -1)])))
    else:
        print('Direction not valid, choose from: \'up\', \'down\', \'left\', \'right\'. Now returning old_list')
        return old_list


def new_speed(multi: float = 1, incr: float = 0) -> float:
    """
    Multiplies the _MAX_VELOCITY to get a new adjusted speed

    @param multi: The speed multiplier
    @param incr: The speed increment
    @return: The new speed
    """
    global turn_multi
    return multi * _MAX_VELOCITY * turn_multi[0] + incr, multi * _MAX_VELOCITY * turn_multi[1] + incr


def is_turning() -> bool:
    return speeds[0] * speeds[1] < 0


def getLegalMoves() -> List[Tuple[int, int]]:
    """
    Gets all the coordinates the robot can move at the current position

    @return: A list of all the legal moves
    """
    return [getNearCoords(angle) for angle in _MVMT_ANGLES if maze.map[getNearCoords(angle)].canBeAvailable]


def getLegalCoordAngle():
    legal_moves = {getNearCoords(angle): angle for angle in _MVMT_ANGLES
                   if maze.map[getNearCoords(angle)].canBeAvailable}
    for key in legal_moves.keys():
        corrected_angle = (math.atan2(getCoords(round_to_int=False)[0] - key[0],
                                      getCoords(round_to_int=False)[1] - key[1])) % (2 * math.pi)
        if corrected_angle > math.pi:
            corrected_angle = 2 * math.pi - corrected_angle
        else:
            corrected_angle = -corrected_angle
        legal_moves[key] = (corrected_angle * (180 / math.pi)) % 360
    return legal_moves


def chooseDirection(move: int = -1):
    """
    Chooses the new direction the robot needs to go towards

    @param move: If set, it contains the next move the robot needs to do, which probably was chosen by this very method
    @return: Doc needs revisit
    """
    global turn_multi, next_angle

    def stayPut():
        nonlocal next_rotation
        global turn_multi
        next_rotation = 0
        turn_multi = [1, 1]
        return 0

    def moveInDirection(direction: int):
        """
        Given a direction to follow in degrees, it executes the corresponding function

        @param direction: The direction starting from 0 in front of the robot and going clockwise(-1 does nothing)
        @return: The result of the operation, 1 if it did something and 0 otherwise
        """
        if direction < 0:
            return stayPut()

        # For some reason left and right are inverted, the following 'if' statement handles that
        global turn_multi, next_move
        nonlocal next_rotation

        if direction == 0:
            turn_multi = [1, 1]
            next_rotation = 0
            next_move = -1
            return 1

        if (direction - getAngle()) % 360 >= 180:
            turn_multi = [1, -1]
        else:
            turn_multi = [-1, 1]

        next_move = 0
        next_rotation = direction

        return 1

    def findBestPath(possible_moves, recursive_counter: int = 0):
        global curr_path, _BLUE, _BLUE2, _PURPLE, _PURPLE2, _RED, _RED2, _GREEN, _GREEN2, _YELLOW, _YELLOW2 \
        ,_ORANGE, _ORANGE2, curr_obj

        if recursive_counter > 1:
            if getCoords()[0] == maze.start[0] and getCoords()[1] == maze.start[1]:
                delay(1000)
                initiateEndSequence()
                delay(1000)
                return
            else:
                maze.checkToExplore()

                try:
                    if len(possible_moves.keys()) == 1:
                        curr_path = {next(iter(possible_moves))}
                    elif recursive_counter <= 3 and struct.unpack('c', receiver.getBytes())[0].decode("utf-8") == 'L':
                        print(curr_path)
                    elif recursive_counter > 2:
                        print('Oh no')
                        print(maze.getBestPath(view_map=True))
                        print(maze)
                        delay(1000)
                        exit()
                        delay(1000)
                except (Exception,):
                    print('I don\'t want to even think about it')

        # If the robot currently doesn't have a path it will try to find a new one
        if curr_path is None or len(curr_path) == 0:
            try:
                curr_path = maze.getBestPath(view_map=False)
            except Exception as exc:
                print(exc)

        # If the path to the nearest unexplored cell doesn't exist, these are the possibilities:
        # - The robot has fully explored the previous area and is now going to go to the next one
        # - The robot has fully explored the map and is now going to go to the starting tile
        if curr_path is None or len(curr_path) == 0:
            if _BLUE.coord is not None:
                maze.to_explore.add(_BLUE.coord)
                _BLUE.coord = None
            elif _BLUE2.coord is not None:
                maze.to_explore.add(_BLUE2.coord)
                _BLUE2.coord = None
            elif _PURPLE.coord is not None:
                maze.to_explore.add(_PURPLE.coord)
                _PURPLE.coord = None
            elif _PURPLE2.coord is not None:
                maze.to_explore.add(_PURPLE2.coord)
                _PURPLE2.coord = None
            elif _RED.coord is not None:
                maze.to_explore.add(_RED.coord)
                _RED.coord = None
            elif _RED2.coord is not None:
                maze.to_explore.add(_RED2.coord)
                _RED2.coord = None
            elif _GREEN.coord is not None:
                maze.to_explore.add(_GREEN.coord)
                _GREEN.coord = None
            elif _GREEN2.coord is not None:
                maze.to_explore.add(_GREEN2.coord)
                _GREEN2.coord = None
            elif _YELLOW.coord is not None:
                maze.to_explore.add(_YELLOW.coord)
                _YELLOW.coord = None
            elif _YELLOW2.coord is not None:
                maze.to_explore.add(_YELLOW2.coord)
                _YELLOW2.coord = None
            elif _ORANGE.coord is not None:
                maze.to_explore.add(_ORANGE.coord)
                _ORANGE.coord = None
            elif _ORANGE2.coord is not None:
                maze.to_explore.add(_ORANGE2.coord)
                _ORANGE2.coord = None
            else:
                maze.to_explore.add(maze.start)
                print('Going to the start')

            findBestPath(possible_moves=possible_moves, recursive_counter=recursive_counter + 1)
            return

        # Under normal circumstances the robot finds the current move that needs to be done by intersecting the legal
        # moves with the path to take
        curr_move = set(possible_moves.keys()).intersection(curr_path)

        try:
            curr_move = next(iter(curr_move))  # Takes the only element in the set
            moveInDirection(direction=possible_moves[curr_move])
            curr_obj = curr_move[0] * 6, curr_move[1] * 6
            curr_path.remove(curr_move)
        except (Exception,):
            # This is just a mess
            maze.checkToExplore()
            print("Best path: ", maze.getBestPath())
            if maze.getBestPath() == None:
                delay(1000)
                initiateEndSequence()
                delay(1000)
                return
                
            elif len(curr_move) == 0 and len(curr_path.intersection(actual_path := maze.getBestPath())) < len(curr_path):
                print('Was trying to follow a wrong path')
                curr_path = actual_path
                findBestPath(possible_moves=possible_moves, recursive_counter=recursive_counter + 1)
                return
            elif receiver.getQueueLength() > 0:
                # Detects LoP
                if struct.unpack('c', receiver.getBytes())[0].decode("utf-8") == 'L':
                    print("Detected controlled LoP", recursive_counter)
                    curr_path = maze.getBestPath()
                    findBestPath(possible_moves=getLegalCoordAngle(), recursive_counter=recursive_counter + 1)
                    receiver.nextPacket()  # Discard the current data packet
                    return
                receiver.nextPacket()  # Discard the current data packet

            print(f'The best path has {len(curr_move)} options to move to?! Resorting to right first')
            raise ValueError('Should have executed rightFirst')

    # Have to put it here to set the default value
    next_rotation = 0

    if move == -1:
        findBestPath(possible_moves=getLegalCoordAngle())

        if (round(next_rotation / 45) - round(getAngle() / 45)) == 0:
            next_angle = (round(next_rotation / 45) * 45) % 360
        else:
            next_angle = next_rotation % 360
    else:
        moveInDirection(move)

    # Currently useless
    # if insideSwamp():
    #     turn_multi = turn_multi[0], turn_multi[1]

    if round(next_angle / 45) % 2 == 0:
        next_angle = round(next_angle / 45) * 45


def static_turn() -> int:
    """
    Establishes the direction and the speed of the rotation and stops at a precision of 0.001°.
    During the rotation the robot stays stationary, since the wheel speeds are opposite at all times

    @return: None
    """
    global turn_multi, speeds, next_move, next_angle

    # Whenever the next_angle or the current angle approaches the north direction(0° or 360°), it gives lots of problems
    # We decided to avoid this problem all together by adding 180 to it when we're pointing near it
    temp_curr_angle = getAngle()
    temp_next_angle = next_angle
    if abs(temp_curr_angle - temp_next_angle) > 180:
        temp_next_angle = (temp_next_angle + 180) % 360
        temp_curr_angle = (temp_curr_angle + 180) % 360

    # The statements below are needed to make almost perfect right angle turns(precise to the 0.001°)
    # --- This make it so that the position of the robot never leaves the actual grid
    # --- The robot will change turning direction when is getting over the threshold
    if abs(temp_curr_angle - temp_next_angle) < 0.001:
        # Reset the speed of the wheels
        turn_multi = [1, 1]
        speeds = new_speed()
        # Reset angle
        next_angle = -1

        chooseDirection(move=next_move)
        next_move = -1

        motor_left.setVelocity(speeds[0])
        motor_right.setVelocity(speeds[1])
        robot.step(timeStep)
        return 1

    # Decides to invert the rotation
    if (temp_next_angle - temp_curr_angle < 0) ^ (turn_multi[0] > 0):
        turn_multi = [turn_multi[0] * -1, turn_multi[1] * -1]

    angular_distance = abs(temp_next_angle - temp_curr_angle)
    if angular_distance > 180:
        angular_distance = abs(angular_distance - 360)

    if angular_distance <= 0.2:
        speeds = new_speed(angular_distance / 10, 0.005)
        return 0

    if angular_distance <= 6:
        speeds = new_speed(math.sqrt(angular_distance / 90))
        return 0

    speeds = new_speed()
    return 0


def dynamic_turn(objective, current):
    global turn_multi

    angle = (math.atan2(current[0] - objective[0] * 6, current[1] - objective[1] * 6)) % (2 * math.pi)
    if angle > math.pi:
        angle = 2 * math.pi - angle
    else:
        angle = -angle
    # Convert the angle to degrees
    angle = (angle * 180 / math.pi)
    # Round the angle to 8th of a circle
    angle = (round(angle / 45) * 45) % 360
    # Get the actual angle considering the cell forward as 0°
    angle = angle - getRotation() * 45

    temp_curr_angle = getAngle()
    temp_next_angle = angle
    if abs(temp_curr_angle - temp_next_angle) > 180:
        temp_next_angle = (temp_next_angle + 180) % 360
        temp_curr_angle = (temp_curr_angle + 180) % 360

    angular_distance = abs(temp_next_angle - temp_curr_angle)
    if angular_distance > 180:
        angular_distance = abs(angular_distance - 360)

    angular_distance /= 180

    # scale_factor = 0.33 + 1 / ((angular_distance + 1) * (angular_distance**3 + 1.5))
    scale_factor = 0.33 + math.e ** (-0.1 * angular_distance) - 0.33 * math.e ** (-0.3 * angular_distance)

    if (temp_curr_angle < temp_next_angle and turn_multi[0] < 0 and turn_multi[1] < 0) or \
            (temp_curr_angle > temp_next_angle and turn_multi[0] > 0 and turn_multi[1] > 0):
        turn_multi = turn_multi[0] * scale_factor, turn_multi[1]
    else:
        turn_multi = turn_multi[0], scale_factor * turn_multi[1]


def delay(ms):
    """
    Time idling necessary to score points
    @param ms : time to wait in ms
    """
    start_timer = robot.getTime()  # Store starting time (in seconds)
    while robot.step(timeStep) != -1 and (robot.getTime() - start_timer) * 1000.0 <= ms:
        pass



def scoreVictim(victim_type: Literal['H', 'S', 'U', 'C', 'F', 'O', 'P'], camera: Literal['left', 'right'], direction,
                msg):
    """
    Send the message to score the victim's point

    @param victim_type: Type of the victim
    @param camera: The camera that saw the victim
    @param direction: The position of the victim in the photo
    @param msg: The message to display if a victim is found
    @return: Boolean that determines the success of the operation
    """

    direction /= 2
    direction += -90 if camera == 'left' else 90
    direction = (direction + getAngle()) % 360

    # alpha = direction * math.pi / 180
    # robot_gps = max(abs(getCoords(divisor=1)[0] - getCoords(divisor=1, round_to_int=False)[0]),
    #                 abs(getCoords(divisor=1)[1] - getCoords(divisor=1, round_to_int=False)[1]))
    # avg_robot_victim = 6.25
    # gps_victim = math.sqrt(
    #     robot_gps ** 2 + avg_robot_victim ** 2 - 2 * robot_gps * avg_robot_victim * math.cos(alpha)
    # )
    # gamma = math.asin(math.sin(alpha) * robot_gps / gps_victim)
    # suppl_beta = ((alpha + gamma) % (2 * math.pi)) * 180 / math.pi

    

    is_letter = victim_type == 'S' or victim_type == 'H' or victim_type == 'U'

    half_fov_cam = left_camera.getFov() * 180 / math.pi / 2

    half_fov_cam *= 1.5 if getRotation() % 2 == 1 else 1

    for v in victim_pos:
        v_min_angle = (v[4] - half_fov_cam) % 360
        v_max_angle = (v[4] + half_fov_cam) % 360
        temp_dir = copy.copy(direction)

        if abs(v_min_angle - temp_dir) > 180 or abs(v_max_angle - temp_dir) > 180 \
                or abs(v_min_angle - v_max_angle) > 180:
            v_min_angle = (v_min_angle + 180) % 360
            v_max_angle = (v_max_angle + 180) % 360
            temp_dir = (temp_dir + 180) % 360

        if v_min_angle < temp_dir < v_max_angle and v[3] == is_letter and \
                math.sqrt((getCoords(divisor=1, round_to_int=False)[0] - v[0]) ** 2 +
                          (getCoords(divisor=1, round_to_int=False)[1] - v[1]) ** 2) < 4:
            return

        del temp_dir
    else:
        victim_pos.add((*getCoords(divisor=1, round_to_int=False), victim_type, is_letter, direction))
        

    motor_left.setVelocity(0)
    motor_right.setVelocity(0)
    delay(1400)

    victim_message = bytes(victim_type, 'utf-8')

    pos_x, pos_z = getCoords(divisor=1)

    print(victim_pos)

    victim_message = struct.pack('i i c', pos_x, pos_z, victim_message)  # Pack the message.
    
    if victim_found == True:
        print("victim found")
    else: 
        print("victim not found")
    
    emitter.send(victim_message)  # Send out the message
    robot.step(timeStep)

    #print(msg)
    # print((*getCoords(divisor=1, round_to_int=False), victim_type, is_letter, direction))

    

def addToSeen(coord: Tuple[int, int], camera: Literal['left', 'right'], victim: str, grid_position: Literal["back", "middle", "front"], victim_found: True):
    opposite_camera = 'right' if camera == 'left' else 'left'
    
    #print(coord, camera, victim)
    
    print(maze.map[coord])
    
    if getRotation() % 2 == 0:
        front_coord = getNearCoords(angle=0, from_coord=coord)
    else:
        if camera == 'left':
            front_coord = getNearCoords(angle=315, from_coord=coord)
        else:
            front_coord = getNearCoords(angle=45, from_coord=coord)

    if (victim == 'S' or victim == 'H' or victim == 'U') and getRotation() % 2 == 0:
        coord = [None, None]

    prev_victims.add((*coord, getRotation(), camera))
    prev_victims.add((*coord, (getRotation() + 1) % 8, camera))
    prev_victims.add((*coord, (getRotation() - 1) % 8, camera))
    prev_victims.add((*front_coord, getRotation(), camera))
    prev_victims.add((*front_coord, (getRotation() + 1) % 8, camera))
    prev_victims.add((*front_coord, (getRotation() - 1) % 8, camera))

    prev_victims.add((*coord, (getRotation() + 4) % 8, opposite_camera))
    prev_victims.add((*coord, (getRotation() + 3) % 8, opposite_camera))
    prev_victims.add((*coord, (getRotation() + 5) % 8, opposite_camera))
    prev_victims.add((*front_coord, (getRotation() + 4) % 8, opposite_camera))
    prev_victims.add((*front_coord, (getRotation() + 3) % 8, opposite_camera))
    prev_victims.add((*front_coord, (getRotation() + 5) % 8, opposite_camera))

def initiateEndSequence():
    global _TUNNELS
    print('initiateEndSequence() was just called')
    # Get the map as a matrix
    #print(maze)
    #print(_TUNNELS)


    map_matrix = maze.getMap(nice_view=False)
    print(map_matrix)
    print(maze.getMap(nice_view=True))
    # Flatten the array with comma separators, then encode it with utf-8 and send it
    # - Get shape as bytes
    shape_bytes = struct.pack('2i', *map_matrix.shape)
    # - Flattening the matrix and join with commas, then encode it
    matrix_bytes = ','.join(map_matrix.flatten()).encode('utf-8')
    # - Prepare the shape + matrix bytes to get the full map message
    map_bytes = shape_bytes + matrix_bytes
    # - Send map data
    emitter.send(map_bytes)
    
    delay(1000)
    # Send map evaluate request to get the map multiplier
    map_evaluate_request = struct.pack('c', b'M')
    emitter.send(map_evaluate_request)
    robot.step(timeStep)
    delay(1000)

    # Send exit message to get the exit bonus
    exit_mes = struct.pack('c', b'E')
    emitter.send(exit_mes)
    delay(1000)
   

    robot.step(timeStep)

    print('Finished execution')
    delay(1000)
    exit()
    delay(1000)
    delay(1000)


########################################################################################################################
# Initialization variables

# Used for initiate and continue the turning process
turn_multi = [1, 1]

# When you move, you can give a 'next move' direction to say for example:
# --- Turn right, then turn right again(90) to turn around in a dead end
# --- Turn left, then go forward(0)
next_move: int = -1

curr_path = copy.copy(set())
curr_obj = None

# Initiates the sequence to avoid an in front of the robot(wall/obstacle/hole)
# - Added later, but it also counts going to another are as an 'obstruction', since it speeds up exploration
obstruction_found = False

# Store the position of the last obstacle found
last_obstacle_coord = None

# Needs to reach a chosen value(def=3) to recognize that there is a hole or a checkpoint,
# since the engine sometimes glitches out and gives a black screen
check_hole_color = 0
check_hole_depth = 0

# If -1 it's undefined, it indicates the expected angle at the end of a right angle turn
next_angle = -1

# The speed is initially set to [0, 0] to waste the first frame
speeds = [0, 0]

# Set of stuff that says to not take a photo near a previously discovered letter(hazards never get recognized two times)
prev_victims = copy.copy(set())
# Set containing all the previously found victims and their coordinates
victim_pos = copy.copy(set())

########################################################################################################################
# Setup robot and its devices

robot = Robot()

# Initiate the emitter and receiver
receiver = Receiver('receiver')
emitter = Emitter("emitter")
receiver.enable(timeStep)  # Enable the receiver. Note that the emitter does not need to call enable()

# Just wheels
wheel_left = robot.getDevice("wheel1 motor")
wheel_right = robot.getDevice("wheel2 motor")
motor_left = Motor("wheel1 motor")
motor_right = Motor("wheel2 motor")
motor_left.setPosition(float("inf"))
motor_right.setPosition(float("inf"))
motor_left.setVelocity(speeds[0])
motor_right.setVelocity(speeds[1])

# The inertial unit gets the orientation of the robot
inertial_unit = InertialUnit('inertial_unit')
inertial_unit.enable(timeStep)

# The color sensor gets one pixel just in front of the robot, positioned to avoid black holes
color_sensor = robot.getDevice("color_sensor")
# noinspection PyUnresolvedReferences
color_sensor.enable(timeStep)

# The GPS gets the current position in the maze in meters
gps = GPS('gps')
gps.enable(timeStep)

# Cameras of robot
left_camera = Camera('left_camera')
right_camera = Camera('right_camera')
left_camera.enable(timeStep)
right_camera.enable(timeStep)

lidar = Lidar('lidar')
lidar.enable(timeStep)
lidar.enablePointCloud()

ds0 = DistanceSensor("ds0")
ds1 = DistanceSensor("ds1")
dsf = DistanceSensor("dsf")
ds0.enable(timeStep)
ds1.enable(timeStep)
dsf.enable(timeStep)

depth_sensor = DistanceSensor("depth_sensor")
depth_sensor.enable(timeStep)

# The first frame is used to update all values from default
robot.step(timeStep)

curr_cloud = lidar.range_image[512 * _LIDAR_LAYER:512 * (_LIDAR_LAYER + 1):]
last_coords = getCoords()
last_positions = []

maze = Net(starting_coord=getCoords(), starting_cell=Cell(explored=True, c_t=_START))
addCells(last_coords)
maze.map[getCoords()].room = 1

########################################################################################################################
# Initialize movement

for node_angle in _MVMT_ANGLES:
    if maze.map[getNearCoords(node_angle)].available:
        maze.to_explore.add(getNearCoords(node_angle))

maze.checkToExplore()
chooseDirection(move=next_move)
start_simul_time = robot.getTime() 

#emergency: If the robot doesn't have enough time to return back it will send the map 2-3 seconds before the time runs out in order to get map bonus points
def emergency_exit():
    current_time = datetime.now()
    time_difference = current_time - starting_time
    time_difference_int = time_difference.seconds
    simul_time = robot.getTime() - start_simul_time
    if ((time_difference_int > 595 or simul_time > 478)): #and current_tile != starting_tile
        print ('Dont have enough time')
        print ('Real time:', time_difference_int)
        print ('Simulation time', simul_time)
        initiateEndSequence()
        delay(1000)

########################################################################################################################
# Main cycle
while robot.step(timeStep) != -1:
    ####################################################################################################################
    # Victim recognition with AI
    # Take a photo with the right camera if there's a wall and this camera hasn't found a victim in a while
    emergency_exit()
    #if compas() == 'south':
    #    print('SOUUUTHTHTHTHHTHTHT')
    a = maze.map[getCoords()]
    print(a)
    current_cell = maze.map.get(getCoords())  # Access the current cell from the map
    """
    if current_cell is not None:
        if victim_found == True:

            if left_camera_t == True:
            
                current_cell.addVictimsToMap(camera='left')  # Call the method to add victims to the current cell
            elif right_camera_t == True:
                current_cell.addVictimsToMap(camera='right')
            

    else:
        print("Error on victim placement in map")      """

    if not (*getCoords(), getRotation(), 'right') in prev_victims and not maze.map[getNearCoords(90)].available:
        right_camera_t = True
        right_camera.saveImage(r'C:\Users\Robotika\Desktop\Robotika\right_img.jpg', 100)
        right_img = cv2.imread(r'C:\Users\Robotika\Desktop\Robotika\right_img.jpg')
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d%H%M%S")
        cv2.imwrite(r'C:\Users\jakov\Documents\Slikenesto\R' + date_time_str + r'.jpg', right_img)

        if v_dir := Victim.detectVictim(img=right_img, camera=right_camera,
                                         is_turning=is_turning() or getRotation() % 2 == 1):
            prediction, message, grid_position, victim_found = Victim.predictionImg([r'C:\Users\Robotika\Desktop\Robotika\right_img.jpg'], "right")

            if prediction != None:
                message = f'{message} on the right at {getCoords()}'
                addToSeen(getCoords(), 'right', prediction, grid_position, victim_found)
                scoreVictim(prediction, 'right', v_dir, message)
    
    # Take a photo with the left camera if there's a wall and this camera hasn't found a victim in a while
    if not (*getCoords(), getRotation(), 'left') in prev_victims and not maze.map[getNearCoords(270)].available:
        left_camera_t = True
        left_camera.saveImage(r'C:\Users\Robotika\Desktop\Robotika\left_img.jpg', 100)
        left_img = cv2.imread(r'C:\Users\Robotika\Desktop\Robotika\left_img.jpg')
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d%H%M%S")
        cv2.imwrite(r'C:\Users\Robotika\Desktop\Robotika\Slikenesto\L' + date_time_str + r'.jpg', left_img)

        if v_dir := Victim.detectVictim(img=left_img, camera=left_camera,
                                         is_turning=is_turning() or getRotation() % 2 == 1):
            prediction, message, grid_position, victim_found = Victim.predictionImg([r'C:\Users\Robotika\Desktop\Robotika\left_img.jpg'], "left")
            
            #print(prediction, message)
            if prediction != None:
                message = f'{message} on the left at {getCoords()}'
                addToSeen(getCoords(), 'left', prediction, grid_position, victim_found)
                scoreVictim(prediction, 'left', v_dir, message)

    ####################################################################################################################
    # If every check fails, it just goes forward
    if curr_obj is not None and next_angle == -1:
        # turn_multi = [1, 1]
        # dynamic_turn(curr_obj, getCoords(divisor=1, round_to_int=False))
        speeds = new_speed()
    else:
        speeds = new_speed()

    ####################################################################################################################
    # If two seconds(or 64 frames) ago you were in the same position as now, it creates an obstacle
    last_positions.insert(0, (getCoords(divisor=1), getAngle()))
    if len(last_positions) > timeStep * 2:
        two_sec_ago_pos = last_positions.pop()
        if two_sec_ago_pos[0] == last_positions[0][0]:
            temp_prev_a = two_sec_ago_pos[1]
            temp_curr_a = last_positions[0][1]
            if (temp_prev_a - 5) % 360 > temp_prev_a or (temp_prev_a + 5) % 360 < temp_prev_a:
                temp_next_a = (temp_prev_a + 90) % 360
                temp_curr_a = (temp_curr_a + 90) % 360

            if temp_prev_a - 5 < temp_curr_a < temp_prev_a + 5:
                obstruction_found = True
                obstacle_coord = getNearCoords(from_coord=last_coords, angle=0)
                last_obstacle_coord = obstacle_coord
                maze += {obstacle_coord: Cell(cc=_TEMP_WALL)}
                maze.removeToExplore(obstacle_coord, getCoords())
                print(f'Stayed still: {temp_prev_a}, {temp_curr_a}')

    ####################################################################################################################
    # Get the lidar distances
    if turn_multi[0] >= 0 and turn_multi[1] >= 0:
        curr_cloud = lidar.range_image[512 * _LIDAR_LAYER:512 * (_LIDAR_LAYER + 1):]
        if getRotation() % 2 == 1:
            curr_cloud = [float('inf')] * 512
    else:
        curr_cloud = [float('inf')] * 512

    ####################################################################################################################
    # Check for the colors on the ground

    # Stores the color of the ground in front of the robot to do checks on it
    curr_color = getColor()
    #print (curr_color)

    # Check if the color is the one of an empty tile and check if there's a chance of hole being there
    if depth_sensor.getValue() > 0.2:
        # The sensor has to find a good value for 6 times to assert that there's a hole
        # - This is necessary due to a glitch that makes tunnels have the same sensor collisions as holes
        if not check_hole_depth >= 6:
            check_hole_depth += 1
        else:
            obstruction_found = True
            if getRotation() % 2 == 1:
                obstruction_found = placeAndHandleHole(hole_coordinate=getNearCoords(from_coord=last_coords, angle=0),
                                                       possible_offset=315, is_void=depth_sensor.getValue() > 0.75)
            else:
                obstruction_found = placeAndHandleHole(
                    hole_coordinate=getNearCoords(from_coord=last_coords, angle=0, distance=2),
                    possible_offset=270, is_void=depth_sensor.getValue() > 0.75)
            check_hole_depth = 0

    else:
        check_hole_depth = 0

    # Otherwise handle the colors as usual and reset the depth_sensor counter
    if not checkColor(curr_color, _EMPTY.color_range):
        if checkColor(curr_color, _HOLE.color_range):
            check_hole_depth = 0
            # The color black needs to be found 3 times to assert that there's a hole
            # - This is necessary since from time to time the sensor glitches out and gives a black pixel for no reason
            if not check_hole_color >= 3:
                check_hole_color += 1
            else:
                obstruction_found = True
                if getRotation() % 2 == 1:
                    placeAndHandleHole(hole_coordinate=getNearCoords(from_coord=last_coords, angle=0),
                                       possible_offset=45)
                else:
                    placeAndHandleHole(hole_coordinate=getNearCoords(from_coord=last_coords, angle=0, distance=2),
                                       possible_offset=90)
                check_hole_color = 0

        else:
            # If the color black is not found reset the color counter to 0 and check the other colors
            check_hole_color = 0

            # The chain of 'or's is useful since if one color is found the other ones are skipped
            checkAndPlaceTile(tile=_SWAMP, color=curr_color) or \
            checkAndPlaceTile(tile=_CHECKPOINT, color=curr_color) or \
            checkAndPlaceTunnel(tunnel=_BLUE, color=curr_color, color_name = "BLUE") or \
            checkAndPlaceTunnel(tunnel=_BLUE2, color=curr_color, color_name = "BLUE") or \
            checkAndPlaceTunnel(tunnel=_PURPLE, color=curr_color,  color_name = "PURPLE") or \
            checkAndPlaceTunnel(tunnel=_PURPLE2, color=curr_color, color_name = "PURPLE") or \
            checkAndPlaceTunnel(tunnel=_RED, color=curr_color, color_name = "RED") or \
            checkAndPlaceTunnel(tunnel=_RED2, color=curr_color, color_name = "RED") or \
            checkAndPlaceTunnel(tunnel=_GREEN, color=curr_color, color_name = "GREEN")or \
            checkAndPlaceTunnel(tunnel=_GREEN2, color=curr_color, color_name = "GREEN")or \
            checkAndPlaceTunnel(tunnel=_YELLOW, color=curr_color, color_name = "YELLOW") or \
            checkAndPlaceTunnel(tunnel=_YELLOW2, color=curr_color, color_name = "YELLOW") or \
            checkAndPlaceTunnel(tunnel=_ORANGE, color=curr_color, color_name = "ORANGE")
            checkAndPlaceTunnel(tunnel=_ORANGE2, color=curr_color, color_name = "ORANGE")

            if checkColor(color=curr_color, color_range=_GREEN.color_range):
                check_hole_depth = 0

    ####################################################################################################################
    # Check for obstacles dangerously near the robot

    if turn_multi[0] >= 0 and turn_multi[1] >= 0:
        # This is link that was used to search the correct angles:
        # https://www.desmos.com/calculator/ege55xeuim
        obstacle_search = lidar.range_image[512 * (_LIDAR_LAYER + 1) - 64:512 * (_LIDAR_LAYER + 1):] + \
                          lidar.range_image[512 * _LIDAR_LAYER:512 * _LIDAR_LAYER + 64:]

        if min(obstacle_search) < _DANGER_ZONE_CLOSENESS:
            obstruction_found = True
            obstacle_coord = getNearCoords(from_coord=last_coords, angle=0)
            last_obstacle_coord = obstacle_coord
            maze += {obstacle_coord: Cell(cc=_TEMP_WALL)}
            maze.removeToExplore(obstacle_coord, getCoords())

    ####################################################################################################################
    # Detect and adapt to LoP

    if receiver.getQueueLength() > 0:  # If receiver queue is not empty
        receivedData = receiver.getBytes()
        tup = struct.unpack('c', receivedData)  # Parse data into character
        if tup[0].decode("utf-8") == 'L':  # 'L' means lack of progress occurred
            print('LoP in the main cycle')
            turn_multi = [1, 1]
            speeds = new_speed(0)
            obstruction_found = False
            maze.updateToExplore(last_pos=None, curr_pos=None, next_to_explore={last_coords, last_obstacle_coord})
            curr_path = maze.getBestPath()
            chooseDirection()
        receiver.nextPacket()  # Discard the current data packet

    ####################################################################################################################
    # If it's in a new node, do things

    if abs(abs(abs(getCoords(round_to_int=False, divisor=1)[0]) - abs(last_coords[0]) * 6) - 6) < 0.5 or \
            abs(abs(abs(getCoords(round_to_int=False, divisor=1)[1]) - abs(last_coords[1]) * 6) - 6) < 0.5:
        curr_coords = getCoords()

        # Add the cell the robot is standing on and its close neighbours
        addCells(curr_coords)
        
        #print(f"Previous coords: {last_coords}, current coords: {curr_coords}")
        #print(type(maze.map[curr_coords]).__name__)
        if last_coords in _TUNNELS:
            previous_colour = _TUNNELS[last_coords]
            #print(f"Colour of previous tile: {previous_colour}")
            maze.changeCurrentRoom(previous_colour)

        if maze.map[curr_coords].explored == False:
            maze.map[curr_coords].explored = True
            maze.map[curr_coords].room = maze.current_room
            #print(f"Unexplored node: {curr_coords} and his room: {maze.map[curr_coords].room}")

        
        #print("-----------------")
        
        
#        else:
 #           print("------------------------")
 #           print(maze.map[curr_coords])
 #           pass

        
        maze.updateToExplore(last_pos=last_coords, curr_pos=curr_coords, next_to_explore=getLegalMoves())

        # Used to avoid bugs
        obstruction_found = False

        chooseDirection()

        last_coords = curr_coords
        #print(curr_coords)

        #print(maze)
        #print(f"{curr_coords} : {maze.map[curr_coords].room}")

        # The prints to get the state at the end of the 'movement' and 'choices' go below here
        # --- Turning is not considered a move, since the robot stays still at the same coordinate

    ####################################################################################################################
    # If the robot is turning, it will continue to do so until it finishes

    if (turn_multi[0] < 0) ^ (turn_multi[1] < 0):
        if static_turn() == 1 and getRotation() % 2 == 0:
            addCells(getCoords())
    else:
        next_angle = -1

    ####################################################################################################################
    # Otherwise if the robot is dangerously near to a wall/obstacle/hole/tunnel it will adapt & go back to the last node

    if obstruction_found:
        #print(f"Room before obstruction: {maze.current_room}")
        #print("Obstruction found")
        maze.current_room = maze.map[last_coords].room
        #print(f"Room after obstruction: {maze.current_room}")
        turn_multi = [-1, -1]

        dynamic_turn((last_coords[0] * 6, last_coords[1] * 6), getCoords(round_to_int=False, divisor=1))
        speeds = new_speed()

        if abs(abs(abs(getCoords(round_to_int=False, divisor=1)[0]) - abs(last_coords[0]) * 6)) < 1 and \
                abs(abs(abs(getCoords(round_to_int=False, divisor=1)[1]) - abs(last_coords[1]) * 6)) < 1:
            turn_multi = [1, 1]
            speeds = new_speed(0)
            curr_path = None
            obstruction_found = False
            chooseDirection()

    ####################################################################################################################
    # Select final speed of the wheels at the end of the frames

    motor_left.setVelocity(speeds[0])
    motor_right.setVelocity(speeds[1])
    

    #print(compas())

