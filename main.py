from collections import deque
import logging
import numpy as np
import os, sys
import pygame
from operator import itemgetter
from pygame.locals import *

SHAPE_SIZE = 3**3 * 2
NUM_PREVIOUS_GENERATIONS = 40
#WINDOW_SIZE = 3**4


class Pattern(object):
  # Patterns are 2D arrays of uint8 with helpful utility methods.
  def __init__(self, width, height, ideal, current):
    self._width = width
    self._height = height
    self._ideal = ideal
    self._current = current

  def _CreateBestChild(self, ideal, current):
    '''Returns a new best child, or None if no such child evolved.'''
    new_patterns = []

    # Generate purely random children.
    for x in xrange(16):
      new_patterns.append(Pattern.RandomPattern(self._width, self._height))

    # Generate random morphlings of the current best.
    for x in xrange(32):
      random_pattern = Pattern.RandomPattern(self._width, self._height)
      morphling = Pattern.CombinePatterns(current, random_pattern, 0.99)
      new_patterns.append(morphling)

    # Combine the previous bests with current best and randomness.
    # for previous_best in self._previous_bests[:NUM_PREVIOUS_GENERATIONS]:
    #   new_patterns.append(Pattern.CombinePatterns(current, previous_best, 0.9))
    #   new_patterns.append(Pattern.CombinePatterns(
    #       previous_best, Pattern.RandomPattern(self._width, self._height), 0.6))

    # Score the children and sort them.
    scored_patterns = [(Pattern.ScoreSimilarity(ideal, pattern), pattern)
                       for pattern in new_patterns]

    # Sort by descending score and take the winner. Only take the 0'th element
    # (the score) because othewise when scores are tied patterns will be used
    # as tie breakers, which numpy doesn't like.
    scored_patterns.sort(key=itemgetter(0), reverse=True)
    best_score, best_pattern = scored_patterns[0]

    current_score = Pattern.ScoreSimilarity(ideal, current)
    if best_score > current_score:
      logging.info("New best score: %d", best_score)
      return True, best_pattern  # 2d array "Truth" is ambiguous
    return False, current

  def IsIdeal(self):
    return np.array_equal(self._ideal, self._current)

  def Evolve(self):
    changed, best_child = self._CreateBestChild(self._ideal, self._current)
    # As we restructure, we ignore the "changed" status returned.
    return changed, Pattern(self._width, self._height, self._ideal, best_child)

  def ToImage(self):
    return np.dstack((self._current, self._current, self._current))

  def DrawOnSurface(self, surface):
    pygame.surfarray.blit_array(surface, self.ToImage())

  def GetEdgePoints(self):
    square = np.uint8(np.zeros(self._current.shape))
    square[0,] = True  # Top
    square[-1,] = True  # Bottom
    square[:,0] = True  # Left
    square[:,-1] = True  # Right
    return np.transpose(np.nonzero(np.logical_and(square, self._current.shape)))

  @staticmethod
  def FindChildSpawns(pattern):
    edge_points = Pattern.FindEdgePoints(pattern)
    height, width = pattern.shape
    # for edge point (x, y)
    #   child x = -1 * width + 2x
    #   child y = -1 * height + 2y
    return [(-1 * width + 2 * x + 1, -1 * height + 2 * y + 1)
            for (x, y) in edge_points]

  @staticmethod
  def ScoreSimilarity(ideal, pattern):
    inverted_ideal = np.invert(ideal)
    inverted_pattern = np.invert(pattern)
    # I DON'T THINK THIS IS RIGHT.
    white = np.sum(np.bitwise_and(ideal, pattern))
    black = np.sum(np.bitwise_and(inverted_ideal, inverted_pattern))
    return white + black                           

  @staticmethod
  def AddPatterns(left, right):
    # Returns a pattern that is the sum 
    pass
    
  @staticmethod
  def CombinePatterns(left, right, weight_left):
    # To probabilistically choose between left mask and right mask for each
    # pixel, we generate a random mask with a threshold so that a value of
    # 255 (white) indicates we should take a pixel from the left pattern and
    # a value of 0 (black) indicates we should take a pixel from the right.
    left_mask = Pattern.RandomPattern(width=left.shape[0],
                                      height=left.shape[1],
                                      threshold=1-weight_left)
    right_mask = np.invert(left_mask)
    return np.add(np.bitwise_and(left_mask, left),
                  np.bitwise_and(right_mask, right))

  @staticmethod
  def IdealDiamond(width, height):
    array = np.zeros((width, height), dtype=np.uint8)
    # TODO(ebensh): Optimize diamond generation.
    # Mx, My for clarity of dimensions for non-square.
    Mx = width
    My = height
    WHITE = 255
    for x in xrange(Mx / 2):
      array[x, (My / 2) - x] = WHITE  # Top left
      array[x, (My / 2) + x] = WHITE  # Bottom left
    for x in xrange(Mx / 2, Mx):
      array[x, x - (Mx / 2)] = WHITE  # Top right
      array[x, My - 1 - (x - (Mx / 2))] = WHITE  # Bottom right
    return array

  @staticmethod
  def IdealPlus(width, height):
    array = np.zeros((width, height), dtype=np.uint8)
    # TODO(ebensh): Optimize plus generation.
    # Mx, My for clarity of dimensions for non-square.
    Mx = width
    My = height
    WHITE = 255
    for x in xrange(Mx):
      array[x, (My / 2)] = WHITE  # Horizontal line
    for y in xrange(My):
      array[Mx / 2, y] = WHITE  # Vertical line
    return array


  @staticmethod
  def RandomPattern(width, height, threshold=0.5):
    MAX = 256  # Black == 0, White == 255
    threshold = int(threshold * MAX)
    pattern = np.uint8(np.random.randint(256, size=(width, height)))
    # Convert to black and white
    pattern[pattern < threshold] = 0
    pattern[pattern >= threshold] = 255
    return pattern


def Main():
  pygame.init()
  logging.basicConfig(level=logging.INFO)
  screen = pygame.display.set_mode((SHAPE_SIZE * NUM_PREVIOUS_GENERATIONS,
                                    SHAPE_SIZE))
  pygame.display.set_caption('Experimental')

  background = pygame.Surface(screen.get_size())
  background = background.convert()
  background.fill((0, 0, 0))

  # SUBSURFACE :)

  patterns = deque([], maxlen=NUM_PREVIOUS_GENERATIONS)
  patterns.appendleft(Pattern(SHAPE_SIZE, SHAPE_SIZE,
                              Pattern.IdealDiamond(SHAPE_SIZE,
                                                   SHAPE_SIZE),
                              Pattern.RandomPattern(SHAPE_SIZE,
                                                    SHAPE_SIZE)))
  clock = pygame.time.Clock()

  while True:
    clock.tick(60)

    # Update
    pattern = patterns[0]  # Start with the last generation
    #if not pattern.IsIdeal():
    changed, new_pattern = pattern.Evolve()
    if changed:
      patterns.appendleft(new_pattern)

      # Draw
      screen.blit(background, (0, 0))
      for i, pattern in enumerate(patterns):
        subsurface = screen.subsurface(Rect(i * SHAPE_SIZE, 0,
                                            SHAPE_SIZE, SHAPE_SIZE))
        pattern.DrawOnSurface(subsurface)
      pygame.display.flip()

    for event in pygame.event.get():
      if event.type == QUIT:
        return
      elif event.type == KEYDOWN and event.key == K_ESCAPE:
        return

if __name__=="__main__":
  Main()
  pygame.quit()

