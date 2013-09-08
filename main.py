import logging
import numpy as np
import os, sys
import pygame
from operator import itemgetter
from pygame.locals import *

INITIAL_SHAPE_SIZE = 3**3
WINDOW_SIZE = 3**4

class StaticPattern(pygame.sprite.Sprite):
  # StaticPatterns don't evolve themselves. They are meant to be statically
  # rendered and then thrown away.
  def __init__(self, rect, pattern):
    pygame.sprite.Sprite.__init__(self)
    self.rect = rect
    logging.error("Spawning static pattern at %s", rect)
    self._current = pattern
    self._size = rect.width
    self.image = pygame.Surface((self._size, self._size))
    pygame.surfarray.blit_array(self.image,
                                Pattern.PatternToImage(self._current))

  def update(self):
    pass


class Pattern(object):
  # Patterns are 2D arrays of uint8 with helpful utility methods.
  def __init__(self, width, height):
    self._width = width
    self._height = height
    self._children = []
    # Set the ideal to be an ideal shape array.
    self._ideal = Pattern.IdealDiamond(self._width, self._height)
    self._current = Pattern.RandomPattern(self._width, self._height)
    self._previous_bests = []

  def _CreateBestChild(self, ideal, current):
    '''Returns a new best child, or None if no such child evolved.'''
    new_patterns = []

    # Generate purely random children.
    for x in xrange(2):
      new_patterns.append(Pattern.RandomPattern(self._width, self._height))

    # Generate random morphlings of the current best.
    for x in xrange(4):
      random_pattern = Pattern.RandomPattern(self._width, self._height)
      morphling = Pattern.CombinePatterns(current, random_pattern, 0.99)
      new_patterns.append(morphling)

    # Combine the previous bests with current best and randomness.
    for previous_best in self._previous_bests[:8]:
      new_patterns.append(Pattern.CombinePatterns(current, previous_best, 0.9))
      new_patterns.append(Pattern.CombinePatterns(
          previous_best, Pattern.RandomPattern(self._width, self._height), 0.6))

    # Score the children and sort them.
    scored_patterns = [(Pattern.ScoreSimilarity(ideal, pattern), pattern)
                       for pattern in new_patterns]

    # Sort by descending score and take the winner. Only take the 0'th element
    # (the score) because when scores are tied patterns will be used as tie
    # breakers, which numpy doesn't like.
    scored_patterns.sort(key=itemgetter(0), reverse=True)
    best_score, best_pattern = scored_patterns[0]

    current_score = Pattern.ScoreSimilarity(ideal, current)
    if best_score > current_score:
      logging.info("New best score: %d", best_score)
      self._previous_bests = self._previous_bests[1:8]
      self._previous_bests.append(best_pattern)
      return True, best_pattern  # 2d array "Truth" is ambiguous
    return False, current

  def _IsIdeal(self):
    return np.array_equal(self._ideal, self._current)

  def Evolve(self):
    if self._IsIdeal():
      return  # Steady state - we have no more evolving to do.

    changed, best_child = self._CreateBestChild(self._ideal, self._current)
    if changed:
      self._current = best_child
    return changed

  def ToImage(self):
    return np.dstack((self._current, self._current, self._current))

  @staticmethod
  def FindEdgePoints(pattern):
    square = np.uint8(np.zeros(pattern.shape))
    square[0,] = True  # Top
    square[-1,] = True  # Bottom
    square[:,0] = True  # Left
    square[:,-1] = True  # Right
    np.logical_and(square, pattern)
    return np.transpose(np.nonzero(np.logical_and(square, pattern)))

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


class PatternSprite(pygame.sprite.Sprite):
  # Patterns should always be stored as 2D arrays.
  # Convert to 3D only when displaying and only when necessary.
  def __init__(self, width, height):
    pygame.sprite.Sprite.__init__(self)
    self._pattern = Pattern(width, height)
    # Pygame Sprite variables.
    self.rect = pygame.Rect(WINDOW_SIZE / 2 - width / 2,
                            WINDOW_SIZE / 2 - height / 2,
                            width, height)
    # Set the initial image to be a surface of a random pattern.
    self.image = pygame.Surface((width, height))

  def Redraw(self):
    pygame.surfarray.blit_array(self.image, self._pattern.ToImage())

  def update(self):
    if self._pattern.Evolve():
      self.Redraw()
      # Redraw children too!
    else:
      logging.info("Generation not better than previous best.")


    # if redraw:
    #   child_spawns = Pattern.FindChildSpawns(self._current)
    #   # Provide a flipped view of the current array to the children.
    #   child_pattern = np.flipud(np.fliplr(self._current))
    #   # Clear out the list of current children, garbage collecting them.
    #   for child in self._children:
    #     child.kill()
    #   self._children = []
    #   for (x, y) in child_spawns:
    #     child = StaticPattern(self.rect.move(x, y), child_pattern)
    #     child.add(self.groups()[0])
    #     self._children.append(child)
            



def Main():
  pygame.init()
  logging.basicConfig(level=logging.INFO)
  screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
  pygame.display.set_caption('Experimental')

  background = pygame.Surface(screen.get_size())
  background = background.convert()
  background.fill((0, 0, 0))

  # SUBSURFACE :)

  pattern = PatternSprite(INITIAL_SHAPE_SIZE, INITIAL_SHAPE_SIZE)
  allsprites = pygame.sprite.RenderPlain((pattern))
  clock = pygame.time.Clock()

  #for steps in xrange(10000):
  while True:
    #clock.tick(60)

    allsprites.update()

    screen.blit(background, (0, 0))
    allsprites.draw(screen)
    pygame.display.flip()

    for event in pygame.event.get():
      if event.type == QUIT:
        return
      elif event.type == KEYDOWN and event.key == K_ESCAPE:
        return

if __name__=="__main__":
  Main()
  pygame.quit()
 
