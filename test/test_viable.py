import unittest
import sys
sys.path.append('/orange/alberto.perezant/liweichang/dev/tmp_test/mcts_hplattice/mcts/')
from run import run

class TestInput(unittest.TestCase):
  def test_chain(self):
    chain = 'hphphphxx'

    with self.assertRaises(AssertionError):
        run(chain, 10, 0.5)

  def test_step(self):
    steps = 5.0

    with self.assertRaises(AssertionError):
        run('hphp', steps, 0.5)

    steps = -1

    with self.assertRaises(AssertionError):
        run('hphp', steps, 0.5)

  def test_weight(self):
    weight = -3

    with self.assertRaises(AssertionError):
        run('hphp', 10, weight)

    
