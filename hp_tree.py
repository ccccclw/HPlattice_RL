import numpy as np

actions = ['L','F','R']

class Node:

    def __init__(self, pos, parent_num, action, level, passby, reward, pure_reward, aa_type):
        self.pos = pos
        self.action = action
        self.parent_num = parent_num
        self.level = level
        self.reward = reward
        self.pure_reward = pure_reward
        self.type = aa_type
        self.passby = passby
        self.children = []
        if self.level == 0 or self.level == 1:
            self.row_num = 0
        else:
            self.row_num = self.parent_num * 3 + actions.index(self.action)

    def is_terminal(self):
        if self.level == len(self.chain):
            return True
        else:
            return False

    def is_root(self):
        if self.level == 0:
            return True
        else:
            return False

    def add_child(self,node):
        self.children.append(node)



def init_tree(chain):
    tree = []
    tree.append(Node([0,0], 0, 'F', 0, 1, 0, 0, chain[0]))
    tree.append(Node([0,1], 0, 'F', 1, 1, 0, 0, chain[1]))
    tree[0].add_child(tree[1])
    return tree
    