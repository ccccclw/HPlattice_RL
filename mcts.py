import numpy as np
from collections import defaultdict
from hp_tree import init_tree
from hp_tree import Node
import random

class MCTS:

    def __init__(self, chain : str, exploration_weight: float):
        self.tree = init_tree(chain)
        self.chain = chain
        self.all_pos = [node.pos for node in self.tree]
        self.path = [self.tree[0]]
        self.exploration_weight = exploration_weight

    def run(self):
        self.select(self.tree[1])
        treelen_a = len(self.tree)
        leaf = self.path[-1]
        if leaf.level < len(self.chain)-1:
            self.expand(leaf)
            treelen_b = len(self.tree)
            reward = self.simulate(treelen_b-treelen_a)
        else:
            reward = leaf.pure_reward
        self.update_reward(reward)
        return reward

    def select(self, node):
        # path = []
        while True:
            self.path.append(node)
            # print("node_pos: ", node.pos,"chilren: ",[node_i.pos for node_i in node.children])
            if len(node.children) == 0:
                break
            elif len(node.children) <= 3:
                node = self._uct_select(node) 
            # node = self.path.append(new_node)
            # else:
            #     # n = node
            #     # self.path.append(n)
            #     new_node = self._uct_select(node) 
            #     self.path.append(new_node)
            # break

    def expand(self, node, random_action=False):
        parent_pos = node.pos
        parent_action = node.action
        new_nodes = []
        new_pos = [[parent_pos[0],parent_pos[1]+1],
                   [parent_pos[0]+1,parent_pos[1]],
                   [parent_pos[0],parent_pos[1]-1],
                   [parent_pos[0]-1,parent_pos[1]]]
        # print(f"parent_pos {node.pos}, {new_pos}, current path: {[node_i.pos for node_i in self.path]}")

        new_pos.remove(self.path[-2].pos)
        # print(f"parent_pos {node.pos}, {new_pos}, current path: {[node_i.pos for node_i in self.path]}")
        tmp_rewards = [self.reward(new_pos[i], node) for i in range(3)]
        tmp_pure_rewards = [self.reward(new_pos[i], node, added_reward=False) for i in range(3)]
        tmp_path_pos = [node_i.pos for node_i in self.path]

        for action_index in range(3):
            if new_pos[action_index] not in tmp_path_pos:
                parent_pos_diff = np.array(new_pos[action_index]) - np.array(node.parent_pos)
                pos_diff = np.array(new_pos[action_index]) - np.array(node.pos)
                #Determine the direction of move
                if 0 in parent_pos_diff:
                    action = 'F'
                else:
                    if parent_pos_diff[0] == 1 and parent_pos_diff[1] == 1:
                        if pos_diff[0] == 1 and pos_diff[1] == 0:
                            action = 'R'
                        elif pos_diff[0] == 0 and pos_diff[1] == 1:
                            action = 'L'
                    if parent_pos_diff[0] == -1 and parent_pos_diff[1] == -1:
                        if pos_diff[0] == -1 and pos_diff[1] == 0:
                            action = 'R'
                        elif pos_diff[0] == 0 and pos_diff[1] == -1:
                            action = 'L'
                    if parent_pos_diff[0] == -1 and parent_pos_diff[1] == 1:
                        if pos_diff[0] == 0 and pos_diff[1] == 1:
                            action = 'R'
                        elif pos_diff[0] == -1 and pos_diff[1] == 0:
                            action = 'L'
                    if parent_pos_diff[0] == 1 and parent_pos_diff[1] == -1:
                        if pos_diff[0] == 0 and pos_diff[1] == -1:
                            action = 'R'
                        elif pos_diff[0] == 1 and pos_diff[1] == 0:
                            action = 'L'

                new_node = Node(new_pos[action_index],node.row_num,node.pos,action,node.level+1,1,tmp_rewards[action_index],tmp_pure_rewards[action_index],self.chain[node.level+1])
                node.add_child(new_node)
                self.tree.append(new_node)
                self.all_pos.append(new_pos[action_index])
                new_nodes.append(new_node)
            else:
                pass
                # print("node in path: ", new_pos[action_index], node.action)
        if len(new_nodes) == 0:
            node.reward = 0
            node.pure_reward = 0
            self.update_penalty()
            raise RuntimeError("Trapped.")
        # print(f"new_node_len: {node.pos, len(new_nodes), [node_i.pos for node_i in new_nodes]}")
        return new_nodes


    def reward(self, pos, node, added_reward=True):
        tmp_all_pos = np.array([tmp_node.pos for tmp_node in self.path[:-1]])
        # print("current path: ", tmp_all_pos, node.pos, node.level)
        pos_diff = np.sum(np.abs(np.array(pos) - tmp_all_pos),axis=1)
        if added_reward:
            if self.chain[node.level+1] == 'P':
                return node.reward
            else:
                return np.sum(np.array(list(self.chain)[:node.level])[pos_diff==1] == 'H') + node.reward
        else:
            if self.chain[node.level+1] == 'P':
                return node.pure_reward
            else:
                return np.sum(np.array(list(self.chain)[:node.level])[pos_diff==1] == 'H') + node.pure_reward


    def simulate(self, new_node):
        if new_node == 0:
            random_node = self.tree[-1]
        else:
            random_node = random.choice(self.tree[-1*new_node:])
        self.path.append(random_node)
        # print("new_random: ", random_node.level, random_node.pos)
        while random_node.level < len(self.chain)-1:
            # print("add_new_random: ", random_node.level, random_node.pos)
            new_nodes = self.expand(random_node)
            random_node = random.choice(new_nodes)
            self.path.append(random_node)

        return random_node.pure_reward
            # node = node.find_random_child()
        
    def update_reward(self, reward):
        for node_i in reversed(self.path):
            node_i.reward += reward
            node_i.passby += 1
        
    def update_penalty(self):
        for node_i in reversed(self.path):
            node_i.reward -= 1

    def _uct_select(self, node):
        # assert all(n in self.children for n in self.children[node])
        log_N_parent = np.log(node.passby)

        def uct(node):
            "Upper confidence bound for trees"
            return node.reward / node.passby + self.exploration_weight * np.sqrt(
                log_N_parent / node.passby
            )
        def uct_bk(node):
            reward_w = [node.reward / node.passby + i * np.sqrt(log_N_parent / node.passby) for i in np.arange(1,20,0.1)]
            return reward_w
        
        # print("rewards: ",[uct(node_i) for node_i in node.children], [node_i.passby for node_i in node.children], "max: ", max(node.children, key=uct).pos, max(node.children, key=uct).passby, node.reward)
        random.shuffle(node.children)
        # reward_ws = np.argmax(np.array([uct(node_i) for node_i in node.children]),axis=0)
        # reward_ws_counter = np.argmax([(reward_ws == i).sum() for i in range(len(node.children))])
        # print("reward_ws_counter: ",reward_ws_counter)
        return max(node.children, key=uct)
        # return node.children[int(reward_ws_counter)]