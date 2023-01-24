import numpy as np
from mcts import MCTS
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import matplotlib
matplotlib.use('agg')

def run(chain, steps, roll_out, exploration_weight):
    mstc = MCTS(chain, roll_out, exploration_weight)
    trial = 0
    values = []
    chain = chain.upper()
    highest_reward = 0
    best_path = [[0,i] for i in range(len(chain))]
    while trial < steps:
        try:
            node_passby = (np.array([node_i.passby for node_i in mstc.tree])>1).sum()
            if node_passby == (3**(len(chain)-2) + 2):
                break
            trial_reward = mstc.run()
            best_path = [node_i.pos for node_i in mstc.path]
            if trial_reward > highest_reward:
                highest_reward = trial_reward
                best_path = [node_i.pos for node_i in mstc.path]

            values.append(trial_reward)
            print(f"path: {[node_i.pos for node_i in mstc.path]}")
            mstc.path = [mstc.path[0]]
            print(f"trial {trial} with trial_reward: {trial_reward} with tree length {len(mstc.tree)}")
            trial += 1
        except RuntimeError:
            print(f"Trapped in trial {trial}")
            mstc.path = [mstc.path[0]]
            trial += 1
    value = max(values)
    # print(value)
    return values, best_path

def plot_grid(path, values, steps, seq, e_weight, save_fig = True):
    x = [t[0] for t in path]
    y = [t[1] for t in path]
    str_seq = seq.upper()
    # assert len(str_seq) == info["chain_length"]
    # H_seq = [t[0] for t in labelled_conf if t[1] == 'H']
    # P_seq = [t[0] for t in labelled_conf if t[1] == 'P']

    # print("x: ", x)
    # print("y: ", y)
    # print("str_seq: ", str_seq)
    # print("H_seq: ", H_seq)
    # print("P_seq: ", P_seq)

    # Width, height in inches.
    fig_width = 5
    fig_height = 5
    # fontsize for legend
    fontsize = "xx-small"
    title_font = 12
    if len(str_seq) > 10:
        fig_width = 10
        fig_height = 10
        fontsize = "x-small"
        title_font = 15
    if len(str_seq) > 20:
        fig_width = 13
        fig_height = 13
        fontsize = "small"
        title_font = 18
    if len(str_seq) > 30:
        fig_width = 16
        fig_height = 16
        title_font = 21
    if len(str_seq) > 40:
        fig_width = 18
        fig_height = 18
        title_font = 24
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.subplots_adjust(top=0.9) # use a lower number to make more vertical space

    # set x and y limit and center origin
    max_xval = max(x)
    max_yval = max(y)
    ax.set_xlim(min(x)-1, max(x)+1)
    ax.set_ylim(min(y)-1, max(y)+1)

    # grid background
    ax.grid(linewidth=0.6, linestyle=':')

    # adjust plots with equal axis ratios
    #ax.axis('equal')
    ax.set_aspect('equal')  # , adjustable='box')

    # x and y axis tick at integer level
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # # figure title
    # # split the seq into chunks of chunk_size for the matplotlib title
    # # NOTE: here the chunks are fixed sequence length not chain length
    # chunks, chunk_size = info['seq_length'], 20
    # # print(f"chunks={chunks}, chunk_size={chunk_size}")
    # pad_len = info['seq_length'] - info['chain_length']
    for i in range(len(list(str_seq))-1):
        ax.plot(x[i:i+2],y[i:i+2],c='k',zorder=0)
    seq_len = len(str_seq)
    for res_index in range(seq_len):
        res = str_seq[res_index]
        if res == 'P':
            ax.scatter(x[res_index],y[res_index],facecolor='royalblue',edgecolor='k',zorder=5)
        else:
            ax.scatter(x[res_index],y[res_index],facecolor='orangered',edgecolor='k',zorder=5)
        for res_index_H in range (res_index, seq_len):
            res_H = str_seq[res_index_H]
            if res_H == 'H' and res == 'H' and abs(res_index_H - res_index) != 1 and (abs(x[res_index_H]-x[res_index])+abs(y[res_index_H]-y[res_index]) == 1):
                ax.plot([x[res_index],x[res_index_H]],[y[res_index],y[res_index_H]],linestyle=':',c='orangered',zorder=0)
    
    title = f"Seq: {str_seq} with Energy: {max(values)}"
    # print("Title: ", title)
    ax.set_title(title, fontsize=title_font)


    if save_fig:
        plt.savefig(os.getcwd()+'/'+str_seq[:6]+f'_n{steps}'+f'_v{len(values)}'+f'_w{e_weight}'+'_best_path.png',bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10,2))
    ax.plot(list(range(len(values))),-1*np.array(values))
    ax.set_yticks(list(range(-1*max(values)-2,2,2)))
    ax.yaxis.grid(True) 
    ax.set_xlabel("step",fontsize=14)
    ax.set_ylabel("Energy",fontsize=14)
    if save_fig:
        plt.savefig(os.getcwd()+'/'+str_seq[:6]+f'_n{steps}'+f'_v{len(values)}'+f'_w{e_weight}'+'_energies.png',bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running MCTS on HP lattice chain")
    parser.add_argument("-s", "--seq", type=str, help="HP lattice sequence")
    parser.add_argument("-n", "--steps", type=int, help="Number of trials")
    parser.add_argument("-r", "--roll_out", type=int, help="Number of roll_out")
    parser.add_argument("-w", "--exploration_weight", type=float, help="UCB exploration weight")
    args = parser.parse_args()
    value, path = run(args.seq, args.steps, args.roll_out, args.exploration_weight)
    plot_grid(path, value, args.steps, args.seq, args.exploration_weight)

