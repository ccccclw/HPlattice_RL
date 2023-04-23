import numpy as np
from mcts import MCTS
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import matplotlib
import glob
import imageio
import shutil
matplotlib.use('agg')

def run(chain, steps, exploration_weight):
    chain = chain.upper()
    assert list(set(chain)) == ['H','P'], "Not a valid HP chain."
    assert type(steps) == int and steps > 0 , "Steps should be positive integer."
    assert exploration_weight >= 0, "Exploration weight is less than 0."
    mstc = MCTS(chain, exploration_weight)
    trial = 0
    values = []
    highest_reward = 0
    best_path = [[0,i] for i in range(len(chain))]
    all_paths = []
    print(f"Input chain with length {len(chain)}: {chain}")
    while trial < steps:
        try:
            node_passby = (np.array([node_i.passby for node_i in mstc.tree])>1).sum()
            if node_passby == (3**(len(chain)-2) + 2):
                break
            trial_reward = mstc.run()
            current_path = [node_i.pos for node_i in mstc.path]
            all_paths.append(current_path)
            values.append(trial_reward)
            if trial_reward > highest_reward:
                highest_reward = trial_reward
                best_path = current_path
            print(f"path: {[node_i.pos for node_i in mstc.path]}")
            mstc.path = [mstc.path[0]]
            print(f"trial {trial} with trial_reward: {trial_reward} with tree length {len(mstc.tree)}")
            trial += 1
            
        except RuntimeError:
            print(f"Trapped in trial {trial}")
            mstc.path = [mstc.path[0]]
            trial += 1
    values = np.array(values)
    best_paths = np.array(all_paths)[np.where(np.array(values)==max(values))]
    best_paths = np.unique(best_paths,axis=0).astype(np.int32)
    # print(value)
    return values, best_path, best_paths

def plot_grid(path, values, steps, seq, e_weight, path_index, every_step = False, save_fig = True):
    str_seq = seq.upper()
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
    x = [t[0] for t in path]
    y = [t[1] for t in path]

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
    global step
    step = steps
    def plot_step(str_seq, x, y, ax, frame=False):
        global step
        if frame:
            step = len(str_seq)
        for i in range(len(list(str_seq))-1):
            ax.plot(x[i:i+2],y[i:i+2],c='k',zorder=0)
        seq_len = len(str_seq)
        energy = 0
        for res_index in range(seq_len):
            res = str_seq[res_index]
            if res == 'P':
                ax.scatter(x[res_index],y[res_index],facecolor='royalblue',edgecolor='k',zorder=5)
            else:
                ax.scatter(x[res_index],y[res_index],facecolor='orangered',edgecolor='k',zorder=5)
            for res_index_H in range (res_index, seq_len):
                res_H = str_seq[res_index_H]
                if res_H == 'H' and res == 'H' and abs(res_index_H - res_index) != 1 and (abs(x[res_index_H]-x[res_index])+abs(y[res_index_H]-y[res_index]) == 1):
                    energy += 1
                    ax.plot([x[res_index],x[res_index_H]],[y[res_index],y[res_index_H]],linestyle=':',c='orangered',zorder=0)
        
        title = f"Seq: {str_seq} with Energy: {energy}"
        # print("Title: ", title)
        ax.set_title(title, fontsize=title_font)


        if save_fig:
            plt.savefig(os.getcwd()+'/'+str_seq[:6]+f'_n{step}'+f'_v{len(values)}'+f'_w{e_weight}'+f'_best_path_E{max(values)}_{path_index}.png',bbox_inches='tight')

    if every_step:
        for i in range(len(str_seq)):
            plot_step(str_seq[:i+1],x,y,ax,frame=True)
    else:
        plot_step(str_seq,x,y,ax)

    fig, ax = plt.subplots(figsize=(10,2))
    ax.plot(list(range(len(values))),-1*np.array(values))
    ax.set_yticks(list(range(-1*max(values)-2,2,2)))
    ax.yaxis.grid(True) 
    ax.set_xlabel("step",fontsize=14)
    ax.set_ylabel("Energy",fontsize=14)
    if save_fig and not every_step:
        plt.savefig(os.getcwd()+'/'+str_seq[:6]+f'_n{step}'+f'_v{len(values)}'+f'_w{e_weight}'+f'_energies_E{max(values)}_{path_index}.png',bbox_inches='tight')

def generate_gif(path, values, steps, seq, e_weight, save_fig = True):
    frames = []
    time = list(range(0,len(path)))
    # for i in range(1,len(path)+1):
    plot_grid(path, values,steps,seq, e_weight, 0, every_step=True,save_fig=True)
    all_es = [glob.glob(os.getcwd()+f"/"+f"*_n{i}_*best_path*")[0] for i in range(1,len(path)+1)]
    for i,t in enumerate(all_es):
        image = imageio.v3.imread(t)
        frames.append(image)
        print(glob.glob(os.getcwd()+f"/"+f"*_n{i+1}_*")[0])
        os.remove(glob.glob(os.getcwd()+f"/"+f"*_n{i+1}_*")[0])
    imageio.mimsave(os.getcwd()+'/energies.gif', # output gif
                frames,          # array of input frames
                fps = 2)         # optional: frames per second

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running MCTS on HP lattice chain")
    parser.add_argument("-s", "--seq", type=str, help="HP lattice sequence")
    parser.add_argument("-n", "--steps", type=int, help="Number of trials")
    parser.add_argument("-w", "--exploration_weight", type=float, help="UCB exploration weight")
    args = parser.parse_args()
    value, path, paths = run(args.seq, args.steps, args.exploration_weight)
    cwd = os.getcwd()
    save_path = os.getcwd()+f'/results_{args.seq.upper()[:6]}_{args.steps}_{args.exploration_weight}'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(os.getcwd()+f'/results_{args.seq.upper()[:6]}_{args.steps}_{args.exploration_weight}')
    os.chdir(os.getcwd()+f'/results_{args.seq.upper()[:6]}_{args.steps}_{args.exploration_weight}')
    for i,path_i in enumerate(paths):
        generate_gif(path_i, value, args.steps, args.seq, args.exploration_weight)
        os.replace(os.getcwd()+'/energies.gif',os.getcwd()+f'/energies_{i}.gif')
        plot_grid(path_i, value, args.steps, args.seq, args.exploration_weight,i)
    os.chdir(cwd)

