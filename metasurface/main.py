"""
Trains a GAN, and visulizes results

Author(s): Wei Chen (wchen459@gmail.com)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from gan import GAN
from topology_plot import plot_grid, plot_random

import sys
sys.path.append('..')
from utils import ElapsedTimer, create_dir, safe_remove, read_cfg


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('mode', type=str, default='train', help='train or test')
    args = parser.parse_args()
    
    config_path = 'config.ini'
    config = read_cfg(config_path, 'GAN')
    
    # Read dataset
    nominal_data = np.load('data/metasurface_nominal.npy')
    deformed_data = np.load('data/metasurface_deformed.npy')
    N = nominal_data.shape[0]
    assert N == deformed_data.shape[0]
    rez = nominal_data.shape[1]
    
    # Prepare save directory
    create_dir('./trained_model')
    save_dir = './trained_model/{}_{}'.format(config['parent_latent_dim'], config['child_latent_dim'])
    create_dir(save_dir)
    
    # Train
    model = GAN(config['parent_latent_dim'], config['child_latent_dim'], config['noise_dim'], rez)
    if args.mode == 'train':
        safe_remove(save_dir)
        timer = ElapsedTimer()
        model.train(nominal_data, deformed_data, 
                    batch_size=config['batch_size'], train_steps=config['train_steps'], 
                    disc_lr=config['disc_lr'], gen_lr=config['gen_lr'], 
                    save_interval=config['save_interval'], save_dir=save_dir)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
        runtime_file = open('{}/runtime.txt'.format(save_dir), 'w')
        runtime_file.write('%s\n' % runtime_mesg)
        runtime_file.close()
    else:
        model.restore(directory=save_dir)
    
    print('Plotting synthesized shapes ...')
    plot_grid(8, gen_func=model.synthesize_parent, d=model.parent_latent_dim, bounds=(0., 1.),
              fname='{}/synthesized_parent'.format(save_dir))
    plot_random(8, gen_func=model.synthesize_parent, d=model.parent_latent_dim, bounds=(0., 1.),
                fname='{}/synthesized_parent_random'.format(save_dir))
    parent_latent = np.random.uniform(low=0., high=1., size=(1, model.parent_latent_dim))
    plot_grid(8, gen_func=lambda x: model.synthesize_child(parent_latent, x, return_parent=True), 
              d=model.child_latent_dim, bounds=(-0.5, 0.5), 
              fname='{}/synthesized_child'.format(save_dir))
    plot_random(8, gen_func=lambda x: model.synthesize_child(parent_latent, x, return_parent=True), 
                d=model.child_latent_dim, bounds=(-0.5, 0.5), 
                fname='{}/synthesized_child_random'.format(save_dir))
    
    
    # from tqdm import tqdm
    # from scipy.stats import wasserstein_distance
    # import seaborn as sns
    
    # from simulation import evaluate
    # from build_data import deform
    
    # print('Comparing the performance distribution of fabricated designs with the ground truth ...')
    # n_parent = 10
    # n_child_per_parent = 100
    # parent_latent = np.random.uniform(low=0., high=1., size=(n_parent, model.parent_latent_dim))
    # parent_designs = model.synthesize_parent(parent_latent)
    # list_wd = []
    # for i in range(n_parent):
    #     nominal_perf = evaluate(parent_designs[i])
    #     print('Comparing nominal design #{}/{}'.format(i+1, n_parent))
    #     # Generated manufactured designs
    #     child_designs = model.synthesize_child(parent_latent[i], n_child_per_parent)
    #     perfs = []
    #     for child_design in tqdm(child_designs):
    #         perf = evaluate(child_design)
    #         perfs.append(perf)
    #     # True manufactured designs
    #     real_perfs = []
    #     for _ in tqdm(range(n_child_per_parent)):
    #         deformed_design = deform(parent_designs[i])
    #         real_perf = evaluate(deformed_design)
    #         real_perfs.append(real_perf)
            
    #     plt.figure()
    #     plt.subplot(211)
    #     plt.plot(parent_designs[i,:,0], parent_designs[i,:,1], '-')
    #     plt.axis('equal')
    #     plt.subplot(212)
    #     sns.kdeplot(real_perfs, linestyle='--', color='#7be276', shade=True, Label='Real')
    #     sns.kdeplot(perfs, linestyle='-', color='#63b1ed', shade=True, Label='Generated')
    #     plt.scatter(real_perfs, np.zeros(len(real_perfs)), c='#7be276', alpha=0.3, marker='o', edgecolors='none')
    #     plt.scatter(perfs, np.zeros(len(perfs)), c='#63b1ed', alpha=0.3, marker='^', edgecolors='none')
    #     plt.axvline(x=nominal_perf, linestyle='--', color='#7be276')
    #     plt.xlabel(r'$C_L/C_D$') 
    #     plt.ylabel('Probability density')
    #     plt.legend(frameon=False)
    #     plt.tight_layout()
    #     plt.savefig('{}/perf_distributions_{}.svg'.format(save_dir, i))
    #     plt.close()
        
    #     wd = wasserstein_distance(perfs, real_perfs)
    #     list_wd.append(wd)
        
    # np.savetxt('{}/wasserstein_distance.csv'.format(save_dir), list_wd, delimiter="\t")
