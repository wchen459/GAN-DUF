"""
Trains a BezierGAN, and visulizes results

Author(s): Wei Chen (wchen459@gmail.com)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from bezier_gan import GAN
from shape_plot import plot_grid

import sys
sys.path.append('..')
from utils import ElapsedTimer, create_dir, safe_remove, read_cfg


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('mode', type=str, default='train', help='train or test')
    args = parser.parse_args()
    
    config_path = 'config.ini'
    config = read_cfg(config_path, 'BezierGAN')
    
    # Read dataset
    nominal_data = np.load('data/airfoil_nominal.npy')
    deformed_data = np.load('data/airfoil_deformed.npy')
    N = nominal_data.shape[0]
    assert N == deformed_data.shape[0]
    n_points = nominal_data.shape[1]
    
    # Prepare save directory
    create_dir('./trained_model')
    save_dir = './trained_model/{}_{}'.format(config['parent_latent_dim'], config['child_latent_dim'])
    create_dir(save_dir)
    
    # Train
    model = GAN(config['parent_latent_dim'], config['child_latent_dim'], config['noise_dim'], n_points, config['bezier_degree'])
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
    plot_grid(5, gen_func=model.synthesize_parent, d=model.parent_latent_dim, bounds=(0., 1.),
              scale=.95, scatter=False, c='k', fname='{}/synthesized_parent'.format(save_dir))
    parent_latent = np.random.uniform(low=0., high=1., size=(1, model.parent_latent_dim))
    plot_grid(5, gen_func=lambda x: model.synthesize_child(parent_latent, x), d=model.child_latent_dim, bounds=(-0.5, 0.5),
              scale=.95, scatter=False, c='k', fname='{}/synthesized_child'.format(save_dir))
            
            
    
    
