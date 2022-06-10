"""
GAN

Author(s): Wei Chen (wchen459@gmail.com)
"""

import time
import numpy as np
import tensorflow as tf

from topology_plot import plot_grid, plot_random


EPSILON = 1e-7

        
class GAN(object):
    
    def __init__(self, parent_latent_dim=5, child_latent_dim=2, noise_dim=10, rez=64):

        self.parent_latent_dim = parent_latent_dim
        self.child_latent_dim = child_latent_dim
        self.noise_dim = noise_dim
        
        self.rez = rez
        
    def generator(self, parent_c, child_c, z, reuse=tf.AUTO_REUSE):
        
        depth = 32*32
        dim = int(self.rez/32)
        kernel_size = (5,5)
        strides = (2,2)
        
        with tf.variable_scope('Generator', reuse=reuse):
                
            cz = tf.concat([parent_c, child_c, z], axis=-1)
            
            x = tf.layers.dense(cz, 1024)
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
    
            x = tf.layers.dense(x, dim*dim*depth)
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.reshape(x, (-1, dim, dim, depth))
    
            x = tf.layers.conv2d_transpose(x, int(depth/2), kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = tf.layers.conv2d_transpose(x, int(depth/4), kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = tf.layers.conv2d_transpose(x, int(depth/8), kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = tf.layers.conv2d_transpose(x, int(depth/16), kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = tf.layers.conv2d_transpose(x, int(depth/32), kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = tf.layers.conv2d_transpose(x, 1, kernel_size, padding='same')
            x = tf.nn.tanh(x, name='child_fake')
            
            return x
        
    def discriminator(self, parent_x, child_x, reuse=tf.AUTO_REUSE):
        
        depth = 32
        dropout = 0.4
        kernel_size = (5,5)
        strides = (2,2)
        
        with tf.variable_scope('Discriminator', reuse=reuse):
                
            x = tf.concat([parent_x, child_x], axis=-1)
        
            x = tf.layers.conv2d(x, depth*1, kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=True)
            
            x = tf.layers.conv2d(x, depth*2, kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=True)
            
            x = tf.layers.conv2d(x, depth*4, kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=True)
            
            x = tf.layers.conv2d(x, depth*8, kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=True)
            
            x = tf.layers.conv2d(x, depth*16, kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=True)
            
            x = tf.layers.conv2d(x, depth*32, kernel_size, strides=strides, padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=True)
            
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 1024)
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            d = tf.layers.dense(x, 1)
            
            q = tf.layers.dense(x, 128)
            q = tf.layers.batch_normalization(q, momentum=0.9)
            q = tf.nn.leaky_relu(q, alpha=0.2)
            q_mean = tf.layers.dense(q, self.parent_latent_dim+self.child_latent_dim)
            q_logstd = tf.layers.dense(q, self.parent_latent_dim+self.child_latent_dim)
            q_logstd = tf.maximum(q_logstd, -16)
            # Reshape to batch_size x 1 x latent_dim
            q_mean = tf.reshape(q_mean, (-1, 1, self.parent_latent_dim+self.child_latent_dim))
            q_logstd = tf.reshape(q_logstd, (-1, 1, self.parent_latent_dim+self.child_latent_dim))
            q = tf.concat([q_mean, q_logstd], axis=1, name='pred_parent_latent') # batch_size x 2 x latent_dim
            
            parent_q = q[:, :, :self.parent_latent_dim]
            child_q = q[:, :, self.parent_latent_dim:]
            
            return d, parent_q, child_q
        
    def train(self, parent_data, child_data, train_steps=2000, batch_size=32, disc_lr=2e-4, gen_lr=2e-4, save_interval=0, save_dir=None):
            
        parent_data = parent_data.astype(np.float32)
        child_data = child_data.astype(np.float32)
        
        # Inputs
        parent_real = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez], name='parent_real')
        parent_real_ = tf.expand_dims(parent_real, axis=-1)
        child_real = tf.placeholder(tf.float32, shape=[None, self.rez, self.rez], name='child_real')
        child_real_ = tf.expand_dims(child_real, axis=-1)
        self.parent_c = tf.placeholder(tf.float32, shape=[None, self.parent_latent_dim], name='parent_latent')
        self.child_c = tf.placeholder(tf.float32, shape=[None, self.child_latent_dim], name='child_latent')
        self.z1 = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='noise1')
        self.z2 = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='noise2')
        
        # Targets
        parent_q_target = tf.placeholder(tf.float32, shape=[None, self.parent_latent_dim])
        child_q_target = tf.placeholder(tf.float32, shape=[None, self.child_latent_dim])
        
        # Outputs
        d_real, _, _ = self.discriminator(parent_real_, child_real_)
        self.parent_fake = self.generator(self.parent_c, tf.zeros_like(self.child_c), self.z1)
        self.child_fake = self.generator(self.parent_c, self.child_c, self.z2)
        d_fake, parent_q, child_q = self.discriminator(self.parent_fake, self.child_fake)
        
        # Losses
        # Cross entropy losses for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        # Cross entropy losses for G
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        # Gaussian loss for Q
        def get_q_loss(q_fake, q_target):
            q_mean = q_fake[:, 0, :]
            q_logstd = q_fake[:, 1, :]
            epsilon = (q_target - q_mean) / (tf.exp(q_logstd) + EPSILON)
            q_loss = q_logstd + 0.5 * tf.square(epsilon)
            q_loss = tf.reduce_mean(q_loss)
            # q_loss = tf.reduce_mean(tf.square(q_target - q_fake[:, 0, :]))
            return q_loss
        q_loss = get_q_loss(parent_q, parent_q_target) + get_q_loss(child_q, child_q_target)
        
        # Optimizers
        d_optimizer = tf.train.AdamOptimizer(learning_rate=disc_lr, beta1=0.5)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=gen_lr, beta1=0.5)
        
        # Generator variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator variables
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        
        # Training operations
        d_train_real = d_optimizer.minimize(d_loss_real, var_list=dis_vars)
        d_train_fake = d_optimizer.minimize(d_loss_fake + 0.1*q_loss, var_list=dis_vars)
        g_train = g_optimizer.minimize(g_loss + 0.1*q_loss, var_list=gen_vars)
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Create summaries to monitor losses
        tf.summary.scalar('D_loss_for_real', d_loss_real)
        tf.summary.scalar('D_loss_for_fake', d_loss_fake)
        tf.summary.scalar('G_loss', g_loss)
        tf.summary.scalar('Q_loss', q_loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Start training
        self.sess = tf.Session()
        
        # Run the initializer
        self.sess.run(init)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('{}/logs'.format(save_dir), graph=self.sess.graph)
    
        for t in range(train_steps):
    
            parent_ind = np.random.choice(parent_data.shape[0], size=batch_size, replace=False)
            parent_batch = parent_data[parent_ind]
            child_ind = np.random.choice(child_data.shape[1], size=batch_size, replace=True)
            child_batch = child_data[parent_ind, child_ind]
            _, dlr = self.sess.run([d_train_real, d_loss_real], feed_dict={parent_real: parent_batch, 
                                                                           child_real: child_batch})
            
            parent_latent_batch = np.random.uniform(low=0., high=1., size=(batch_size, self.parent_latent_dim))
            child_latent_batch = np.random.normal(scale=0.5, size=(batch_size, self.child_latent_dim))
            noise1_batch = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            noise2_batch = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            _, dlf, qdl = self.sess.run([d_train_fake, d_loss_fake, q_loss],
                                        feed_dict={self.parent_c: parent_latent_batch, 
                                                   self.child_c: child_latent_batch, 
                                                   self.z1: noise1_batch, 
                                                   self.z2: noise2_batch, 
                                                   parent_q_target: parent_latent_batch, 
                                                   child_q_target: child_latent_batch})
            
            _, gl, qgl, summary_str = self.sess.run([g_train, g_loss, q_loss, merged_summary_op],
                                                    feed_dict={parent_real: parent_batch, 
                                                               child_real: child_batch,
                                                               self.parent_c: parent_latent_batch, 
                                                               self.child_c: child_latent_batch, 
                                                               self.z1: noise1_batch, 
                                                               self.z2: noise2_batch, 
                                                               parent_q_target: parent_latent_batch, 
                                                               child_q_target: child_latent_batch})
            
            summary_writer.add_summary(summary_str, t+1)
            
            # Show messages
            log_mesg = "%d: [D] real %f fake %f q %f" % (t+1, dlr, dlf, qdl)
            log_mesg = "%s  [G] fake %f q %f" % (log_mesg, gl, qgl)
            print(log_mesg)
            
            if save_interval>0 and (t+1)%save_interval==0 or t+1==train_steps:
                
                # Save the variables to disk.
                save_path = saver.save(self.sess, '{}/model'.format(save_dir))
                print('Model saved in path: %s' % save_path)
                print('Plotting results ...')
                # plot_grid(5, gen_func=self.synthesize_parent, d=self.parent_latent_dim, bounds=(0., 1.),
                #           fname='{}/synthesized_parent_{}'.format(save_dir, t+1))
                plot_random(5, gen_func=self.synthesize_parent, d=self.parent_latent_dim, bounds=(0., 1.),
                            fname='{}/synthesized_parent_random_{}'.format(save_dir, t+1))
                parent_latent = np.random.uniform(low=0., high=1., size=(1, self.parent_latent_dim))
                # plot_grid(5, gen_func=lambda x: self.synthesize_child(parent_latent, x, return_parent=True), 
                #           d=self.child_latent_dim, bounds=(-0.5, 0.5), 
                #           fname='{}/synthesized_child_{}'.format(save_dir, t+1))
                plot_random(5, gen_func=lambda x: self.synthesize_child(parent_latent, x, return_parent=True), 
                            d=self.child_latent_dim, bounds=(-0.5, 0.5), 
                            fname='{}/synthesized_child_random_{}'.format(save_dir, t+1))

                from sklearn.decomposition import PCA
                from matplotlib import pyplot as plt
                
                N = 200
                X = parent_data[:N,::2,::2].reshape(N,-1)
                Y = self.synthesize_parent(N)[:,::2,::2].reshape(N,-1)
                XY = np.concatenate([X, Y], axis=0)
                XY_embedded = PCA(n_components=2).fit_transform(XY)
                X_embedded = XY_embedded[:N]
                Y_embedded = XY_embedded[N:]
                plt.figure()
                plt.scatter(X_embedded[:,0], X_embedded[:,1], c='b', s=5, label='data', alpha=0.2, edgecolors='none')
                plt.scatter(Y_embedded[:,0], Y_embedded[:,1], c='r', s=5, label='synth', alpha=0.2, edgecolors='none')
                plt.legend()
                plt.axis('equal')
                plt.savefig('{}/projection_{}.svg'.format(save_dir, t+1))
                plt.close()
        
        summary_writer.close()
                    
    def restore(self, directory='.'):
        
        self.sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/model.meta'.format(directory))
        saver.restore(self.sess, tf.train.latest_checkpoint('{}/'.format(directory)))
        
        # Access and create placeholders variables            
        graph = tf.get_default_graph()
        self.parent_c = graph.get_tensor_by_name('parent_latent:0')
        self.child_c = graph.get_tensor_by_name('child_latent:0')
        self.z2 = graph.get_tensor_by_name('noise2:0')
        self.child_fake = graph.get_tensor_by_name('Generator_1/child_fake:0')

    def synthesize_parent(self, parent_latent, noise=None):
        if isinstance(parent_latent, int):
            N = parent_latent
            parent_latent = np.random.uniform(low=0., high=1., size=(N, self.parent_latent_dim))
        else:
            parent_latent = np.array(parent_latent, ndmin=2)
            N = parent_latent.shape[0]
        if noise is None:
            noise = np.zeros((N, self.noise_dim))
        parent_generated = self.sess.run(self.child_fake, feed_dict={self.parent_c: parent_latent, 
                                                                     self.child_c: np.zeros((N, self.child_latent_dim)),
                                                                     self.z2: noise})
        return np.squeeze(parent_generated)

    def synthesize_child(self, parent_latent, child_latent, noise=None, return_parent=False):
        
        if isinstance(child_latent, int):
            N = child_latent
            child_latent = np.random.normal(scale=0.5, size=(N, self.child_latent_dim))
        else:
            child_latent = np.array(child_latent, ndmin=2)
            N = child_latent.shape[0]
            
        parent_latent = parent_latent.reshape(1, self.parent_latent_dim)
        parent_latent_ = np.tile(parent_latent, [N, 1])
            
        if noise is None:
            noise = np.zeros((N, self.noise_dim))
            
        child_generated = self.sess.run(self.child_fake, feed_dict={self.parent_c: parent_latent_, 
                                                                    self.child_c: child_latent,
                                                                    self.z2: noise})
        
        if return_parent:
            parent_generated = self.sess.run(self.child_fake, feed_dict={self.parent_c: parent_latent, 
                                                                         self.child_c: np.zeros((1, self.child_latent_dim)),
                                                                         self.z2: np.zeros((1, self.noise_dim))})
            return np.squeeze(child_generated), np.squeeze(parent_latent), np.squeeze(parent_generated)
        else:
            return np.squeeze(child_generated)
