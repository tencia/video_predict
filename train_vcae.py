import sys
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne as nn

import utils as u
import models as m

# 01/03/2016
# trains a variational autoencoder with convolutions at the input and output layers
# for better use of images. works well on MNIST and moving MNIST, does not work on
# natural images

def main(L=2, img_size=64, pxsh=0., z_dim=32, n_hid=1024, num_epochs=12, binary='True',
        init_from='', data_file='', batch_size=128, save_to='params'):
    binary = binary.lower()=='true'

    # Create VAE model
    input_var = T.tensor4('inputs')
    print("Building model and compiling functions...")
    print("L = {}, z_dim = {}, n_hid = {}, binary={}".format(L, z_dim, n_hid, binary))
    l_tup = l_z_mu, l_z_ls, l_x_mu_list, l_x_ls_list, l_x_list, l_x = \
           m.build_vcae(input_var, L=L, binary=binary, z_dim=z_dim, n_hid=n_hid)
        
    if len(init_from) > 0:
        print('loading from {}'.format(init_from))
        load_params(l_x, init_from)

    # compile functions
    loss, _ = u.build_vae_loss(input_var, *l_tup, deterministic=False, binary=binary, L=L)
    test_loss, test_prediction = u.build_vae_loss(input_var, *l_tup, deterministic=True,
            binary=binary, L=L)
    params = nn.layers.get_all_params(l_x, trainable=True)
    updates = nn.updates.adam(loss, params, learning_rate=3e-5)
    train_fn = theano.function([input_var], loss, updates=updates)
    val_fn = theano.function([input_var], test_loss)
    ae_fn = theano.function([input_var], test_prediction)

    # run training loop
    print('training for {} epochs'.format(num_epochs))
    data = u.DataH5PyStreamer(data_file, batch_size=batch_size)
    hist = u.train_with_hdf5(data, num_epochs=num_epochs, train_fn=train_fn, test_fn=val_fn,
            tr_transform=lambda x: u.raw_to_floatX(x[0], pixel_shift=pxsh, center=False),
            te_transform=lambda x: u.raw_to_floatX(x[0], pixel_shift=pxsh, center=True))

    # generate examples, save training history
    te_stream = data.streamer(shuffled=True)
    imb, = next(te_stream.get_epoch_iterator())
    orig_images = u.raw_to_floatX(imb, pixel_shift=pxsh)
    autoencoded_images = ae_fn(orig_images)
    for i in range(autoencoded_images.shape[0]):
        u.get_image_pair(orig_images, autoencoded_images, index=i, shift=pxsh) \
                .save('output_{}.jpg'.format(i))
    hist = np.asarray(hist)
    np.savetxt('vcae_train_hist.csv', np.asarray(hist), delimiter=',', fmt='%.5f')
    u.save_params(l_x, os.path.join(save_to, 'vcae_{}.npz'.format(hist[-1,-1])))

if __name__ == '__main__':
    # make all arguments of main(...) command line arguments (with type inferred from
    # the default value) - this doesn't work on bools so those are strings when
    # passed into main.
    import argparse, inspect
    parser = argparse.ArgumentParser(description='Command line options')
    ma = inspect.getargspec(main)
    for arg_name,arg_type in zip(ma.args[-len(ma.defaults):],[type(de) for de in ma.defaults]):
        parser.add_argument('--{}'.format(arg_name), type=arg_type, dest=arg_name)
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
