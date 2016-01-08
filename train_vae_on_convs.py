import numpy as np
import sys
import os
import theano
import theano.tensor as T
import lasagne as nn

import utils as u
import models as m
import config as c

#
# trains a vae using convolutional features (and a separately trained network
# for reversing the convolutional feature extraction.

def main(data_file = '', num_epochs=10, batch_size = 128, L=2, z_dim=256,
        n_hid=1500, binary='false', img_size = 64, init_from = '', save_to='params',
        split_layer='conv7', pxsh = 0.5, specstr = c.pf_cae_specstr,
        cae_weights=c.pf_cae_params, deconv_weights = c.pf_deconv_params):
    binary = binary.lower() == 'true'

    # pre-trained function for extracting convolutional features from images
    cae = m.build_cae(input_var=None, specstr=specstr, shape=(img_size,img_size))
    laydict = dict((l.name, l) for l in nn.layers.get_all_layers(cae))
    convshape = nn.layers.get_output_shape(laydict[split_layer])
    convs_from_img, _ = m.encoder_decoder(cae_weights, specstr=specstr, layersplit=split_layer,
            shape=(img_size, img_size))
    # pre-trained function for returning to images from convolutional features
    img_from_convs = m.deconvoluter(deconv_weights, specstr=specstr, shape=convshape)

    # Create VAE model
    print("Building model and compiling functions...")
    print("L = {}, z_dim = {}, n_hid = {}, binary={}".format(L, z_dim, n_hid, binary))
    input_var = T.tensor4('inputs')
    c,w,h = convshape[1], convshape[2], convshape[3]
    l_tup = l_z_mu, l_z_ls, l_x_mu_list, l_x_ls_list, l_x_list, l_x = \
            m.build_vae(input_var, L=L, binary=binary, z_dim=z_dim, n_hid=n_hid,
                   shape=(w,h), channels=c)

    if len(init_from) > 0:
        print("loading from {}".format(init_from))
        u.load_params(l_x, init_from)
    
    # build loss, updates, training, prediction functions
    loss,_ = u.build_vae_loss(input_var, *l_tup, deterministic=False, binary=binary, L=L)
    test_loss, test_prediction = u.build_vae_loss(input_var, *l_tup, deterministic=True,
            binary=binary, L=L)

    lr = theano.shared(nn.utils.floatX(1e-5))
    params = nn.layers.get_all_params(l_x, trainable=True)
    updates = nn.updates.adam(loss, params, learning_rate=lr)
    train_fn = theano.function([input_var], loss, updates=updates)
    val_fn = theano.function([input_var], test_loss)
    ae_fn = theano.function([input_var], test_prediction)

    # run training loop
    def data_transform(x, do_center):
        floatx_ims = u.raw_to_floatX(x, pixel_shift=pxsh, square=True, center=do_center)
        return convs_from_img(floatx_ims)

    print("training for {} epochs".format(num_epochs))
    data = u.DataH5PyStreamer(data_file, batch_size=batch_size)
    hist = u.train_with_hdf5(data, num_epochs=num_epochs, train_fn=train_fn, test_fn=val_fn,
                             tr_transform=lambda x: data_transform(x[0], do_center=False),
                             te_transform=lambda x: data_transform(x[0], do_center=True))

    # generate examples, save training history
    te_stream = data.streamer(shuffled=True)
    imb, = next(te_stream.get_epoch_iterator())
    orig_feats = data_transform(imb, do_center=True)
    reconstructed_feats = ae_fn(orig_feats).reshape(orig_feats.shape)
    orig_feats_deconv = img_from_convs(orig_feats)
    reconstructed_feats_deconv = img_from_convs(reconstructed_feats)
    for i in range(reconstructed_feats_deconv.shape[0]):
        u.get_image_pair(orig_feats_deconv, reconstructed_feats_deconv, index=i, shift=pxsh)\
                .save('output_{}.jpg'.format(i))
    hist = np.asarray(hist)
    np.savetxt('vae_convs_train_hist.csv', np.asarray(hist), delimiter=',', fmt='%.5f')
    u.save_params(l_x, os.path.join(save_to, 'vae_convs_{}.npz'.format(hist[-1,-1])))

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
