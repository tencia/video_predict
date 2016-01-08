import sys
import os
import numpy as np
import lasagne as nn
import theano
import theano.tensor as T
from PIL import Image

import utils as u
import models as m
import config as c

# 01/03/2016
# Trains a convnet for going from the conv features back to images
# because just using the bottom of the autoencoder doesn't work if
# splitting on any layer other than 'encode'

def main(data_file='', img_size = 64, num_epochs = 10, batch_size = 128,
        pxsh = 0.5, split_layer = 'conv7', specstr=c.pf_cae_specstr,
        cae_params=c.pf_cae_params, save_to='params'):
 
    # transform function to go from images -> conv feats
    conv_feats,_ = m.encoder_decoder(cae_params, specstr=specstr,
                                   layersplit=split_layer, shape=(img_size,img_size))

    # build pretrained net for images -> convfeats in order to get the input shape
    # for the reverse function
    print('compiling functions')
    conv_net = m.build_cae(input_var=None, specstr=specstr, shape=(img_size,img_size))
    cae_layer_dict = dict((l.name, l) for l in nn.layers.get_all_layers(conv_net))
    shape = nn.layers.get_output_shape(cae_layer_dict[split_layer])

    # build net for convfeats -> images
    imgs_var = T.tensor4('images')
    convs_var = T.tensor4('conv_features')
    deconv_net = m.build_deconv_net(input_var=convs_var, shape=shape, specstr=specstr)
    loss = nn.objectives.squared_error(imgs_var, nn.layers.get_output(deconv_net)).mean()
    te_loss = nn.objectives.squared_error(imgs_var, nn.layers.get_output(deconv_net,
        deterministic=True)).mean()
    params = nn.layers.get_all_params(deconv_net, trainable=True)
    lr = theano.shared(nn.utils.floatX(3e-3))
    updates = nn.updates.adam(loss, params, learning_rate=lr)

    # compile functions
    train_fn = theano.function([convs_var, imgs_var], loss, updates=updates)
    val_fn = theano.function([convs_var, imgs_var], te_loss)
    deconv_fn = theano.function([convs_var], nn.layers.get_output(deconv_net,
        deterministic=True))

    # run training loop
    print("training for {} epochs".format(num_epochs))
    def data_transform(x, do_center):
        floatx_ims = u.raw_to_floatX(x, pixel_shift=pxsh, square=True, center=do_center)
        return (conv_feats(floatx_ims), floatx_ims)
    data = u.DataH5PyStreamer(data_file, batch_size=batch_size)
    hist = u.train_with_hdf5(data, num_epochs=num_epochs, train_fn=train_fn, test_fn=val_fn,
            tr_transform=lambda x: data_transform(x[0], do_center=False),
            te_transform=lambda x: data_transform(x[0], do_center=True))

    # generate examples, save training history and params
    te_stream = data.streamer(shuffled=True)
    imb, = next(te_stream.get_epoch_iterator())
    imb = data_transform(imb, True)[0]
    result = deconv_fn(imb)
    for i in range(result.shape[0]):
        Image.fromarray(u.get_picture_array(result, index=i, shift=pxsh)) \
                .save('output_{}.jpg'.format(i))
    hist = np.asarray(hist)
    np.savetxt('deconv_train_hist.csv', np.asarray(hist), delimiter=',', fmt='%.5f')
    u.save_params(deconv_net, os.path.join(save_to, 'deconv_{}.npz'.format(hist[-1,-1])))

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
