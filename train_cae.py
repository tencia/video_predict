import sys
import os
import numpy as np
import lasagne as nn
import theano
import theano.tensor as T
import utils as u
import models as m

# 01/03/2016
# trains convolutional autoencoder, goes directly from image to z-space
# this autoencodes fine but doesn't generate/interpolate well

default_specstr='d-0.2,16-5,d-0.2,32-3,d-0.2,64-3,8-1,d-0.2,{}'

def main(specstr=default_specstr, z_dim=256, num_epochs=10, ch=3, init_from='',
        img_size=64, pxsh=0.5, data_file='', batch_size=8, save_to='params'):

    # build expressions for the output, loss, gradient
    input_var = T.tensor4('inputs')
    print('building specstr {} - zdim {}'.format(specstr, z_dim))
    cae = m.build_cae_nopoolinv(input_var, shape=(img_size,img_size), channels=ch, 
            specstr=specstr.format(z_dim))
    l_list = nn.layers.get_all_layers(cae)
    pred = nn.layers.get_output(cae)
    loss = nn.objectives.squared_error(pred, input_var.flatten(2)).mean()
    params = nn.layers.get_all_params(cae, trainable=True)
    grads = nn.updates.total_norm_constraint(T.grad(loss, params), 10)
    updates = nn.updates.adam(grads, params, learning_rate=1e-3)
    te_pred = nn.layers.get_output(cae, deterministic=True)
    te_loss = nn.objectives.squared_error(te_pred, input_var.flatten(2)).mean()

    # training functions
    print('compiling functions')
    train_fn = theano.function([input_var], loss, updates=updates)
    val_fn = theano.function([input_var], te_loss)

    # compile functions for encode/decode to test later
    enc_layer = l_list[next(i for i in xrange(len(l_list)) if l_list[i].name=='encode')]
    enc_fn = theano.function([input_var], nn.layers.get_output(enc_layer, deterministic=True))
    dec_fn = lambda z: nn.layers.get_output(cae, deterministic=True,
        inputs={l_list[0]:np.zeros((z.shape[0],ch,img_size,img_size),dtype=theano.config.floatX),
            enc_layer:z}).eval().reshape(-1,ch,img_size,img_size)

    # load params if requested, run training
    if len(init_from) > 0:
        print('loading params from {}'.format(init_from))
        load_params(cae, init_from)
    data = u.DataH5PyStreamer(data_file, batch_size=batch_size)
    print('training for {} epochs'.format(num_epochs))
    hist = u.train_with_hdf5(data, num_epochs=num_epochs,
        train_fn = train_fn,
        test_fn = val_fn,
        tr_transform=lambda x: u.raw_to_floatX(x[0], pixel_shift=pxsh, center=False),
        te_transform=lambda x: u.raw_to_floatX(x[0], pixel_shift=pxsh, center=True))

    # generate examples, save training history
    te_stream = data.streamer(shuffled=True)
    imb, = next(te_stream.get_epoch_iterator())
    tg = u.raw_to_floatX(imb, pixel_shift=pxsh, square=True, center=True)
    pr = dec_fn(enc_fn(tg))
    for i in range(pr.shape[0]):
        u.get_image_pair(tg, pr,index=i,shift=pxsh).save('output_{}.jpg'.format(i))
    hist = np.asarray(hist)
    np.savetxt('cae_train_hist.csv', np.asarray(hist), delimiter=',', fmt='%.5f')
    u.save_params(cae, os.path.join(save_to, 'cae_{}.npz'.format(hist[-1,-1])))

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
