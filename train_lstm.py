import os
import sys
import numpy as np

import lasagne as nn
import theano
import theano.tensor as T

import utils as u
import models as m
import config as c

def main(save_to='params', 
         dataset = 'mm',
         kl_loss='true', # use kl-div in z-space instead of mse
         diffs = 'false',
         seq_length = 30,
         num_epochs=1,
         lstm_n_hid=1024,
         max_per_epoch=3
        ):
    kl_loss = kl_loss.lower() == 'true'
    diffs = diffs.lower() == 'true'

    # set up functions for data pre-processing and model training
    input_var = T.tensor4('inputs')

    # different experimental setup for moving mnist vs pulp fiction dataests
    if dataset == 'pf':
        img_size = 64
        cae_weights = c.pf_cae_params
        cae_specstr = c.pf_cae_specstr
        split_layer = 'conv7'
        inpvar = T.tensor4('input')
        net = m.build_cae(inpvar, specstr=cae_specstr, shape=(img_size, img_size))
        convs_from_img,_ = m.encoder_decoder(cae_weights, specstr=cae_specstr,
                layersplit=split_layer, shape=(img_size, img_size), poolinv=True)
        laydict = dict((l.name, l) for l in nn.layers.get_all_layers(net))
        zdec_in_shape = nn.layers.get_output_shape(laydict[split_layer])
        deconv_weights = c.pf_deconv_params
        vae_weights = c.pf_vae_params
        img_from_convs = m.deconvoluter(deconv_weights, specstr=cae_specstr, shape=zdec_in_shape)
        L=2
        vae_n_hid = 1500
        binary = False
        z_dim = 256
        l_tup = l_z_mu, l_z_ls, l_x_mu_list, l_x_ls_list, l_x_list, l_x = \
               m.build_vae(input_var, L=L, binary=binary, z_dim=z_dim, n_hid=vae_n_hid,
                        shape=(zdec_in_shape[2], zdec_in_shape[3]), channels=zdec_in_shape[1])
        u.load_params(l_x, vae_weights)
        datafile = 'data/pf.hdf5'
        frame_skip=3 # every 3rd frame in sequence
        z_decode_layer = l_x_mu_list[0]
        pixel_shift = 0.5
        samples_per_image = 4
        tr_batch_size = 16 # must be a multiple of samples_per_image
    elif dataset == 'mm':
        img_size = 64
        cvae_weights = c.mm_cvae_params
        L=2
        vae_n_hid = 1024
        binary = True
        z_dim = 32
        zdec_in_shape = (None, 1, img_size, img_size)
        l_tup = l_z_mu, l_z_ls, l_x_mu_list, l_x_ls_list, l_x_list, l_x = \
            m.build_vcae(input_var, L=L, z_dim=z_dim, n_hid=vae_n_hid, binary=binary,
                       shape=(zdec_in_shape[2], zdec_in_shape[3]), channels=zdec_in_shape[1])
        u.load_params(l_x, cvae_weights)
        datafile = 'data/moving_mnist.hdf5'
        frame_skip=1
        w,h=img_size,img_size # of raw input image in the hdf5 file
        z_decode_layer = l_x_list[0]
        pixel_shift = 0
        samples_per_image = 1
        tr_batch_size = 128 # must be a multiple of samples_per_image

    # functions for moving to/from image or conv-space, and z-space
    z_mat = T.matrix('z')
    zenc = theano.function([input_var], nn.layers.get_output(l_z_mu, deterministic=True))
    zdec = theano.function([z_mat], nn.layers.get_output(z_decode_layer, {l_z_mu:z_mat},
        deterministic=True).reshape((-1, zdec_in_shape[1]) + zdec_in_shape[2:]))
    zenc_ls = theano.function([input_var], nn.layers.get_output(l_z_ls, deterministic=True))

    # functions for encoding sequences of z's
    print 'compiling functions'
    z_var = T.tensor3('z_in')
    z_ls_var = T.tensor3('z_ls_in')
    tgt_mu_var = T.tensor3('z_tgt')
    tgt_ls_var = T.tensor3('z_ls_tgt')
    learning_rate = theano.shared(nn.utils.floatX(1e-4))

    # separate function definitions if we are using MSE and predicting only z, or KL divergence
    # and predicting both mean and sigma of z
    if kl_loss:
        def kl(p_mu, p_sigma, q_mu, q_sigma):
            return 0.5 * T.sum(T.sqr(p_sigma)/T.sqr(q_sigma) + T.sqr(q_mu - p_mu)/T.sqr(q_sigma)
                               - 1 + 2*T.log(q_sigma) - 2*T.log(p_sigma))
        lstm, _ = m.Z_VLSTM(z_var, z_ls_var, z_dim=z_dim, nhid=lstm_n_hid, training=True)
        z_mu_expr, z_ls_expr = nn.layers.get_output([lstm['output_mu'], lstm['output_ls']])
        z_mu_expr_det, z_ls_expr_det = nn.layers.get_output([lstm['output_mu'],
            lstm['output_ls']], deterministic=True)
        loss = kl(tgt_mu_var, T.exp(tgt_ls_var), z_mu_expr, T.exp(z_ls_expr))
        te_loss = kl(tgt_mu_var, T.exp(tgt_ls_var), z_mu_expr_det, T.exp(z_ls_expr_det))
        params = nn.layers.get_all_params(lstm['output'], trainable=True)
        updates = nn.updates.adam(loss, params, learning_rate=learning_rate)
        train_fn = theano.function([z_var, z_ls_var, tgt_mu_var, tgt_ls_var], loss, 
                updates=updates)
        test_fn = theano.function([z_var, z_ls_var, tgt_mu_var, tgt_ls_var], te_loss)
    else:
        lstm, _ = m.Z_LSTM(z_var, z_dim=z_dim, nhid=lstm_n_hid, training=True)
        loss = nn.objectives.squared_error(nn.layers.get_output(lstm['output']),
                tgt_mu_var).mean()
        te_loss = nn.objectives.squared_error(nn.layers.get_output(lstm['output'],
            deterministic=True), tgt_mu_var).mean()
        params = nn.layers.get_all_params(lstm['output'], trainable=True)
        updates = nn.updates.adam(loss, params, learning_rate=learning_rate)
        train_fn = theano.function([z_var, tgt_mu_var], loss, updates=updates)
        test_fn = theano.function([z_var, tgt_mu_var], te_loss)

    if dataset == 'pf':
        z_from_img = lambda x: zenc(convs_from_img(x))
        z_ls_from_img = lambda x: zenc_ls(convs_from_img(x))
    elif dataset == 'mm':
        z_from_img = zenc
        z_ls_from_img = zenc_ls

    # training loop
    print('training for {} epochs'.format(num_epochs))
    nbatch = (seq_length+1) * tr_batch_size * frame_skip / samples_per_image
    data = u.DataH5PyStreamer(datafile, batch_size=nbatch)

    # for taking arrays of uint8 (non square) and converting them to batches of sequences
    def transform_data(ims_batch, center=False):
        imb = u.raw_to_floatX(ims_batch, pixel_shift=pixel_shift,
                center=center)[np.random.randint(frame_skip)::frame_skip]
        zbatch = np.zeros((tr_batch_size, seq_length+1, z_dim), dtype=theano.config.floatX)
        zsigbatch = np.zeros((tr_batch_size, seq_length+1, z_dim), dtype=theano.config.floatX)
        for i in xrange(samples_per_image):
            chunk = tr_batch_size/samples_per_image
            if diffs:
                zf = z_from_img(imb).reshape((chunk, seq_length+1, -1))
                zbatch[i*chunk:(i+1)*chunk, 1:] = zf[:,1:] - zf[:,:-1]
                if kl_loss:
                    zls = z_ls_from_img(imb).reshape((chunk, seq_length+1, -1))
                    zsigbatch[i*chunk:(i+1)*chunk, 1:] = zls[:,1:] - zls[:,:-1]
            else:
                zbatch[i*chunk:(i+1)*chunk] = z_from_img(imb).reshape((chunk, seq_length+1, -1))
                if kl_loss:
                    zsigbatch[i*chunk:(i+1)*chunk] = z_ls_from_img(imb).reshape((chunk,
                        seq_length+1, -1))
        if kl_loss:
            return zbatch[:,:-1,:], zsigbatch[:,:-1,:], zbatch[:,1:,:], zsigbatch[:,1:,:]
        return zbatch[:,:-1,:], zbatch[:,1:,:]

    # we need sequences of images, so we do not shuffle data during trainin
    hist = u.train_with_hdf5(data, num_epochs=num_epochs, train_fn=train_fn, test_fn=test_fn,
                     train_shuffle=False,
                     max_per_epoch=max_per_epoch,
                     tr_transform=lambda x: transform_data(x[0], center=False),
                     te_transform=lambda x: transform_data(x[0], center=True))

    hist = np.asarray(hist)
    u.save_params(lstm['output'], os.path.join(save_to, 'lstm_{}.npz'.format(hist[-1,-1])))

    # build functions to sample from LSTM
    # separate cell_init and hid_init from the other learned model parameters
    all_param_values = nn.layers.get_all_param_values(lstm['output'])
    init_indices = [i for i,p in enumerate(nn.layers.get_all_params(lstm['output']))
            if 'init' in str(p)]
    init_values = [all_param_values[i] for i in init_indices]
    params_noinit = [p for i,p in enumerate(all_param_values) if i not in init_indices]

    # build model without learnable init values, and load non-init parameters
    if kl_loss:
        lstm_sample, state_vars = m.Z_VLSTM(z_var, z_ls_var, z_dim=z_dim, nhid=lstm_n_hid,
                training=False)
    else:
        lstm_sample, state_vars = m.Z_LSTM(z_var, z_dim=z_dim, nhid=lstm_n_hid, training=False)
    nn.layers.set_all_param_values(lstm_sample['output'], params_noinit)

    # extract layers representing thee hidden and cell states, and have sample_fn
    # return their outputs
    state_layers_keys = [k for k in lstm_sample.keys() if 'hidfinal' in k or 'cellfinal' in k]
    state_layers_keys = sorted(state_layers_keys)
    state_layers_keys = sorted(state_layers_keys, key = lambda x:int(x.split('_')[1]))
    state_layers = [lstm_sample[s] for s in state_layers_keys]
    if kl_loss:
        sample_fn = theano.function([z_var, z_ls_var] + state_vars,
                nn.layers.get_output([lstm['output_mu'], lstm['output_ls']] + state_layers,
                    deterministic=True))
    else:
        sample_fn = theano.function([z_var] + state_vars,
                nn.layers.get_output([lstm['output']] + state_layers, deterministic=True))

    from images2gif import writeGif
    from PIL import Image

    if dataset == 'pf':
        img_from_z = lambda z: img_from_convs(zdec(z))
    elif dataset == 'mm':
        img_from_z = zdec

    te_stream = data.streamer(shuffled=True)
    imb, = next(te_stream.get_epoch_iterator())
    z_tup = transform_data(imb, center=True)
    if kl_loss:
        z_in, z_ls_in = z_tup[0], z_tup[1]
        segment_idx = np.random.randint(z_in.shape[0])
        z_last, z_ls_last = z_in[segment_idx:segment_idx+1], z_ls_in[segment_idx:segment_idx+1]
        z_vars = [z_last, z_ls_last]
    else:
        z_in = z_tup[0]
        segment_idx = np.random.randint(z_in.shape[0])
        z_last = z_in[segment_idx:segment_idx+1]
        z_vars = [z_last]
    images = []
    state_values = [np.dot(np.ones((z_last.shape[0],1), dtype=theano.config.floatX), s)
            for s in init_values]
    output_list = sample_fn(*(z_vars + state_values))

    # use whole sequence of predictions for output
    z_pred = output_list[0]
    state_values = output_list[2 if kl_loss else 1:]

    rec = img_from_z(z_pred.reshape(-1, z_dim))
    for k in xrange(rec.shape[0]):
        images.append(Image.fromarray(u.get_picture_array(rec, index=k, shift=pixel_shift)))
    k += 1
    # slice prediction to feed into lstm
    z_pred = z_pred[:,-1:,:]
    if kl_loss:
        z_ls_pred = output_list[1][:,-1:,:]
        z_vars = [z_pred, z_ls_pred]
    else:
        z_vars = [z_pred]
    for i in xrange(30): # predict 30 frames after the end of the priming video
        output_list = sample_fn(*(z_vars + state_values))
        z_pred = output_list[0]
        state_values = output_list[2 if kl_loss else 1:]
        rec = img_from_z(z_pred.reshape(-1, z_dim))
        images.append(Image.fromarray(u.get_picture_array(rec, index=0, shift=pixel_shift)))
        if kl_loss:
            z_ls_pred = output_list[1]
            z_vars = [z_pred, z_ls_pred]
        else:
            z_vars = [z_pred]
    writeGif("sample.gif",images,duration=0.1,dither=0)

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
