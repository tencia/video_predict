import lasagne as nn
import theano
import theano.tensor as T
from gaussian_sample_layer import GaussianSampleLayer
from lstm_sampleable_layer import LSTMSampleableLayer
import utils as u

# Z_LSTM - takes (batchsize, seqlen, z_dim), sequence to sequence
def Z_LSTM(input_var, z_dim=256, nhid=512, layers=2, gradclip=10, training=True):
    ret = {}
    state_vars = []
    ret['input'] = layer = nn.layers.InputLayer(input_var=input_var, shape=(None, None, z_dim))
    batchsize, seqlen, _ = layer.input_var.shape
    for lay in xrange(layers):
        ret['drop_{}'.format(lay)] = layer = nn.layers.DropoutLayer(layer, p=0.3)
        if training:
            ret['lstm_{}'.format(lay)] = layer = LSTMSampleableLayer(layer, nhid,
                grad_clipping=gradclip, learn_init=True)
        else:
            cell_var = T.matrix('cell_var_{}'.format(lay))
            hid_var = T.matrix('hid_var_{}'.format(lay))
            state_vars.append(cell_var)
            state_vars.append(hid_var)
            ret['lstm_{}'.format(lay)] = layer = LSTMSampleableLayer(layer, nhid,
                cell_init=cell_var, hid_init=hid_var)
        ret['cell_{}'.format(lay)] = nn.layers.SliceLayer(layer, axis=2,
                indices=slice(None,nhid))
        ret['hid_{}'.format(lay)] = layer = nn.layers.SliceLayer(layer, axis=2,
                indices=slice(nhid,None))
    ret['reshape'] = layer = nn.layers.ReshapeLayer(layer, (-1, nhid))
    ret['project'] = layer = nn.layers.DenseLayer(layer, num_units=z_dim, nonlinearity=None)
    ret['output'] = layer = nn.layers.ReshapeLayer(layer, (batchsize, seqlen, z_dim))
    # final state slice layers for passing to next instance of lstm
    for lay in xrange(layers):
        ret['cellfinal_{}'.format(lay)] = nn.layers.SliceLayer(ret['cell_{}'.format(lay)],
                axis=1, indices=-1)
        ret['hidfinal_{}'.format(lay)] = nn.layers.SliceLayer(ret['hid_{}'.format(lay)], 
                axis=1, indices=-1)
    return ret, state_vars

# Z_LSTM - takes two variables each of (batchsize, seqlen, z_dim)
# one for mean, one for log-sigma
def Z_VLSTM(input_var, input_ls_var, z_dim=256, nhid=512, layers=2, gradclip=10, training=True):
    ret = {}
    state_vars = []
    ret['input_mu'] = nn.layers.InputLayer(input_var = input_var, shape = (None, None, z_dim))
    ret['input_ls'] = nn.layers.InputLayer(input_var = input_ls_var, shape=(None, None, z_dim))
    ret['input_merge'] = layer = nn.layers.ConcatLayer([ret['input_mu'],
        ret['input_ls']], axis=2)
    batchsize, seqlen, _ = ret['input_mu'].input_var.shape
    for lay in xrange(layers):
        ret['drop_{}'.format(lay)] = layer = nn.layers.DropoutLayer(layer, p=0.3)
        if training:
            ret['lstm_{}'.format(lay)] = layer = LSTMSampleableLayer(layer, nhid,
                grad_clipping=gradclip, learn_init=True)
        else:
            cell_var = T.matrix('cell_var_{}'.format(lay))
            hid_var = T.matrix('hid_var_{}'.format(lay))
            state_vars.append(cell_var)
            state_vars.append(hid_var)
            ret['lstm_{}'.format(lay)] = layer = LSTMSampleableLayer(layer, nhid,
                cell_init=cell_var, hid_init=hid_var)
        ret['cell_{}'.format(lay)] = nn.layers.SliceLayer(layer, axis=2, 
                indices=slice(None,nhid))
        ret['hid_{}'.format(lay)] = layer = nn.layers.SliceLayer(layer, axis=2, 
                indices=slice(nhid,None))
    ret['reshape'] = layer = nn.layers.ReshapeLayer(layer, (-1, nhid))
    ret['project_mu'] = nn.layers.DenseLayer(layer, num_units=z_dim, nonlinearity=None)
    ret['output_mu'] = nn.layers.ReshapeLayer(ret['project_mu'], (batchsize, seqlen, z_dim))
    ret['project_ls'] = nn.layers.DenseLayer(layer, num_units=z_dim, nonlinearity=None)
    ret['output_ls'] = nn.layers.ReshapeLayer(ret['project_ls'], (batchsize, seqlen, z_dim))
    ret['output'] = nn.layers.ConcatLayer([ret['output_mu'], ret['output_ls']], axis=2)
    # final state slice layers for passing to next instance of lstm
    for lay in xrange(layers):
        ret['cellfinal_{}'.format(lay)] = nn.layers.SliceLayer(ret['cell_{}'.format(lay)], 
                axis=1, indices=-1)
        ret['hidfinal_{}'.format(lay)] = nn.layers.SliceLayer(ret['hid_{}'.format(lay)], 
                axis=1, indices=-1)
    return ret, state_vars

# variational autoencoder (permutation-invariant) - used for going from convolutional
# features to latent variable space
def build_vae(input_var, L=2, binary=True, shape=(28,28), channels=1, z_dim=2, n_hid=1024):
    x_dim = shape[0] * shape[1] * channels
    l_input = nn.layers.InputLayer(shape=(None,channels,shape[0], shape[1]),
            input_var=input_var, name='input')
    l_enc_hid = nn.layers.DenseLayer(l_input, num_units=n_hid,
            nonlinearity=nn.nonlinearities.tanh if binary else T.nnet.softplus,
            name='enc_hid')
    l_enc_mu = nn.layers.DenseLayer(l_enc_hid, num_units=z_dim,
            nonlinearity = None, name='enc_mu')
    l_enc_logsigma = nn.layers.DenseLayer(l_enc_hid, num_units=z_dim,
            nonlinearity = None, name='enc_logsigma')
    l_dec_mu_list = []
    l_dec_logsigma_list = []
    l_output_list = []
    # tie the weights of all L versions so they are the "same" layer
    W_dec_hid = None
    b_dec_hid = None
    W_dec_mu = None
    b_dec_mu = None
    W_dec_ls = None
    b_dec_ls = None
    for i in xrange(L):
        l_Z = GaussianSampleLayer(l_enc_mu, l_enc_logsigma, name='Z')
        l_dec_hid = nn.layers.DenseLayer(l_Z, num_units=n_hid,
                nonlinearity = nn.nonlinearities.tanh if binary else T.nnet.softplus,
                W=nn.init.GlorotUniform() if W_dec_hid is None else W_dec_hid,
                b=nn.init.Constant(0.) if b_dec_hid is None else b_dec_hid,
                name='dec_hid')
        if binary:
            l_output = nn.layers.DenseLayer(l_dec_hid, num_units = x_dim,
                    nonlinearity = nn.nonlinearities.sigmoid,
                    W = nn.init.GlorotUniform() if W_dec_mu is None else W_dec_mu,
                    b = nn.init.Constant(0.) if b_dec_mu is None else b_dec_mu,
                    name = 'dec_output')
            l_output_list.append(l_output)
            if W_dec_hid is None:
                W_dec_hid = l_dec_hid.W
                b_dec_hid = l_dec_hid.b
                W_dec_mu = l_output.W
                b_dec_mu = l_output.b
        else:
            l_dec_mu = nn.layers.DenseLayer(l_dec_hid, num_units=x_dim,
                    nonlinearity = nn.nonlinearities.tanh,
                    W = nn.init.GlorotUniform() if W_dec_mu is None else W_dec_mu,
                    b = nn.init.Constant(0) if b_dec_mu is None else b_dec_mu,
                    name = 'dec_mu')
            # relu_shift is for numerical stability - if training data has any
            # dimensions where stdev=0, allowing logsigma to approach -inf
            # will cause the loss function to become NAN. So we set the limit
            # stdev >= exp(-1 * relu_shift)
            relu_shift = 10
            l_dec_logsigma = nn.layers.DenseLayer(l_dec_hid, num_units=x_dim,
                    W = nn.init.GlorotUniform() if W_dec_ls is None else W_dec_ls,
                    b = nn.init.Constant(0) if b_dec_ls is None else b_dec_ls,
                    nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift,
                    name='dec_logsigma')
            l_output = GaussianSampleLayer(l_dec_mu, l_dec_logsigma,
                    name='dec_output')
            l_dec_mu_list.append(l_dec_mu)
            l_dec_logsigma_list.append(l_dec_logsigma)
            l_output_list.append(l_output)
            if W_dec_hid is None:
                W_dec_hid = l_dec_hid.W
                b_dec_hid = l_dec_hid.b
                W_dec_mu = l_dec_mu.W
                b_dec_mu = l_dec_mu.b
                W_dec_ls = l_dec_logsigma.W
                b_dec_ls = l_dec_logsigma.b
    l_output = nn.layers.ElemwiseSumLayer(l_output_list, coeffs=1./L, name='output')
    return l_enc_mu, l_enc_logsigma, l_dec_mu_list, l_dec_logsigma_list, l_output_list, l_output

# convolution autoencoder -autoencodes images using MSE loss (not variational
# lower bond loss). Used for extracting convolutional features to use in VAE
def build_cae(input_var=None, specstr='', shape=(100,100),channels=3, batchnorm=False):
    l_input = nn.layers.InputLayer(shape=(None,channels,shape[0], shape[1]),
            input_var=input_var, name='input')
    l_last = l_input
    to_invert=[]
    specs=map(lambda s: s.split('-'), specstr.split(','))
    layerIdx = 1
    for spec in specs:
        if len(spec) == 2 and spec[0] == 'd':
            # do not append, because we don't do the inverse of a dropout
            # layer on the way up
            l_last = nn.layers.DropoutLayer(l_last, p=float(spec[1]), rescale=True,
                    name='dropout{}'.format(layerIdx))
        elif len(spec) == 2 and spec[0] == 'p':
            l_last = nn.layers.MaxPool2DLayer(l_last, pool_size=int(spec[1]),
                    name='pool{}'.format(layerIdx))
            to_invert.append(l_last)
        elif len(spec) == 2:
            nfilt = int(spec[0])
            fsize = int(spec[1])
            l_last = nn.layers.Conv2DLayer(l_last, num_filters=nfilt,
                    filter_size=(fsize,fsize),
                    nonlinearity=None,
                    b=None,
                    name='conv{}'.format(layerIdx))
            to_invert.append(l_last)
            if batchnorm:
                l_last = nn.layers.BatchNormLayer(l_last, name='bn{}'.format(layerIdx))
            else:
                l_last = nn.layers.BiasLayer(l_last, name='bias{}'.format(layerIdx))
            l_last = nn.layers.NonlinearityLayer(l_last,
                    nonlinearity=nn.nonlinearities.tanh,
                    name='nl{}'.format(layerIdx))
        elif len(spec) == 1:
            l_last = nn.layers.DenseLayer(l_last, num_units=int(spec[0]),
                    nonlinearity=None,
                    b=None,
                    name='dense{}'.format(layerIdx))
            to_invert.append(l_last)
            l_last = nn.layers.BiasLayer(l_last,
                    name='bias{}'.format(layerIdx))
            l_last = nn.layers.NonlinearityLayer(l_last,
                    nonlinearity=nn.nonlinearities.tanh,
                    name='nl{}'.format(layerIdx))
        layerIdx += 1
    #l_last.nonlinearity=nn.nonlinearities.linear
    l_last.name='encode'
    for lay in to_invert[::-1]:
        l_last=nn.layers.InverseLayer(l_last, lay, name='inv_{}'.format(lay.name))
        if not 'pool' in lay.name:
            if batchnorm:
                l_last = nn.layers.BatchNormLayer(l_last, name='inv_bn_{}'.format(lay.name))
            else:
                l_last = nn.layers.BiasLayer(l_last, name='inv_bias_{}'.format(lay.name))
            l_last = nn.layers.NonlinearityLayer(l_last,
                    nonlinearity=nn.nonlinearities.tanh,
                    name='inv_nl_{}'.format(lay.name))
    l_output = nn.layers.ReshapeLayer(l_last, shape=(([0], -1)), name='output')
    return l_output

# convolution autoencoder -autoencodes images using MSE loss (not variational
# lower bond loss). Used for extracting convolutional features to use in VAE
# Avoids using InverseLayer and instead uses Upscale2D
def build_cae_nopoolinv(input_var, specstr, shape, channels=3, batchnorm=False):
    l_input = nn.layers.InputLayer(shape=(None,channels,shape[0], shape[1]),
            input_var=input_var, name='input')
    l_last = l_input
    to_invert=[]
    specs=map(lambda s: s.split('-'), specstr.split(','))
    layerIdx = 1
    for spec in specs:
        if len(spec) == 2 and spec[0] == 'd':
            # do not append, because we don't do the inverse of a dropout
            # layer on the way up
            l_last = nn.layers.DropoutLayer(l_last, p=float(spec[1]), rescale=True,
                    name='dropout{}'.format(layerIdx))
        elif len(spec) == 2 and spec[0] == 'p':
            l_last = nn.layers.MaxPool2DLayer(l_last, pool_size=int(spec[1]),
                    name='pool{}'.format(layerIdx))
            to_invert.append(l_last)
        elif len(spec) == 2:
            nfilt = int(spec[0])
            fsize = int(spec[1])
            l_last = nn.layers.Conv2DLayer(l_last, num_filters=nfilt,
                    filter_size=(fsize,fsize),
                    nonlinearity=None,
                    b=None,
                    name='conv{}'.format(layerIdx))
            to_invert.append(l_last)
            if batchnorm:
                l_last = nn.layers.BatchNormLayer(l_last, name='bn{}'.format(layerIdx))
            else:
                l_last = nn.layers.BiasLayer(l_last, name='bias{}'.format(layerIdx))
            l_last = nn.layers.NonlinearityLayer(l_last,
                    nonlinearity=nn.nonlinearities.tanh,
                    name='nl{}'.format(layerIdx))
        elif len(spec) == 1:
            l_last = nn.layers.DenseLayer(l_last, num_units=int(spec[0]),
                    nonlinearity=None,
                    b=None,
                    name='dense{}'.format(layerIdx))
            to_invert.append(l_last)
            l_last = nn.layers.BiasLayer(l_last,
                    name='bias{}'.format(layerIdx))
            l_last = nn.layers.NonlinearityLayer(l_last,
                    nonlinearity=nn.nonlinearities.tanh,
                    name='nl{}'.format(layerIdx))
        layerIdx += 1
    #l_last.nonlinearity=nn.nonlinearities.linear
    l_last.name='encode'
    for lay in to_invert[::-1]:
        if 'pool' in lay.name:
            l_last = nn.layers.Upscale2DLayer(l_last, scale_factor=lay.pool_size)
        else:
            l_last=nn.layers.InverseLayer(l_last, lay, name='inv_{}'.format(lay.name))
            if batchnorm:
                l_last = nn.layers.BatchNormLayer(l_last, name='inv_bn_{}'.format(lay.name))
            else:
                l_last = nn.layers.BiasLayer(l_last, name='inv_bias_{}'.format(lay.name))
            l_last = nn.layers.NonlinearityLayer(l_last,
                    nonlinearity=nn.nonlinearities.tanh,
                    name='inv_nl_{}'.format(lay.name))
    l_output = nn.layers.ReshapeLayer(l_last, shape=(([0], -1)), name='output')
    return l_output

# variational convolutional autoencoder - just like VAE but has convolutional layer at top
# and bottom to improve performance on images
def build_vcae(input_var, L=1, binary=True, shape=(64,64), channels=1, z_dim=32, \
        n_hid=1024, nfilts=32, fsize=3):
    w,h=shape[0],shape[1]
    l_input = nn.layers.InputLayer(shape=(None,channels, w, h),
            input_var=input_var, name='input')
    l_dr = nn.layers.DropoutLayer(l_input, p=0.3)
    l_conv = nn.layers.Conv2DLayer(l_dr, num_filters=nfilts, filter_size=fsize,
            nonlinearity=nn.nonlinearities.tanh, name='conv', pad=fsize/2)
    l_enc_hid = nn.layers.DenseLayer(l_conv, num_units=n_hid,
            nonlinearity=nn.nonlinearities.tanh if binary else T.nnet.softplus,
            name='enc_hid')
    l_enc_mu = nn.layers.DenseLayer(l_enc_hid, num_units=z_dim,
            nonlinearity = None, name='enc_mu')
    l_enc_logsigma = nn.layers.DenseLayer(l_enc_hid, num_units=z_dim,
            nonlinearity = None, name='enc_logsigma')
    l_dec_mu_list = []
    l_dec_logsigma_list = []
    l_output_list = []
    # tie the weights of all L versions so they are the "same" layer
    W_dec_hid = None
    b_dec_hid = None
    W_dec_mu = None
    b_dec_mu = None
    W_dec_ls = None
    b_dec_ls = None
    for i in xrange(L):
        l_Z = GaussianSampleLayer(l_enc_mu, l_enc_logsigma, name='Z')
        l_dec_hid = nn.layers.DenseLayer(l_Z, num_units=n_hid,
                nonlinearity = nn.nonlinearities.tanh if binary else T.nnet.softplus,
                W=nn.init.GlorotUniform() if W_dec_hid is None else W_dec_hid,
                b=nn.init.Constant(0.) if b_dec_hid is None else b_dec_hid,
                name='dec_hid')
        l_dec_out = nn.layers.DenseLayer(l_dec_hid, num_units=nfilts*w*h,
                nonlinearity=nn.nonlinearities.tanh, name='dec_out')
        l_resh = nn.layers.ReshapeLayer(l_dec_out, shape=(([0],nfilts,w,h)), name='dec_resh')
        if binary:
            l_output = nn.layers.Conv2DLayer(l_resh, num_filters=channels,
                    filter_size=fsize, pad=fsize/2,
                    nonlinearity = nn.nonlinearities.sigmoid,
                    W = nn.init.GlorotUniform() if W_dec_mu is None else W_dec_mu,
                    b = nn.init.Constant(0.) if b_dec_mu is None else b_dec_mu,
                    name = 'dec_output')
            l_output_list.append(l_output)
            if W_dec_hid is None:
                W_dec_hid = l_dec_hid.W
                b_dec_hid = l_dec_hid.b
                W_dec_mu = l_output.W
                b_dec_mu = l_output.b
        else:
            l_dec_mu = nn.layers.Conv2DLayer(l_resh, num_filters=channels, 
                    filter_size=fsize, pad=fsize/2, nonlinearity = None,
                    W = nn.init.GlorotUniform() if W_dec_mu is None else W_dec_mu,
                    b = nn.init.Constant(0) if b_dec_mu is None else b_dec_mu,
                    name = 'dec_mu')
            # relu_shift is for numerical stability - if training data has any
            # dimensions where stdev=0, allowing logsigma to approach -inf
            # will cause the loss function to become NAN. So we set the limit
            # stdev >= exp(-1 * relu_shift)
            relu_shift = 10
            l_dec_logsigma = nn.layers.Conv2DLayer(l_resh, num_filters=channels,
                    filter_size=fsize, pad=fsize/2,
                    nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift,
                    W = nn.init.GlorotUniform() if W_dec_ls is None else W_dec_ls,
                    b = nn.init.Constant(0) if b_dec_ls is None else b_dec_ls,
                    name='dec_logsigma')
            l_output = GaussianSampleLayer(l_dec_mu, l_dec_logsigma, name='dec_output')
            l_dec_mu_list.append(l_dec_mu)
            l_dec_logsigma_list.append(l_dec_logsigma)
            l_output_list.append(l_output)
            if W_dec_hid is None:
                W_dec_hid = l_dec_hid.W
                b_dec_hid = l_dec_hid.b
                W_dec_mu = l_dec_mu.W
                b_dec_mu = l_dec_mu.b
                W_dec_ls = l_dec_logsigma.W
                b_dec_ls = l_dec_logsigma.b
    l_output = nn.layers.ElemwiseSumLayer(l_output_list, coeffs=1./L, name='output')
    return l_enc_mu, l_enc_logsigma, l_dec_mu_list, l_dec_logsigma_list, l_output_list, l_output

# deconvolutional net trained for recovering images given convolutional features
def build_deconv_net(input_var, shape, specstr):
    specs=map(lambda s: s.split('-'), specstr.split(','))[::-1]
    l_last = nn.layers.InputLayer(shape=shape, input_var = input_var, name='input')
    layerIdx=1
    for spec in specs:
        if len(spec) == 2 and spec[0] == 'd':
            l_last = nn.layers.DropoutLayer(l_last, p=float(spec[1]), rescale=True,
                name='dropout{}'.format(layerIdx))
        elif len(spec) == 2 and spec[0] == 'p':
            l_last = nn.layers.Upscale2DLayer(l_last, scale_factor=int(spec[1]),
                    name='upscale{}'.format(layerIdx))
        elif len(spec) == 2:
            nfilt=int(spec[0])
            fsize=int(spec[1])
            l_last = nn.layers.Conv2DLayer(l_last, num_filters=nfilt,
                filter_size=(fsize,fsize), W=nn.init.GlorotUniform(),
                nonlinearity=nn.nonlinearities.tanh, pad='full',
                name='conv{}'.format(layerIdx))
        layerIdx +=1
    l_last = nn.layers.Conv2DLayer(l_last, num_filters=3,
        filter_size=(3,3), nonlinearity=nn.nonlinearities.tanh, pad=1,
        name='conv{}'.format(layerIdx))
    return l_last

# builds a deconv-net and returns function for going from conv-space to images
def deconvoluter(params_fn, specstr, shape):
    input_var = T.tensor4('input')
    decnet = build_deconv_net(input_var, shape=shape, specstr=specstr)
    u.load_params(decnet, params_fn)
    return theano.function([input_var], nn.layers.get_output(decnet))

# builds a CAE and returns functions to go from image space to the layer denoted by
# layersplit, and from that layer back to images. However, the second function only
# works correctly if layersplit='encode', due to the structure of the Lasagne layer
# implementation, so if we split on a different layer then it is necessary to
# build a separate function for going from conv-space to images.
def encoder_decoder(paramsfile, specstr, channels=3, layersplit='encode', shape=(64,64),
        poolinv=False):
    inp = T.tensor4('inputs')
    w,h=shape
    build_fn = build_cae if poolinv else build_cae_nopoolinv
    network = build_fn(inp, shape=shape,channels=channels,specstr=specstr)
    u.load_params(network, paramsfile)
    laylist = nn.layers.get_all_layers(network)
    enc_layer_idx = next(i for i in xrange(len(laylist)) if laylist[i].name==layersplit)
    enc_layer = laylist[enc_layer_idx]
    return (lambda x: nn.layers.get_output(enc_layer, inputs=x,deterministic=True).eval(),
            lambda x: nn.layers.get_output(network,
                inputs={laylist[0]:np.zeros((x.shape[0],channels,w,h),
                    dtype=theano.config.floatX),
                    enc_layer:x}, deterministic=True).eval().reshape(-1,channels,w,h))

