#!/usr/bin/env python3
''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from imageio import imread
from scipy import linalg
import pathlib
import urllib
import warnings
import torch


class InvalidFIDException(Exception):
    pass


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.io.gfile.GFile(pth, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')


# -------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3


# -------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=32, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    if verbose:
        print(" done")
    return pred_arr


# -------------------------------------------------------------------------------


def get_stats_fromdataloader(dataloader, sess, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    pred_dict = {}
    mu_dict = {}
    sigma_dict = {}

    for i, (imgs, labels, indices) in enumerate(dataloader):
        if verbose:
            print("\rPropagating batch {}".format(i + 1), end="", flush=True)
        labels = labels.cpu().numpy()
        batch_size = imgs.size(0)
        imgs = imgs * 0.5 + 0.5
        imgs = imgs * 255.0
        imgs = imgs.numpy().astype(np.uint8)
        imgs = np.transpose(imgs, (0, 2, 3, 1))
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': imgs})
        pred_arr = pred.reshape(batch_size, -1)

        unique_labels = np.unique(labels)
        for lab in unique_labels:
            if lab not in pred_dict:
                pred_dict[lab] = []
            pred_dict[lab].append(pred_arr[labels == lab, ::])

    if verbose:
        print(" done")
    for lab in pred_dict.keys():
        pred_dict[lab] = np.vstack(pred_dict[lab])
        mu_dict[lab] = np.mean(pred_dict[lab], axis=0)
        sigma_dict[lab] = np.cov(pred_dict[lab], rowvar=False)

    return mu_dict, sigma_dict

# -------------------------------------------------------------------------------

def calculate_frechet_distance(mu1_dict, sigma1_dict, mu2_dict, sigma2_dict, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Dict of Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : Dict of The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: Dict of The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: Dict of The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : Mean Frechet Distance.
    """

    fid_list = []
    for key in mu1_dict.keys():
        mu1 = mu1_dict[key]
        mu2 = mu2_dict[key]
        sigma1 = sigma1_dict[key]
        sigma2 = sigma2_dict[key]

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        fid_class = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        fid_list.append(fid_class)
    fid_list = np.array(fid_list)

    return np.mean(fid_list)


# -------------------------------------------------------------------------------


def calculate_activation_statistics(images, labels, sess, batch_size=32, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    unique_labels = np.unique(labels)
    mu_dict = {}
    sigma_dict = {}
    for lab in unique_labels:
        indices = np.nonzero(labels == lab)[0]
        img_class = images[indices, ::]
        act = get_activations(img_class, sess, batch_size, verbose)
        mu_class = np.mean(act, axis=0)
        sigma_class = np.cov(act, rowvar=False)
        mu_dict[lab] = mu_class
        sigma_dict[lab] = sigma_class
    return mu_dict, sigma_dict


# ------------------
# The following methods are implemented to obtain a batched version of the activations.
# This has the advantage to reduce memory requirements, at the cost of slightly reduced efficiency.
# - Pyrestone
# ------------------


def load_image_batch(files):
    """Convenience method for batch-loading images
    Params:
    -- files    : list of paths to image files. Images need to have same dimensions for all files.
    Returns:
    -- A numpy array of dimensions (num_images,hi, wi, 3) representing the image pixel values.
    """
    return np.array([imread(str(fn)).astype(np.float32) for fn in files])


# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
# -------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def get_stats(samples, sess, is_loader=False):
    if is_loader:
        m, s = get_stats_fromdataloader(samples, sess)
    elif isinstance(samples, tuple):
        x = np.array(samples[0])
        y = np.array(samples[1])
        m, s = calculate_activation_statistics(x, y, sess)
    elif samples.endswith('.pkl'):
        with open(samples, "rb") as fp:
            dat = pickle.load(fp)
            m = dat['mu']
            s = dat['sigma']
        print('Cache loaded successfully ...')
    else:
        raise ValueError('samples should either be dataloader, tuple of img, label list or cached pickle file')
    return m, s


def compute_fid(cache_path, samples, inception_path, dataloader):

    inception_path = check_or_download_inception(inception_path)
    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if not os.path.exists(cache_path):
            m1, s1 = get_stats(dataloader, sess, is_loader=True)
            print('Caching ...')
            stats = {'mu': m1, 'sigma': s1}
            with open(cache_path, "wb") as fp:
                pickle.dump(stats, fp)
        else:
            print('Using cache ...')
            m1, s1 = get_stats(cache_path, sess)

        m2, s2 = get_stats(samples, sess)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value
