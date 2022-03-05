from . import fid
from . import intra_fid
from . import inception_tf
from pathlib import Path
import os.path as osp
import numpy as np


class Evaluator(object):
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader
        Path(self.config.inception_cache_path).mkdir(parents=True, exist_ok=True)

        print('Initializing inception ...')
        inception_tf.initialize_inception()

    def compute_metrics(self, samples):

        if self.config.conditional:
            IS_mean, IS_std = inception_tf.get_inception_score(np.array(samples[0]), splits=10,
                                                               batch_size=8, mem_fraction=1)
            print('Inception score: {} +/- {}'.format(IS_mean, IS_std))

            fid_score = fid.compute_fid(osp.join(self.config.inception_cache_path, 'stats.npy'), samples[0],
                                        self.config.inception_cache_path, self.dataloader)
            print('FID score: {}'.format(fid_score))

            intra_fid_score = intra_fid.compute_fid(osp.join(self.config.inception_cache_path, 'stats_intra.pkl'), samples,
                                                    self.config.inception_cache_path, self.dataloader)
            print('Intra FID score: {}'.format(intra_fid_score))
            return IS_mean, IS_std, fid_score, intra_fid_score
        else:
            IS_mean, IS_std = inception_tf.get_inception_score(np.array(samples), splits=10,
                                                               batch_size=8, mem_fraction=1)
            print('Inception score: {} +/- {}'.format(IS_mean, IS_std))
            fid_score = fid.compute_fid(osp.join(self.config.inception_cache_path, 'stats.npy'), samples,
                                        self.config.inception_cache_path, self.dataloader)
            print('FID score: {}'.format(fid_score))
            return IS_mean, IS_std, fid_score
