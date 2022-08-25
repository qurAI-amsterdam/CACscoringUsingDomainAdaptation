# Copyright: (c) 2019, University Medical Center Utrecht
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np
import torch
from os import path, makedirs
from time import time
from datetime import datetime, timedelta
from argparse import ArgumentParser

from networks.ConvNet import SingleVoxelRemover
from dataset.CalciumDataset import Dataset, MutableSubsetOfSlices, read_metadata_file, classifier_to_overlay_labels
from dataset.CalciumDataset import balanced_minibatch_of_cropped_slices_iterator as minibatch_iterator
from networks.ConvNet import DilatedConvNet as ConvNet
from utils.io import read_image, write_image
from utils.util import get_T
from dataset.extractors import SliceExtractor
from networks.Gan import transfer_encoder_to_model, Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------------------------------------------------------------

# Configuration
config = {
    'model': 'combined',
    'experiment': 'DeeplySupervised_NLST_newImp_adam',
    # 'experiment': 'DeeplySupervised_NLST_originalsettings',
# 'experiment': 'DeeplySupervised_NLST_originalsettings2_initializer',
    # 'model': 'combined',
    # 'experiment': 'DeeplySupervised_NLST_addTrain',
    'random_seed': 389725,
    'slice_size_voxels': 1,
    'slice_padding_voxels': ConvNet.padding,
    'loading_from_theano':False,
    'use_trainedG':True,
    'minibatch_size': 4,
    'train_data': 'training',
    'classes': 7,
    'ortho': True,
}

# Command line arguments
parser = ArgumentParser()


parser.add_argument('--inputdir', default='./data/ContrastCT_IDR/CCTA')
parser.add_argument('--scratchdir', default='./data/ContrastCT_IDR/resampled_CCTA')
parser.add_argument('--logdir', default='./logs')
parser.add_argument('--test_data', default='testing')
parser.add_argument('--G_experiment', default='GAN_NLST_mixCT_adv_cls_loss_MMD_firstD2.0K_clip0.1_EpochN200_lr5.0e-4')
parser.add_argument('--use_aorta_T', action='store_true')
parser.add_argument('--restore_epoch', type=int, default=700)
parser.add_argument('--train_scans', type=int, default=1012)
parser.add_argument('--kernels', default='all')

# ----------------------------------------------------------------------------------------------------------------------

# Set config values from command line arguments
for k, v in vars(parser.parse_args()).items():
    config[k] = v

# Set further directories
config['imagedir'] = path.join(config['scratchdir'], 'images_resampled')
config['overlaydir'] = path.join(config['scratchdir'], 'annotations_resampled')

# Initialization
overall_start_time = time()

if config['random_seed'] is not None:
    np.random.seed(config['random_seed'])

# Compile network
convnet = ConvNet(config, n_classes=config['classes'], deep_supervision=True)
# Restore network state
if config["loading_from_theano"]:
    model_file = r"D:\data\NLST\resampled_CT\training\networks_1012\slices_combined_DeeplySupervised_FullImage_AllKernels_NLST/epoch700.pkl"
    convnet.load_weight_from_theano(model_file)
else:
    logdir = path.join(config["logdir"], config['experiment'])
    model_file = '{}/model_latest.pth'.format(logdir)
    convnet.load_state_dict( torch.load(model_file) )
print('Restored network state from {}'.format(model_file))

if config['use_trainedG']:
    G = Generator()

    # G_experiment = 'GAN_NLST_mixCT_adv_cls_loss_MMD_firstD2.0K_clip0.1_EpochN200_lr5.0e-4'
    G_experiment = config["G_experiment"]
    G_file = '{}/{}/G_15.pth'.format( config["logdir"], G_experiment)

    config['model'] = G_experiment
    G.load_state_dict( torch.load(G_file) )
    convnet = transfer_encoder_to_model(convnet, G)
    print("Transfer encoder {}".format(G_file))

convnet.to(device).eval()

# Create test dataset
metadata = read_metadata_file(path.join(config['inputdir'], 'dataset.csv'))
test_data = Dataset(config['test_data'], metadata, config, kernels=config['kernels'])

# Make sure directory for results exists
resultdir = path.join(test_data.resultdir, 'calcium_masks')
if not path.exists(resultdir):
    makedirs(resultdir)
print('Saving to {}'.format(resultdir))


classify = convnet.mainClassifier

remove_single_voxels = SingleVoxelRemover()

for k, uid in enumerate(sorted(test_data.uids)):
    print('{}/{}'.format(k + 1, len(test_data.uids)))

    result_filename = path.join(resultdir, uid + '.mhd')
    if path.exists(result_filename):
        print('Result file already exists, skipping...')
        continue

    # Load image
    image_filename = path.join(test_data.imagedir, uid + '.mhd')
    if not path.exists(image_filename):
        print('Image file does not exist, skipping...')
        continue

    if config['use_aorta_T']:
        imT = get_T( image_filename )
    else:
        imT = 130

    start_time = time()
    image, spacing, origin = read_image(image_filename)
    slice_extractor = SliceExtractor(image)
    print('  > {} image loading'.format(timedelta(seconds=round(time() - start_time))))

    # Process axial, sagittal and coronal slices to obtain voxelwise probabilities
    n_features = convnet.features_per_orientation
    features = np.zeros(image.shape + (3 * n_features,), dtype='float16')
    features.fill(0)  # needed to actually reserve the memory

    for axis in range(3):
        start_time = time()

        image_shape = image.shape
        if axis == 1:
            image_shape = (image_shape[1], image_shape[0], image_shape[2])
        elif axis == 2:
            image_shape = (image_shape[2], image_shape[0], image_shape[1])

        n_slices = image_shape[0]
        slice_indices = list(range(0, n_slices))
        slice_x = image_shape[1] + config['slice_padding_voxels']
        slice_y = image_shape[2] + config['slice_padding_voxels']
        so = config['slice_padding_voxels'] // 2  # slice offset

        for start_batch in range(0, n_slices, config['minibatch_size']):
            end_batch = start_batch + config['minibatch_size']
            batch_of_slices = slice_indices[start_batch:end_batch]

            slices = np.empty(shape=(len(batch_of_slices), 1, slice_x, slice_y), dtype=slice_extractor.dtype)
            slices.fill(-1000)

            for i in range(len(batch_of_slices)):
                slices[i, 0, so:-so, so:-so] = slice_extractor.extract_slice(batch_of_slices[i], axis=axis)
            # normalize the slice
            slices = (np.clip(slices, a_min=-1000, a_max=3000) - 130) / 1130.0
            f = convnet.subnets[axis](torch.tensor(slices).to(device) )
            f = f.cpu().detach().numpy()
            if axis == 0:
                features[start_batch:end_batch, :, :, 0*n_features:1*n_features] = np.transpose(f, (0, 2, 3, 1))
            elif axis == 1:
                features[:, start_batch:end_batch, :, 1*n_features:2*n_features] = np.transpose(f, (2, 0, 3, 1))
            elif axis == 2:
                features[:, :, start_batch:end_batch, 2*n_features:3*n_features] = np.transpose(f, (2, 3, 0, 1))

        print('  > {} classification along axis {}'.format(timedelta(seconds=round(time() - start_time)), axis))

    # Iterate over the slices in batches to turn features into final probabilities
    start_time = time()

    result = np.zeros_like(image, dtype='int16')
    n_slices = image.shape[0]
    slice_indices = list(range(0, n_slices))

    for start_batch in range(0, n_slices, config['minibatch_size']):
        end_batch = start_batch + config['minibatch_size']
        batch_of_slices = slice_indices[start_batch:end_batch]
        inputs = np.transpose(features[start_batch:end_batch, :, :, :], (0, 3, 1, 2))
        batch_probs = classify( torch.tensor(inputs, dtype=torch.float).to(device) )
        batch_probs = torch.reshape(batch_probs, (batch_probs.shape[0], -1, inputs.shape[-2], inputs.shape[-1]))
        batch_probs = batch_probs.cpu().detach().numpy()
        result[start_batch:end_batch, :, :] = classifier_to_overlay_labels(np.argmax(batch_probs, axis=1))

    print('  > {} voxel classification'.format(timedelta(seconds=round(time() - start_time))))

    # Remove single voxels and voxels < 130HU
    start_time = time()

    result[image < imT] = 0

    # Store mask
    write_image(result_filename, result.astype('int16'), spacing, origin)
    print('  > {} post processing and saving mask'.format(timedelta(seconds=round(time() - start_time))))

print('Done with everything, took {} in total'.format(timedelta(seconds=round(time()-overall_start_time)))),
print('({:%d %b %Y %H:%M:%S})'.format(datetime.now()))