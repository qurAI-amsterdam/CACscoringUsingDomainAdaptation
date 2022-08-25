# Copyright: (c) 2019, University Medical Center Utrecht
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np
import pandas as pd
import torch
from os import path, makedirs
from time import time
from datetime import datetime, timedelta
from argparse import ArgumentParser

from networks.ConvNet import SingleVoxelRemover
from dataset.CalciumDataset import Dataset, MutableSubsetOfSlices, read_metadata_file, classifier_to_overlay_labels
from networks.Gan import transfer_encoder_to_model
from networks.ConvNet import DilatedConvNet as ConvNet
from utils.io import read_image, write_image
from dataset.extractors import SliceExtractor
from utils.util import Evaluation_fun
def val_encoder_fun( config, encoder, model_file, writer, epoch, device, T = 130):
    # transfer the trained encoder to CNNnet1 and testing

    config['inputdir'] = './data/CCTA_val/CCTA'
    config['scratchdir'] = './data/CCTA_val/resampled_CCTA'
    config['test_data'] = 'testing'
    config['train_data'] = 'training'
    config['train_scans'] =10
    config['kernels']='all'
    config['minibatch_size'] = 16


    # Set further directories
    config['imagedir'] = path.join(config['scratchdir'], 'images_resampled')
    config['overlaydir'] = path.join(config['scratchdir'], 'annotations_resampled')

    # Compile network
    convnet = ConvNet(config, n_classes=config['classes'], deep_supervision=True)
    # Restore network state
    convnet.load_state_dict( torch.load(model_file) )
    # print('Restored network state from {}'.format(model_file))
    transfer_encoder_to_model(convnet, encoder)
    # print('Transfer encoder to convnet1')

    convnet.to(device).eval()

    # Create test dataset
    metadata = read_metadata_file(path.join(config['inputdir'], 'dataset.csv'))
    test_data = Dataset(config['test_data'], metadata, config, kernels=config['kernels'])

    classify = convnet.mainClassifier
    remove_single_voxels = SingleVoxelRemover()

    eval_res = []
    for k, uid in enumerate(sorted(test_data.uids)):
        # Load image
        image_filename = path.join(test_data.imagedir, uid + '.mhd')
        mask_filename = path.join(test_data.overlaydir, uid + '.mhd')
        if not path.exists(image_filename):
            print('Image file does not exist, skipping...')
            continue

        image, spacing, origin = read_image(image_filename)
        voxel_volume = np.prod(spacing)
        mask, _, _ = read_image( mask_filename )
        slice_extractor = SliceExtractor(image)

        with torch.no_grad():
            # diasable the gradient calculation

            # Process axial, sagittal and coronal slices to obtain voxelwise probabilities
            n_features = convnet.features_per_orientation
            features = np.zeros(image.shape + (3 * n_features,), dtype='float16')
            features.fill(0)  # needed to actually reserve the memory

            for axis in range(3):
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

            # Iterate over the slices in batches to turn features into final probabilities
            result = np.zeros_like(image, dtype='int16')
            n_slices = image.shape[0]
            slice_indices = list(range(0, n_slices))

            for start_batch in range(0, n_slices, config['minibatch_size']):
                end_batch = start_batch + config['minibatch_size']
                # batch_of_slices = slice_indices[start_batch:end_batch]
                inputs = np.transpose(features[start_batch:end_batch, :, :, :], (0, 3, 1, 2))
                batch_probs = classify( torch.tensor(inputs, dtype=torch.float).to(device) )
                batch_probs = torch.reshape(batch_probs, (batch_probs.shape[0], -1, inputs.shape[-2], inputs.shape[-1]))
                batch_probs = batch_probs.cpu().detach().numpy()
                result[start_batch:end_batch, :, :] = classifier_to_overlay_labels(np.argmax(batch_probs, axis=1))

            result[image < T] = 0
            lesions = remove_single_voxels( result )
            result[lesions == 0] = 0

            out_dict = Evaluation_fun(result, mask, voxel_volume=voxel_volume)
            eval_res.append(out_dict)

    df = pd.DataFrame( eval_res )
    mean_res = dict( df.mean() )
    cac_res = {k: mean_res[k] for k in list(mean_res)[:12]}
    message = '> validation performance: '
    for k in cac_res:
        writer.add_scalar('Validation/{}'.format(k), cac_res[k], epoch)
        message += '{}:{}---'.format(k, cac_res[k])
    print(message)
