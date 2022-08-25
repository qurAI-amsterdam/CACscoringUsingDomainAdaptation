import json, tqdm
import numpy as np
from time import time
from argparse import ArgumentParser
from datetime import datetime, timedelta
from os import path, makedirs
from utils.util import count_parameters, loss_fun, accuracy_fun, sensitivity_fun
from torch.utils.tensorboard import SummaryWriter
import torch
import random

from dataset.CalciumDataset import Dataset,  read_metadata_file
from dataset.CalciumDataset import MutableSubsetOfSlices_nomask, MutableSubsetOfSlices
from utils.GAN_validation import val_encoder_fun
from dataset.CalciumDataset import get_a_batch_of_cropped_slices_iterator as get_batch
from dataset.CalciumDataset import balanced_minibatch_of_cropped_slices_iterator as get_balanced_batch
from networks.ConvNet import DilatedConvNet as ConvNet
from networks.Gan import Generator, Discriminator, restore_from_torch, NeckNet
from networks.Loss import MMD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------------------------------------------------------------

# Configuration
config = {
    'model': 'combined',
    'experiment': 'GAN_mixCT_adv_cls_loss_MMD',
    'random_seed': 148243,
    'slice_size_voxels': 25,
    'slice_padding_voxels': ConvNet.padding,
    "restore_epoch": None,
    'restore_G': True,
    'images_per_subset': 5,
    'epochs': 200,
    'batches_per_epoch': 5,
    'batch_size': 10,
    'minibatch_size': 16,
    'lr': 0.0005,
    'lr_decay': 1.0,
    'lr_step': 5,
    'save_every_m':5,
    'ct_data': 'ct',
    'ccta_data': 'ccta',
    'classes': 7,
}
# set seeds
seed = config['random_seed']
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

mu_dis, mu_gen = 1, 1

# Command line arguments
parser = ArgumentParser()

parser.add_argument('--inputdir', default='./data/NCCT_CCTA/mixCT')
parser.add_argument('--scratchdir', default='./data/NCCT_CCTA/resampled_mixCT')
parser.add_argument('--logdir', default='./logs')
parser.add_argument('--N_Diters', default=10, type=int)
parser.add_argument('--clip_value', default=0.1, type=float, help="low and upper clip value for D weights")
parser.add_argument('--w_cls', default=0.5, type=float, help="weight for classification loss")
parser.add_argument('--w_mmd', default=1.0, type=float, help="weight for classification loss")
parser.add_argument('--kernels', default='all')  # soft/sharp

# ----------------------------------------------------------------------------------------------------------------------

# Set config values from command line arguments
for k, v in vars(parser.parse_args()).items():
    config[k] = v

# Set further directories
config['imagedir'] = path.join(config['scratchdir'], 'images_resampled')
config['overlaydir'] = path.join(config['scratchdir'], 'annotations_resampled')

experiment = '{}_firstD{}K_clip{}_EpochN{}_lr{}e-4'.format(config['experiment'], config['N_Diters']/10, config['clip_value'], config['epochs'],config['lr']*1e4)

logdir = path.join( config["logdir"], experiment)
writer = SummaryWriter(logdir)
# Initialization
overall_start_time = time()

# Load datasets and turn lists of lesions into lists of voxels
metadata = read_metadata_file(path.join(config['inputdir'], 'dataset.csv'))

ct_data = Dataset(config['ct_data'], metadata, config, mode='slices', kernels=config['kernels'])
ncct_subset = MutableSubsetOfSlices(ct_data, config['images_per_subset'])

ccta_data = Dataset(config['ccta_data'], metadata, config, mode='images_only', kernels=config['kernels'])
ccta_subset = MutableSubsetOfSlices_nomask(ccta_data, config['images_per_subset'] )

# Compile network
modelG = Generator()
modelD = Discriminator()
modelG.to( device )
modelD.to( device )

mmd_fn = MMD()

lr_G, lr_D = config['lr']*0.1, config['lr']

optimizerG = torch.optim.RMSprop(modelG.parameters(), lr=lr_G,  weight_decay=0.0005)
optimizerD = torch.optim.RMSprop(modelD.parameters(), lr=lr_D,  weight_decay=0.0005)

lr_schedule_G = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=config['lr_step'], gamma=config["lr_decay"])
lr_schedule_D = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=config['lr_step'], gamma=config["lr_decay"])
print('Successfully compiled three convolutional neural network with {} parameters'.format( count_parameters(modelG) ))

modelfile = './logs/DeeplySupervised_multiDataset/model_latest.pth'
if config['restore_G']:
    restore_from_torch(modelG, modelfile)
    print('Restored Generator network state from {}'.format( modelfile ))

neckNet = NeckNet()
restore_from_torch(neckNet, modelfile)
print('Restored Neck network state from {}'.format( modelfile ))
neckNet.set_model_noTrain()
neckNet.to(device)
neckNet.eval()

# Train the network in epochs
first_epoch = config['restore_epoch'] + 1 if config['restore_epoch'] is not None else 1
print('Setting initial learning rate of G {}; learning rate of D {}'.format(lr_G, lr_D))

# Before we start training, lets store the configuration in a file
config_file = path.join(logdir, 'settings_from_epoch{}.json'.format(first_epoch))
if not path.exists(logdir):
    makedirs(logdir)
with open(config_file, 'w') as f:
    json.dump(config, f, sort_keys=True, indent=2, separators=(',', ': '))

start_time = time()
N_Diters = config['N_Diters']
D_model_file = '{}/D_firsttrained.pth'.format(logdir)
if path.exists(D_model_file):
    D_dict = torch.load( D_model_file )
    modelD.load_state_dict( D_dict )
    print("Restore first trained D from {}".format(D_model_file))
else:
    for i in range( N_Diters ):
        # Train Discriminator first with N epoch
        print('Epoch {}/{}'.format(i, N_Diters ))

        # Update subset once in a while
        epoch_time = time()
        if i > first_epoch:
            print('Updating training subset...')
            ncct_subset.swap_images(config['images_per_subset'])
            ccta_subset.swap_images(config['images_per_subset'])

        performance_D = []
        for p in modelD.parameters():
            p.requires_grad = True
        for n in tqdm.tqdm( range( 100 ) ):
            #Get data from CT and CCTA, CT as real data and CCTA as fake data
            for [batch_ct,_], batch_ccta in zip(get_batch(ncct_subset,n_minibatches=config['batch_size'], T=130, axis=None, withGT=True),\
                                            get_batch(ccta_subset, n_minibatches=config['batch_size'], T=130, axis=None)):
                # set gradients to zero
                modelD.zero_grad()
                outG_ct = modelG( batch_ct.to(device) )
                outG_ccta = modelG( batch_ccta.to(device) )

                logitD_ct = modelD(outG_ct.detach())
                logitG_ccta = modelD(outG_ccta.detach())

                dis_loss = -1 * mu_dis * (torch.mean(logitD_ct) - torch.mean(logitG_ccta))
                dis_loss.backward()

                #update parameters of Discriminator
                optimizerD.step()
                performance_D.append( dis_loss.cpu().detach().numpy() )

                ### compact the space of Discriminator
                for p in modelD.parameters():
                    p.data.clamp_( -config['clip_value'], config['clip_value']) # clipping the weight for Discriminator
        performance_D = np.mean(np.asarray(performance_D), axis=0)
        print(' > Discriminator: loss: {} '.format( performance_D ))

# Start alternatively training the Discriminator and generator
for epoch in range(first_epoch, config['epochs'] + 1):
    # Start the actual epoch
    print('Epoch {}/{}'.format(epoch, config['epochs']))

    # Update subset once in a while
    epoch_time = time()
    if epoch > first_epoch:
        print('Updating training subset...')
        ncct_subset.swap_images(config['images_per_subset'])
        ccta_subset.swap_images(config['images_per_subset'])

    if epoch < 100:  # change this later
        d_iters = 20
    else:
        d_iters = 5
    performance_D, performance_G = [], []
    cls_losses, mmd_losses = [], []
    mean_represent_ct, mean_represent_ccta = [], []
    for n in tqdm.tqdm( range(config['batches_per_epoch']) ):
        # train discriminator for a number of iteration
        for p in modelD.parameters():
            p.requires_grad =  True

        for d_i in range(d_iters):
            #Get data from CT and CCTA, CT as real data and CCTA as fake data
            for [batch_ct,_], batch_ccta in zip(get_batch(ncct_subset,n_minibatches=config['batch_size'], T=130, axis=None, withGT=True),\
                                            get_batch(ccta_subset, n_minibatches=config['batch_size'], T=130, axis=None)):
                # set gradients to zero
                modelD.zero_grad()
                outG_ct = modelG( batch_ct.to(device) )
                outG_ccta = modelG( batch_ccta.to(device) )

                logitD_ct = modelD(outG_ct.detach())
                logitG_ccta = modelD(outG_ccta.detach())

                # dis_loss = -1 * mu_dis * torch.mean( logitD_ct - logitG_ccta )
                dis_loss = -1 * mu_dis * (torch.mean(logitD_ct) - torch.mean(logitG_ccta))
                dis_loss.backward()

                #update parameters of Discriminator
                optimizerD.step()

                performance_D.append( dis_loss.cpu().detach().numpy() )
                mean_represent_ct.append( torch.mean(outG_ct).item() )

                for p in modelD.parameters():  # Disable modeD training
                    p.data.clamp_(-config['clip_value'], config['clip_value'])  # clipping the weight for Discriminator

        ### train generator
        for p in modelD.parameters():  # Freeze the discriminator
            p.requires_grad = False

        # Get data from  CCTA
        for [batch_ct, labels], batch_ccta in zip(get_balanced_batch(ncct_subset, n_minibatches=config['batch_size'], axis=None ), \
                                        get_batch(ccta_subset, n_minibatches=config['batch_size'], T=130, axis=None)):
            modelG.zero_grad()
            outG_ccta = modelG( batch_ccta.to(device) )
            logitG_ccta = modelD( outG_ccta )
            gen_loss = -1 * mu_gen * torch.mean( logitG_ccta )      # including a l2 regularization of mean feature distance between CT and CCTA


            # the classification loss on source domain
            outG_ct = modelG(batch_ct.to(device))
            preds_ct = neckNet(outG_ct)
            cls_loss = neckNet.loss_fun(preds_ct, labels.to(device))*config['w_cls']


            #the similarity loss
            preds_ccta = neckNet(outG_ccta)
            mmd_loss = mmd_fn( torch.flatten(preds_ct, start_dim=1 ), torch.flatten(preds_ccta, start_dim=1))
            mmd_loss = mmd_loss*config['w_mmd']


            tt_loss = gen_loss + cls_loss + mmd_loss
            tt_loss.backward()

            optimizerG.step()

            performance_G.append(gen_loss.cpu().detach().numpy())
            mean_represent_ccta.append(torch.mean(outG_ccta).item())
            cls_losses.append( cls_loss.cpu().detach().numpy() )
            mmd_losses.append(mmd_loss.cpu().detach().numpy())

    # Compute statistics over entire epoch
    performance_D = np.mean(np.asarray(performance_D), axis=0)
    performance_G = np.mean(np.asarray(performance_G), axis=0)
    cls_losses = np.mean(cls_losses, axis=0)
    mmd_losses = np.mean(mmd_losses, axis=0)
    print(' > Discriminator: loss: {}; -------Generator: loss: {}; --------- classification loss: {}; --------- mmd loss: {};'.format(
        performance_D, performance_G, cls_losses, mmd_losses))
    print(' > took {} for epoch calculation'.format(timedelta(seconds=round(time() - epoch_time))))

    # write to tensorboard
    writer.add_scalar('Train/Loss_D', performance_D, epoch)
    writer.add_scalar('Train/Loss_G', performance_G, epoch)
    writer.add_scalar('Train/Loss_cls', performance_G, epoch)
    writer.add_scalar('Train/Loss_mmd', mmd_losses, epoch)

    val_encoder_fun(config, modelG, modelfile, writer, epoch, device, T=130)

    if epoch % config['save_every_m'] == 0:
        g_model_file = '{}/G_{}.pth'.format(logdir, epoch)
        torch.save(modelG.state_dict(), g_model_file)

        D_model_file = '{}/D_{}.pth'.format(logdir, epoch)
        torch.save(modelD.state_dict(), D_model_file)

    # update learning rate
    lr_schedule_G.step()
    lr_schedule_D.step()

    # Stop time, report runtime for entire epoch
    print(' > took {}'.format(timedelta(seconds=round(time() - start_time))))

writer.flush()
writer.close()
print('Done with everything, took {} in total'.format(timedelta(seconds=round(time()-overall_start_time)))),
print('({:%d %b %Y %H:%M:%S})'.format(datetime.now()) )