import numpy as np
import torch
import torch.nn as nn
from os import path

import pickle

class Softmax2d(nn.Module):
    def __init__(self):
        super(Softmax2d, self).__init__()

    def forward(self, x):
        e_x = torch.exp(x - x.max( dim=1, keepdim=True)[0])
        return e_x / torch.sum(e_x, dim=1, keepdim=True)

class SingleVoxelRemover:
    """Theano-based function that removes voxels without neighbors from a binary mask"""
    def __init__(self):
        self.conv3 = torch.nn.Conv3d(1,1,kernel_size=(3,3,3),stride=1,padding=1, bias=False)
        self.conv3.requires_grad_(False)
        torch.nn.init.ones_(self.conv3.weight)
        self.conv3.weight[:,:,1,1,1] = 0

    def __call__(self, mask):
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.float)
        if len(mask.shape) == 3:
            mask = torch.reshape(mask, (1,1, mask.shape[0], mask.shape[1], mask.shape[2]))

        out = self.conv3(mask)*mask
        out = torch.clamp(out, 0, 1)
        out = torch.squeeze(out)
        return out.detach().numpy()

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.0)

class DilatedConvNet( nn.Module ):
    padding = 128 + 2
    def __init__(self, config, n_classes= 7, deep_supervision=True):
        super().__init__()
        self.n_classes = n_classes
        self.config = config

        dilations = (2, 4, 8, 16, 32)
        self.deepSV = deep_supervision
        n_prefilter_layers = 2
        n_filters = 32
        n_dense_filters = 128
        self.features_per_orientation = 32
        self.model_dir = path.join(
            config['scratchdir'],
            config['train_data'],
            'networks_' + str(config['train_scans']),
            'slices_' + config['model'] + '_' + config['experiment']
        )

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss = nn.NLLLoss()

        subnets = []
        for i in range(3):  # axial, sagittal, coronal
            modules = []
            #analyze the image
            inchannel = 1
            for i in range(n_prefilter_layers):
                modules.append( nn.Conv2d(in_channels=inchannel, out_channels=n_filters, kernel_size=(3,3)) )
                modules.append( nn.ELU() )
                inchannel = n_filters

            for dilation in dilations:
                # modules.append( nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=(3,3), dilation=(dilation, dilation)) )
                modules.append(nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=(3, 3),
                                         dilation=(dilation, dilation)))
                modules.append( nn.ELU() )

            modules.append( nn.Conv2d( in_channels=n_filters, out_channels=n_filters, kernel_size=(3,3) ))
            modules.append( nn.ELU() )

            modules.append(nn.Conv2d(in_channels=n_filters, out_channels=self.features_per_orientation, kernel_size=(1, 1)))
            modules.append(nn.ELU())
            subnets.append( nn.Sequential(*modules) )
        self.subnets = nn.ModuleList( subnets )

        self.auxClassifier0 = nn.Sequential(
            nn.Dropout(0.35),
            nn.Conv2d( self.features_per_orientation, self.n_classes, kernel_size=(1,1) ),
            Softmax2d()
        )

        self.auxClassifier1 = nn.Sequential(
            nn.Dropout(0.35),
            nn.Conv2d(self.features_per_orientation, self.n_classes, kernel_size=(1, 1)),
            # nn.Softmax2d()
            Softmax2d()
        )

        self.auxClassifier2 = nn.Sequential(
            nn.Dropout(0.35),
            nn.Conv2d(self.features_per_orientation, self.n_classes, kernel_size=(1, 1)),
            # nn.Softmax2d()
            Softmax2d()
        )

        self.dropout = nn.Dropout( 0.35 )
        self.conv1 = nn.Conv2d( self.features_per_orientation*3, n_dense_filters, kernel_size=(1,1) )
        self.Elu = nn.ELU()
        self.conv2 = nn.Conv2d( n_dense_filters, n_classes, kernel_size=(1,1) )
        # self.softmax = nn.Softmax()
        self.softmax2d = Softmax2d()

        self.init_weight()

    def init_weight(self):
        for i in range(3):
            self.subnets[i].apply(init_weights)
        self.auxClassifier0.apply(init_weights)
        self.auxClassifier1.apply(init_weights)
        self.auxClassifier2.apply(init_weights)

        torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.conv1.bias.data.fill_(0.0)
        self.conv2.bias.data.fill_(0.0)

    def forward(self, x):
        # x size is [b, 3, x, y]
        # axial, sagittal, coronal
        features, center_feature = [], []
        center_xy = (self.config['slice_size_voxels'] - 1) // 2
        for i in range(x.shape[1]):
            outi = self.subnets[i]( x[:, i:i+1, :] )
            features.append( outi )
            center_feature.append( outi[:,:, center_xy:center_xy+1, center_xy:center_xy+1] )
            # center_feature.append(outi)

        out = torch.cat(center_feature, dim=1)
        out = self.dropout(out)
        out = self.Elu( self.conv1(out) )
        out = self.dropout( out )
        out = self.conv2(out)
        # out = self.softmax(out)

        if self.deepSV:
            aux_outs = []
            aux_outs.append( self.auxClassifier0( features[0]) )
            aux_outs.append( self.auxClassifier1(features[1]) )
            aux_outs.append( self.auxClassifier2(features[2]) )
            return out, aux_outs
        else:
            return out

    def mainClassifier(self, features):
        out = self.Elu( self.conv1(features) )
        out = self.softmax2d( self.conv2(out) )
        return out

    def loss_fun(self, preds, labels, deep_sv=True):

        center_label = labels[:, 0, labels.shape[2] // 2, labels.shape[3] // 2].long()
        if isinstance(preds, tuple):
            preds, preds_aux = preds[0].squeeze(), preds[1]

        # mainloss = self.loss( torch.softmax(preds, dim=1), center_label)
        mainloss = self.loss_ce(preds, center_label)

        if deep_sv:
            for i in range(3):
                predi = preds_aux[i]
                predi_flatten = torch.flatten(predi.permute(1, 0, 2, 3), start_dim=1).permute(1, 0)
                labeli_flaten = torch.flatten(labels[:, i, :, :]).long()
                # lossi = self.loss(predi_flatten, labeli_flaten)
                lossi = self.loss_ce(predi_flatten, labeli_flaten)
                mainloss += 0.05 * lossi

        return mainloss

    def load_weight_from_theano(self, pklfile):
        #failed, the Dilated conve in Theano and pytorch are different.
        with open(pklfile, "rb") as f:
            params0 = pickle.load(f, encoding='latin1')
            params1 = pickle.load(f, encoding='latin1')
            params2 = pickle.load(f, encoding='latin1')
            params3 = pickle.load(f, encoding='latin1')
        N = len(self.subnets[0])
        for i in range(0, N, 2):
            # j = int(i / 2)
            if i > 3 and i < 13:
                wnp = np.copy(np.transpose(params0[i], (1, 0, 2, 3)))
            else:
                wnp = np.copy(params0[i][:, :, ::-1, ::-1])

            self.subnets[0][i].weight = nn.Parameter( torch.from_numpy(wnp) )
            biasnp = params0[i+ 1].copy()
            self.subnets[0][i].bias= nn.Parameter(torch.from_numpy(biasnp))

        for i in range(0, N, 2):
            if i > 3 and i < 13:
                wnp = np.copy(np.transpose(params0[i], (1, 0, 2, 3)))
            else:
                wnp = np.copy(params0[i][:, :, ::-1, ::-1])

            self.subnets[1][i].weight= nn.Parameter( torch.from_numpy(wnp) )
            biasnp = params1[i+ 1].copy()
            self.subnets[1][i].bias= nn.Parameter(torch.from_numpy(biasnp))

        for i in range(0, N, 2):
            if i > 3 and i < 13:
                wnp = np.copy(np.transpose(params0[i], (1, 0, 2, 3)))
            else:
                wnp = np.copy(params0[i][:, :, ::-1, ::-1])

            self.subnets[2][i].weight= nn.Parameter( torch.from_numpy(wnp) )
            biasnp = params2[i+ 1].copy()
            self.subnets[2][i].bias= nn.Parameter(torch.from_numpy(biasnp))

        self.auxClassifier0[1].weight= nn.Parameter( torch.from_numpy( params0[-2][:, :, ::-1, ::-1].copy() ) )
        self.auxClassifier0[1].bias= nn.Parameter(torch.from_numpy( params0[-1].copy()))

        self.auxClassifier1[1].weight= nn.Parameter(torch.from_numpy(params1[-2][:, :, ::-1, ::-1].copy()))
        self.auxClassifier1[1].bias= nn.Parameter(torch.from_numpy(params1[-1].copy()))

        self.auxClassifier2[1].weight= nn.Parameter(torch.from_numpy(params2[-2][:, :, ::-1, ::-1].copy()))
        self.auxClassifier2[1].bias= nn.Parameter(torch.from_numpy(params2[-1].copy()))

        self.conv1.weight= nn.Parameter(torch.from_numpy(params3[-4][:, :, ::-1, ::-1].copy()))
        self.conv1.bias= nn.Parameter(torch.from_numpy(params3[-3].copy()))

        self.conv2.weight= nn.Parameter(torch.from_numpy(params3[-2][:, :, ::-1, ::-1].copy()))
        self.conv2.bias= nn.Parameter(torch.from_numpy(params3[-1].copy()))

from torch.nn.parameter import Parameter
from utils.util import load_lasagne_weights

class LasagneBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.beta = Parameter(torch.Tensor(num_features))
        self.gamma = Parameter(torch.Tensor(num_features))
        self.mu = Parameter(torch.Tensor(num_features))
        self.denom = Parameter(torch.Tensor(num_features))

    def forward(self, x):
        mu = self.mu.view(1, -1, 1, 1)
        denom = self.denom.view(1, -1, 1, 1)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        y = (x - mu) * denom * gamma + beta
        return y


class LasagneBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.beta = Parameter(torch.Tensor(num_features))
        self.gamma = Parameter(torch.Tensor(num_features))
        self.mu = Parameter(torch.Tensor(num_features))
        self.denom = Parameter(torch.Tensor(num_features))

    def forward(self, x):
        y = (x - self.mu.view(1, -1)) * self.denom.view(1, -1) * self.gamma.view(1, -1) + self.beta.view(1, -1)
        return y

class UndilatedConvNet( nn.Module ):
    """Second stage network"""
    is3D = False

    def __init__(self, config, compile_train_func=True):
        super().__init__()
        self.config = config
        self.model_dir = path.join(
            config['scratchdir'],
            config['train_data'],
            'networks_' + str(config['train_scans']),
            'voxels_' + config['model'] + '_' + config['experiment']
        )
        self.param_file = path.join(self.model_dir, 'epoch{}.pkl')
        # af = torch.nn.ELU(inplace=True)
        self.loss_ce = nn.CrossEntropyLoss()

        subnets = []
        for i in range(3):  # axial, sagittal, coronal
            modules = []
            # analyze the image
            modules.append(nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(5, 5), bias=False))
            modules.append(nn.ELU())
            modules.append(LasagneBatchNorm2d(num_features=24) )

            modules.append( nn.MaxPool2d(kernel_size=(2,2)))
            modules.append(nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), bias=False))
            modules.append(nn.ELU())
            modules.append(LasagneBatchNorm2d(num_features=32))

            modules.append(nn.MaxPool2d(kernel_size=(2, 2)))
            modules.append(nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), bias=False))
            modules.append(nn.ELU())
            modules.append(LasagneBatchNorm2d(num_features=48))

            modules.append(nn.MaxPool2d(kernel_size=(2, 2)))
            modules.append(nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(1, 1), bias=False))
            modules.append(nn.ELU())
            modules.append(LasagneBatchNorm2d(num_features=32))

            modules.append(nn.Flatten())
            subnets.append(nn.Sequential(*modules))
        self.subnets = nn.ModuleList(subnets)

        self.classifier = nn.Sequential( nn.Dropout(p=0.5),
                    # with 3*32*6*6,
                    nn.Linear(3456, out_features=256, bias=False), nn.ReLU(),
                    LasagneBatchNorm1d(num_features=256),
                    nn.Linear(256, 2), nn.Softmax(1) )
    def acc_sen_fun(self, pred, label):
        pred = torch.argmax(pred.squeeze(), dim=1)
        label = label > 0.5
        pred = pred > 0.5
        acc =  torch.sum(pred == label) / label.shape[0]
        sen = (torch.sum(label[ pred ]) + 1e-5) / (torch.sum( label ) + 1e-5)
        return acc, sen

    def load_weight_from_theano(self, pklfile, class2mapfile):
        from collections import OrderedDict
        values = load_lasagne_weights(pklfile)[0]

        state_dict = OrderedDict()
        with open(class2mapfile, 'r') as fh:
            for line in fh:
                k, idx, operation = line.split(',')

                k = k.strip()
                idx = int(idx.strip())
                operation = operation.strip()
                val = values[idx]
                if operation == 'flip':
                    val = np.copy(val[:, :, ::-1, ::-1])
                elif operation == 'transpose':
                    val = np.copy(val.T)
                else:
                    pass

                # state_dict[k] = torch.from_numpy(val).to(device)
                state_dict[k] = torch.from_numpy(val)
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        # x size is [b, 3, x, y]
        # axial, sagittal, coronal
        features= []

        for i in range(x.shape[1]):
            outi = self.subnets[i](x[:, i:i + 1, :])
            features.append(outi)

        out = torch.cat(features, dim=1)
        out = self.classifier( out )
        return out


if __name__ == "__main__":
    from argparse import ArgumentParser

    config = {  # cnnNet1
        'model': 'combined',
        'experiment': 'DeeplySupervised_FullImage_AllKernels',
        'random_seed': 148243,
        'slice_size_voxels': 25,
        'slice_padding_voxels': 130,
        'restore_epoch': None,
        'images_per_subset': 60,
        'epochs': 700,
        'batches_per_epoch': 5,
        'batch_size': 10,
        'minibatch_size': 64,
        'lr': 0.0005,
        'lr_decay': 1.0,
        'train_data': 'training',
        'valid_data': 'validation',
        'classes': 7,
        "scratchdir": 'D:/data/Cardiac_Exams_UMCU/resampled_CSCT',
        "inputdir": 'D:/data/Cardiac_Exams_UMCU/CSCT',
        "train_scans": 10,

    }

    file = "../logs/DeeplySupervised_multiDataset/S1epoch1500.pkl"
    model1 = DilatedConvNet(config)
    model1.load_weight_from_theano(file)
    torch.save(model1.state_dict(), file.replace("S1epoch1500.pkl", "model_latest.pth"))

    x = torch.rand( config["batch_size"], 3, 155, 155 )
    out1 = model1(x)
    print(out1[0].shape)

    config = {  # Configuration cnnNet2
        'model': '2classes',
        'experiment': 'UndilatedDeep65_OTF_FullImage_AllKernels_AllKernelMask',
        'random_seed': 312475,
        'patch_size_mm': 65 * 0.66,
        'patch_size_voxels': 65,
        'min_vol': 1.5,
        'max_vol': 100000000,
        'restore_epoch': None,
        'images_per_subset': 101,
        'epochs': 250,
        'batches_per_epoch': 10,
        'batch_size': 50,
        'minibatch_size': 64,
        'lr': 0.0001,
        'lr_decay': 1.0,
        'train_data': 'training',
        'valid_data': 'validation',
        "scratchdir": 'D:/data/NLST/resampled_CT',
        'train_scans': 100
    }
    # parser.add_argument('--inputdir', default='D:/data/NLST/CT')
    # parser.add_argument('--scratchdir', default='D:/data/NLST/resampled_CT')
    # parser.add_argument('--visdom', default=None)
    # parser.add_argument('--kernels', default='all')  # soft/sharp
    # parser.add_argument('--stage1', default='combined_DeeplySupervised_FullImage_AllKernels_NLST_1012_700')

    file = r"D:\project\Works\TransferLearning\Calcium_scoring_automatic\code/S2epoch1000.pkl"
    class2mapfile = r"D:\project\Works\TransferLearning\Calcium_scoring_tl\classifier2_mapping.txt"
    model2 = UndilatedConvNet(config)


    x = torch.rand(config["batch_size"], 3, config["patch_size_voxels"], config["patch_size_voxels"] )
    out1 = model2(x)

    model2.load_weight_from_theano(file, class2mapfile)
    out2 = model2(x)
    print(out1, out2)
    # print(x.shape, out.shape)
    # dict_ps = model2.state_dict()
    # for k in dict_ps:
    #     print(k, dict_ps[k].shape)


    # from networks.Gan import transfer_encoder_to_model, Generator
    #
    # encoder = Generator()
    # g_file = '../logs/GAN_mixCT_clipDweights/G_10.pth'
    # trainded_dict = torch.load(g_file)
    # encoder.load_state_dict( trainded_dict )
    # model3 = transfer_encoder_to_model(model=model2, encoder=encoder)
    #
    # print(torch.max( model3.state_dict()['subnets.0.0.weight'].cpu() - trainded_dict['subnets.0.0.weight'].cpu() ) )

