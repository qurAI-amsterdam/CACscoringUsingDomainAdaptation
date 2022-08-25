import torch
from torch import nn
from os import path

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.0)

class Generator( nn.Module ):
    padding = 128 + 2
    def __init__(self, ):
        super().__init__()

        dilations = (2, 4, 8, 16, 32)
        n_prefilter_layers = 2
        n_filters = 32
        self.features_per_orientation = 32

        subnets = []
        for i in range(3):  # axial, sagittal, coronal
            modules = []
            inchannel = 1
            for i in range(n_prefilter_layers):
                modules.append( nn.Conv2d(in_channels=inchannel, out_channels=n_filters, kernel_size=(3,3)) )
                modules.append( nn.ELU() )
                inchannel = n_filters

            for dilation in dilations:
                modules.append(nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=(3, 3),
                                         dilation=(dilation, dilation)))
                modules.append( nn.ELU() )

            modules.append( nn.Conv2d( in_channels=n_filters, out_channels=n_filters, kernel_size=(3,3) ))
            modules.append( nn.ELU() )

            modules.append(nn.Conv2d(in_channels=n_filters, out_channels=self.features_per_orientation, kernel_size=(1, 1)))
            modules.append(nn.ELU())
            subnets.append( nn.Sequential(*modules) )
        self.subnets = nn.ModuleList( subnets )

    def init_weight(self):
        for i in range(3):
            self.subnets[i].apply(init_weights)

    def forward(self, x):
        # x size is [b, 3, x, y]
        # axial, sagittal, coronal
        features, center_feature = [], []
        for i in range(x.shape[1]):
            outi = self.subnets[i]( x[:, i:i+1, :] )
            features.append( outi )
        out = torch.cat(features, dim=1)
        return out

from networks.ConvNet import Softmax2d
class NeckNet( nn.Module ):
    padding = 128 + 2
    def __init__(self, n_classes= 7):
        super().__init__()
        self.n_classes = n_classes

        n_dense_filters = 128
        self.features_per_orientation = 32
        self.loss_ce = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout( 0.35 )
        self.conv1 = nn.Conv2d( self.features_per_orientation*3, n_dense_filters, kernel_size=(1,1) )
        self.Elu = nn.ELU()
        self.conv2 = nn.Conv2d( n_dense_filters, n_classes, kernel_size=(1,1) )
        # self.softmax = nn.Softmax()
        self.softmax2d = Softmax2d()

    def forward(self, feature):
        # x size is [b, 3, x, y]
        # axial, sagittal, coronal
        center_xy = feature.shape[-1] // 2
        center_feature = feature[:, :, center_xy:center_xy + 1, center_xy:center_xy + 1]

        out = self.dropout(center_feature)
        out = self.Elu( self.conv1(out) )
        out = self.dropout( out )
        out = self.conv2(out)
        # out = self.softmax(out)
        out = self.softmax2d(out)

        return out

    def loss_fun(self, preds, labels ):
        center_label = labels[:, 0, labels.shape[2] // 2, labels.shape[3] // 2].long()
        if isinstance(preds, tuple):
            preds, preds_aux = preds[0].squeeze(), preds[1]
        preds = preds.squeeze()
        mainloss = self.loss_ce(preds, center_label)
        return mainloss

    def set_model_noTrain(self):
        for p in self.parameters():
            p.requires_grad = False

    def mainClassifier(self, features):
        out = self.Elu( self.conv1(features) )
        out = self.softmax2d( self.conv2(out) )
        return out

def restore_from_torch(model, model_file):
    pretrained_dict = torch.load(model_file)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return model

def transfer_encoder_to_model(model, encoder):
    # where encoder is part of model
    pretrained_dict = encoder.state_dict()
    model_dict = model.state_dict()

    # merge the encoder dict to model:
    for k, v in pretrained_dict.items():
        if k in model_dict:
            model_dict[k] = v

    # 2. load the merged state dict
    model.load_state_dict(model_dict)
    return model

class Discriminator(nn.Module):
    def __init__(self, ):
        super().__init__()
        in_channel = 96
        n_filters = 32
        n_class = 1

        modules = []

        modules.append(nn.Conv2d(in_channels=in_channel, out_channels=n_filters*4, kernel_size=(3, 3))) #23
        modules.append( nn.BatchNorm2d(n_filters*4) )
        modules.append( nn.ReLU() )

        modules.append(nn.Conv2d(in_channels=n_filters * 4, out_channels=n_filters * 4, kernel_size=(3, 3)))  #21
        modules.append(nn.BatchNorm2d(n_filters * 4))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(in_channels=n_filters * 4, out_channels=n_filters * 8, kernel_size=(3, 3)))   #19
        modules.append(nn.BatchNorm2d(n_filters * 8))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(in_channels=n_filters * 8, out_channels=n_filters * 4, kernel_size=(3, 3)))    #17
        modules.append(nn.BatchNorm2d(n_filters * 4))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(in_channels=n_filters * 4, out_channels=n_filters * 2, kernel_size=(3, 3)))  # 15
        modules.append(nn.BatchNorm2d(n_filters * 2))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(in_channels=n_filters * 2, out_channels=n_filters , kernel_size=(3, 3)))  # 13
        modules.append(nn.BatchNorm2d(n_filters ))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(in_channels=n_filters, out_channels=n_class, kernel_size=(3, 3)))  # 11
        modules.append(nn.BatchNorm2d(n_class))
        modules.append(nn.ReLU())

        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=1*11*11, out_features=1))

        self.module_seq = nn.Sequential( *modules )

    def forward(self, x):
        out = self.module_seq(x)
        return out

if __name__ == "__main__":
    # print(np.random.rand(3,2))
    A = torch.rand(16, 3, 155, 155)
    m = Generator()
    # m = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
    out = m(A)
    d = Discriminator()
    neck = NeckNet()
    out_feature = neck.mainClassifier(out)

    dout = d(out)
    print( A.shape, out.shape, out_feature.shape, dout.shape)
    print(torch.mean(out))