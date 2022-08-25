import os
import torch
from torch import nn

import numpy as np
from skimage.measure import label
from scipy.ndimage.morphology import binary_closing, generate_binary_structure

import pickle
def load_lasagne_weights(fname):
    all_weights = list()
    with open(fname, 'rb') as f:
        while True:
            try:
                values = pickle.load(f, encoding='bytes')
                all_weights.append(values)
            except EOFError:
                break
    return all_weights

def getLargestCC(segmentation, connectivity=None, Ifclosing=False):
    labels = label(segmentation, connectivity=connectivity)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    if Ifclosing:
        largestCC = binary_closing(largestCC, generate_binary_structure(3,2))
    return largestCC

def random_item(items):
    return items[np.random.randint(len(items))]

def count_parameters(model):
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()

    return num_params

def loss_fun(preds, labels, deep_sv=True):
    loss_ce = nn.CrossEntropyLoss()
    center_label = labels[:, 0, labels.shape[2] // 2, labels.shape[3] // 2].long()
    if isinstance(preds, tuple):
        preds, preds_aux = preds[0].squeeze(), preds[1]

    mainloss = loss_ce(preds, center_label)

    if deep_sv:
        for i in range(3):
            predi = preds_aux[i]
            predi_flatten = torch.flatten(predi.permute(1,0,2,3), start_dim=1).permute(1,0)
            labeli_flaten = torch.flatten(labels[:,i,:,:]).long()
            lossi = loss_ce( predi_flatten, labeli_flaten )
            mainloss += 0.05*lossi

    return mainloss

def sensitivity_fun(preds, labels ):
    if isinstance( preds, tuple):
        preds, labels = torch.clone(preds[0]), torch.clone((labels))
    else:
        preds, labels = torch.clone(preds), torch.clone((labels))
    center_label = labels[:, 0, labels.shape[2] // 2, labels.shape[3] // 2].long() >0.5
    center_pred = torch.argmax(preds.squeeze(), dim=1) > 0.5

    # sensitivity = TP / (TP+ FN)
    return (torch.sum( center_label[center_pred] )+1e-5) / (torch.sum(center_label)+1e-5)

def accuracy_fun(preds, labels, binary=False):
    if isinstance( preds, tuple):
        preds, labels = torch.clone(preds[0]), torch.clone((labels))
    else:
        preds, labels = torch.clone(preds), torch.clone((labels))
    center_label = labels[:, 0, labels.shape[2] // 2, labels.shape[3] // 2].long()
    # center_pred = torch.argmax(preds[:,:, preds.shape[2]//2, preds.shape[3]//2], 1)
    # center_pred = torch.argmax( preds[:, :, preds.shape[2] // 2, preds.shape[3] // 2], dim=1)
    center_pred = torch.argmax(preds.squeeze(), dim=1)
    if binary:
        center_label = center_label > 0.5
        center_pred = center_pred > 0.5
        return torch.sum(center_pred==center_label) / preds.shape[0]
    else:
        return torch.sum(center_pred == center_label) / preds.shape[0]

def get_T( imf ):
    #imf is the image file with full path
    uid = os.path.split( imf )[-1][:-4]
    dir = os.path.split( imf )[-2]
    im_fold = os.path.split(dir)[-1]

    dir = dir.replace( im_fold, 'threshold')
    file = '{}/{}.txt'.format(dir, uid)
    l2 = np.loadtxt(file)
    mean, std = l2[6], l2[7]

    return int( mean+3*std )
    # return mean+3*std


def Evaluation_fun(test, truth, voxel_volume = 1):
    # test, truth are the mask of test and truth
    labels_dict = {8: 'LAD', 9: "LCX", 10: "RCA", 15: "TAC", 53: "MVC", 58: "AVC"}
    out_dict = {}

    # calculate evaluation for CAC
    maski_truth = (truth >7) & (truth <11)
    maski_test = (test >= 8) & (test <= 10)
    N_test = np.sum(maski_test)
    N_truth = np.sum(maski_truth)
    TP = np.sum(maski_test & maski_truth)
    FP = N_test - TP
    FN = N_truth - TP
    sensitivity = (TP + 1e-5) / (TP + FN + 1e-5)
    F1 = (2 * TP + 1e-5) / (2 * TP + FP + FN + 1e-5)

    out_dict['CAC_sensitivity'] = sensitivity
    out_dict['CAC_FPvolume'] = FP * voxel_volume
    out_dict['CAC_F1'] = F1


    # calculate evaluations per label
    for k in labels_dict:
        maski_truth = truth == k
        maski_test = test == k
        TP = np.sum( maski_test & maski_truth)
        FP = np.sum(maski_test) - TP
        FN = np.sum(maski_truth) - TP
        sensitivity  = (TP + 1e-5) / (TP + FN + 1e-5)
        F1 = (2*TP+ 1e-5)/(2*TP + FP+ FN + 1e-5)

        out_dict[ '{}_sensitivity'.format(labels_dict[k]) ] = sensitivity
        out_dict['{}_FPvolume'.format(labels_dict[k])] = FP*voxel_volume
        out_dict['{}_F1'.format(labels_dict[k])] = F1

    out_dict['CAC volume detection'] = N_test * voxel_volume
    out_dict['CAC volume reference'] = N_truth * voxel_volume
    return out_dict

def count_TP_lesion( label_truth, cac_test):
    num_truth = np.max(label_truth)
    TP = 0
    for i in range(1, num_truth+1):
        if np.max(cac_test[label_truth==i]) > 0:
            TP+=1

    return TP



def Evaluation_leison_fun(test, truth):
    # test, truth are the mask of test and truth
    # labels_dict = {8: 'LAD', 9: "LCX", 10: "RCA", 15: "TAC", 53: "MVC", 58: "AVC"}
    out_dict = {}

    # calculate evaluation for CAC
    cac_truth = (truth >7) & (truth <11)
    cac_test = (test > 7) & (test < 11)

    labels_truth = label( cac_truth )
    labels_test = label( cac_test )
    num_truth, num_test = np.max(labels_truth), np.max(labels_test)
    if num_truth == 0:
        TP = 0
    else:
        TP = count_TP_lesion(labels_truth, cac_test)
    FN = num_truth - TP
    FP = num_test - TP
    FP = 0 if FP < 0 else FP

    sensitivity = (TP + 1e-5) / (TP + FN + 1e-5)
    F1 = (2 * TP + 1e-5) / (2 * TP + FP + FN + 1e-5)

    out_dict['CAC_Lsensitivity'] = sensitivity
    out_dict['CAC_LFP'] = FP
    out_dict['CAC_LF1'] = F1
    out_dict['Leisions in Detection'] = num_test
    out_dict['Leisions in Reference'] = num_truth

    return out_dict


# import numpy as np
from skimage import measure


def regiongrow_lesions(mask, img, spacing, max_vol=1000, Threshold = 130, minVoxel=None):
    mask = mask.astype(int)
    img_bin = ( img >= Threshold ).astype(int)
    img_labels = measure.label(img_bin)
    ovl_labels, num_lesions = measure.label(mask, return_num=True)
    maxvol = np.ceil(max_vol / np.product(spacing)).astype(int)

    for label in range(1, num_lesions + 1):
        location = np.where(ovl_labels == label)

        label_img = []
        for n in range(len(location[0])):
            label_img.append(img_labels[location[0][n], location[1][n], location[2][n]])
        label_img = np.array(label_img)
        label_img = label_img[np.array(np.nonzero(label_img))]
        if len(label_img[0]) > 0:
            label_img = np.argmax(np.bincount(np.asarray(label_img[0])))
        else:
            continue

        location_lesion = np.array(np.where(img_labels == label_img))

        label_mask = []
        for n in range(len(location_lesion[0])):
            label_mask.append(mask[location_lesion[0][n], location_lesion[1][n], location_lesion[2][n]])
        label_mask = np.array(label_mask)
        label_mask = label_mask[np.array(np.nonzero(label_mask))]
        #        print(label_mask)
        if len(label_mask[0]) > 0:
            label_mask = np.argmax(np.bincount(np.asarray(label_mask[0])))
        else:
            continue

        # discard if lesion grows larger than maximum volume

        # if len(location_lesion[0]) > maxvol or len(location_lesion[0]) < minVoxel:
        if minVoxel is None:
            if len(location_lesion[0]) > maxvol:
                # treat the lesion as False-positive and remove it.
                mask[location_lesion[0, :], location_lesion[1, :], location_lesion[2, :]] = 0

            else:
                mask[location_lesion[0, :], location_lesion[1, :], location_lesion[2, :]] = label_mask
        else:
            if len(location_lesion[0]) > maxvol or len(location_lesion[0]) < minVoxel:
                # treat the lesion as False-positive and remove it.
                mask[location_lesion[0, :], location_lesion[1, :], location_lesion[2, :]] = 0
            else:
                mask[location_lesion[0, :], location_lesion[1, :], location_lesion[2, :]] = label_mask
    return mask.astype(int)

def remove_small_lesions(mask, spacing, min_voxel = 2):
    mask_bin = mask > 0
    ovl_labels, num_lesions = measure.label(mask_bin, return_num=True)
    th = min_voxel
    # th = (np.ceil(min_voxel / np.product(spacing)).astype(int))
    # if th < 2:
    #     th = 2

    for label in range(1, num_lesions + 1):
        location = np.where(ovl_labels == label)
        if len(location[0]) <= th:
            for n in range(len(location[0])):
                mask[location[0][n], location[1][n], location[2][n]] = 0
    return mask



if __name__ == "__main__":
    # print(np.random.rand(3,2))
    # A = torch.rand(8,7)
    # B = torch.randint( 7, (8,1, 2,2))
    # sen = sensitivity_fun(A, B)
    # print(A, B[:,:,1,1], sen)
    # print( sen)

    imf = "D:/data/ContrastCT_IDR/resampled_CCTA/images_resampled/3089.mhd"
    dir = os.path.split(imf)[0]
    print(os.path.split(dir))
    print( get_T(imf) )