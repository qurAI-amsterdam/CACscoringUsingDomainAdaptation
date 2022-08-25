import SimpleITK as sitk
import numpy as np
import glob, os
import pandas as pd

from utils.util import Evaluation_fun

# dir = "D:/data/Cardiac_Exams_UMCU/resampled_CSCT"
# Gtruth = 'annotations_resampled'
# # model = 'predictions_SwinTransformer_SwinTransformer_128_1012_700'
# model = 'predictions_SwinTransformer_splitChannel_SwinTransformer_splitChannel_128_1012_700'
# # model = 'predictions_combined_DeeplySupervised_FullImage_AllKernels_1012_700'

# dir = "D:/data/NLST/resampled_CT"
# ctDir = 'D:/data/NLST/CT'

dir = "D:/data/ContrastCT_IDR/resampled_CCTA"
ctDir = 'D:/data/ContrastCT_IDR/CCTA'

# dir = "D:/data/Contrast_CT/resampled_CCTA"
# dir = "D:/data/Cardiac_Exams_UMCU/resampled_CSCT"
# ctDir = "D:/data/Cardiac_Exams_UMCU/CSCT"

# Gtruth = 'annotations_resampled'

# model = "predictions_combined_DeeplySupervised_NLST_1012_700"
# model = "predictions_combined_DeeplySupervised_NLST_originalsettings_1012_700"
# model = "predictions_combined_DeeplySupervised_NLST_originalsettings2_initializer_1012_700"
# model = "predictions_combined_DeeplySupervised_NLST_theano_1012_700"
# model = "predictions_combined_DeeplySupervised_NLST_newImp_adam_1012_700"
# model = "predictions_SwinTransformer_SwinTransformer_128_NLST_1012_700"
# model = "predictions_combined_DeeplySupervised_FullImage_AllKernels_NLST_1012_700"
# model = "predictions_combined_DeeplySupervised_NLST_newImp_adam_1012_700"
# model = "predictions_GAN_mixCT_clipDweights_useTrainedG_firstD1K_DeeplySupervised_NLST_newImp_adam_1012_700"
# model = "predictions_GAN_mixCT_clipDweights_useTrainedG_firstD2.0K_clip0.5_EpochN200_lr5.0e-4_Bz16_DeeplySupervised_NLST_newImp_adam_1012_700_V0"
# model = "predictions_GAN_mixCT_clipDweights_useTrainedG_firstD1K_DeeplySupervised_NLST_newImp_adam_1012_700_Threshold_rmSingleVoxels"
# model = "predictions_GAN_mixCT_clipDweights_useTrainedG_firstD2.0K_clip0.5_EpochN200_lr5.0e-4_Bz16_DeeplySupervised_NLST_newImp_adam_1012_700"
# model = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700"
# model = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700_bb"
# model = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700_rg"
# model = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700_rg_nrs"
# model = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700_rg_nrs_500"
# model = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700_rg_nrs_500_sl_1_heart"
# model = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700_rg_nrs_500_heart"
# model, Gtruth = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700", 'annotations_resampled'
# model, Gtruth = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700_rg_nrs_500_heart", 'annotations_resampled'
# model, Gtruth = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700_rg_nrs_500_sl_1_heart", 'annotations_resampled_sl_1'
# model, Gtruth = "predictions_GAN_mixCT_adv_cls_MMD_EpochN200_lr5.0e-4wcls_2.0wmmd0.0_DeeplySupervised_NLST_newImp_adam_1012_700_rg_nrs_500_sl_2_heart", 'annotations_resampled_sl_2'

model, Gtruth = "predictions_GAN_NLST_mixCT_adv_cls_loss_MMD_firstD2.0K_clip0.1_EpochN200_lr5.0e-4_DeeplySupervised_NLST_newImp_adam_1012_700", 'annotations_resampled_sl_2'

files = glob.glob('{}/{}/calcium_masks/*.mhd'.format(dir, model))

csvf = "{}/dataset.csv".format(ctDir)
df_cases =  pd.read_csv(csvf )
# print(df_cases)
testing_group = df_cases[df_cases['Subset']=='testing']
uids = testing_group["Series Instance UID"]

output = pd.DataFrame()
for uid in uids:
    ftest = '{}/{}/calcium_masks/{}.mhd'.format(dir, model, uid)
    ftruth = ftest.replace(model, Gtruth)
    ftruth = ftruth.replace('calcium_masks', '')
    try:
        im_truth = sitk.ReadImage(ftruth)
        mask_truth = sitk.GetArrayFromImage(im_truth)
        mask_test = sitk.GetArrayFromImage(sitk.ReadImage(ftest))
    except:
        print("can not load data: ", ftest)
        continue
    # calculate evaluation for various category
    try:
        voxel_volume = np.product(im_truth.GetSpacing())
        out_dict = Evaluation_fun(mask_test, mask_truth, voxel_volume=voxel_volume)
    except:
        print("Evaluation failed")
        continue

    case = os.path.split(ftruth)[-1]
    out_dict["case"] = case.replace('.mhd', ';')
    str = ""
    for k, v in out_dict.items():
        str += "{}:{};".format(k, v)
    output = output.append(out_dict, ignore_index=True)
    print(str)

output.to_csv('./docs/{}_eva.csv'.format(model), encoding='utf-8')
ax = output.plot.scatter( x = 'CAC volume detection', y = 'CAC volume reference' )
fig = ax.get_figure()
fig.savefig('./docs/{}_volume.jpg'.format(Gtruth))
print( output.mean() )