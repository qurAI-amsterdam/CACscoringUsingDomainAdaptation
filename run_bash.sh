
# work with cardic NCCT and CCTA mixed data
#train with adv_cls_mmd_loss

CUDA_VISIBLE_DEVICES=7 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --N_Diters 20 --clip_value 0.5 > ./nohup/TrainGAN_adv_cls_mmd_ND25_C0.5.out 2> ./nohup/TrainGAN_adv_cls_mmd_ND25_C0.5.err < /dev/null &

CUDA_VISIBLE_DEVICES=1 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 0.5 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=7 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 0.5 --w_mmd 0.1 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.1.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.1.err < /dev/null &
CUDA_VISIBLE_DEVICES=5 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 0.5 --w_mmd 0.2 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.2.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.2.err < /dev/null &

#train with adv_cls_loss
CUDA_VISIBLE_DEVICES=1 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 0.5 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=7 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 1.0 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=5 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 0.8 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=1 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 0.2 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &

CUDA_VISIBLE_DEVICES=1 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 0.0 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=2 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 1.5 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=3 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 2 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=4 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 3 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=5 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 4 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=7 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 5 --w_mmd 0.0 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &


CUDA_VISIBLE_DEVICES=0 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 2 --w_mmd 0.05 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=2 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 2 --w_mmd 0.1 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=1 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 2 --w_mmd 0.2 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=5 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 2 --w_mmd 0.3 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &
CUDA_VISIBLE_DEVICES=7 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 2 --w_mmd 0.4 > ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.out 2> ./nohup/TrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &

# work with NLST_CCTA mixed data
CUDA_VISIBLE_DEVICES=7 no_proxy=localhost nohup python trainGAN_adv_cls_MMD.py --w_cls 2 --w_mmd 0.0 --inputdir ./data/chestNCCT_CCTA/mixCT --scratchdir ./data/chestNCCT_CCTA/resampled_mixCT --N_Diters 20 > ./nohup/NLSTTrainGAN_adv_cls_mmd_wcls0.5_wmmd0.0.err < /dev/null &