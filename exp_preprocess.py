import argparse
from ast import arg
import logging
import os
from datetime import datetime

import numpy as np

import evaluation.EvalScanNet as EvalScanNet
import preprocess.neuris_data as neuris_data
import utils.utils_geometry as GeoUtils
import utils.utils_image as ImageUtils
import utils.utils_io as IOUtils
import utils.utils_normal as NormalUtils
from confs.path import lis_name_scenes
from evaluation.renderer import render_depthmaps_pyrender

import glob
import cv2

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='private')
    parser.add_argument('--video_name', type=str, default=None)
    parser.add_argument('--video_ext', type=str, default='MOV')
    parser.add_argument('--ori_img_width', type=int, default=1920)
    parser.add_argument('--ori_img_height', type=int, default=1080)
    parser.add_argument('--no_crop_image', action='store_true')
    parser.add_argument('--crop_width', type=int, default=1360)
    parser.add_argument('--crop_height', type=int, default=1020)
    parser.add_argument('--reso_level', type=int, default=0)
    parser.add_argument('--no_split_image', action='store_true')
    parser.add_argument('--no_sample_image', action='store_true')
    parser.add_argument('--sample_interval', type=int, default=10)
    parser.add_argument('--no_sfm', action='store_true')
    parser.add_argument('--no_mask', action='store_true')
    parser.add_argument('--rvm_mask_dir', type=str, default=None)
    parser.add_argument('--no_prepare_neus', action='store_true')
    args = parser.parse_args()
    

    dataset_type = args.data_type
    
    if dataset_type == 'scannet':
        dir_root_scannet = '/media/hp/HKUCS2/Dataset/ScanNet'
        dir_root_neus = f'{dir_root_scannet}/sample_neus'

        for scene_name in lis_name_scenes:
            dir_scan = f'{dir_root_scannet}/{scene_name}'
            dir_neus = f'{dir_root_neus}/{scene_name}'
            neuris_data.prepare_neuris_data_from_scannet(dir_scan, dir_neus, sample_interval=6, 
                                                b_sample = True, 
                                                b_generate_neus_data = True,
                                                b_pred_normal = True, 
                                                b_detect_planes = False,
                                                b_unify_labels = False) 

    if dataset_type == 'private':
        video_name = args.video_name
        video_ext = args.video_ext
        original_size_img = (args.ori_img_width, args.ori_img_height)
        cropped_size_img = (args.crop_width, args.crop_height)
        reso_level = args.reso_level
        b_split_image = not args.no_split_image
        b_sample_image = not args.no_sample_image
        sample_interval = args.sample_interval
        b_sfm = not args.no_sfm
        b_crop_image = not args.no_crop_image
        rvm_mask_dir = args.rvm_mask_dir
        b_mask = not args.no_mask
        b_prepare_neus = not args.no_prepare_neus
        print(original_size_img, cropped_size_img)


        # example of processing iPhone video
        # put a video under folder tmp_sfm_mvs or put your images under tmp_sfm_mvs/images
        dir_neuris = f'/home/lixin/support_libs/NeuRIS/dataset/private/{video_name}'
        dir_neuris = os.path.abspath(dir_neuris)
        dir_sfm_mvs = os.path.abspath(f'{dir_neuris}/tmp_sfm_mvs')


        # split video into frames and sample images
        path_video = f'{dir_sfm_mvs}/{video_name}.{video_ext}'
        dir_split = f'{dir_sfm_mvs}/images_split'
        dir_mvs_sample = f'{dir_sfm_mvs}/images' # for mvs reconstruction
        dir_neuris_sample = f'{dir_sfm_mvs}/images_calibrated' # remove uncalbrated images
        dir_neuris_sample_cropped = f'{dir_neuris}/image'

        if b_split_image:
            ImageUtils.split_video_to_frames(path_video, dir_split)

        # sample images
        if b_sample_image:
            rename_mode = 'order_04d'
            ext_source = '.png'
            ext_target = '.png'
            ImageUtils.convert_images_type(dir_split, dir_mvs_sample, rename_mode,
                                            target_img_size = None, ext_source = ext_source, ext_target =ext_target,
                                            sample_interval = sample_interval)

        # SfM camera calibration
        if b_sfm:
            os.system(f'python ./preprocess/sfm_mvs.py --dir_mvs {dir_sfm_mvs} --image_width {original_size_img[0]} --image_height {original_size_img[1]} --reso_level {reso_level}')

        if b_crop_image:
            neuris_data.crop_images_neuris(dir_imgs = dir_neuris_sample, 
                                dir_imgs_crop = dir_neuris_sample_cropped, 
                                path_intrin = f'{dir_sfm_mvs}/intrinsics.txt', 
                                path_intrin_crop = f'{dir_neuris}/intrinsics.txt', 
                                crop_size = cropped_size_img)

            # crop depth
            if IOUtils.checkExistence(f'{dir_sfm_mvs}/depth_calibrated'):
                ImageUtils.crop_images(dir_images_origin = f'{dir_sfm_mvs}/depth_calibrated',
                                            dir_images_crop = f'{dir_neuris}/depth', 
                                            crop_size = cropped_size_img, 
                                            img_ext = '.npy')

        if b_mask:
            dir_mvs_mask_sample = f'{dir_sfm_mvs}/mask'
            dir_mvs_image_sample = f'{dir_sfm_mvs}/images'
            dir_neuris_mask_sample_cropped = f'{dir_neuris}/mask'
            os.makedirs(dir_mvs_mask_sample, exist_ok=True)
            os.makedirs(dir_neuris_mask_sample_cropped, exist_ok=True)
            img_names = sorted(glob.glob(f'{dir_mvs_image_sample}/*.png'))
            for img_full_name in img_names:
                img_name = img_full_name.split('/')[-1]
                mask_name = f"{rvm_mask_dir}/alpha/{img_name}"
                human_mask = cv2.imread(mask_name)
                scene_mask = 255 - human_mask
                cv2.imwrite(f"{dir_mvs_mask_sample}/{img_name}", scene_mask)
                # os.system(f"cp {rvm_mask_dir}/alpha/{img_name} {dir_mvs_mask_sample}/{img_name}")
            assert b_crop_image
            ImageUtils.crop_images(dir_images_origin = dir_mvs_mask_sample,
                                   dir_images_crop = dir_neuris_mask_sample_cropped, 
                                   crop_size = cropped_size_img,
                                   img_ext = '.png')

        if b_prepare_neus:
            neuris_data.prepare_neuris_data_from_private_data(dir_neuris, cropped_size_img, 
                                                            b_generate_neus_data = True,
                                                                b_pred_normal = True, 
                                                                b_detect_planes = False)

    print('Done')
