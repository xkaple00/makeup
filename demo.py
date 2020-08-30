import argparse
from pathlib import Path

from PIL import Image
from psgan import Inference
from fire import Fire
import numpy as np

import faceutils as futils
from psgan import PostProcess, PreProcess
from setup import setup_config, setup_argparser
import sys
import cv2


def main(save_path='results/transferred_image',
            save_path_image='results/transferred_image_raw',
            save_path_face='results/transferred_face',
            save_path_image_1 = 'results/save_path_image_1',
            save_path_image_2 = 'results/save_path_image_2'):
    parser = setup_argparser()
    parser.add_argument(
        "--source_path",
        # default="./assets/images/non-makeup/xfsy_0106.png",
        default="../dataset/womakeup2.jpg",
        metavar="FILE",
        help="path to source image")
    parser.add_argument(
        "--reference_path_1",
        default="../dataset/makeup1.jpg",
        metavar="FILE",
        help="path to reference image 1")
    parser.add_argument(
        "--reference_path_2",
        default="../dataset/makeup2.jpg",
        metavar="FILE",
        help="path to reference image 2")
    parser.add_argument(
        "--reference_dir",
        default="assets/images/makeup",
        help="path to reference images")
    parser.add_argument(
        "--transfer_lips",
        default=1,
        help="transfer makeup from lips")
    parser.add_argument(
        "--transfer_skin",
        default=1,
        help="transfer makeup from skin")
    parser.add_argument(
        "--transfer_eyes",
        default=1,
        help="transfer makeup from eyes")
    parser.add_argument(
        "--n_ref_images",
        default=1,
        help="number of reference images to transfer makeup from, accessible values 1 or 2")
    parser.add_argument(
        "--speed",
        action="store_true",
        help="test speed")
    parser.add_argument(
        "--device",
        default="cpu",
        help="device used for inference")
    parser.add_argument(
        "--model_path",
        default="assets/models/G.pth",
        help="model for loading")

    args = parser.parse_args()
    config = setup_config(args)

    assert args.n_ref_images in [1,2], "Number of reference images must be 1 or 2"

    # Using the second cpu
    inference = Inference(
        config, args.device, args.model_path)
    postprocess = PostProcess(config)
    preprocess = PreProcess(config)

    source = Image.open(args.source_path).convert("RGB")

    # reference_paths = list(Path(args.reference_dir).glob("*"))
    # np.random.shuffle(reference_paths)
    # for reference_path in reference_paths:

    for blending_ratio in np.arange(0., 1.2, 0.2):
        print('reference_path_1', args.reference_path_1)
        print('reference_path_2', args.reference_path_2)

        # if not reference_path_1.is_file():
        #     print(reference_path_1, "is not a valid file.")
        #     continue

        # if not reference_path_2.is_file():
        #     print(reference_path_2, "is not a valid file.")
        #     continue

        reference_1 = Image.open(args.reference_path_1).convert("RGB")
        if args.n_ref_images == 2:
            reference_2 = Image.open(args.reference_path_2).convert("RGB")
        else:
            reference_2 = source

        # Transfer the psgan from reference to source.
        image_1, face_1, binary_masks_1 = inference.transfer(source, reference_1, with_face=True)
        image_2, face_2, binary_masks_2 = inference.transfer(source, reference_2, with_face=True)

        source_crop = source.crop(
            (face_1.left(), face_1.top(), face_1.right(), face_1.bottom()))

        source_crop = source_crop.resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), Image.ANTIALIAS)

        print('args.transfer_lips', type(args.transfer_lips))

        binary_mask_1 = np.expand_dims((binary_masks_1[0] * 1 + binary_masks_1[1] * 1 + binary_masks_1[2] * 0).cpu().numpy()[0],-1)

        cv2.imwrite('results/binary_mask_1.png', binary_mask_1*255)
        binary_mask_2 = np.ones(binary_mask_1.shape) - binary_mask_1
        cv2.imwrite('results/binary_mask_2.png', binary_mask_2*255)

        image = Image.blend(image_1, image_2, 1-blending_ratio)

        image_1.save(save_path_image_1 + '.png')
        image_2.save(save_path_image_2 + '.png')
        
        # image_cv = cv2.addWeighted(np.array(image_1)[~binary_mask_1]==source, blending_ratio, np.array(image_2)[~binary_mask_1]==source, 1.-blending_ratio, gamma=0, dtype = cv2.CV_32F)

        image_cv = cv2.addWeighted(np.where(binary_mask_1>0.5, image_1, source_crop), blending_ratio, np.where(binary_mask_1>0.5, image_2, source_crop), 1.-blending_ratio, gamma=0, dtype = cv2.CV_32F)
        
        # image_cv = image_cv[:,:,::-1] #BGR to RGB

        image_cv = Image.fromarray(np.uint8(image_cv))

        image_cv = postprocess(source_crop, image_cv)

        image_cv.save('results/'+'image_cv'+str(round(blending_ratio,2))+'.png')

        print('image, face')
        print("binary_masks_1", binary_masks_1.shape)
        image.save(save_path_image + '_' + str(round(blending_ratio, 2)) + '.png')

        source_crop = source.crop(
            (face_1.left(), face_1.top(), face_1.right(), face_1.bottom()))
        image = postprocess(source_crop, image)
        image.save(save_path + '_' + str(round(blending_ratio, 2)) + '.png')

        if args.speed:
            import time
            start = time.time()
            for _ in range(100):
                inference.transfer(source, reference_1)
            print("Time cost for 100 iters: ", time.time() - start)


if __name__ == '__main__':
    main()
