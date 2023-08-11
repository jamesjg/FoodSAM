import sys

# sys.path.append('.')
# sys.path.append('./SAM')
# sys.path.append('./mmseg')
import argparse
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from mmcv.utils import DictAction
import argparse
import os
import numpy as np
from typing import Any, Dict, List
import shutil, logging
from FoodSAM_tools.panoramic_segment import panoramic_segment
from FoodSAM_tools.object_detection import object_detect
from FoodSAM_tools.predict_semantic_mask import semantic_predict

parser = argparse.ArgumentParser(
    description=(
        "Runs SAM automatic mask generation and  instance segmentation on an input image or directory of images, "
    )
)
parser.add_argument(
    "--data_root",
    type=str,
    default='dataset/FoodSeg103/Images',
    help="Path to folder of images and masks.",
)
parser.add_argument(
    "--img_dir",
    type=str,
    default='img_dir/test',
    help="Path to img.",
)
parser.add_argument(
    "--ann_dir",
    type=str,
    default='ann_dir/test',
    help="Path to ann.",
)

parser.add_argument(
    "--img_path",
    type=str,
    default=None,
    help="Path to a single input image.",
)
parser.add_argument(
    "--output",
    type=str,
    default='Output/Panoramic_Results',
    help=(
        "Path to the directory where masks will be output. Output will be a folder"
    ),
)
parser.add_argument(
    "--SAM_checkpoint",
    type=str,
    default="ckpts/sam_vit_h_4b8939.pth",
    help="The path to the SAM checkpoint to use for mask generation.",
)
parser.add_argument('--semantic_config', default="configs/SETR_MLA_768x768_80k_base.py", help='test config file path of mmseg')
parser.add_argument('--semantic_checkpoint', default="ckpts/SETR_MLA/iter_80000.pth", help='checkpoint file of mmseg')
parser.add_argument(
    "--model-type",
    type=str,
    default='vit_h',
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


parser.add_argument(
    '--aug-test', action='store_true', help='Use Flip and Multi scale aug')

parser.add_argument(
    '--options', nargs='+', action=DictAction, help='custom options')
parser.add_argument(
    '--eval-options',
    nargs='+',
    action=DictAction,
    help='custom options for evaluation')
parser.add_argument('--color_list_path', type=str, default="FoodSAM/FoodSAM_tools/color_list.npy")

parser.add_argument(
    "--category_txt",
    default="FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt" ,
)
parser.add_argument(
    "--num_class",
    default=104,
)
parser.add_argument(
    "--area_thr",
    default=0 ,
)
parser.add_argument(
    "--ratio_thr",
    default=0.5 ,
)
parser.add_argument(
    "--top_k",
    default=80 ,
)

parser.add_argument(
    "--detection_config",
    default="configs/Unified_learned_OCIM_RS200_6x+2x.yaml",
    metavar="FILE",
    help="path to config file",
    )
parser.add_argument(
    "--confidence-threshold",
    type=float,
    default=0.5,
    help="Minimum score for instance predictions to be shown",
)
parser.add_argument(
    "--opts",
    help="Modify config options using the command-line 'KEY VALUE' pairs",
    default=["MODEL.WEIGHTS", "ckpts/Unified_learned_OCIM_RS200_6x+2x.pth"] ,
    nargs=argparse.REMAINDER,
)

amg_settings = parser.add_argument_group("AMG Settings")
amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    os.makedirs(os.path.join(path, "sam_mask"), exist_ok=True)
    masks_array = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        masks_array.append(mask.copy())
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, "sam_mask" ,filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)

    masks_array = np.stack(masks_array, axis=0)
    np.save(os.path.join(path, "sam_mask" ,"masks.npy"), masks_array)
    metadata_path = os.path.join(path, "sam_metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))
    return


def create_logger(save_folder):
    
    log_file = f"sam_process.log"
    final_log_file = os.path.join(save_folder, log_file)

    logging.basicConfig(
        format=
        '[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(final_log_file, mode='w'),
            logging.StreamHandler()
        ])                        
    logger = logging.getLogger()
    print(f"Create Logger success in {final_log_file}")
    return logger

def main(args: argparse.Namespace) -> None:
    print(args)
    os.makedirs(args.output, exist_ok=True)
    logger = create_logger(args.output)
    logger.info("running sam!")
    sam = sam_model_registry[args.model_type](checkpoint=args.SAM_checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    
    assert args.data_root or args.img_path
    if args.img_path:
        targets = [args.img_path]
    else:
        img_folder = os.path.join(args.data_root, args.img_dir)
        targets = [
            f for f in os.listdir(img_folder) if not os.path.isdir(os.path.join(img_folder, f))
        ]
        targets = [os.path.join(img_folder, f) for f in targets]

    for t in targets:
        logger.info(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            logger.error(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = generator.generate(image)
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        os.makedirs(save_base, exist_ok=True)
        write_masks_to_folder(masks, save_base)
        shutil.copyfile(t, os.path.join(save_base, "input.jpg"))
    logger.info("sam done!\n")

    logger.info("running semantic seg model!")
    semantic_predict(args.data_root, args.img_dir, args.ann_dir, args.semantic_config, args.options, args.aug_test, args.semantic_checkpoint, args.eval_options, args.output, args.color_list_path, args.img_path)
    logger.info("semantic predict done!\n")

    logger.info("running object detection model")
    object_detect(args)
    logger.info("object detection done!\n")


    logger.info("panoramic segmentation!")
    panoramic_segment(args.output, args.category_txt, args.color_list_path, num_class=args.num_class, area_thr=args.area_thr, ratio_thr=args.ratio_thr, top_k=args.top_k)
    logger.info("panoramic segmentation done!\n")

    logger.info("The results saved in {}!\n".format(args.output))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
