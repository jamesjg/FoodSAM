import argparse
import os.path as osp
import os
import tempfile
import mmcv
import torch
import numpy as np
import sys
sys.path.append('.')
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import  inference_segmentor, init_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
def save_result(img_path,
                result,
                color_list_path,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                vis_save_name='pred_vis.png',
                mask_save_name='pred_mask.png'):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        color_list_path: path of (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. 
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img_path)
    img = img.copy()
    seg = result[0]

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    color_list = np.load(color_list_path)
    color_list[0] = [238, 239, 20] # set special backgrond color
    for label, color in enumerate(color_list):
        color_seg[seg == label, :] = color_list[label]
    # convert to BGR
    # color_seg = color_seg[..., ::-1]

    img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, os.path.join(out_file, vis_save_name))
        mmcv.imwrite(seg, os.path.join(out_file, mask_save_name))


    if not (show or out_file):
        print('show==False and out_file is not specified, only '
                        'result image will be returned')
        return img
    
def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name
def single_gpu_test(model,
                    data_loader,
                    color_list_path,
                    show=False,
                    out_dir=None,
                    efficient_test=False,):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'].split('.')[0])
                else:
                    out_file = None

                save_result(
                    img_show,
                    result,
                    color_list_path=color_list_path,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def semantic_predict(data_root, img_dir, ann_dir, config, options, aug_test,  checkpoint, eval_options, output, color_list_path, 
                     img_path=None):
    cfg = mmcv.Config.fromfile(config)
    if options is not None:
        cfg.merge_from_dict(options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True


    #test only one image
    if img_path:
        model = init_segmentor(config, checkpoint)
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        result = inference_segmentor(model, img_path)
        output_dir = os.path.join(output, os.path.basename(img_path).split('.')[0])
        save_result(
                    img_path,
                    result,
                    color_list_path=color_list_path,
                    show=False,
                    out_file=output_dir)

    else:
        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model,checkpoint, map_location='cpu')
        # build the dataloader
        cfg.data.test.data_root = data_root
        cfg.data.test.img_dir = img_dir
        cfg.data.test.ann_dir = ann_dir
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        efficient_test = False
        if eval_options is not None:
            efficient_test = eval_options.get('efficient_test', False)

        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, color_list_path,  out_dir=output,
                                efficient_test=efficient_test)

    
    




