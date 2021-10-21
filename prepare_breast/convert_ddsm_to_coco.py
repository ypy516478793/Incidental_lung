import os
import glob
import numpy as np
import cv2
import mmcv
import pandas
import warnings

# from config.cfg_loader import proj_paths_json
from pycocotools import mask as coco_api_mask


# def convert_npz_to_png(data_path):
#     for dir in glob.glob(os.path.join(data_path, '*')):
#         num_files = len(glob.glob(os.path.join(dir, '*')))
#         if num_files < 2:
#             print(os.path.basename(dir), num_files)
#             continue
# 
#         save_path = os.path.join(dir, os.path.basename(dir)+'.png')
#         if not os.path.exists(save_path):
#             mamm_img = np.load(os.path.join(dir, "image.npz"),
#                                allow_pickle=True)["image"]
#             cv2.imwrite(save_path, mamm_img)


def mask2polygon(_mask):
    contours, hierarchy = cv2.findContours(
        _mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    return segmentation


def area(_mask):
    rle = coco_api_mask.encode(np.asfortranarray(_mask))
    area = coco_api_mask.area(rle)
    return area


def get_info_lesion(df, ROI_ID):
    _, _, patient_id, left_or_right, image_view, abnormality_id = ROI_ID.split(
        '_')
    rslt_df = df[(df['patient_id'] == ('P_' + patient_id)) &
                 (df['left or right breast'] == left_or_right) &
                 (df['image view'] == image_view) &
                 (df['abnormality id'] == int(abnormality_id))]

    return rslt_df


def convert_ddsm_to_coco(out_file, data_root, annotation_filename):
    save_path = os.path.join(data_root, out_file)
    if os.path.exists(save_path):
        warnings.warn(f"{save_path} has already existed")
        return

    images = []
    annotations = []
    obj_count = 0

    df = pandas.read_csv(os.path.join(data_root, annotation_filename))

    for idx, dir_path in enumerate(mmcv.track_iter_progress(glob.glob(os.path.join(data_root, '*')))):
        filename = os.path.basename(dir_path)
        img_path = os.path.join(dir_path, filename + '.png')
        if not os.path.exists(img_path):
            continue
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=os.path.join(filename, filename+'.png'),
            height=height,
            width=width))

        bboxes = []
        labels = []
        masks = []

        for roi_idx, mask_path in enumerate(glob.glob(os.path.join(dir_path, 'mask*.npz'))):
            while True:
                roi_idx += 1

                rslt_df = get_info_lesion(df, f'{filename}_{roi_idx}')

                if len(rslt_df) == 0:
                    print(f'No ROI was found for ROI_ID: {filename}_{roi_idx}')
                    continue

                label = rslt_df['pathology'].to_numpy()[0]
                if label == 'MALIGNANT':
                    cat_id = 0
                elif label in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
                    cat_id = 1
                else:
                    raise ValueError(
                        f'Label: {label} is unrecognized for ROI_ID: {filename}_{roi_idx}')

                break

            mask_arr = np.load(mask_path, allow_pickle=True)["mask"]
            seg_poly = mask2polygon(mask_arr)
            seg_area = area(mask_arr)

            flat_seg_poly = [el for sublist in seg_poly for el in sublist]
            px = flat_seg_poly[::2]
            py = flat_seg_poly[1::2]
            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            seg_poly = [[el + 0.5 for el in poly] for poly in seg_poly]

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=cat_id,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=seg_area,
                segmentation=seg_poly,
                iscrowd=0)

            annotations.append(data_anno)

            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': 'malignant'}, {'id': 1, 'name': 'benign'}])
    mmcv.dump(coco_format_json, os.path.join(data_root, out_file))


# def read_annotation_json(json_file):
#     data = mmcv.load(json_file)
# 
#     categories = data['categories']
# 
#     img_annotations = []
# 
#     for img_data in data['images']:
#         img_id = img_data['id']
# 
#         annotations = [
#             annotation for annotation in data['annotations'] if annotation['id'] == img_id]
# 
#         img_annotations.append((img_data, annotations))
# 
#     return img_annotations, categories
# 
# 
# def save_detection_gt_for_eval(data_root, detection_gt_root):
#     img_annotations, categories = read_annotation_json(json_file=os.path.join(
#         data_root, 'annotation_coco_with_classes.json'))
# 
#     print(len(img_annotations))
#     for img, anns in mmcv.track_iter_progress(img_annotations):
#         print(img['file_name'], len(anns))
#         img_filename, _ = os.path.splitext(os.path.basename(img['file_name']))
# 
#         save_path = os.path.join(
#             detection_gt_root, f'{img_filename}.txt')
#         if os.path.exists(save_path):
#             continue
#         with open(save_path, 'w') as f:
#             for ann in anns:
#                 x, y, w, h = (str(el) for el in ann['bbox'])
#                 c = [el['name']
#                      for el in categories if el['id'] == ann['category_id']][0]
#                 f.write(' '.join((c, x, y, w, h, '\n')))


if __name__ == '__main__':
    # data_root = proj_paths_json['DATA']['root']
    # processed_cbis_ddsm_root = os.path.join(
    #     data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])

    processed_cbis_ddsm_root = './' ##! specify the path of CBIS-DDSM processed data !##

    mass_train_root = os.path.join(processed_cbis_ddsm_root, 'mass', 'train') # path to `train` directory of processed CBIS-DDSM (the data you have sent me before)
    mass_test_root = os.path.join(processed_cbis_ddsm_root, 'mass', 'test')   # path to `test` directory of processed CBIS-DDSM

    # convert_npz_to_png(data_path=mass_train_root) # convert .npz files to .png format
    # convert_npz_to_png(data_path=mass_test_root)

    convert_ddsm_to_coco(out_file='annotation_coco_with_classes.json',
                         data_root=mass_train_root,
                         annotation_filename='mass_case_description_train_set.csv') # the annotation file can be downloaded from the CBIS-DDSM website and must 
																					# be placed in the `mass_train_root`

    convert_ddsm_to_coco(out_file='annotation_coco_with_classes.json',
                         data_root=mass_test_root,
                         annotation_filename='mass_case_description_test_set.csv') # same as above but for `mass_test_root`

    # experiment_root = proj_paths_json['EXPERIMENT']['root']
    # processed_cbis_ddsm_detection_gt_root = os.path.join(
    #     experiment_root,
    #     proj_paths_json['EXPERIMENT']['mmdet_processed_CBIS_DDSM']['root'],
    #     proj_paths_json['EXPERIMENT']['mmdet_processed_CBIS_DDSM']['det_gt'])
    # train_det_gt_root = os.path.join(
    #     processed_cbis_ddsm_detection_gt_root, 'train')
    # test_det_gt_root = os.path.join(
    #     processed_cbis_ddsm_detection_gt_root, 'test')
    # if not os.path.exists(train_det_gt_root):
    #     os.makedirs(train_det_gt_root, exist_ok=True)
    # if not os.path.exists(test_det_gt_root):
    #     os.makedirs(test_det_gt_root, exist_ok=True)

    # save_detection_gt_for_eval(
    #     data_root=mass_train_root, detection_gt_root=train_det_gt_root)
    # save_detection_gt_for_eval(
    #     data_root=mass_test_root, detection_gt_root=test_det_gt_root)
