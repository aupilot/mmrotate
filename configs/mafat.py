# the new config inherits the base configs to highlight the necessary modification
_base_ = './rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'

# 1. dataset settings
dataset_type = 'DOTADataset'
classes = ('small_vehicle', 'medium_vehicle', 'large_vehicle', 'bus',
           'double_trailer_truck', 'container', 'heavy_equipment',
           'pylon', 'small_aircraft', 'large_aircraft', 'small_vessel', 'medium_vessel', 'large_vessel')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/Volumes/Datasets/MAFAT.cropped/annfiles',
        img_prefix='/Volumes/Datasets/MAFAT.cropped/images'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/Volumes/Datasets/MAFAT.cropped/annfiles',
        img_prefix='/Volumes/Datasets/MAFAT.cropped/images'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/Volumes/Datasets/MAFAT/converted/labelTxt',
        img_prefix='/Volumes/Datasets/MAFAT/converted/images'))

# 2. model settings
model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        # explicitly over-write all the `num_classes` field from default 15 to 5.
        num_classes=len(classes)))
