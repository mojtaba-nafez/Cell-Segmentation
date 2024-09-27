# Cell-Classification Team
## Segmentation Project

Step 1:
    To download & preprocess the data:

        `python data\preprocessing.py`

Step 2:
    To train the model:

        `python train.py --model <model-name> --[other options]`


Note:
    You can see the optional arguments that are available to pass to train.py file by running this:
        `python train.py --model <model-name> --help`

--> Currently, you can use below models as <model-name>:
        unet: for the simple UNet model.
        unet_bn: for the UNet with Batch Normalization and more Conv2D layers.
        deeplabv3plus_resnet: for the DeepLabV3+ with ResNet backbone.
        deeplabv3plus_xception: for the DeepLabV3+ with XCepttion backbone.
        deeplabV3plus_attention: for the DeepLabV3+ with ResNet backbone and attention implemented.

*About dataset: 2018 Data Science Bowl
    This dataset contains a large number of segmented nuclei images. The images were
    acquired under a variety of conditions and vary in the cell type, magnification, and
    imaging modality (brightfield vs. fluorescence). The dataset is designed to challenge an
    algorithm's ability to generalize across these variations.

    Each image is represented by an associated ImageId. Files belonging to an image are
    contained in a folder with this ImageId. Within this folder are two subfolders:

    	images: contains the image file.
    	masks: contains the segmented masks of each nucleus. This folder is only included in the training set. Each mask contains one
    	nucleus.
    Masks are not allowed to overlap (no pixel belongs to two masks).
    -  note that in this project, for simplicity, we merged all masks of each image together.
    -  for more info about the dataset and the competition take a look at below page:
            https://www.kaggle.com/c/data-science-bowl-2018/data