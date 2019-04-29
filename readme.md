# Watermark Removal
This code attempts to remove watermarks in a very simple and naive way given a
large set of images all with the same watermark. It is built to used as a
baseline for determining the effectiveness of other, more sophisticated
techniques.

## Process
Watermarks are typically applied in the following way:

watermaked_image = alpha * watermark + (1 - alpha) * image    (1)

And can be removed by estimating the alpha and watermark mask via:

image = (watermarked_image - alpha * watermark) / (1 - alpha)    (2)

## Algorithm
### 1. Get the median of the gradients:
This step takes the median of all the single image gradients. Assuming the
watermark is in the same location on each image, this leaves behind the gradient
of the watermark.

### 2. Watermark is reconstructed:
The gradients are then used to reconstuct the original watermark by
integrating them. This reconstructed watermark is the alpha value. The alpha
thresholded to get the mask.

### 3. Recover the image:
The image is recovered using equation 2.

## How to use:
```
usage: rm_watermarks.py [-h] [-o OUTPUT_FOLDER] [-f] images_folder

positional arguments:
  images_folder         Path to a watermarked image dataset folder

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FOLDER, --output-folder OUTPUT_FOLDER
                        Reconstucted images folder
  -f, --force           Deletes the contents of the output folder
```

## Warning
This repository is not to enfringe on any copyright laws. The development
and release of this code is for academic purposes and for the development of
more robust watermark techniques only.
