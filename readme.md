# Watermark Removal

## Process
Watermarks are typically applied in the following way:

watermaked_image = alpha * watermark + (1 - alpha) * image    (1)

And can be removed by estimating the alpha and watermark mask with:

image = (watermarked_image - alpha * watermark) / (1 - alpha)    (2)

## Algorithm
  1. Calculate the median of the x, y gradients
  2. Derivative of gradients -> alpha
  3. Threshold alpha -> watermark mask
  4. Recover image with equation 2

## Usage:
```
rm_watermarks.py -h
```
