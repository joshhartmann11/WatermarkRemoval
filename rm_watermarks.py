"""
This code attempts to remove watermarks given a
large set of images watermarked with the following process:

watermaked_image = alpha * watermark + (1 - alpha) * image

The watermark can be removed by estimating the alpha and watermark mask with:
image = (watermarked_image - alpha * watermark) / (1 - alpha)

To estimate alpha and mask:
1. Calculate the median of the x,y gradients of the image
2. Intgrate the median of the gradients -> alpha
3. Threshold alpha -> mask
4. Recover the imag using equation 2
"""
import os
import argparse
import cv2
import numpy as np

DEBUG = True

def debug(func, *args, **kwargs):
    if DEBUG:
        func(*args, **kwargs)


def save_image(path, image):
    cv2.imwrite(path, image)


def load_images(dir):
    image_paths = [os.path.join(dir, f) for f in os.listdir(dir)]
    images = np.array([cv2.imread(i) for i in image_paths])
    return images


def normalize_images(images):
    no = images.copy()
    no = no.astype(np.float64)
    no = (no - np.min(no)) / (np.max(no) - np.min(no))
    return no


def get_gradient_median(images, ksize=1):
    grad_x = np.empty(images.shape)
    grad_y = np.empty(images.shape)
    for i in range(images.shape[0]):
        grad_x[i] = cv2.Sobel(images[i], cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y[i] = cv2.Sobel(images[i], cv2.CV_64F, 0, 1, ksize=ksize)
    wm_grad_x = np.median(grad_x, axis=0)
    wm_grad_y = np.median(grad_y, axis=0)
    return wm_grad_x, wm_grad_y


def integrate_gradients(grad_x, grad_y):
    elev = 0
    int_x = np.empty(grad_x.shape)
    for z in range(grad_x.shape[2]):
        for x in range(grad_x.shape[0]):
            for y in range(grad_x.shape[1]):
                elev_prev = elev
                elev += grad_x[x, y, z]
                int_x[x, y, z] = (elev + elev_prev) / 2
            elev = 0

    int_y = np.empty(grad_y.shape)
    for z in range(grad_y.shape[2]):
        for y in range(grad_y.shape[1]):
            for x in range(grad_y.shape[0]):
                elev_prev = elev
                elev += grad_y[x, y, z]
                int_y[x, y, z] = (elev + elev_prev) / 2
            elev = 0

    int_y[int_y < 0] = 0
    int_x[int_x < 0] = 0

    return int_x, int_y


def estimate_watermark(int_x, int_y):
    alpha = np.mean((int_x, int_y), axis=0)
    mean = np.mean(alpha, axis=2)
    mean_8 = 255 * (mean - np.min(mean)) / (np.max(mean) - np.min(mean))
    mean_8 = mean_8.astype(np.uint8)
    _, mask = cv2.threshold(mean_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = normalize_images(mask)
    mask = np.concatenate((mask[..., None], mask[..., None], mask[..., None]), axis=2)
    return alpha, mask


def subtract_watermark(wm, alpha, images, dir):
    os.makedirs(dir)
    for i in range(images.shape[0]):
        result = (images[i] - (wm * alpha)) / (1 - alpha)
        path = os.path.join(dir, f"image_{i}.png")
        save_image(path, result*255)


def main(images_folder, output_folder):
    debug(print, "Loading images...")
    images = load_images(images_folder)

    debug(print, f"Normalizing {images.shape[0]} images...")
    images_no = normalize_images(images)

    debug(print, "Getting the median of gradients...")
    grad_x, grad_y = get_gradient_median(images_no)

    debug(print, "Integrating the median gradients...")
    int_x, int_y = integrate_gradients(grad_x, grad_y)

    debug(print, "Estimating the watermark...")
    alpha, wm = estimate_watermark(int_x, int_y)

    debug(save_image, "debug/alpha.png", alpha * 255)
    debug(save_image, "debug/watermark.png", wm * 255)
    debug(print, f"Subtracting watermarks to \"{output_folder}\"...")
    subtract_watermark(wm, alpha, images_no, output_folder)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_folder", help="Path to a watermarked image dataset folder", type=str)
    parser.add_argument("-o", "--output-folder", help="Reconstucted images folder", type=str, default="output")
    parser.add_argument("-f", "--force", help="Deletes the contents of the output folder", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.force and os.path.exists(args.output_folder):
        for i in os.listdir(args.output_folder):
            os.remove(os.path.join(args.output_folder, i))
        os.rmdir(args.output_folder)
    main(args.images_folder, args.output_folder)
