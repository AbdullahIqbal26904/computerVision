import numpy as np
import cv2 as cv  # noqa: F401


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with uint8 values in the range [0, 255] and
    return a copy of the image with data type float32 and values in the range [0, 1]
    """
    image_float = image.astype(np.float32)
    image_float = image_float / 255.0
    return image_float


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with float32 values in the range [0, 1] and
    return a copy of the image with uint8 values in the range [0, 255]. Values outside the range
    should be clipped (i.e. a float of 1.1 should be converted to a uint8 of 255, and a float of
    -0.1 should be converted to a uint8 of 0).
    """
    image_uint = image * 255
    image_uint = np.clip(image_uint, 0, 255)  # clip function limits the value in between 0 & 255
    image_uint = image_uint.astype(np.uint8)
    return image_uint


def crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image cropped to the
    given rectangle. Any part of the rectangle that falls outside the image should be considered
    black (i.e. 0 intensity in all channels).
    """
    cropped_image = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for v in range(w):
            row, col = y + i, x + v
            if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:  # if image is in bound else keep zeros
                # print(image[row, col])
                cropped_image[i, v] = image[row, col]  # Copy pixel if within bounds
    return cropped_image


def scale_by_half_using_numpy(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image taking every
    other pixel in each row and column. For example, if the original image has shape (H, W, 3),
    the returned image should have shape (H // 2, W // 2, 3).
    """
    H, W, bgr = image.shape
    i = 0
    v = 0
    scaled_image = np.zeros((H // 2, W // 2, 3), dtype=np.uint8)
    for row in range(0, len(image), 2):
        for col in range(0, len(image[0]), 2):
            scaled_image[i][v] = image[row][col]
            v += 1
        i += 1
        v = 0
    # print(scaled_image.shape)
    return scaled_image


def scale_by_half_using_cv(image: np.ndarray) -> np.ndarray:
    """Using cv.resize, take an image and return a copy of the image scaled down by a factor of 2,
    mimicking the behavior of scale_by_half_using_numpy_slicing. Pay attention to the
    'interpolation' argument of cv.resize (see the OpenCV documentation for details).
    """
    height, width = image.shape[:2]

    scaled_image = cv.resize(image, (width // 2, height // 2), interpolation=cv.INTER_AREA)

    # print(resized_image.shape)
    return scaled_image


def horizontal_mirror_image(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image flipped
    horizontally (i.e. a mirror image). The behavior should match cv.flip(image, 1).
    """
    H, W, bgr = image.shape
    print(H, W, bgr)
    mirror_img = np.zeros_like(image)
    for row in range(0, H):
        for col in range(0, W // 2):
            temp = image[row][col]
            mirror_img[row][col] = image[row][(W - 1) - col]
            mirror_img[row][(W - 1) - col] = temp
    return mirror_img


def rotate_counterclockwise_90(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image rotated
    counterclockwise by 90 degrees. The behavior should match
    cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE).
    """
    # print(image.shape)
    new_img = np.zeros((1200, 1524, 3), dtype=np.uint8)
    for row in range(len(image)):
        for col in range(len(image[1])):
            new_img[col, row] = image[row, col]
    return new_img


def swap_b_r(image: np.ndarray) -> np.ndarray:
    """Given an OpenCV image in BGR channel format, return a copy of the image with the blue and red
    channels swapped. You may use any numpy or opencv functions you like.
    """
    # print(len(image))
    # print(len(image[0]))
    new_img = np.copy(image)
    for row in range(len(new_img)):
        for col in range(len(new_img[0])):
            temp = new_img[row][col][2]
            new_img[row][col][2] = new_img[row][col][0]
            new_img[row][col][0] = temp
    return new_img


def blues(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the blue
    channel
    """
    blue_array = np.zeros((len(image), len(image[0]), 3), dtype=np.uint8)
    for row in range(len(image)):
        for col in range(len(image[0])):
            blue_array[row][col][0] = image[row][col][0]
    return blue_array


def greens(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the green
    channel
    """
    green_array = np.zeros((len(image), len(image[0]), 3), dtype=np.uint8)
    for row in range(len(image)):
        for col in range(len(image[0])):
            green_array[row][col][1] = image[row][col][1]
    return green_array


def reds(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the red
    channel
    """
    red_array = np.zeros((len(image), len(image[0]), 3), dtype=np.uint8)
    for row in range(len(image)):
        for col in range(len(image[0])):
            red_array[row][col][2] = image[row][col][2]
    return red_array


def scale_saturation(image: np.ndarray, scale: float) -> np.ndarray:
    """Take an OpenCV image in BGR channel format. Convert to HSV and multiply the saturation
    channel by the given scale factor, then convert back to BGR.
    """
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    float_img = uint8_to_float(hsv_image)
    for i in range(len(image)):
        for x in range(len(image[0])):
            float_img[i][x][1] = float_img[i][x][1] * scale
    # print(hsv_image)
    uint_image = float_to_uint8(float_img)
    bgr_image = cv.cvtColor(uint_image, cv.COLOR_HSV2BGR)
    return bgr_image
    # raise NotImplementedError("your code here")


def grayscale(image: np.ndarray) -> np.ndarray:
    """Using numpy, reproduce the OpenCV function cv.cvtColor(image, cv.COLOR_BGR2GRAY) to convert
    the given image to grayscale. The returned image should still be in BGR channel format.
    """
    gray_array = np.zeros((len(image), len(image[1]), 3), dtype=np.uint8)
    for row in range(len(image)):
        for col in range(len(image[0])):
            gray_val = image[row][col][0] * 0.114 + image[row][col][1] * 0.587 + image[row][col][2] * 0.299
            gray_array[row][col][0] = gray_val
            gray_array[row][col][1] = gray_val
            gray_array[row][col][2] = gray_val
    return gray_array


def tile_bgr(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a 2x2 tiled copy of the image, with the
    original image in the top-left, the blue channel in the top-right, the green channel in the
    bottom-left, and the red channel in the bottom-right. If the original image has shape (H, W, 3),
    the returned image has shape (2 * H, 2 * W, 3).
    """
    blue_img = blues(image)
    red_img = reds(image)
    green_img = greens(image)
    top_row = np.hstack((image, blue_img))
    bottom_row = np.hstack((green_img, red_img))
    tiled_image = np.vstack((top_row, bottom_row))
    print(tiled_image.shape)
    return tiled_image


def main():
    image = cv.imread("bouquet.png")

    while True:
        print("\nChoose an option:")
        print("1 - Convert uint8 to float")
        print("2 - Convert float to uint8")
        print("3 - Crop Image (1st parameters)")
        print("4 - Crop Image (2nd parameters)")
        print("5 - Scale Image by Half (Numpy)")
        print("6 - Scale Image by Half (OpenCV)")
        print("7 - Mirror Image Horizontally")
        print("8 - Rotate Image Counterclockwise 90°")
        print("9 - Swap Blue and Red Channels")
        print("10 - Extract Blue Channel")
        print("11 - Extract Green Channel")
        print("12 - Extract Red Channel")
        print("13 - Scale Saturation (2.0)")
        print("14 - Scale Saturation (0.0)")
        print("15 - Convert to Grayscale")
        print("16 - Convert to title_bgr")
        print("0 - Exit")

        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Invalid input! Please enter a number.")
            continue

        processed_image = None

        if choice == 1:
            processed_image = uint8_to_float(image)
            print(f"Float conversion: {processed_image}")
            break
        elif choice == 2:
            processed_image = float_to_uint8(image)
            print(f"uint conversion: {processed_image}")
            break
        elif choice == 3:
            processed_image = crop(image, 373, 1424, 200, 100)
        elif choice == 4:
            processed_image = crop(image,-50, 500, 200, 200)
        elif choice == 5:
            processed_image = scale_by_half_using_numpy(image)
        elif choice == 6:
            processed_image = scale_by_half_using_cv(image)
        elif choice == 7:
            processed_image = horizontal_mirror_image(image)
        elif choice == 8:
            processed_image = rotate_counterclockwise_90(image)
        elif choice == 9:
            processed_image = swap_b_r(image)
        elif choice == 10:
            processed_image = blues(image)
        elif choice == 11:
            processed_image = greens(image)
        elif choice == 12:
            processed_image = reds(image)
        elif choice == 13:
            processed_image = scale_saturation(image, 2.0)
        elif choice == 14:
            processed_image = scale_saturation(image, 0.0)
        elif choice == 15:
            processed_image = grayscale(image)
        elif choice == 16:
            processed_image = tile_bgr(image)
        elif choice == 0:
            print("Exiting program.")
            break
        else:
            print("Invalid choice! Please enter a valid option.")
            continue

        if processed_image is not None and choice != 1 and choice != 2:
            cv.imshow("Processed Image", processed_image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("Error: The function did not return an image.")


if __name__ == "__main__":
    main()
