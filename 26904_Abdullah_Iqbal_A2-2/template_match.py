import cv2 as cv
import numpy as np
import os


# NOTE: these values are just placeholders, you may need to adjust them
_DEFAULT_THRESHOLD = 0.70
_DEFAULT_NMS_WINDOW = 7


def non_maximal_suppression_2d(values: np.ndarray, window_size: int = _DEFAULT_NMS_WINDOW) -> np.ndarray:
    """Performs non-maximal suppression on the given values matrix. That is, anywhere that
    values[i,j] is a *local maximum* within a (window x window) box, it is kept, and anywhere
    that it is not a local maximum it is 'suppressed' (set to zero).

    The original values matrix is not modified, but a new matrix is returned.
    """
    suppressed = np.zeros_like(values)
    rows, cols = values.shape
    print(f"original ye hai: {rows,cols}")
    for i in range(rows):
        for j in range(cols):
            start_row = max(0, i - window_size)
            end_row = min(rows, i + window_size + 1)
            start_col = max(0, j - window_size)
            end_col = min(cols, j + window_size + 1)
            window = values[start_row:end_row, start_col:end_col]
            # print(window)
            if values[i, j] == np.max(window):
                suppressed[i, j] = values[i, j]

    return suppressed


def apply_threshold(values: np.ndarray, threshold: float = _DEFAULT_THRESHOLD) -> np.ndarray:
    """Applies a threshold to the given values matrix. That is, anywhere that values[i,j] is greater
    than the threshold, it is kept with the same value, and anywhere that it is below the threshold
    it is set to zero.

    The original values matrix is not modified, but a new matrix is returned.
    """
    new_score = np.copy(values)
    for i in range(len(new_score)):
        for x in range(len(values[0])):
            if new_score[i][x] < threshold:
                new_score[i][x] = 0
    return new_score


def find_objects_by_template_matching(
        image: np.ndarray, template: np.ndarray, threshold: float, nms_window: int
) -> list[tuple[int, int]]:
    """Finds copies of the given template in the given image by template-matching. Returns a list of
    (x, y) coordinates of the top-left corner of each match. The main steps of this function are:

    1. Use cv.matchTemplate to get a score map. This map is a 2D array where score[i,j] gives a
       measure of how well the template matches the image at position (i,j). Depending on the choice
       of 'method' in cv.matchTemplate, the score can be positive or negative, and the best match
       can be either the maximum or minimum value in the score map.
    2. Normalize the score map so that the best match is 1 and the worst match is 0
    3. Apply the threshold to the score map to throw away any values below the threshold (i.e. set
       pixels to zero if their score is below the threshold). Use a call to apply_threshold for
       this.
    3. Use non-maximal suppression to keep only local maxima in a (nms_window x nms_window) window
       (i.e. set pixels to zero if they are not the maximum value among their neighbors). Use a call
       to non_maximal_suppression_2d for this.
    4. Use np.where() to find all remaining nonzero pixels --> these are our matches.
    """
    score_map = cv.matchTemplate(scene, template, cv.TM_CCOEFF)
    score_map = cv.normalize(score_map,None,alpha=0,beta=1,norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
    new_score_map = apply_threshold(score_map)
    nms = non_maximal_suppression_2d(new_score_map)
    y_coords, x_coords = np.where(nms > 0)
    return list(zip(x_coords, y_coords))
    # raise NotImplementedError("Your code here.")


def visualize_matches(scene: np.ndarray, obj_hw: tuple[int, int], xy: list[tuple[int, int]]):
    """Visualizes the matches found by find_objects_by_template_matching."""
    count = len(xy)
    h, w = obj_hw
    for x, y in xy:
        cv.rectangle(scene, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Add text in the bottom left corner by using x=10 and y=the height of the scene - 20 pixels
    cv.putText(
        scene,
        f"Found {count} matches",
        (10, scene.shape[0] - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )

    cv.imshow("Matches", scene)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to the scene", required=True)
    parser.add_argument("--template", help="Path to the template", required=True)
    parser.add_argument(
        "--threshold", help="Threshold for matches", type=float, default=_DEFAULT_THRESHOLD
    )
    parser.add_argument(
        "--nms-window",
        help="Window size for non-maximal suppression",
        type=int,
        default=_DEFAULT_NMS_WINDOW,
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not os.path.exists(args.template):
        raise FileNotFoundError(f"Image not found: {args.template}")

    if args.nms_window % 2 == 0:
        raise ValueError("The window size must be odd.")

    if args.nms_window < 1:
        raise ValueError("The window size must be greater than or equal to 1.")

    scene = cv.imread(args.image)
    object = cv.imread(args.template)
    xy = find_objects_by_template_matching(scene, object, args.threshold, args.nms_window)
    print("scene shape: ", scene.shape)
    print("coin shape: ", object.shape)

    visualize_matches(scene, object.shape[:2], xy)


