import cv2
from .config import *
import numpy as np


def get_channel_histogram(channel, bins, range):
    """Calculate and normalize the histogram of a given channel."""
    hist = cv2.calcHist([channel], [0], None, [bins], range)
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def calculate_histograms(hsv_image):
    """Calculate histograms for each channel in the HSV image."""
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]

    hue_hist = get_channel_histogram(h_channel, HUE_BINS, HUE_RANGE)
    s_hist = get_channel_histogram(s_channel, SAT_BINS, SAT_RANGE)
    v_hist = get_channel_histogram(v_channel, VAL_BINS, VAL_RANGE)

    return hue_hist, s_hist, v_hist


def dynamic_color_ranges(hsv_image):
    """Set dynamic color ranges based on the histogram."""
    hue_hist, s_hist, v_hist = calculate_histograms(hsv_image)

    green_range = GREEN_RANGE
    blue_range = BLUE_RANGE

    green_peak = np.argmax(hue_hist[green_range[0] : green_range[1]]) + green_range[0]
    blue_peak = np.argmax(hue_hist[blue_range[0] : blue_range[1]]) + blue_range[0]

    lower_green_hue, upper_green_hue = max(0, green_peak - HUE_PEAK_ADJUST), min(
        179, green_peak + HUE_PEAK_ADJUST
    )
    lower_blue_hue, upper_blue_hue = max(0, blue_peak - HUE_PEAK_ADJUST), min(
        179, blue_peak + HUE_PEAK_ADJUST
    )

    s_peak = np.argmax(s_hist)
    v_peak = np.argmax(v_hist)

    lower_s, upper_s = max(0, s_peak - SAT_PEAK_ADJUST), min(
        255, s_peak + SAT_PEAK_ADJUST
    )
    lower_v, upper_v = max(0, v_peak - VAL_PEAK_ADJUST), min(
        255, v_peak + VAL_PEAK_ADJUST
    )

    lower_green = np.array([lower_green_hue, lower_s, lower_v])
    upper_green = np.array([upper_green_hue, upper_s, upper_v])

    lower_blue = np.array([lower_blue_hue, lower_s, lower_v])
    upper_blue = np.array([upper_blue_hue, upper_s, upper_v])

    return lower_green, upper_green, lower_blue, upper_blue


def segment_image(image):
    """Segment the image based on dynamic HSV ranges."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green, upper_green, lower_blue, upper_blue = dynamic_color_ranges(hsv_image)

    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    land_mask = cv2.bitwise_not(cv2.bitwise_or(green_mask, blue_mask))

    return green_mask, blue_mask, land_mask


def calculate_percentage_change(area1, area2):
    """Calculate the percentage change between two areas."""
    change = ((area2 - area1) / area1) * 100 if area1 != 0 else 0
    return change


def edge_detection(image):
    """Apply Canny edge detection to the image."""
    edges = cv2.Canny(image, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    return edges


def refine_mask(mask):
    """Apply morphological operations to refine the mask."""
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def contour_detection(image, green_mask, blue_mask, land_mask):
    """Detect and draw filled contours on the image."""
    contours_green, _ = cv2.findContours(
        green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_blue, _ = cv2.findContours(
        blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_land, _ = cv2.findContours(
        land_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    image_with_green_contours = image.copy()
    image_with_blue_contours = image.copy()
    image_with_land_contours = image.copy()

    cv2.drawContours(
        image_with_green_contours, contours_green, -1, (0, 255, 0), thickness=cv2.FILLED
    )
    cv2.drawContours(
        image_with_blue_contours, contours_blue, -1, (0, 0, 255), thickness=cv2.FILLED
    )
    cv2.drawContours(
        image_with_land_contours, contours_land, -1, (255, 165, 0), thickness=cv2.FILLED
    )

    return (
        image_with_green_contours,
        image_with_blue_contours,
        image_with_land_contours,
        contours_green,
        contours_blue,
        contours_land,
    )
