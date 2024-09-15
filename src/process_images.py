import cv2
from .image_processing import *
from .report_generation import *
from .plots import *


def process_images(image_paths):
    """Load and segment multiple images, then calculate and plot changes."""
    if len(image_paths) != 2:
        raise ValueError("image_paths array must contain exactly 2 images.")

    try:
        image1 = cv2.imread(image_paths[0])
        image2 = cv2.imread(image_paths[1])

        if image1 is None or image2 is None:
            raise FileNotFoundError("One or both images could not be loaded.")

        image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        green_mask1, blue_mask1, land_mask1 = segment_image(image1)
        green_mask2, blue_mask2, land_mask2 = segment_image(image2)

        green_mask1 = refine_mask(green_mask1)
        blue_mask1 = refine_mask(blue_mask1)
        land_mask1 = refine_mask(land_mask1)

        green_mask2 = refine_mask(green_mask2)
        blue_mask2 = refine_mask(blue_mask2)
        land_mask2 = refine_mask(land_mask2)

        (
            green_contours1,
            blue_contours1,
            land_contours1,
            contours_green1,
            contours_blue1,
            contours_land1,
        ) = contour_detection(image1_rgb, green_mask1, blue_mask1, land_mask1)
        (
            green_contours2,
            blue_contours2,
            land_contours2,
            contours_green2,
            contours_blue2,
            contours_land2,
        ) = contour_detection(image2_rgb, green_mask2, blue_mask2, land_mask2)

        green_area1 = sum(cv2.contourArea(c) for c in contours_green1)
        green_area2 = sum(cv2.contourArea(c) for c in contours_green2)
        blue_area1 = sum(cv2.contourArea(c) for c in contours_blue1)
        blue_area2 = sum(cv2.contourArea(c) for c in contours_blue2)
        land_area1 = sum(cv2.contourArea(c) for c in contours_land1)
        land_area2 = sum(cv2.contourArea(c) for c in contours_land2)

        generate_report(
            green_area1, green_area2, blue_area1, blue_area2, land_area1, land_area2
        )

        images = [
            image1_rgb,
            image2_rgb,
            green_mask1,
            green_mask2,
            blue_mask1,
            blue_mask2,
            land_mask1,
            land_mask2,
        ]
        titles = [
            "Original Image 1",
            "Original Image 2",
            "Vegetation Mask 1",
            "Vegetation Mask 2",
            "Water Mask 1",
            "Water Mask 2",
            "Land Mask 1",
            "Land Mask 2",
        ]

        plot_images(images, titles)

        # Plot vegetation contours
        plot_contour_images(
            green_contours1,
            green_contours2,
            "Vegetation Contours Image 1",
            "Vegetation Contours Image 2",
        )

        # Plot water contours
        plot_contour_images(
            blue_contours1,
            blue_contours2,
            "Water Contours Image 1",
            "Water Contours Image 2",
        )

        # Plot land contours
        plot_contour_images(
            land_contours1,
            land_contours2,
            "Land Contours Image 1",
            "Land Contours Image 2",
        )

        plot_percentage_changes(
            green_area1, green_area2, blue_area1, blue_area2, land_area1, land_area2
        )

    except Exception as e:
        print(f"An error occurred: {e}")
