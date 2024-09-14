import cv2
import numpy as np
import matplotlib.pyplot as plt
import images as img


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

    hue_hist = get_channel_histogram(h_channel, 180, [0, 180])
    s_hist = get_channel_histogram(s_channel, 256, [0, 256])
    v_hist = get_channel_histogram(v_channel, 256, [0, 256])

    return hue_hist, s_hist, v_hist


def dynamic_color_ranges(hsv_image):
    """Set dynamic color ranges based on the histogram."""
    hue_hist, s_hist, v_hist = calculate_histograms(hsv_image)

    green_range = (35, 85)
    blue_range = (85, 130)

    green_peak = np.argmax(hue_hist[green_range[0] : green_range[1]]) + green_range[0]
    blue_peak = np.argmax(hue_hist[blue_range[0] : blue_range[1]]) + blue_range[0]

    lower_green_hue, upper_green_hue = max(0, green_peak - 10), min(
        179, green_peak + 10
    )
    lower_blue_hue, upper_blue_hue = max(0, blue_peak - 10), min(179, blue_peak + 10)

    s_peak = np.argmax(s_hist)
    v_peak = np.argmax(v_hist)

    lower_s, upper_s = max(0, s_peak - 50), min(255, s_peak + 50)
    lower_v, upper_v = max(0, v_peak - 50), min(255, v_peak + 50)

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
    edges = cv2.Canny(image, 100, 200)
    return edges


def morphological_operations(mask):
    """Apply morphological operations to refine the mask."""
    kernel = np.ones((5, 5), np.uint8)
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


def plot_contour_images(image1, image2, title1, title2):
    """Plot contour images in a 1x2 grid."""
    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()


def plot_images(images, titles):
    """Plot the images with titles."""
    plt.figure(figsize=(16, 9))
    for i in range(len(images)):
        plt.subplot(2, 4, i + 1)
        cmap = (
            "Greens"
            if "Vegetation" in titles[i]
            else (
                "Blues"
                if "Water" in titles[i]
                else "Oranges" if "Land" in titles[i] else None
            )
        )
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()


def plot_percentage_changes(
    green_area1, green_area2, blue_area1, blue_area2, land_area1, land_area2
):
    """Plot the percentage changes in vegetation, water, and land areas."""
    plt.figure(figsize=(16, 9))

    plt.subplot(1, 3, 1)
    plt.bar(
        ["Before", "After"],
        [green_area1, green_area2],
        color="green",
    )
    plt.xlabel("Image")
    plt.ylabel("Area")
    plt.title("Vegetation Area")

    plt.subplot(1, 3, 2)
    plt.bar(
        ["Before", "After"],
        [blue_area1, blue_area2],
        color="blue",
    )
    plt.xlabel("Image")
    plt.ylabel("Area")
    plt.title("Water Area")

    plt.subplot(1, 3, 3)
    plt.bar(
        ["Before", "After"],
        [land_area1, land_area2],
        color="orange",
    )
    plt.xlabel("Image")
    plt.ylabel("Area")
    plt.title("Land Area")

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()


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

        green_mask1 = morphological_operations(green_mask1)
        blue_mask1 = morphological_operations(blue_mask1)
        land_mask1 = morphological_operations(land_mask1)

        green_mask2 = morphological_operations(green_mask2)
        blue_mask2 = morphological_operations(blue_mask2)
        land_mask2 = morphological_operations(land_mask2)

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

        veg_change = calculate_percentage_change(green_area1, green_area2)
        water_change = calculate_percentage_change(blue_area1, blue_area2)
        land_change = calculate_percentage_change(land_area1, land_area2)

        report = [
            f"Percentage change in vegetation: {veg_change:.2f}%",
            f"Percentage change in water: {water_change:.2f}%",
            f"Percentage change in land: {land_change:.2f}%",
        ]
        with open("image_comparison_report.txt", "w") as file:
            file.write("\n".join(report))

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
        plot_percentage_changes(
            green_area1, green_area2, blue_area1, blue_area2, land_area1, land_area2
        )

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

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    image_paths = [img.before_zambia_drought, img.after_zambia_drought]
    process_images(image_paths)
