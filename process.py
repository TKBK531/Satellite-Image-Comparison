import cv2
import numpy as np
import matplotlib.pyplot as plt
import images as img


# Function to calculate the histogram of a given channel
def get_channel_histogram(channel, bins, range):
    hist = cv2.calcHist([channel], [0], None, [bins], range)
    hist = cv2.normalize(hist, hist).flatten()
    return hist


# Function to set dynamic color ranges based on the histogram
def dynamic_color_ranges(hsv_image):
    # Extract the H, S, and V channels
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]

    # Calculate histograms for H, S, and V channels
    hue_hist = get_channel_histogram(h_channel, 180, [0, 180])
    s_hist = get_channel_histogram(s_channel, 256, [0, 256])
    v_hist = get_channel_histogram(v_channel, 256, [0, 256])

    # Define initial ranges for green and blue
    green_range = (35, 85)  # Typical range for green in HSV
    blue_range = (85, 130)  # Typical range for blue in HSV

    # Find the dominant peaks in the hue histogram within these ranges
    green_peak = np.argmax(hue_hist[green_range[0] : green_range[1]]) + green_range[0]
    blue_peak = np.argmax(hue_hist[blue_range[0] : blue_range[1]]) + blue_range[0]

    # Adjust the ranges around the peaks
    lower_green_hue, upper_green_hue = (green_peak - 10, green_peak + 10)
    lower_blue_hue, upper_blue_hue = (blue_peak - 10, blue_peak + 10)

    # Find the dominant peaks in the S and V histograms
    s_peak = np.argmax(s_hist)
    v_peak = np.argmax(v_hist)

    # Adjust the ranges around the peaks
    lower_s, upper_s = (s_peak - 50, s_peak + 50)
    lower_v, upper_v = (v_peak - 50, v_peak + 50)

    # Ensure the ranges are within valid bounds
    lower_s = max(0, lower_s)
    upper_s = min(255, upper_s)
    lower_v = max(0, lower_v)
    upper_v = min(255, upper_v)

    # Define the lower and upper boundaries for green and blue with dynamic S and V
    lower_green = np.array([lower_green_hue, lower_s, lower_v])
    upper_green = np.array([upper_green_hue, upper_s, upper_v])

    lower_blue = np.array([lower_blue_hue, lower_s, lower_v])
    upper_blue = np.array([upper_blue_hue, upper_s, upper_v])

    return lower_green, upper_green, lower_blue, upper_blue


# Function to segment the image based on dynamic HSV ranges
def segment_image(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get dynamic color ranges
    lower_green, upper_green, lower_blue, upper_blue = dynamic_color_ranges(hsv_image)

    # Create masks for green (vegetation) and blue (water) areas
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Create a land mask (anything that is not green or blue)
    land_mask = cv2.bitwise_not(cv2.bitwise_or(green_mask, blue_mask))

    return green_mask, blue_mask, land_mask


# Function to calculate the RGB histogram of an image
def get_rgb_histogram(image):
    # Calculate the histogram for each channel
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

    # Normalize the histograms
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    return hist_r, hist_g, hist_b


# Function to calculate the percentage change between two masks
def calculate_percentage_change(mask1, mask2):
    area1 = np.sum(mask1 > 0)
    area2 = np.sum(mask2 > 0)
    change = ((area2 - area1) / area1) * 100 if area1 != 0 else 0
    return change


# Function to compare histograms and generate a report
def compare_histograms(hist1, hist2, channel_name):
    diff = np.abs(hist1 - hist2)
    total_diff = np.sum(diff)
    return f"Total difference in {channel_name} channel: {total_diff:.2f}"


# Load and segment multiple images
def process_images(image_paths):

    if len(image_paths) != 2:
        raise ValueError("image_paths array must contain exactly 2 images.")

    # Load the images
    image1 = cv2.imread(image_paths[0])
    image2 = cv2.imread(image_paths[1])

    # Convert the images to HSV color space
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Segment the images
    green_mask1, blue_mask1, land_mask1 = segment_image(image1)
    green_mask2, blue_mask2, land_mask2 = segment_image(image2)

    # Convert to RGB format for displaying with matplotlib
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Calculate the hue histograms
    hue_hist1 = get_channel_histogram(hsv_image1[:, :, 0], 180, [0, 180])
    hue_hist2 = get_channel_histogram(hsv_image2[:, :, 0], 180, [0, 180])

    # Calculate the RGB histograms
    hist_r1, hist_g1, hist_b1 = get_rgb_histogram(image1_rgb)
    hist_r2, hist_g2, hist_b2 = get_rgb_histogram(image2_rgb)

    # Compare histograms and generate a report
    report = []

    # Calculate percentage changes in vegetation, water, and land
    veg_change = calculate_percentage_change(green_mask1, green_mask2)
    water_change = calculate_percentage_change(blue_mask1, blue_mask2)
    land_change = calculate_percentage_change(land_mask1, land_mask2)

    report.append(f"Percentage change in vegetation: {veg_change:.2f}%")
    report.append(f"Percentage change in water: {water_change:.2f}%")
    report.append(f"Percentage change in land: {land_change:.2f}%")

    # Save the report to a text file
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

    plt.figure(figsize=(16, 9))

    for i in range(8):
        plt.subplot(2, 4, i + 1)
        if "Vegetation" in titles[i]:
            cmap = "Greens"
        elif "Water" in titles[i]:
            cmap = "Blues"
        else:
            cmap = "Oranges"

        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Histogram data with colors
    histograms = [
        (hue_hist1, hue_hist2, "Hue Histogram", "Hue Value", "purple"),
        (hist_r1, hist_r2, "Red Channel Histogram", "Pixel Value", "r"),
        (hist_g1, hist_g2, "Green Channel Histogram", "Pixel Value", "g"),
        (hist_b1, hist_b2, "Blue Channel Histogram", "Pixel Value", "b"),
    ]

    # Visualize the histograms
    plt.figure(figsize=(16, 9))

    for i, (hist1, hist2, title, xlabel, color) in enumerate(histograms, 1):
        plt.subplot(2, 2, i)
        plt.plot(hist1, color=color, label=f"{title.split()[0]} - Image 1")
        plt.plot(
            hist2,
            color=color,
            linestyle="dashed",
            label=f"{title.split()[0]} - Image 2",
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_paths = [img.before_zambia_drought, img.after_zambia_drought]
    # image_paths = [img.before_donana_park, img.after_donana_park]
    process_images(image_paths)
