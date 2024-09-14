import matplotlib.pyplot as plt


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

    # Data for plotting
    areas = [
        (green_area1, green_area2, "Vegetation Area", "green"),
        (blue_area1, blue_area2, "Water Area", "blue"),
        (land_area1, land_area2, "Land Area", "orange"),
    ]

    # Loop through the data and create subplots
    for i, (area1, area2, title, color) in enumerate(areas, start=1):
        plt.subplot(1, 3, i)
        plt.bar(["Before", "After"], [area1, area2], color=color)
        plt.xlabel("Image")
        plt.ylabel("Area")
        plt.title(title)

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()
