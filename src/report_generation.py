from .image_processing import calculate_percentage_change
from .config import *

# Assumptions based on area changes
ASSUMPTIONS = {
    "severe_drought": "Both water and vegetation areas have significantly decreased, indicating a severe drought.",
    "mild_drought": "Both water and vegetation areas have decreased, indicating a possible drought.",
    "urbanization": "The land area has increased, indicating a potential future urbanization spot.",
    "good_rain": "The vegetation area has increased, indicating good rainfall.",
    "water_increase": "The water area has increased, indicating possible flooding or reservoir filling.",
    "stable": "The areas have remained relatively stable with no significant changes.",
}


def generate_report(
    green_area1, green_area2, blue_area1, blue_area2, land_area1, land_area2
):
    """Generate a detailed report of the area changes."""
    veg_change = calculate_percentage_change(green_area1, green_area2)
    water_change = calculate_percentage_change(blue_area1, blue_area2)
    land_change = calculate_percentage_change(land_area1, land_area2)

    # Determine the assumption based on the changes
    if (
        veg_change < SEVERE_DROUGHT_THRESHOLD
        and water_change < SEVERE_DROUGHT_THRESHOLD
    ):
        assumption = ASSUMPTIONS["severe_drought"]
    elif (
        veg_change < POSSIBLE_DROUGHT_THRESHOLD
        and water_change < POSSIBLE_DROUGHT_THRESHOLD
    ):
        assumption = ASSUMPTIONS["mild_drought"]
    elif land_change > URBANIZATION_THRESHOLD:
        assumption = ASSUMPTIONS["urbanization"]
    elif veg_change > 0 and water_change <= 0:
        assumption = ASSUMPTIONS["good_rain"]
    elif water_change > 0 and veg_change <= 0:
        assumption = ASSUMPTIONS["water_increase"]
    else:
        assumption = ASSUMPTIONS["stable"]

    report = [
        "Image Comparison Report",
        "=======================",
        "",
        "Vegetation Area:",
        f"  Before: {green_area1:.2f} pixels",
        f"  After: {green_area2:.2f} pixels",
        f"  Change: {veg_change:.2f}%",
        "",
        "Water Area:",
        f"  Before: {blue_area1:.2f} pixels",
        f"  After: {blue_area2:.2f} pixels",
        f"  Change: {water_change:.2f}%",
        "",
        "Land Area:",
        f"  Before: {land_area1:.2f} pixels",
        f"  After: {land_area2:.2f} pixels",
        f"  Change: {land_change:.2f}%",
        "",
        "Summary:",
        f"  The vegetation area changed by {veg_change:.2f}%.",
        f"  The water area changed by {water_change:.2f}%.",
        f"  The land area changed by {land_change:.2f}%.",
        "",
        "Assumption:",
        f"  {assumption}",
    ]

    with open("image_comparison_report.txt", "w") as file:
        file.write("\n".join(report))
