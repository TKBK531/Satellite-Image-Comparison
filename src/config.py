# Constants for color ranges in HSV space
GREEN_RANGE = (35, 85)  # Range of hue values for green color
BLUE_RANGE = (85, 130)  # Range of hue values for blue color

# Constants for histogram bins and ranges
HUE_BINS = 360  # Number of bins for hue histogram
HUE_RANGE = [0, 360]  # Range of hue values
SAT_BINS = 256  # Number of bins for saturation histogram
SAT_RANGE = [0, 256]  # Range of saturation values
VAL_BINS = 256  # Number of bins for value histogram
VAL_RANGE = [0, 256]  # Range of value (brightness) values

# Constants for peak adjustment in histograms
HUE_PEAK_ADJUST = 15  # Adjustment value for hue peak
SAT_PEAK_ADJUST = 50  # Adjustment value for saturation peak
VAL_PEAK_ADJUST = 50  # Adjustment value for value (brightness) peak

# Constants for morphological operations
MORPH_KERNEL_SIZE = (3, 3)  # Kernel size for morphological operations

# Constants for Canny edge detection
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 200


# Assumption thresholds in percentage change
SEVERE_DROUGHT_THRESHOLD = -20
POSSIBLE_DROUGHT_THRESHOLD = -10

URBANIZATION_THRESHOLD = 10
