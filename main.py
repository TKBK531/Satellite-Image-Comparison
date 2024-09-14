# from process import process_images
import src.images as img
from src.process_images import process_images

img_paths = [img.before_zambia_drought, img.after_zambia_drought]
process_images(img_paths)
