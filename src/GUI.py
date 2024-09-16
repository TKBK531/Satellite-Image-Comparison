import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from .process_images import process_images


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SATELLITE IMAGE ANALYSIS")

        self.image_paths = [None, None]
        self.scale_factor = tk.DoubleVar()

        self.create_widgets()

    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.root, text="SATELLITE IMAGE ANALYSIS", font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)

        # Frame for Image Selection
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=10)

        tk.Label(image_frame, text="Select Image 1:").grid(
            row=0, column=0, padx=10, pady=10
        )
        self.image1_button = tk.Button(
            image_frame, text="Browse", command=lambda: self.select_image(0)
        )
        self.image1_button.grid(row=0, column=1, padx=10, pady=10)
        self.image1_preview = tk.Label(image_frame)
        self.image1_preview.grid(row=0, column=2, padx=10, pady=10)

        tk.Label(image_frame, text="Select Image 2:").grid(
            row=1, column=0, padx=10, pady=10
        )
        self.image2_button = tk.Button(
            image_frame, text="Browse", command=lambda: self.select_image(1)
        )
        self.image2_button.grid(row=1, column=1, padx=10, pady=10)
        self.image2_preview = tk.Label(image_frame)
        self.image2_preview.grid(row=1, column=2, padx=10, pady=10)

        # Frame for Scale Factor
        scale_frame = tk.Frame(self.root)
        scale_frame.pack(pady=10)

        tk.Label(scale_frame, text="Scale Factor (Pixels per Meter):").grid(
            row=0, column=0, padx=10, pady=10
        )
        self.scale_entry = tk.Entry(scale_frame, textvariable=self.scale_factor)
        self.scale_entry.grid(row=0, column=1, padx=10, pady=10)

        # Process Button
        process_frame = tk.Frame(self.root)
        process_frame.pack(pady=10)

        self.process_button = tk.Button(
            process_frame, text="Process Images", command=self.process_images
        )
        self.process_button.grid(row=0, column=0, padx=10, pady=10)

    def select_image(self, index):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tiff;")]
        )
        if file_path:
            self.image_paths[index] = file_path
            if index == 0:
                self.display_image(file_path, self.image1_preview)
            else:
                self.display_image(file_path, self.image2_preview)
            messagebox.showinfo("Selected Image", f"Image {index + 1} selected.")

    def display_image(self, file_path, label):
        image = Image.open(file_path)
        image.thumbnail((100, 100))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def process_images(self):
        if None in self.image_paths:
            messagebox.showerror("Error", "Please select both images.")
            return

        scale_factor = self.scale_factor.get()
        if scale_factor < 0:
            messagebox.showerror("Error", "Scale factor cannot be negative.")
            return

        try:
            process_images(self.image_paths, scale_factor)
            messagebox.showinfo("Success", "Images processed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


def run_app():
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
