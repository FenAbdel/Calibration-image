import os
from PIL import Image

input_folder = "./images"
output_folder = "./compressed_images"
os.makedirs(output_folder, exist_ok=True)  

new_size = (800, 600)  
quality = 50           


for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with Image.open(input_path) as img:
            img = img.resize(new_size, Image.Resampling.LANCZOS)              
            if filename.lower().endswith('.png'):
                img.save(output_path, "PNG", optimize=True)
            else:
                img.save(output_path, "JPEG", quality=quality, optimize=True)

        print(f"Compressed {filename} and saved to {output_path}")

print("All images have been compressed!")
