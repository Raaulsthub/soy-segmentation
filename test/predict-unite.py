from PIL import Image
import os

# Define the paths to the directories containing images
image_dir = "./images"
predicted_dir = "./predicted-images"

# Create a list to store the image filenames
image_filenames = [f"img{i}.jpg" for i in range(1, 5)]  # 4 images instead of 6

# Create a new image for the grid
grid_width = 4  # Number of columns (changed to 4)
grid_height = 2  # Number of rows
image_size = (200, 200)  # Size of each image in the grid
padding = 10  # Space between images

# Calculate the size of the grid image
grid_width_px = (image_size[0] + padding) * grid_width - padding
grid_height_px = (image_size[1] + padding) * grid_height - padding

grid_image = Image.new("RGB", (grid_width_px, grid_height_px), "white")

# Paste the images into the grid
for i, filename in enumerate(image_filenames):
    row = i // grid_width
    col = i % grid_width
    img_path = os.path.join(image_dir, filename)
    img = Image.open(img_path)
    img = img.resize(image_size)
    x = col * (image_size[0] + padding)
    y = row * (image_size[1] + padding)
    grid_image.paste(img, (x, y))

    predicted_img_path = os.path.join(predicted_dir, filename)
    predicted_img = Image.open(predicted_img_path)
    predicted_img = predicted_img.resize(image_size)
    x = col * (image_size[0] + padding)
    y = (row + grid_height // 2) * (image_size[1] + padding)
    grid_image.paste(predicted_img, (x, y))

# Save the grid image
grid_image.save("image_grid.jpg")

# Display the grid image
grid_image.show()
