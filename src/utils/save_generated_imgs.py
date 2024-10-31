import os
import torchvision.utils as vutils
from PIL import Image

def save_generated_images(fake_images, epoch, output_dir="../output"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a grid from fake images
    grid = vutils.make_grid(fake_images, normalize=True)
    
    # Convert the grid to a PIL image and save it
    ndarr = grid.mul(255).add(0.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(ndarr)
    img.save(f"{output_dir}/epoch_{epoch+1}.png")
