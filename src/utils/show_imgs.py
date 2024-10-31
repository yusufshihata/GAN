from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_image(img, num_images=16, size=(1, 28, 28)):
    unflat_img = img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_images], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()