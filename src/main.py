import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.show_imgs import show_image
from utils.weights_init import weights_init
from utils.save_generated_imgs import save_generated_images

# Use the GPU
device = torch.device('cuda')

NOISE_DIM = 64

# Define hyperparameters
RANDOM_SEED = 42
LR = 0.02
EPOCHS = 50
BATCH_SIZE = 128
BETAS = (0.5, 0.99)

# Define the Data Augmentations
data_aug = transforms.Compose([
    transforms.RandomRotation((-20, 20)),
    transforms.ToTensor()
])

# Load the MNIST dataset
dataset = datasets.MNIST(root='../data/', download=True, train=True, transform=data_aug)

# Load the Dataset into batches
trainset = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define the Discriminator net architecture
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, img):
        return self.model(img)

# Define the Generator net architecture
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise_dim, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2),
            nn.Tanh()
        )
    
    def forward(self, z):
        z = z.view(-1, self.noise_dim, 1, 1)
        return self.model(z)

# Initialize the models
D = Discriminator().to(device).apply(weights_init)
G = Generator(NOISE_DIM).to(device).apply(weights_init)

# Define the Training Optimizers
Gopt = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
Dopt = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

# Define the Loss functions criterion
criterion = nn.BCEWithLogitsLoss()

# Training Loop
for epoch in range(EPOCHS):

    # Define the Total loss for each model to average later
    DTotalLoss = 0.0
    GTotalLoss = 0.0

    for realImg, _ in trainset:

        # Define a ground truth tensors for fake and real
        batch_size = realImg.size(0)
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        realImg = realImg.to(device) # Transfer the image to the gpu

        # Define the noise grid
        noise = torch.randn(batch_size, NOISE_DIM, device=device)

        # Train the discriminator
        Dopt.zero_grad()

        fakeImg = G(noise)
        Dpred = D(fakeImg)
        DFakeLoss = criterion(Dpred, fake)

        Dpred = D(realImg)
        DRealLoss = criterion(Dpred, real)

        DLoss = (DRealLoss + DFakeLoss) / 2

        DLoss.backward()
        Dopt.step()

        DTotalLoss += DLoss

        # Train the Generator
        Gopt.zero_grad()

        noise = torch.randn(batch_size, NOISE_DIM, device=device)

        fakeImg = G(noise)
        Dpred = D(fakeImg)
        GLoss = criterion(Dpred, real)

        GLoss.backward()
        Gopt.step()

        GTotalLoss += GLoss
    
    # Define the average loss for each epoch
    DAvgLoss = DTotalLoss / len(dataset)
    GAvgLoss = GTotalLoss / len(dataset)

    print(f"Epoch: {epoch} | DLoss: {DAvgLoss} | GLoss: {GAvgLoss}")

    show_image(fakeImg)

    # Save the Result images from each epoch
    save_generated_images(fakeImg, epoch)
