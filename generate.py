import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# Copy the Generator class definition (needs to be exactly the same as in training)
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 7*7*64)
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.conv = nn.Conv2d(16, 1, kernel_size=7)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)
        
        x = self.ct1(x)
        x = F.relu(x)
        
        x = self.ct2(x)
        x = F.relu(x)
        
        return self.conv(x)

""" 
Latent dimension (latent_dim)
==================================
- The size of the random noise vector that is used as input to generate images

"""

def generate_images(num_images=6, latent_dim=100, save_dir='generated_samples'):
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize generator and load weights
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    generator = Generator(latent_dim).to(device)
    
    # Load the saved model with weights_only=True for security
    generator.load_state_dict(
        torch.load('generator.pt', map_location=device, weights_only=True)
    )
    generator.eval()
    
    # Generate images
    with torch.no_grad():
        # Create random noise
        z = torch.randn(num_images, latent_dim).to(device)
        
        # Generate images
        samples = generator(z).cpu()
        
        # Plot and save
        fig = plt.figure(figsize=(15, 5))
        for i in range(num_images):
            plt.subplot(1, num_images, i+1)
            plt.imshow(samples[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.axis('off')
        
        # Save the grid of images
        plt.savefig(os.path.join(save_dir, 'generated_samples.png'))
        plt.close()
        
        # Save individual images
        for i in range(num_images):
            plt.figure(figsize=(5, 5))
            plt.imshow(samples[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'))
            plt.close()
    
    print(f"Generated {num_images} images in {save_dir}")

if __name__ == "__main__":
    # Generate 6 images
    generate_images(num_images=6) 