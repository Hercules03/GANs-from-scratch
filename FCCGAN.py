import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        # Fully connected layers to map noise to intermediate features
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 512)
        self.fc3 = nn.Linear(512, 4096)
        self.bn_fc = nn.BatchNorm1d(4096)
        
        # Reshape layer
        self.reshape = lambda x: x.view(-1, 256, 4, 4)
        
        # Transposed convolution layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Fully connected path
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.bn_fc(self.fc3(x)))
        
        # Reshape for convolution
        x = self.reshape(x)
        
        # Transposed convolution path
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        
        return x

class Discriminator(nn.Module):
    def __init__(self, use_pooling=True):
        super().__init__()
        self.use_pooling = use_pooling
        
        # Convolutional layers
        if use_pooling:
            # Version with pooling
            self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            # Version with strided convolutions
            self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        # Print shape for debugging
        batch_size = x.size(0)
        
        # Convolutional path
        if self.use_pooling:
            x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
            x = self.pool(x)
            x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
            x = self.pool(x)
            x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
            x = self.pool(x)
        else:
            x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
            x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Fully connected path
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.sigmoid(self.fc4(x))
        
        return x

class FCCGAN:
    def __init__(self, latent_dim=100, use_pooling=True):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator(use_pooling)
        
        # Create fixed noise for visualization
        self.fixed_noise = torch.randn(8, self.latent_dim)
        
    def generate(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim)
        if next(self.generator.parameters()).is_cuda:
            z = z.cuda()
        return self.generator(z)

    def get_device(self):
        return next(self.generator.parameters()).device
        
    def plot_epoch_samples(self, epoch):
        try:
            output_dir = "epoch_samples"
            os.makedirs(output_dir, exist_ok=True)
            
            # Move fixed noise to the same device as the generator
            device = self.get_device()
            fixed_noise = self.fixed_noise.to(device)
            
            # Generate images
            self.generator.eval()
            with torch.no_grad():
                fake_images = self.generator(fixed_noise)
            self.generator.train()
            
            # Convert images for plotting
            fake_images = fake_images.cpu()
            
            # Create figure
            fig = plt.figure(figsize=(12, 6))
            for i in range(fake_images.size(0)):
                plt.subplot(2, 4, i+1)
                plt.tight_layout()
                
                # Convert from PyTorch's [C,H,W] to matplotlib's [H,W,C] format
                img = fake_images[i].permute(1, 2, 0)
                
                # Rescale from [-1, 1] to [0, 1]
                img = (img + 1) / 2
                img = torch.clamp(img, 0, 1)
                
                plt.imshow(img)
                plt.title(f"Sample {i+1}")
                plt.axis("off")
            
            # Save the figure
            save_path = os.path.join(output_dir, f'epoch_{epoch}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            print(f"Successfully saved epoch samples to: {save_path}")
            
        except Exception as e:
            print(f"Error in plot_epoch_samples: {str(e)}")
            import traceback
            traceback.print_exc()

def train_fccgan(model, num_epochs=150, batch_size=32, lr=0.0001, beta1=0.5, beta2=0.999, save_interval=10):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.generator.to(device)
    model.discriminator.to(device)

    # Create directories for saving samples and models
    if not os.path.exists('samples'):
        os.makedirs('samples')
    if not os.path.exists('models'):
        os.makedirs('models')

    # Data preprocessing and loader setup remains the same
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                           shuffle=True, num_workers=2)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    print("Starting Training...")
    for epoch in range(num_epochs):
        g_losses = []
        d_losses = []
        
        for i, (real_images, _) in enumerate(trainloader):
            batch_size = real_images.size(0)
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            real_images = real_images.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            d_output_real = model.discriminator(real_images)
            d_loss_real = criterion(d_output_real, real_labels)
            
            z = torch.randn(batch_size, model.latent_dim).to(device)
            fake_images = model.generator(z)
            d_output_fake = model.discriminator(fake_images.detach())
            d_loss_fake = criterion(d_output_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            g_output = model.discriminator(fake_images)
            g_loss = criterion(g_output, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(trainloader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')

        # End of epoch processing
        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        print(f'Epoch [{epoch}/{num_epochs}] Average losses - D: {avg_d_loss:.4f} G: {avg_g_loss:.4f}')

        # Plot samples at the end of each epoch
        model.plot_epoch_samples(epoch)

        # Save checkpoints at intervals
        if epoch % save_interval == 0:
            with torch.no_grad():
                fake_images = model.generator(torch.randn(16, model.latent_dim).to(device))
                save_image(fake_images.data[:16], f'samples/fake_images_epoch_{epoch}.png', 
                         normalize=True, nrow=4)

            torch.save(model.generator.state_dict(), f'models/generator_epoch_{epoch}.pth')
            torch.save(model.discriminator.state_dict(), f'models/discriminator_epoch_{epoch}.pth')

    # Save final model state
    print("Training finished! Saving final model...")
    final_save_path = 'models/generator_final.pth'
    torch.save(model.generator.state_dict(), final_save_path)
    print(f"Final generator model saved to: {final_save_path}")
    
    return model

# Configuration and usage remains the same
config = {
    'num_epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'beta1': 0.5,
    'beta2': 0.999,
    'latent_dim': 100,
    'save_interval': 10
}

if __name__ == '__main__':
    fccgan = FCCGAN(latent_dim=config['latent_dim'], use_pooling=True)
    trained_model = train_fccgan(
        model=fccgan,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        beta1=config['beta1'],
        beta2=config['beta2'],
        save_interval=config['save_interval']
    )