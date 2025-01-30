import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import pytorch_lightning as pl

BATCH_SIZE=128
AVAIL_GPUS=min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count()/2)

class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir="./data", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert images into tensors
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize the tensor with the mean(0.1307) and the standard deviation(0.3081) of the MNIST dataset
            ]
        )
        
        # Override MNIST URLs directly in torchvision
        MNIST.mirrors = ['https://ossci-datasets.s3.amazonaws.com/mnist/']
        
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True) # Download the MNIST dataset and treat it as the training set
        MNIST(self.data_dir, train=False, download=True)    # Download the MNIST dataset and treat it as the test set
    
    def setup(self, stage=None):
        # Assign train/val set
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

# Detective: fake or no fake -> 1 output [0, 1]
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)
    
    
# Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 7*7*64)    # [n, 64, 7, 7]
        
        # nn.ConvTranspose2d(input_channels, output_channels, Kernel_size, Stride)
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7) # [n, 1, 28, 28]
        
    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)    #256
        
        #Up sample (transposed conv) 16*16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)
        
        # Upsample to 34*34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)
        
        # Convolution to 28*28 (1 feature map)
        return self.conv(x)
    
class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr_g=0.0002, lr_d=0.0001):  # Different learning rates for G and D
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()
        
        # Create random noise
        self.validation_z = torch.randn(6, self.hparams.latent_dim)
        
    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        
        # Train Discriminator First
        opt_d.zero_grad()
        
        # Real images
        real_labels = torch.ones(batch_size, 1).type_as(real_imgs) * 0.9  # Label smoothing
        real_pred = self.discriminator(real_imgs)
        real_loss = self.adversarial_loss(real_pred, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, self.hparams.latent_dim).type_as(real_imgs)
        fake_imgs = self(z)
        fake_labels = torch.zeros(batch_size, 1).type_as(real_imgs)
        fake_pred = self.discriminator(fake_imgs.detach())
        fake_loss = self.adversarial_loss(fake_pred, fake_labels)
        
        # Combined D loss
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Train Generator
        opt_g.zero_grad()
        
        # Generate new fake images
        z = torch.randn(batch_size, self.hparams.latent_dim).type_as(real_imgs)
        fake_imgs = self(z)
        fake_pred = self.discriminator(fake_imgs)
        
        # Use real labels for generator loss
        g_loss = self.adversarial_loss(fake_pred, real_labels)
        self.manual_backward(g_loss)
        opt_g.step()
        
        # Log losses
        self.log_dict({
            "g_loss": g_loss,
            "d_loss": d_loss,
            "d_real": real_loss,
            "d_fake": fake_loss
        })
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.hparams.lr_g,
            betas=(0.5, 0.999)  # Standard betas for GANs
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.hparams.lr_d,
            betas=(0.5, 0.999)
        )
        return [opt_g, opt_d], []
    
    def plot_imgs(self):
        try:
            # Save in current directory
            output_dir = "generated_images"
            
            z = self.validation_z.type_as(self.generator.fc1.weight)
            sample_imgs = self(z).cpu()
            
            fig = plt.figure(figsize=(10, 6))
            for i in range(sample_imgs.size(0)):
                plt.subplot(2, 3, i+1)
                plt.tight_layout()
                plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap="gray_r", interpolation="none")
                plt.title(f"Generated Image {i+1}")
                plt.xticks([])
                plt.yticks([])
                plt.axis("off")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the plot with epoch number
            save_path = os.path.join(output_dir, f'epoch_{self.current_epoch}.png')
            plt.savefig(save_path)
            plt.close()  # Close the figure to free memory
            
            print(f"Successfully saved image to: {save_path}")
            
        except Exception as e:
            print(f"Error in plot_imgs: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_train_epoch_end(self):
        print(f"Epoch {self.current_epoch} ended, generating images...")
        self.plot_imgs()
        
if __name__ == '__main__':
    dm = MNISTDataModule()
    model = GAN(lr_g=0.0002, lr_d=0.0001)  # Slightly slower learning rate for discriminator
    
    trainer = pl.Trainer(
        max_epochs=20,  # Increased epochs
        accelerator='mps' if torch.backends.mps.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10
    )
    trainer.fit(model, dm)
    
    # Check for generated images in current directory
    output_dir = "generated_images"
    if os.path.exists(output_dir):
        print(f"\nGenerated images are in: {os.path.abspath(output_dir)}")
        print(f"Images generated: {os.listdir(output_dir)}")
    else:
        print(f"\nNo generated_images directory found at: {os.path.abspath(output_dir)}")
        
    # Save the model
    torch.save(model.state_dict(), 'gan_model.pt')
    print("Model saved to gan_model.pt")
    torch.save(model.generator.state_dict(), 'generator.pt')
    print("Generator model saved to generator.pt")