import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim=200):  # Increased latent dimension
        super().__init__()
        
        # Starting with 4x4 spatial dimensions
        self.fc1 = nn.Linear(latent_dim, 4*4*512)
        
        # Upsampling layers to reach 32x32
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Final layer to generate 3 channels
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 512, 4, 4)
        x = self.conv_blocks(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 4x4 -> 1x1
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_blocks(x)
        return x.view(-1, 1)

class CIFAR10GAN(pl.LightningModule):
    def __init__(self, latent_dim=200, lr_g=0.0002, lr_d=0.0002):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()
        
        # Fixed noise for validation
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        
    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        
        # Train Discriminator
        opt_d.zero_grad()
        
        real_labels = torch.ones(batch_size, 1).type_as(real_imgs) * 0.9  # Label smoothing
        real_pred = self.discriminator(real_imgs)
        real_loss = self.adversarial_loss(real_pred, real_labels)
        
        z = torch.randn(batch_size, self.hparams.latent_dim).type_as(real_imgs)
        fake_imgs = self(z)
        fake_labels = torch.zeros(batch_size, 1).type_as(real_imgs)
        fake_pred = self.discriminator(fake_imgs.detach())
        fake_loss = self.adversarial_loss(fake_pred, fake_labels)
        
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Train Generator
        opt_g.zero_grad()
        
        z = torch.randn(batch_size, self.hparams.latent_dim).type_as(real_imgs)
        fake_imgs = self(z)
        fake_pred = self.discriminator(fake_imgs)
        
        g_loss = self.adversarial_loss(fake_pred, real_labels)
        self.manual_backward(g_loss)
        opt_g.step()
        
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
            betas=(0.5, 0.999)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.hparams.lr_d,
            betas=(0.5, 0.999)
        )
        return [opt_g, opt_d], []
    
    def plot_imgs(self):
        try:
            output_dir = "generated_images"
            os.makedirs(output_dir, exist_ok=True)
            
            z = self.validation_z.type_as(self.generator.fc1.weight)
            sample_imgs = self(z).cpu()
            
            fig = plt.figure(figsize=(12, 6))
            for i in range(sample_imgs.size(0)):
                plt.subplot(2, 4, i+1)
                plt.tight_layout()
                img = sample_imgs.detach()[i].permute(1, 2, 0)
                img = (img + 1) / 2  # Rescale from [-1, 1] to [0, 1]
                plt.imshow(img)
                plt.title(f"Generated {i+1}")
                plt.axis("off")
            
            save_path = os.path.join(output_dir, f'epoch_{self.current_epoch}.png')
            plt.savefig(save_path)
            plt.close()
            
            print(f"Successfully saved image to: {save_path}")
            
        except Exception as e:
            print(f"Error in plot_imgs: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_train_epoch_end(self):
        print(f"Epoch {self.current_epoch} ended, generating images...")
        self.plot_imgs()

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=64):  # Increased batch size
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # No need for resize since we're using native CIFAR10 dimensions
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = torch.utils.data.random_split(
                cifar_full, [45000, 5000]
            )
        
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )
    
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=4)

if __name__ == "__main__":
    # Set parameters
    BATCH_SIZE = 64  # Increased batch size
    LATENT_DIM = 200  # Increased latent dimension
    
    # Initialize datamodule and model
    dm = CIFAR10DataModule(batch_size=BATCH_SIZE)
    
    # Prepare the data
    dm.prepare_data()
    dm.setup()
    
    # Get a batch of training data
    train_loader = dm.train_dataloader()
    images, labels = next(iter(train_loader))
    
    # Print dimensions
    print("\nCIFAR10 Training Data Dimensions:")
    print(f"Batch size: {images.size(0)}")
    print(f"Number of channels: {images.size(1)}")
    print(f"Height: {images.size(2)}")
    print(f"Width: {images.size(3)}")
    print(f"\nSingle image shape: {images[0].shape}")
    
    # CIFAR10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Plot a sample image
    plt.figure(figsize=(8, 8))
    
    # Get the first image and its label
    img = images[0]
    label = labels[0]
    
    # Denormalize the image
    img = img.permute(1, 2, 0)  # Change from [C,H,W] to [H,W,C]
    img = img * 0.5 + 0.5  # Denormalize: pixel * std + mean
    
    plt.imshow(img)
    plt.title(f'Class: {classes[label]}')
    plt.axis('off')
    
    # Save the plot
    os.makedirs("training_samples", exist_ok=True)
    plt.savefig('training_samples/cifar10_sample.png')
    plt.close()
    print("\nSample image saved as 'training_samples/cifar10_sample.png'")
    

    model = CIFAR10GAN(
        latent_dim=LATENT_DIM,
        lr_g=0.0002,  # Slightly increased learning rates
        lr_d=0.0002
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=200,  # Increased epochs
        accelerator='mps' if torch.backends.mps.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(model, dm)
    
    # Save the trained models
    torch.save(model.state_dict(), 'cifar10_gan_model.pt')
    torch.save(model.generator.state_dict(), 'cifar10_generator.pt')
    print("Models saved successfully")
