import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

class HighResGenerator(nn.Module):
    def __init__(self, latent_dim, output_size=112):
        super().__init__()
        self.output_size = output_size
        
        # Initial dense layer
        self.fc1 = nn.Linear(latent_dim, 14*14*128)
        
        # Upsampling layers
        self.ct1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 28x28
        self.bn1 = nn.BatchNorm2d(64)
        
        self.ct2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 56x56
        self.bn2 = nn.BatchNorm2d(32)
        
        self.ct3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)   # 112x112
        self.bn3 = nn.BatchNorm2d(16)
        
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1, 128, 14, 14)
        
        x = self.ct1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.ct2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.ct3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = torch.tanh(self.final_conv(x))
        return x

class HighResDiscriminator(nn.Module):
    def __init__(self, input_size=112):
        super().__init__()
        
        def conv_output_size(size, kernel=4, stride=2, padding=1):
            return (size + 2*padding - kernel) // stride + 1
        
        # Calculate sizes through the network
        size1 = conv_output_size(input_size)
        size2 = conv_output_size(size1)
        size3 = conv_output_size(size2)
        size4 = conv_output_size(size3)
        
        self.final_flat_size = size4 * size4 * 256
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Dense layers
        self.fc1 = nn.Linear(self.final_flat_size, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        x = x.view(-1, self.final_flat_size)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class HighResGAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr_g=0.0001, lr_d=0.0001, output_size=112):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        # Initialize generator and discriminator
        self.generator = HighResGenerator(latent_dim=self.hparams.latent_dim, 
                                        output_size=output_size)
        self.discriminator = HighResDiscriminator(input_size=output_size)
        
        # Fixed noise for validation
        self.validation_z = torch.randn(6, self.hparams.latent_dim)
        
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
        
        # Real images
        real_labels = torch.ones(batch_size, 1).type_as(real_imgs) * 0.9
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
            
            fig = plt.figure(figsize=(15, 10))
            for i in range(sample_imgs.size(0)):
                plt.subplot(2, 3, i+1)
                plt.tight_layout()
                plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap="gray_r", interpolation="none")
                plt.title(f"Generated Image {i+1}")
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

class HighResDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=32, target_size=112):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_size = target_size
        
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(
                mnist_full, [55000, 5000]
            )
        
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

if __name__ == "__main__":
    # Set parameters
    BATCH_SIZE = 32  # Reduced batch size for higher resolution
    OUTPUT_SIZE = 112  # Higher resolution output
    LATENT_DIM = 100
    
    # Initialize datamodule and model
    dm = HighResDataModule(batch_size=BATCH_SIZE, target_size=OUTPUT_SIZE)
    model = HighResGAN(
        latent_dim=LATENT_DIM,
        lr_g=0.0001,  # Reduced learning rates for stability
        lr_d=0.0001,
        output_size=OUTPUT_SIZE
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=50,  # Increased epochs for better convergence
        accelerator='mps' if torch.backends.mps.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(model, dm)
    
    # Save the trained models
    torch.save(model.state_dict(), 'highres_gan_model.pt')
    torch.save(model.generator.state_dict(), 'highres_generator.pt')
    print("Models saved successfully")