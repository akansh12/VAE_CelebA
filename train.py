import torch
import torch.nn.functional as F
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import numpy as np
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
LR = 0.005
IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 32
EMBEDDING_DIM = 128
EPOCHS = 50
KLD_WEIGHT = 0.00025

NUM_FRAMES = 50
FPS = 5
LABELS = ["Attractive", "Blond_Hair", "Eyeglasses", "Male", "Smiling" ]

DATASET_PATH = "./data/img_align_celeba/"
DATASET_ATTRS_PATH = "./data/list_attr_celeba.txt"

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, attrs_file, LABELS = None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.LABELS = LABELS
        self.attrs = self._extract_attrs(attrs_file)
        self.imgs = glob.glob(os.path.join(root_dir, "*.jpg"))
        self.imgs.sort()
        

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, np.array(self.attrs[idx])[self.LABELS_IDX]

    def _extract_attrs(self, attrs_file):
        attrs = []
        with open(attrs_file, "r") as f:
            lines = f.readlines()
            self.LABELS_IDX = []
            for i, label in enumerate(lines[1].split()):
                if label in self.LABELS:
                    self.LABELS_IDX.append(i)
            for line in lines[2:]:
                split = line.split()
                filename = split[0]
                values = split[1:]
                label = [0 if v == "-1" else 1 for v in values]
                attrs.append(label)
        return attrs

    def show(self, idx):
        plt.imshow(self[idx][0])
        plt.show()

def loss_function(VAELossParams, kld_weight):
    recons, input, mu, log_var = VAELossParams
    recons_loss = F.mse_loss(recons, input)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kld_weight * kld_loss
    
    return {
        "loss": loss,
        "Reconstruction_Loss": recons_loss.detach(),
        "KLD": -kld_loss.detach(),
    }

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1):
        super(ConvBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.block(x)
    
class ConvTBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1):
        super(ConvTBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.block(x)
    
class CelebVAE(torch.nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims = None):
        super(CelebVAE, self).__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.encoder = torch.nn.Sequential(
            *[
                ConvBlock(in_f, out_f) for in_f, out_f in zip([in_channels] + hidden_dims[:-1], hidden_dims)
            ]
        )
        self.fc_mu = torch.nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_log_var = torch.nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.decoder_input = torch.nn.Linear(latent_dim, hidden_dims[-1]*4)
        self.decoder = torch.nn.Sequential(
            *[
                ConvTBlock(in_f, out_f) for in_f, out_f in zip(hidden_dims[::-1][:-1], hidden_dims[:-1][::-1])
            ]
        )
        self.final_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=hidden_dims[0],
                out_channels=hidden_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(hidden_dims[0]),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_dims[0],
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.Tanh(),
        )
    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return [mu, log_var]
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps*std + mu
    
    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]        


output_dir = "../output/"
os.makedirs(output_dir, exist_ok=True)

training_progress_dir = os.path.join(output_dir, "training_progress")
os.makedirs(training_progress_dir, exist_ok=True)

model_weights_dir = os.path.join(output_dir, "model_weights")
os.makedirs(model_weights_dir, exist_ok=True)

MODEL_BEST_WEIGHTS_PATH = os.path.join(model_weights_dir, "best_vae_celeba.pt")
MODEL_WEIGHTS_PATH = os.path.join(model_weights_dir, "vae_celeba.pt")


train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)


celeba_dataset = CelebADataset(DATASET_PATH, DATASET_ATTRS_PATH, LABELS = LABELS, transform=train_transforms)
val_size = int(len(celeba_dataset) * 0.1)
train_size = len(celeba_dataset) - val_size

train_dataset, val_dataset = random_split(celeba_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


model = CelebVAE(CHANNELS, EMBEDDING_DIM)
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

def validate(model, val_dataloader, DEVICE):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(val_dataloader):
            data = data.to(DEVICE)
            recon_batch, data, mu, log_var = model(data)
            loss = loss_function([recon_batch, data, mu, log_var], KLD_WEIGHT)["loss"]
            val_loss += loss.item()
    return val_loss / len(val_dataloader)


best_val_loss = float("inf")
TOTAL_LOSS = []
TRAIN_LOSS_Recon = []
TRAIN_loss_KLD = []
VAL_LOSS = []
print("Training Started!!")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, x in enumerate(tqdm(train_dataloader)):
        input = x[0].to(DEVICE)
        optimizer.zero_grad()
        predictions = model(input)
        total_loss = loss_function(predictions, KLD_WEIGHT)
        # Backward pass
        total_loss["loss"].backward()
        # Optimizer variable updates
        optimizer.step()
        running_loss += total_loss["loss"].item()
        train_loss = running_loss / len(train_dataloader)
        val_loss = validate(model, val_dataloader, DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"vae-celeba": model.state_dict()},
                MODEL_BEST_WEIGHTS_PATH,
            )
    torch.save(
        {"vae-celeba": model.state_dict()},
        MODEL_WEIGHTS_PATH,
    )
    print(
        f"Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_dataloader)}, "
        f"Total Loss: {total_loss['loss'].detach().item():.4f}, "
        f"Reconstruction Loss: {total_loss['Reconstruction_Loss']:.4f}, "
        f"KL Divergence Loss: {total_loss['KLD']:.4f}",
        f"Val Loss: {val_loss:.4f}",
    )
    TOTAL_LOSS.append(total_loss['loss'].detach().item())
    TRAIN_LOSS_Recon.append(total_loss['Reconstruction_Loss'])
    TRAIN_loss_KLD.append(total_loss['KLD'])
    VAL_LOSS.append(val_loss)
    scheduler.step()









