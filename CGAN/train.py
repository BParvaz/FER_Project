from pathlib import Path
import os, csv, time
import torch
import pandas 
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import glob
from PIL import Image, ImageDraw, ImageFont
from utils.versioning import make_run_dirs
"""
FERDataset
A class which inherits the pytorch dataset for storing and manipulating our input
"""
# Inherits Pytorch dataset
class FERDataset(Dataset):
    def __init__(self, dataframe, transform = None):
        # Constructor
        # (String) df: Instance of pandas dataframe with data already loaded
        # (Function) transform: Image processing function 
        
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]

        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8)
        img = pixels.reshape(48, 48) #48x48 to match FER2013

        # from greyscale to RGB
        img = Image.fromarray(img).convert("L")

        label = int(row['emotion'])

        # transform img
        if self.transform:
            img = self.transform(img)

        return img, label

    def get_emotion_class(self,emotion):
        return FERDataset(self.df[self.df['emotion'] == emotion],transform=self.transform)

    @property
    def classes(self):
        classes = self.df['emotion'].unique()
        classes.sort()
        return classes
    
    # Helper function to return distribution of each class in a dataset
    @property
    def dist(self):
        return self.df['emotion'].value_counts()
    
fer2013_dataframe = pandas.read_csv('./data/FER2013/train.csv')
# we're going one back
dataset = FERDataset(dataframe=fer2013_dataframe)
from torchvision.models import resnet18
resnet18().fc.in_features
num_ftrs = resnet18().fc.in_features
latent_space = 256

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
])
dataset = FERDataset(dataframe=fer2013_dataframe,transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
img_shape = dataset[0][0].shape

class ProjectionDiscriminator(nn.Module):
    def __init__(self, img_shape, num_classes):
        super().__init__()
        in_dim = int(np.prod(img_shape))

        # feature extractor h(x)
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # unconditional score f(x)
        self.fc = nn.Linear(256, 1)

        # label embedding e(y)
        self.embed = nn.Embedding(num_classes, 256)

    def forward(self, img, labels):
        h = self.feature(img.view(img.size(0), -1))   # h(x)
        out = self.fc(h).squeeze(1)                   # f(x)

        # projection term <h(x), e(y)>
        emb = self.embed(labels)
        proj = torch.sum(h * emb, dim=1)

        return out + proj


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_channels=1):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.init_size = 6  # 6x6 feature map
        self.fc = nn.Linear(latent_dim + num_classes, 256 * self.init_size * self.init_size)

        self.net = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 12x12
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 24x24
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # upsample to 48x48 but KEEP features
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),    # 48x48
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # refine at 48x48
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # final projection to image
            nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )


    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat((z, c), dim=1)

        out = self.fc(x)
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        img = self.net(out)

        return img
    
generator = Generator(latent_space,len(dataset.classes)).cuda()

discriminator = ProjectionDiscriminator(img_shape,len(dataset.classes)).cuda()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

def generator_train_step(discriminator, generator, g_optimizer, batch_size,
                         latent_space, num_classes):
    generator.train()
    discriminator.eval()  # optional
    z = torch.randn(batch_size, latent_space, device=next(generator.parameters()).device)
    gen_labels = torch.randint(0, num_classes, (batch_size,), device=z.device)

    fake_images = generator(z, gen_labels)
    D_fake = discriminator(fake_images, gen_labels).view(batch_size)

    g_loss = -D_fake.mean()

    g_optimizer.zero_grad(set_to_none=True)
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item()

def discriminator_train_step(discriminator, generator, d_optimizer, real_images, labels,
                             latent_space, lambda_gp=10.0):
    discriminator.train()
    generator.eval()  
    B = real_images.size(0)

    z = torch.randn(B, latent_space, device=real_images.device)
    fake_images = generator(z, labels) 

    D_real = discriminator(real_images, labels).view(B)
    D_fake = discriminator(fake_images.detach(), labels).view(B)

    # GP
    alpha = torch.rand(B, 1, 1, 1, device=real_images.device)
    x_hat = alpha * real_images + (1 - alpha) * fake_images.detach()
    x_hat.requires_grad_(True)

    D_hat = discriminator(x_hat, labels).view(B)
    grads = torch.autograd.grad(
        outputs=D_hat.sum(), inputs=x_hat,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()

    d_loss = D_fake.mean() - D_real.mean() + lambda_gp * gp
    gap = D_real.mean() - D_fake.mean()
    
    d_optimizer.zero_grad(set_to_none=True)
    d_loss.backward()
    d_optimizer.step()

    
    return d_loss.item(), D_real.mean().item(), D_fake.mean().item(), gp.item(), gap.item()


def ensure_dir(p): os.makedirs(p, exist_ok=True)

def init_csv(path):
    ensure_dir(os.path.dirname(path))
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "epoch", "g_loss", "d_loss", "d_real", "d_fake", "gp", "gap"])

def append_csv(path, epoch, g_loss, d_loss, d_real, d_fake, gp, gap):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([time.time(), epoch, g_loss, d_loss, d_real, d_fake, gp, gap])

def save_samples(generator, epoch, latent_space, num_classes, out_dir, device):
    ensure_dir(out_dir)
    generator.eval()
    with torch.no_grad():
        n = num_classes
        z = torch.randn(n, latent_space, device=device)
        labels = torch.arange(n, device=device)
        imgs = generator(z, labels)
        save_image(imgs, os.path.join(out_dir, f"epoch_{epoch:04d}.png"), nrow=4, normalize=True)
    generator.train()

def save_checkpoint(path, epoch, generator, discriminator, g_optimizer, d_optimizer):
    ensure_dir(os.path.dirname(path))
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
    }, path)
def create_final_gif(samples_dir: str | Path, out_dir: str | Path, tag: str) -> Path:
    samples_dir = Path(samples_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(samples_dir.glob("epoch_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNGs found matching {samples_dir / 'epoch_*.png'}")

    frames = []
    font = ImageFont.load_default()

    for p in pngs:
        img = Image.open(p).convert("RGB")
        draw = ImageDraw.Draw(img)

        # epoch from filename: epoch_0005.png -> 0005
        epoch = p.stem.split("_")[-1]
        text = f"Epoch {epoch}"

        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = img.width - w - 5
        y = img.height - h - 5

        draw.text((x + 1, y + 1), text, fill="black", font=font)
        draw.text((x, y), text, fill="white", font=font)

        frames.append(img)

    gif_path = out_dir / f"training_{tag}.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    print(f"Saved {gif_path}")
    return gif_path

def create_final_graph(
    csv_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    tag: str,
    smooth_window: int = 1,
    show: bool = False,
) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pandas.read_csv(csv_path)

    required = ["epoch", "g_loss", "d_loss", "d_real", "d_fake", "gp", "gap"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Ensure numeric + sort by epoch
    for c in required + (["time"] if "time" in df.columns else []):
        df[c] = pandas.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)

    def smooth(series: pandas.Series) -> pandas.Series:
        w = max(int(smooth_window), 1)
        if w <= 1:
            return series
        return series.rolling(window=w, min_periods=1, center=False).mean()

    epoch = df["epoch"]

    def save_plot(y_cols: list[str], title: str, filename: str) -> None:
        plt.figure()
        for c in y_cols:
            plt.plot(epoch, smooth(df[c]), label=c)
        plt.xlabel("epoch")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        if show:
            plt.show()
        plt.close()

    save_plot(["g_loss", "d_loss"], "Generator / Critic Loss", f"losses_{tag}.png")
    save_plot(["d_real", "d_fake"], "Critic Scores: real vs fake", f"critic_scores_{tag}.png")
    save_plot(["gp"], "Gradient Penalty (gp)", f"gp_{tag}.png")
    save_plot(["gap"], "Gap", f"gap_{tag}.png")
    save_plot(["g_loss", "d_loss", "gp", "gap"], "Overview", f"overview_{tag}.png")

    print(f"Saved plots to {out_dir}")
    return out_dir

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move models to gpu:
    generator.to(device)
    discriminator.to(device)

    # constant setup
    num_epochs = 250
    n_critic = 5
    lambda_gp = 10.0

    # set up logging
    run = make_run_dirs("logs")
    log_csv = str(run.metrics_csv)
    sample_dir = str(run.samples_dir)
    ckpt_dir = str(run.checkpoints_dir)
    plots_dir = str(run.plots_dir)
    gif_dir = str(run.gif_dir)
    save_ckpt_every = 5

    init_csv(log_csv)
    ensure_dir(sample_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(plots_dir)
    ensure_dir(gif_dir)

    # begin training
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}...", flush=True)

        for i, (real_images, labels) in enumerate(data_loader):
            real_images = real_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # critic updates
            for _ in range(n_critic):
                d_loss, d_real, d_fake, gp, gap = discriminator_train_step(
                    discriminator=discriminator,
                    generator=generator,
                    d_optimizer=d_optimizer,
                    real_images=real_images,
                    labels=labels,
                    latent_space=latent_space,
                    lambda_gp=lambda_gp
                )

            # generator update
            g_loss = generator_train_step(
                discriminator=discriminator,
                generator=generator,
                g_optimizer=g_optimizer,
                batch_size=real_images.size(0),
                latent_space=latent_space,
                num_classes=len(dataset.classes)
            )

        print(f"Epoch {epoch} | G: {g_loss:.4f} | D: {d_loss:.4f} | "
              f"D_real: {d_real:.2f} | D_fake: {d_fake:.2f} | GP: {gp:.3f} | Gap: {gap:.2f}",
              flush=True)

        append_csv(log_csv, epoch, g_loss, d_loss, d_real, d_fake, gp, gap)
        save_samples(generator, epoch, latent_space, len(dataset.classes), sample_dir, device)

        # always update latest
        save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), epoch,
                        generator, discriminator, g_optimizer, d_optimizer)

        # periodic snapshots
        if (epoch + 1) % save_ckpt_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"), epoch,
                            generator, discriminator, g_optimizer, d_optimizer)
            
    create_final_gif(samples_dir=sample_dir, out_dir=gif_dir, tag=run.tag)
    create_final_graph(csv_path=log_csv, out_dir=plots_dir, tag=run.tag)

if __name__ == "__main__":
    main()
