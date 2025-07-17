import pytorch_lightning as pl
from torchunique import sharedobject
import torch
from torch import distributed as dist
import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning import Trainer


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(self.size)
        y = torch.randint(0, 2, (1,)).float()
        return x, y

#####################################################
# Example Usage Beginning

class TestLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.cache_dict: dict = sharedobject.Unique(dict())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.cache_dict.setdefault("fname", []).append(f"rank{dist.get_rank()} batch_idx{batch_idx:02}")

    def on_predict_epoch_end(self):
        self.cache_dict.barrier() # Force ddp synchronization using barrie. This is to prevent data inconsistencies over the dist ranks.
        cache_dict = self.cache_dict.to_here()["fname"] # Use to_here to get internal objects.
        print(sorted(cache_dict), "epoch_end", dist.get_rank()) # The results on each rank are the same.

######################################################

def prepare_data():
    dataset = RandomDataset(size=32, length=32)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = prepare_data()

    model = TestLightningModule()

    trainer = Trainer(
        accelerator="gpu",
        devices=[0, 1],
        strategy="ddp",
        max_epochs=1,
        log_every_n_steps=10,
    )

    trainer.predict(model, train_loader)