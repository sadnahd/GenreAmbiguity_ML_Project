import torch
from torch.utils.data import Dataset

class RandomVerticalChunkDataset(Dataset):
    """
    For training:
    - takes a Subset of an ImageFolder
    - returns one random vertical chunk from each original image
    - then applies the transform
    """
    def __init__(self, subset, transform=None, chunk_width=224):
        self.subset = subset
        self.transform = transform
        self.chunk_width = chunk_width

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]   # image is still PIL because base ImageFolder has no transform
        width, height = image.size

        crop_width = min(self.chunk_width, width)

        if width > crop_width:
            left = torch.randint(0, width - crop_width + 1, (1,)).item()
        else:
            left = 0

        image = image.crop((left, 0, left + crop_width, height))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class MultiVerticalChunkDataset(Dataset):
    """
    For validation:
    - takes a Subset of an ImageFolder
    - returns multiple deterministic vertical chunks from each original image
    - output shape for image is (num_chunks, C, H, W)
    """
    def __init__(self, subset, transform=None, chunk_width=224, num_chunks=4):
        self.subset = subset
        self.transform = transform
        self.chunk_width = chunk_width
        self.num_chunks = num_chunks

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        width, height = image.size

        crop_width = min(self.chunk_width, width)

        if width <= crop_width or self.num_chunks == 1:
            left_positions = [0]
        else:
            max_left = width - crop_width
            left_positions = torch.linspace(0, max_left, steps=self.num_chunks).round().int().tolist()

        chunks = []
        for left in left_positions:
            chunk = image.crop((left, 0, left + crop_width, height))
            if self.transform is not None:
                chunk = self.transform(chunk)
            chunks.append(chunk)

        chunks = torch.stack(chunks, dim=0)   # (num_chunks, C, H, W)
        return chunks, label
