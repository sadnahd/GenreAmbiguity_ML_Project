import random
from torchvision import transforms

class SpecAugmentBands:
    """
    Apply simple spectrogram-specific augmentation:
    - frequency masking: masks horizontal bands
    - time masking: masks vertical bands

    Expects a tensor of shape (C, H, W).
    """
    def __init__(
        self,
        freq_mask_param=16,
        time_mask_param=24,
        num_freq_masks=2,
        num_time_masks=2,
        p=0.5
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x

        _, H, W = x.shape

        # Frequency masking: mask along height
        for _ in range(self.num_freq_masks):
            mask_h = random.randint(0, self.freq_mask_param)
            if mask_h > 0 and mask_h < H:
                top = random.randint(0, H - mask_h)
                x[:, top:top + mask_h, :] = 0

        # Time masking: mask along width
        for _ in range(self.num_time_masks):
            mask_w = random.randint(0, self.time_mask_param)
            if mask_w > 0 and mask_w < W:
                left = random.randint(0, W - mask_w)
                x[:, :, left:left + mask_w] = 0

        return x


resnet_transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

resnet_transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.02), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    SpecAugmentBands(
        freq_mask_param=16,
        time_mask_param=24,
        num_freq_masks=2,
        num_time_masks=2,
        p=0.6
    ),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

vit_transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

vit_transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.02), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    SpecAugmentBands(
        freq_mask_param=16,
        time_mask_param=24,
        num_freq_masks=2,
        num_time_masks=2,
        p=0.6
    ),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])