import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

class RealEstateDataset(Dataset):
    """Dataset for real estate images. Labels: House"""
    def __init__(self, root_dir, transform=None):
        self.root_path = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_path.rglob("*.jpg"))

        quality_label_names = sorted(list(set(p.parent.parent.name for p in self.image_paths)))
        house_section_label_names = sorted(list(set(p.parent.name for p in self.image_paths)))
        
        self.quality_label_map = {name: idx for idx, name in enumerate(quality_label_names)}
        self.section_label_map = {name: idx for idx, name in enumerate(house_section_label_names)}

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        house_section_label = img_path.parent.name
        quality_label = img_path.parent.parent.name

        section_idx = self.section_label_map[house_section_label]
        quality_idx = self.quality_label_map[quality_label]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(quality_idx, dtype=torch.long), torch.tensor(section_idx, dtype=torch.long)
    
#import this transform where the dataset is used, so we can reuse it in the model training code as well
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),
    transforms.ToTensor()])

if __name__ == "__main__":

    # Example test of dataset
    img_dir = 'data/'
    image_dataset = RealEstateDataset(img_dir, image_transform)
    num_images = len(image_dataset)
    print(f"length of dataset: {num_images}")
    img, label1, label2 = image_dataset[random.randint(0, num_images - 1)]
    print()
    print({'Image shape': img.shape})
    print({'Assessment Quality': label1})
    print({'House Section': label2})
    fix, ax = plt.subplots()
    ax.set_title(f"Assessment Quality: {label1}, House Section: {label2}")
    ax.axis('off')
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
