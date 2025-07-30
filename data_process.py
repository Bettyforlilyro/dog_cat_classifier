from modelscope.msdatasets import MsDataset
import csv
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ms_train_dataset = MsDataset.load(dataset_name='cats_and_dogs', namespace='tany0699', subset_name='default', split='train', trust_remote_code=True)
ms_val_dataset = MsDataset.load(dataset_name='cats_and_dogs', namespace='tany0699', subset_name='default', split='validation', trust_remote_code=True)
batch_size = 64


class MyDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_transform(image)

    def __getitem__(self, index):
        image_path, label = self.data[index]['image:FILE'], self.data[index]['category']
        image = self.preprocess_image(image_path)
        return image, int(label)

    def __len__(self):
        return len(self.data)


train_dataset = MyDataset(ms_train_dataset)
val_dataset = MyDataset(ms_val_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


