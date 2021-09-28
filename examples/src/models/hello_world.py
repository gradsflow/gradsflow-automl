from timm import create_model
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData

from gradsflow import AutoDataset, Model
from gradsflow.data.image import get_augmentations

image_size = (128, 128)
transform = get_augmentations(image_size)
train_ds = FakeData(size=100, image_size=[3, *image_size], transform=transform)
val_ds = FakeData(size=100, image_size=[3, *image_size], transform=transform)
train_dl = DataLoader(train_ds)
val_dl = DataLoader(val_ds)

num_classes = train_ds.num_classes
autodataset = AutoDataset(train_dl, val_dl, num_classes=num_classes)

cnn = create_model("resnet18", pretrained=False, num_classes=num_classes)

model = Model(cnn)
model.compile("crossentropyloss", "adam")
model.fit(autodataset, epochs=10)
