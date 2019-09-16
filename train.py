import argparse
import json
import os

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

from model import build_classifier

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_dir", action="store", help="Directory that contains data"
)

parser.add_argument(
    "--save-dir",
    action="store",
    dest="save_dir",
    default="",
    help="Directory to save checkpoints",
)


parser.add_argument(
    "--arch",
    action="store",
    default="resnet",
    dest="architecture",
    help="Choose architecture to use for pretrained network",
)
parser.add_argument(
    "--learning_rate",
    action="store",
    default="0.01",
    dest="lr",
    type=float,
    help="Learning rate for the Neural Network",
)

parser.add_argument(
    "--hidden_units",
    action="append",
    dest="hidden_layers",
    default=[],
    help="Set the hidden units values",
    type=int,
)

parser.add_argument(
    "--epochs",
    action="store",
    dest="epochs",
    default=20,
    type=int,
    help="Number of epochs",
)

parser.add_argument(
    "--gpu",
    action="store_true",
    default=False,
    help="Set to use a GPU defaults to False",
    dest="use_gpu",
)

results = parser.parse_args()


data_dir = results.data_dir
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x), transform=data_transforms[x]
    )
    for x in ["train", "test", "valid"]
}


dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
    for x in image_datasets
}


with open("cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)


architecture = results.architecture
if architecture == "resnet":
    print("Using the Resent Architecture\n")
    model = models.resnet50(pretrained=True)
    num_in_features = 2048
elif architecture == "vgg":
    print("Using the VGG architecture\n")
    model = models.vgg19(pretrained=True)
    num_in_features = 25088
else:
    print("Using the default: Resnet50")
    model = models.resnet50(pretrained=True)
    num_in_features = 2048


pretrained_model = model
for param in model.parameters():
    param.requires_grad = False

hidden_layers = results.hidden_layers
classifier = build_classifier(num_in_features, 102, hidden_layers, drop_p=0.2)
criterion = nn.NLLLoss()

lr = results.lr
if architecture == "resnet":
    model.fc = classifier
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
else:
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)


use_gpu = results.use_gpu
gpu_available = torch.cuda.is_available()
if use_gpu and gpu_available:
    print("Using the GPU available\n")
    device = torch.device("cuda")
else:
    if gpu_available:
        print("You have chosen to use the CPU for training")
    if use_gpu:
        print("You do not have a GPU to use for training")
    device = torch.device("cpu")

epochs = results.epochs
steps = 0
running_loss = 0
print_every = 20
model.to(device)

trainloader = dataloaders["train"]
validloader = dataloaders["valid"]

# Train Model
print(":::  Training Model\n")
for e in range(epochs):
    for images, labels in trainloader:
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval()

            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)

                logps = model(images)
                loss = criterion(logps, labels)
                validation_loss += loss.item()

                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(
                "Epoch: {}/{}.. ".format(e + 1, epochs),
                "Training Loss: {:.3f}..".format(running_loss / print_every),
                "Validation Loss: {:.3f}..".format(
                    validation_loss / len(validloader)
                ),
                "Validation Accuracy: {:.3f}".format(
                    accuracy / len(validloader)
                ),
            )
            running_loss = 0
            model.train()

# Save Model
print("::: Save Model\n")
model.class_to_idx = image_datasets["train"].class_to_idx

checkpoint = {
    "state-dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epochs": epochs,
    "classifier": classifier,
    "batch_size": 32,
    "pretrained-model": pretrained_model,
    "class-mapping": image_datasets["train"].class_to_idx,
    "class-names": image_datasets["train"].classes,
}


save_dir = os.path.join(results.save_dir, "checkpoint.pth")
torch.save(checkpoint, save_dir)
print('::: Done Training\n ::: model saved to {save_dir}'.format(save_dir))

