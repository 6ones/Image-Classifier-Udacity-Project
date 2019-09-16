import argparse
import json

import numpy as np
import torch
from PIL import Image

from utils import process_image, view_image

parser = argparse.ArgumentParser()

parser.add_argument("input_file", action="store", help="Image to be predicted")
parser.add_argument(
    "checkpoint", action="store", help="Checkpoint containing saved model"
)

parser.add_argument(
    "--top_k",
    action="store",
    default=5,
    dest="top_k",
    type=int,
    help="Return top K most likely classes",
)
parser.add_argument(
    "--category_names",
    action="store",
    dest="category_names",
    help="JSON file containing category names",
)

parser.add_argument(
    "--gpu",
    action="store_true",
    default=False,
    help="Set to use a GPU defaults to False",
    dest="use_gpu",
)

results = parser.parse_args()


def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    model = checkpoint["pretrained-model"]
    try:
        model.fc = checkpoint["classifier"]
    except AttributeError:
        model.classifier = checkpoint["classifier"]

    model.load_state_dict(checkpoint["state-dict"])
    class_names = checkpoint["class-names"]
    return model, class_names


def predict(image_path, model, top_k=5):

    model.to(device)
    model.eval()

    img = Image.open(image_path)
    img = process_image(img)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    inputs = img.to(device)
    logits = model.forward(inputs)
    ps = torch.exp(logits)
    topk = ps.cpu().topk(top_k)
    return (e.data.numpy().squeeze().tolist() for e in topk)


checkpoint_file = results.checkpoint
model, class_names = load_checkpoint(checkpoint_file)


use_gpu = results.use_gpu
gpu_available = torch.cuda.is_available()

if use_gpu and gpu_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    if gpu_available:
        print("You have chosen to use the CPU for training")
    if use_gpu:
        print("You do not have a GPU to use for training")


image_path = results.input_file
top_k = results.top_k
category_names = results.category_names

probs, classes = predict(image_path, model, top_k=top_k)

with open(category_names, 'r') as f:
    cat_to_names = json.load(f)

print(f'Probabilities: {probs}\n Classes:  {classes}')
flower_names = [cat_to_names[class_names[e]] for e in classes]

print(flower_names)
# view_image(image_path, flower_names, probs)
