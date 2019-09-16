from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def process_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image = transform(image)
    return image

def view_image(image_path, flower_name, probs):
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 10), ncols=1, nrows=2)
    image = Image.open(image_path)
    ax1.set_title(flower_name)
    ax1.imshow(image)
    ax1.axis("off")
    y_pos = np.arrange(len(probs))
    ax2.barh(y_pos, probs, align="center")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(flower_name)
    ax2.invert_yaxis()
    ax2.set_title("Class Probability")


