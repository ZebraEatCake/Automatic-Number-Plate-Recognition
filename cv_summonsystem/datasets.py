import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import xml.etree.ElementTree as ET

LABEL_MAP = {
    'License_Plate': 1,
    'License Plate': 1
}

class NumberPlateDataset(Dataset):
    def __init__(self, root, transforms=None, isAnnotated=True):
        self.root = root
        self.transforms = transforms
        self.isAnnotated = isAnnotated
        self.images = []

        if self.isAnnotated:
            # Include only images with annotation
            for fname in os.listdir(root):
                if not fname.endswith(".jpg"):
                    continue
                xml_path = os.path.join(root, fname.replace(".jpg", ".xml"))
                if not os.path.exists(xml_path):
                    continue

                tree = ET.parse(xml_path)
                root_xml = tree.getroot()
                if any(obj.find("name").text in LABEL_MAP for obj in root_xml.findall("object")):
                    self.images.append(fname)
        else:
            # Use all images (no annotations available)
            self.images = sorted(os.listdir(root))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        if self.isAnnotated:
            xml_path = img_path.replace(".jpg", ".xml")
            boxes = []
            labels = []

            tree = ET.parse(xml_path)
            root_xml = tree.getroot()

            for obj in root_xml.findall("object"):
                name = obj.find("name").text
                if name not in LABEL_MAP:
                    continue
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(LABEL_MAP[name])

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([idx])
            }

            return img, target
        else:
            return img, img_name
