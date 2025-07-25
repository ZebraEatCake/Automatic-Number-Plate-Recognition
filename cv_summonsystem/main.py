# pip install -q torch torchvision torchmetrics tqdm

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

from training import train_model
from testing import test_model
from datasets import NumberPlateDataset

def get_transform():
    return T.Compose([T.ToTensor()])

def collate_fn(batch):
    return tuple(zip(*batch))

def main(train=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading datasets...")
    train_dataset = NumberPlateDataset("data/train", transforms=get_transform(), isAnnotated=True)
    valid_dataset = NumberPlateDataset("data/valid", transforms=get_transform(), isAnnotated=True)
    test_dataset = NumberPlateDataset("data/test", transforms=get_transform(), isAnnotated=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Load model
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    num_classes = 2  # 1 number plate class + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Set hyperparameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3)

    if train:
        train_model(
            model, 
            train_loader, 
            valid_loader, 
            device, 
            optimizer,
            lr_scheduler, 
            num_epochs=2
        )
    else:
        # Load best model before testing
        best_model = torch.load("best_model.pth")
        model.load_state_dict(best_model["model_state_dict"])
        test_model(model, test_loader, device)

if __name__ == "__main__":
    main(train=False)  # Set to False to test only
