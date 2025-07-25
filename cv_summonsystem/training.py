import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import matplotlib.pyplot as plt

map_history = []
map_50_history = []

def train_model(model, train_loader, valid_loader, device, optimizer, lr_scheduler, num_epochs=10):
    best_val_map = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        print(f"Epoch {epoch+1} Train Loss: {epoch_loss / len(train_loader):.4f}")

        # Validation Phase
        val_map = evaluate_map(model, valid_loader, device)
        print(f"\nValidation mAP Results:")
        print(f"  mAP@0.50-0.95 : {val_map['map']:.4f}")
        print(f"  mAP@0.50      : {val_map['map_50']:.4f}")
        
        # Save best model
        if val_map['map'] > best_val_map:
            best_val_map = val_map['map']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_map': best_val_map
            }, 'best_model.pth')
            print("Saved new best model.")

        lr_scheduler.step(val_map['map'])

    epochs = list(range(1, len(map_history) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, map_history, label="mAP@[0.5:0.95]")
    plt.plot(epochs, map_50_history, label="mAP@0.50", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("mAP over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_map(model, valid_loader, device):
    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad():
        for images, targets in tqdm(valid_loader, desc="Validating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            metric.update(outputs, targets)
    result = metric.compute()
    val_map = result["map"].item()
    val_map50 = result["map_50"].item()
    
    map_history.append(val_map)
    map_50_history.append(val_map50)
    return result