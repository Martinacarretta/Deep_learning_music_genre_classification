import wandb
import torch
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def test(model, test_loader, device="cuda", save: bool = True, save_path: str = "confusion_matrix.png"):
    all_labels = []
    all_predictions = []

    # Run the model on the test set
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels, other in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Accumulate labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print(f"Accuracy of the model on the {total} test images: {correct / total:%}")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Visualize and save the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.show()

    # Log accuracy to wandb
    wandb.log({"test_accuracy": correct / total})
