from tqdm.auto import tqdm
import numpy as np
import wandb
import torch.optim.lr_scheduler as lr_scheduler
import torch

def train(model, train_loader, val_loader, criterion, optimizer, config):
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    total_batches = len(train_loader) * config['epochs']
    example_ct = 0
    batch_ct = 0
    for epoch in tqdm(range(config['epochs'])):
        # Training
        model.train()
        for batch_idx, (images, labels, other) in enumerate(train_loader):
            loss, accuracy = train_batch(images, labels, model, optimizer, criterion)
            example_ct += len(images)
            batch_ct += 1

            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch, accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, other in val_loader:
                images, labels = images.to(config['device']), labels.to(config['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Log validation loss and accuracy
        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_accuracy": accuracy}, step=example_ct)

        # Update learning rate scheduler based on validation loss
        scheduler.step(val_loss)


def train_batch(images, labels, model, optimizer, criterion, device="cuda"):

    images, labels = images.to(device), labels.to(device) # Move data to device (CPU/GPU)

    outputs = model(images) # Forward pass
    loss = criterion(outputs, labels) # Compute loss

    _, predicted = outputs.max(1) # Get predictions
    correct = (predicted == labels).sum().item() # Count correct predictions
    accuracy = correct / labels.size(0) # Compute accuracy

    optimizer.zero_grad() # Zero the gradients
    loss.backward() # Backward pass
    optimizer.step() # Update the weights

    return loss, accuracy

def train_log(loss, example_ct, epoch, accuracy):
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy}, step=example_ct)  # keep track of the epoch, loss and accuracy when training
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
