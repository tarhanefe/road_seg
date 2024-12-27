from tqdm import tqdm
import torch
import torch.nn.functional as F
import sklearn
import numpy as np

def train(model, optimizer, lr_scheduler, train_loader, val_loader, NUM_EPOCHS,device):
    """
    Function to train the model

    Args:
    model : torch model : Model to be trained
    optimizer : torch optimizer : Optimizer to be used
    lr_scheduler : torch scheduler : Learning rate scheduler to be used
    train_loader : torch DataLoader : Training data loader
    val_loader : torch DataLoader : Validation data loader
    NUM_EPOCHS : int : Number of epochs
    device : torch device : Device to be used for training
    """

    model.to(device)
    metrics = {}
    metrics["train_loss"] = []
    metrics["f1_score"] = []
    for epoch in range(NUM_EPOCHS):  
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 10)
        model.train()
        epoch_loss = 0.0
        for idx, batch in enumerate(tqdm(train_loader)):
                image = batch["pixel_values"].to(device)
                mask = batch["labels"].to(device)
                optimizer.zero_grad()
                outputs = model(pixel_values=image, labels=mask)
                loss, logits = outputs.loss, outputs.logits
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        lr_scheduler.step()
        metrics["train_loss"].append(epoch_loss / len(train_loader))
        print("Train Loss:", epoch_loss / len(train_loader))
        model.eval()
        with torch.no_grad():
                total_loss = 0
                msks = []
                pres = []
                for i, batch in enumerate(tqdm(val_loader)):
                    image = batch["pixel_values"].to(device)
                    mask = batch["labels"].to(device)
                    output = model(pixel_values=image, labels=mask)
                    loss, logits = output.loss, output.logits
                    preds = F.interpolate(logits, size=(608, 608), mode='bilinear', align_corners=False)
                    mask = F.interpolate(mask.unsqueeze(1).float(), size=(608, 608), mode='nearest').squeeze(1)[0]
                    preds = preds.argmax(dim=1)[0].cpu().numpy()
                    total_loss += loss.item()
                    msks.append(mask.cpu().numpy())
                    pres.append(preds)
                f1_score = sklearn.metrics.f1_score(np.vstack(msks).flatten(), np.vstack(pres).flatten(), average='binary')
                metrics["f1_score"].append(f1_score)
                print(f"Validation Loss: {total_loss / len(val_loader)}, F1 Score: {f1_score}")
                print("-" * 10)
    model_dict = model.state_dict()
    torch.save(model_dict, "/home/efe/Desktop/ml-project-2-middle_earth/model_dicts/segformer.pth")
    return model, metrics