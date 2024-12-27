import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.mask_to_submission import *

def test(test_loader,model,device,output_dir,submission_file):
    """
    Function to generate predictions for the test set and create a submission file

    Args:
    test_loader : torch DataLoader : DataLoader for the test set
    model : torch model : Model to be used for prediction
    device : torch device : Device to be used for prediction
    output_dir : str : Directory to save the predicted masks
    submission_file : str : Path to save the submission file
    """
    
    os.makedirs(output_dir, exist_ok=True)
    print("Generating predictions for the test set...")
    model.eval()
    image_filenames = []
    for idx, batch in enumerate(tqdm(test_loader)):    
        with torch.no_grad():
            image = batch["pixel_values"].to(device)
            image90 = torch.rot90(image, 1, [2, 3])
            image180 = torch.rot90(image, 2, [2, 3])
            image270 = torch.rot90(image, 3, [2, 3])

            output = model(pixel_values=image)
            logits = output.logits
            logits = F.interpolate(logits, size=(608, 608), mode='bilinear', align_corners=False)

            output90 = model(pixel_values=image90)
            logits90 = output90.logits
            logits90 = F.interpolate(logits90, size=(608, 608), mode='bilinear', align_corners=False)
            logits90 = torch.rot90(logits90, 3, [2, 3])

            output180 = model(pixel_values=image180)
            logits180 = output180.logits
            logits180 = F.interpolate(logits180, size=(608, 608), mode='bilinear', align_corners=False)
            logits180 = torch.rot90(logits180, 2, [2, 3])
            
            output270 = model(pixel_values=image270)
            logits270 = output270.logits
            logits270 = F.interpolate(logits270, size=(608, 608), mode='bilinear', align_corners=False)
            logits270 = torch.rot90(logits270, 1, [2, 3])

            logits = (logits + logits90 + logits180 + logits270) / 4
            preds = logits.argmax(dim=1)[0].cpu().numpy()
                        
            output_path = os.path.join(output_dir, f"pred_mask_{idx + 1:03d}.png")
            binary_mask_image = Image.fromarray((preds * 255).astype(np.uint8))  # Convert to 0-255 scale and uint8
            binary_mask_image.save(output_path)

            # Append filename for submission
            image_filename = f"/home/efe/Desktop/ml-project-2-middle_earth/submission/predicted/segformer_predicted/pred_mask_{idx + 1:03d}.png"
            image_filenames.append(image_filename)

    # Create submission file
    print("Creating submission file...")
    masks_to_submission(submission_file, *image_filenames)
    print(f"Submission file saved to {submission_file}")
