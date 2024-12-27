# This file contains the necessary code to preprocess the data and create the datasets for training, validation and testing.
import os 
import numpy as np 
import torch 
import matplotlib.pyplot as plt
import sklearn
import tqdm
import albumentations as A 
from tqdm import tqdm
from PIL import Image
from transformers import SegformerImageProcessor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import cv2 as cv

# Set the random seed for reproducibility   
random_state = 0
torch.manual_seed(random_state)
np.random.seed(random_state)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The current used device is: " + device.type)


# Define the transformations to be applied to the images and segmentation maps
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),                                    # Randomly flip horizontally
    A.VerticalFlip(p=0.5),                                      # Randomly flip vertically
    A.ShiftScaleRotate(shift_limit=0.0,                         # No shifting
                       scale_limit=0.0,                         # No scaling
                       rotate_limit=90,                         # Rotate within [-60, 60] degrees
                       p=1.0),                                  # Always apply
    A.RandomBrightnessContrast(p=1),                            # Randomly change brightness and contrast
    A.RandomGamma(p=1),                                         # Randomly change gamma
])

validation_transform = A.Compose([
])

# Split the data into training and validation sets
path = "/home/efe/Desktop/ml-project-2-middle_earth/train/"
def tr_te_split(root_dir):
    """
    Function to split the data into training and validation sets
    
    Args:
        root_dir : str : Path to the root directory containing the images and annotations
    
    Returns:
        train_images : list : List of paths to the training images
        val_images : list : List of paths to the validation images
        train_annotations : list : List of paths to the training annotations
        val_annotations : list : List of paths to the validation annotations
    """
    img_dir = os.path.join(root_dir, "images")
    ann_dir = os.path.join(root_dir, "groundtruth")
    images = sorted([root_dir + "images/" + f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    annotations = sorted([root_dir + "groundtruth/" + f for f in os.listdir(ann_dir) if f.endswith('.png')])
    assert len(images) == len(annotations), "Number of images and masks must be equal."
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size=0.2, random_state=0)
    return train_images, val_images, train_annotations, val_annotations

train_images, val_images, train_annotations, val_annotations = tr_te_split(path)


# Define the datasets for training, validation and testing
class TrainDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, image_list,mask_list,image_processor):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.image_processor = image_processor
        self.images = image_list
        self.masks = mask_list
        assert len(self.images) == len(self.masks), "There must be as many images as there are segmentation maps"

    def __len__(self):
        """
        Returns the number of images in the dataset
        
        Returns:
            int : Number of images in the dataset
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns the image and segmentation map at the given index
        
        Args:
            idx : int : Index of the image and segmentation map to be returned
        
        Returns:
            dict : Dictionary containing the image and segmentation map
        """

        image = Image.open(self.images[idx])
        image = np.array(image)
        segmentation_map = Image.open( self.masks[idx])
        segmentation_map = np.array(segmentation_map)
        segmentation_map = (segmentation_map > 125).astype(np.uint8) 
        augmented = train_transform(image=image, mask=segmentation_map)
        image = augmented['image']
        segmentation_map = augmented['mask']
        image = Image.fromarray(image)
        segmentation_map = Image.fromarray(segmentation_map)
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() 

        return encoded_inputs
    

class ValidationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, image_list,mask_list,image_processor):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """

        self.image_processor = image_processor
        self.images = image_list
        self.masks = mask_list
        assert len(self.images) == len(self.masks), "There must be as many images as there are segmentation maps"

    def __len__(self):
        """
        Returns the number of images in the dataset

        Returns:
            int : Number of images in the dataset
        """

        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns the image and segmentation map at the given index

        Args:
            idx : int : Index of the image and segmentation map to be returned
        
        Returns:
            dict : Dictionary containing the image and segmentation map
        """
        image = Image.open(self.images[idx])
        image = np.array(image)
        segmentation_map = Image.open( self.masks[idx])
        segmentation_map = np.array(segmentation_map)
        segmentation_map = (segmentation_map > 125).astype(np.uint8) 
        augmented = validation_transform(image=image, mask=segmentation_map)
        image = augmented['image']
        segmentation_map = augmented['mask']
        image = Image.fromarray(image)
        segmentation_map = Image.fromarray(segmentation_map)
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs
   
class TestDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, image_processor):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """

        self.root_dir = root_dir
        self.image_processor = image_processor

        self.img_dir = self.root_dir
        self.images = []
        for i in range(1,51):
            self.images.append(f"test_{i}.png")

    def __len__(self):
        """
        Returns the number of images in the dataset

        Returns:
            int : Number of images in the dataset
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns the image and segmentation map at the given index
        
        Args:
            idx : int : Index of the image and segmentation map to be returned
        """

        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        encoded_inputs = self.image_processor(image, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() 
        return encoded_inputs

feature_extractor = SegformerImageProcessor(reduce_labels=False)
train_dataset = TrainDataset(image_list=train_images,mask_list=train_annotations,image_processor=feature_extractor)
val_dataset = ValidationDataset(image_list=val_images,mask_list=val_annotations,image_processor=feature_extractor)