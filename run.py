import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
from src.preprocess import tr_te_split
from src.preprocess import TrainDataset, ValidationDataset, TestDataset
from src.model import def_model
from src.train import train
from src.test import test


def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset and preprocess
    train_path = "/home/efe/Desktop/ml-project-2-middle_earth/train/"
    test_path = "/home/efe/Desktop/ml-project-2-middle_earth/test"
    
    train_images, val_images, train_annotations, val_annotations = tr_te_split(train_path)
    feature_extractor = SegformerImageProcessor(reduce_labels=False)
    train_dataset = TrainDataset(
        image_list=train_images,
        mask_list=train_annotations,
        image_processor=feature_extractor
    )
    val_dataset = ValidationDataset(
        image_list=val_images,
        mask_list=val_annotations,
        image_processor=feature_extractor
    )

    # Load the model
    model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
    model = def_model(model_name)

    # Define hyperparameters
    LEARNING_RATE = 1e-3
    STEP_SIZE = 15
    GAMMA = 0.5
    NUM_EPOCHS = 150
    BATCH_SIZE = 2

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # Train the model
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    model, metrics = train(
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        NUM_EPOCHS,
        device
    )

    # Create the test set
    test_set = TestDataset(
        test_path,
        feature_extractor
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Make predictions
    output_dir = "/home/efe/Desktop/ml-project-2-middle_earth/submission/predicted/segformer_predicted"
    submission_file = "/home/efe/Desktop/ml-project-2-middle_earth/submission/submission.csv"

    test(test_loader = test_loader,
         model = model,
         device = device,
         output_dir = output_dir,
         submission_file = submission_file)


if __name__ == "__main__":
    main()
