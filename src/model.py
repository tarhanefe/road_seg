# This file contains the model definition for the Segformer model
from transformers import SegformerForSemanticSegmentation
model_name = "nvidia/segformer-b3-finetuned-ade-512-512"

def def_model(model_name):
    """
    Function to define the Segformer model

    Args:
    model_name : str : Name of the model to be used
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=2,
        id2label={0: "no_road", 1: "road"},
        label2id={"no_road": 0, "road": 1},
        ignore_mismatched_sizes=True)
    return model