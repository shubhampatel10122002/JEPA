# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:32:04 2024

@author: 22011
"""

from dataset import create_wall_dataloader
from evaluator_vari2 import ProbingEvaluator
import torch
#from models import MockModel
import glob
from jepa_main_vari2 import JEPAModel, Encoder

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    #data_path = "/scratch/DL24FA"
    data_path = "."

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


# def load_model():
#     """Load or initialize the model."""
#     # TODO: Replace MockModel with your trained model
    
#     return model

def load_model():
    # Path to the uploaded model checkpoint
    
    model_checkpoint_path = "./model_vari2_weights.pth"
    #model_checkpoint_path = "./continued_training_best_final.pth"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the JEPA model
    model = JEPAModel(momentum=0.99).to(device)

    # Load the model weights
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode
    model.eval()
    print("Model loaded successfully.")

    return model




def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
