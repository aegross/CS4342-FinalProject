import numpy as np
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt  # if necessary

# https://huggingface.co/Falcom/animal-classifier?library=transformers
# use a pipeline as a high-level helper
from transformers import pipeline
# or load the model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

pipe = pipeline("image-classification", model="Falcom/animal-classifier")
processor = AutoImageProcessor.from_pretrained("Falcom/animal-classifier")
model = AutoModelForImageClassification.from_pretrained("Falcom/animal-classifier")

# https://huggingface.co/docs/transformers/main/en/model_doc/vit#transformers.ViTForImageClassification
# assuming that what's here + what's already been used in deep-model.py will be helpful for implementation