import json
import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from encord_active.lib.db.predictions import FrameClassification, Prediction
from encord_active.lib.project import Project

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

# Import encord project
project_path = r'EASOTA'
project = Project(Path(project_path)).load()

project_ontology = json.loads(
    (project.file_structure.project_dir/'ontology_output.json').read_text()
)

ontology = json.loads(
    project.file_structure.ontology.read_text(encoding="utf-8")
)

# Image classes
classes = []
for option in ontology["classifications"][0]["attributes"][0]["options"]:
    classes.append(option["value"])


# Define the data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Function to extract image label from label_rows metadata
def get_label(classification_answers):
    k = list(classification_answers.keys())[0]
    classification_answers = classification_answers[k]
    answers = classification_answers['classifications'][0]['answers']
    label = answers[0]['value']
    return label


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()

        # input shape (3, 256, 256)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # shape (16, 128, 128)

        # input shape (16, 128, 128)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # output shape (32, 64, 64)

        # input shape (32, 64, 64)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # output shape (64, 32, 32)

        # input shape (64, 32, 32)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # output shape (32, 16, 16)

        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Create an instance of the model
model = CNN()

# Load the saved state dictionary file
model_path = 'cnn_model.pth'
model.load_state_dict(torch.load(model_path))

# Variables
my_predictions = []  # To store predicted labels
predictions_to_import = []  # Predictions to be imported into encord-active
image_paths = []  # List of all image paths

# List of all image True classes
image_labels = [
    get_label(lr['classification_answers'])
    for lr in project.label_rows.values()
]


# Make predictions
for item in tqdm(project.file_structure.iter_labels()):
    for data_unit_hash, image_path in item.iter_data_unit():
        data_unit_hash, image_path = str(data_unit_hash), str(image_path)
        image_paths.append(image_path)
        image = cv2.imread(image_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transformed = test_transforms(image)

    model.eval()
    output = model(image_transformed.to(device).unsqueeze(dim=0))
    class_id = output.argmax(dim=1, keepdim=True)[0][0].item()
    model_prediction = project_ontology['classifications'][classes[class_id]]
    my_predictions.append(classes[class_id])
    confidence = output.softmax(1).tolist()[0][class_id]
    predictions_to_import.append(
        Prediction(
            data_hash=data_unit_hash,
            confidence=confidence,
            classification=FrameClassification(
                feature_hash=model_prediction['feature_hash'],
                attribute_hash=model_prediction['attribute_hash'],
                option_hash=model_prediction['option_hash'],
            ),
        )
    )

with open(f"{project_path}/predictions.pkl", "wb") as f:
    pickle.dump(predictions_to_import, f)


# Metrics
print(classification_report(
        image_labels,
        my_predictions, target_names=classes
    )
)

report = classification_report(
    image_labels,
    my_predictions,
    target_names=classes,
    output_dict=True
)

mean_f1_score = report['macro avg']['f1-score']
mean_recall = report['macro avg']['recall']
mean_precision = report['macro avg']['precision']

print("Mean F1-score: ", mean_f1_score)
print("Mean recall: ", mean_recall)
print("Mean precision: ", mean_precision)


cm = confusion_matrix(image_labels, my_predictions,)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the confusion matrix
im = ax.imshow(cm, cmap='Blues')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Set labels, ticks, and tick labels
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes,
       yticklabels=classes,
       xlabel='Predicted label',
       ylabel='True label')

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2. else "black")

# Add title
ax.set_title("Confusion matrix")

# Show the plot
plt.show()
