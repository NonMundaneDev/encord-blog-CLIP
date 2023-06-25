import json
import os
import pickle
from pathlib import Path
import shutil

import cv2
import clip
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch
from torchvision import transforms
from tqdm import tqdm

from encord_active.lib.db.predictions import FrameClassification, Prediction
from encord_active.lib.project import Project

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

# load clip model
model, preprocess = clip.load("ViT-B/32", device=device)

# Import encord project
project_path = r'EAemotions'
project = Project(Path(project_path)).load()

##
# To store the ontology as ontology_output.json in the current(project)
# working directory, you can do
#
# ```encord-active print --json ontology```
#

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


##
# encode class texts
def generate_encoded_texts_from_classes():
    tkns = [f'A photo of a {class_} face' for class_ in classes]
    texts = clip.tokenize(tkns).to(device)
    return texts


encoded_texts = generate_encoded_texts_from_classes()


# Function to extract image label from label_rows metadata
def get_label(classification_answers):
    k = list(classification_answers.keys())[0]
    classification_answers = classification_answers[k]
    answers = classification_answers['classifications'][0]['answers']
    label = answers[0]['value']
    return label


# Variables
my_predictions = []  # To store predicted labels
predictions_to_import = []  # Predictions to be imported into encord-active
image_paths = []  # List of all image paths
image_labels = []  # List of all image True classes

# Image Transformer function
transform_f = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# Make predictions
for lr in tqdm(project.label_rows.values()):
    label_hash = lr['label_hash']
    data_unit_hash = list(lr['data_units'].keys())[0]
    image_path = os.path.join(
        project.file_structure.data,
        label_hash,
        'images',
        data_unit_hash+'.jpg'
    )
    image_paths.append(image_path)
    image_labels.append(get_label(lr['classification_answers']))
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_transformed = transform_f(image)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(
            image_transformed.to(device).unsqueeze(dim=0),
            encoded_texts
        )
        class_id = logits_per_image.argmax(dim=1, keepdim=True)[0][0].item()
        model_prediction = project_ontology['classifications'][classes[class_id]]
        my_predictions.append(classes[class_id])
        confidence = logits_per_image.softmax(1).tolist()[0][class_id]

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


# Export predictions
with open(f"{project_path}/predictions.pkl", "wb") as f:
    pickle.dump(predictions_to_import, f)


# Create Dataset from CLIP prediction for Training CNN
def create_dataset(path="Clip_GT_labels", train_ratio=0.6):
    parent_folder = path
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    train_folder = os.path.join(parent_folder, "Train")
    val_folder = os.path.join(parent_folder, "Val")
    test_folder = os.path.join(parent_folder, "Test")

    # Allocate Ration
    train_len = int(train_ratio * len(image_paths))
    test_ratio = (1 - train_ratio) / 2
    test_len = int(test_ratio * len(image_paths))

    train_image_paths = image_paths[:train_len]
    train_classes = my_predictions[:train_len]

    val_image_paths = image_paths[train_len: train_len + test_len]
    val_classes = my_predictions[train_len: train_len + test_len]

    test_image_paths = image_paths[train_len + test_len:]
    test_classes = my_predictions[train_len + test_len: ]

    print("Creating Dataset using CLIP predictions as GT labels")

    # Train set
    for image_path, label in tqdm(zip(train_image_paths, train_classes)):
        print("Creating Train Dataset")
        folder_name = os.path.join(train_folder, label)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        image_name = f"{image_path}".split(os.path.sep)[-1]
        filepath = os.path.join(folder_name, image_name)
        shutil.copy(image_path, filepath)

    # Val set
    for image_path, label in tqdm(zip(val_image_paths, val_classes)):
        print("Creating Validation Dataset")
        folder_name = os.path.join(val_folder, label)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        image_name = f"{image_path}".split(os.path.sep)[-1]
        filepath = os.path.join(folder_name, image_name)
        shutil.copy(image_path, filepath)

    # Test set
    for image_path, label in tqdm(zip(test_image_paths, test_classes)):
        print("Creating Test Dataset")
        folder_name = os.path.join(test_folder, label)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        image_name = f"{image_path}".split(os.path.sep)[-1]
        filepath = os.path.join(folder_name, image_name)
        shutil.copy(image_path, filepath)

    print("Dataset Created Succesfully from CLIP Predictions")


create_dataset()


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
