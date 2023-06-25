# A script to import predictions for an image classification model

import json
import pickle
from pathlib import Path
import random

import cv2
import torch
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

from encord_active.lib.db.predictions import FrameClassification, Prediction
from encord_active.lib.project import Project

project_path = '</path/to/local/project/[open-source][test]-Caltech-101>' # Enter your project path
checkpoint_file = ''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

model = resnet18()
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 101)
model.load_state_dict(torch.load('<path/to/local/checkpoints/caltech101 random values on images 4.ckpt>', map_location=device)) # E nter the path to the checkpoint
model.to(device)

transform_f = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224, 224)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

project = Project(Path(project_path)).load()

my_predictions = []
predictions_to_store = []

project_ontology = json.loads((project.file_structure.project_dir/'ontology_output.json').read_text())

ontology = json.loads(project.file_structure.ontology.read_text(encoding="utf-8"))
classes = []
for option in ontology["classifications"][0]["attributes"][0]["options"]:
    classes.append(option["value"])

for lr in tqdm(project.label_rows.values()):
    label_hash = lr['label_hash']
    data_unit_hash = lr['data_hash']
    image_path = project.file_structure.data / label_hash / 'images' / (data_unit_hash+'.jpg')
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_transformed = transform_f(image)

    model.eval()
    output = model(image_transformed.to(device).unsqueeze(dim=0))
    class_id = output.argmax(dim=1, keepdim=True)[0][0].item()
    model_prediction = project_ontology['classifications'][classes[class_id]]
    confidence = output.softmax(1).tolist()[0][class_id]
    predictions_to_store.append(
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
    pickle.dump(predictions_to_store, f)
    