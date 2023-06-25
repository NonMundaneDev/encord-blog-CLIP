# 🔎 Evaluating Foundation Models (CLIP) using Encord-Active

This repository is for a blog article on evaluating Foundation Models (CLIP) using Encord Active. 🟣 Encord Active is an open-source framework for computer vision model testing, evaluation, and validation.

Link to dataset to the working directory - [`emotions-1`](https://www.dropbox.com/sh/rovspvmbtxg2mdx/AAAk9UM8GI57KhRD5ljehGDNa?dl=0).

## 🚀 Steps

1. Create a Virtual environment

   ```bash
    python3.9 -m venv ea-venv

    # On Linux/MacOS
    source ea-venv/bin/activate

    # On Windows
    ea-venv\Scripts\activate
   ```

2. Install encord active

    ```bash
    python -m pip install encord-active
    ```

3. Install some other required libraries

   ```bash
    # Install tqdm
    python -m pip install tqdm

    # Install CLIP
    python -m pip install git+https://github.com/openai/CLIP.git
   ```

4. Unzip dataset

    ```bash
    unzip emotions-1.zip
    ```

5. Create First encord project (CLIP classification)

    ```bash
    # Change directory
    cd ./ea_foundation_models

    encord-active init --name EAemotions --transformer classification_transformer.py ./emotions-1

    # Change directory
    cd ./EAemotions

    # Store ontology
    encord-active print --json ontology
    ```

6. Execute CLIP prediction script

    ```bash
    # Go back to root folder
    cd ..

    # execute script
    python milestone-1a.py
    ```

7. Import CLIP prediction into encord-active project

    ```bash
    # Change to Project directory
    cd ./EAemotions

    # Import Predictions
    encord-active import predictions predictions.pkl

    # Start encord-active webapp server
    encord-active visualize
    ```

8. Train CNN algorithm on Dataset built woth predictions from CLIP as GT Labels

   ```bash
   # Change to root folder
   cd ..

   # Except training script
   python milestone-1b.py
   ```

9. Create a new encord project

    ```bash
    # Create project
    encord-active init --name EAsota --transformer classification_transformer.py Clip_GT_labels\Test

    # Change to project directory
    cd EAsota

    # Store ontology
    encord-active print --json ontology
    ```

10. Execute prediction script

    ```bash
    # change to root directory
    cd ..

    # execute script
    python milestone-1c.py
    ```

11. Import predictions into encord-active project

    ```bash
    # Change to Project directory
    cd ./EAsota

    # Import Predictions
    encord-active import predictions predictions.pkl

    # Start encord-active webapp server
    encord-active visualize
    ```

## 🛠️ Contributors

✅ @[Franklin Obasi](https://github.com/franklinobasy)


✅ @[Stephen Oladele](https://github.com/NonMundaneDev/)