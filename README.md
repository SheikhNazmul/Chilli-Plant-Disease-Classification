# 🌶️ Chilli Plant Disease Classification

A deep-learning project for classifying chilli plant leaf images into eight disease/health categories using transfer learning with **EfficientNetB0, MobileNetV2, and ResNet50**.

> **Project status:** Research/learning project. Results depend on the dataset, training configuration, and runtime. No accuracy numbers are claimed here because they should be measured from a completed run and verified outputs.

## Overview

This notebook explores an image-classification pipeline for chilli plant leaf conditions. It uses TensorFlow/Keras, directory-based image loading, data augmentation, pretrained CNN backbones, model evaluation, confusion matrices, classification reports, and an experimental Gradio interface.

## Models explored

- **EfficientNetB0** — ImageNet-pretrained feature extractor with global average pooling, a 256-unit ReLU dense layer, dropout, and an 8-class softmax output.
- **MobileNetV2** — ImageNet-pretrained backbone with MobileNetV2 preprocessing and a custom classification head.
- **ResNet50** — ImageNet-pretrained backbone with ResNet50 preprocessing and a custom classification head.

In the notebook, the pretrained backbones are frozen during the initial training stage. The models are compiled with Adam and categorical cross-entropy and trained for 10 epochs in the provided cells.

## Classes

The notebook's Gradio section defines these eight output labels:

1. `Chilli__Whitefly`
2. `Chilli__Yellowish`
3. `Chilli__Anthracnos`
4. `Chilli__Damping_Off`
5. `Chilli__Leaf_Curl_Virus`
6. `Chilli__Leaf_Spot`
7. `Chilli__Veinal_Mottle_Virus`
8. `Chilli___healthy`

These labels are reproduced from the notebook. Confirm that the dataset's actual directory names and class ordering match the inference label list before using the demo.

## Workflow

1. Load the dataset from a ZIP archive.
2. Create training, validation, and test generators from class-organized folders.
3. Resize images to **224 × 224** and use batches of **32**.
4. Apply rotation, width/height shifts, and horizontal flipping to training images.
5. Train the transfer-learning models.
6. Plot training/validation accuracy and loss.
7. Evaluate on the test generator.
8. Generate a confusion matrix and classification report.
9. Try image-level predictions and a Gradio interface.

## Repository contents

- `chilli_diseases.ipynb` — Colab/Jupyter notebook containing experiments and the demo code.

## Run in Google Colab

1. Open the notebook from this repository.
2. Upload your dataset ZIP to Google Drive, or update the notebook to point to a dataset location accessible from Colab.
3. Review and update all paths before running. The original notebook uses Google Drive and Colab paths such as `/content/drive/MyDrive/archive.zip` and `/content/dataset/`.
4. Run the notebook cells in order. Training deep-learning models generally requires a GPU for reasonable speed.
5. Check the evaluation outputs and confirm the correct model file is loaded before running the Gradio demo.

### Expected dataset layout

```text
Chilli_Plant_Diseases_Dataset/
├── train/
│   ├── class_name_1/
│   └── ...
├── valid/
│   ├── class_name_1/
│   └── ...
└── test/
    ├── class_name_1/
    └── ...
```

The dataset itself is not included in this repository. Use a dataset you have permission to access and share, and follow its license/attribution requirements.

## Requirements

The notebook uses these main libraries:

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Pillow
- Gradio

Install in a notebook cell if needed:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow gradio
```

The exact compatible package versions depend on the Colab/runtime environment. For reproducibility, record the versions used for a successful run.

## Important notes before reuse

- **Review the notebook paths.** Several cells use Google Drive-specific paths and saved model paths that may not exist in a fresh environment.
- **Verify model saving.** Some save statements in the notebook refer to a different model variable or filename than the model being trained. Confirm each save/load pair before relying on it.
- **Verify class order.** The hard-coded Gradio class list must exactly match the training generator's class-index order.
- **Check preprocessing consistency.** Each architecture may require its own preprocessing; keep training and inference preprocessing aligned.
- **Evaluate honestly.** Report test accuracy, precision, recall, F1-score, and confusion matrix only from a completed, correctly configured run.
- **Not agricultural advice.** Predictions are experimental and should not replace assessment by a qualified agricultural expert.

## Future improvements

- Refactor notebook paths into configurable variables.
- Add a reproducible environment file and dataset citation.
- Add per-model metrics in a comparison table.
- Correct and verify model save/load paths.
- Add Grad-CAM visual explanations and a standalone inference script.
- Package the Gradio demo with a tested model artifact and deployment instructions.

## Author

**Sheikh Nazmul Islam**  
GitHub: [@SheikhNazmul](https://github.com/SheikhNazmul)

---

If this project helps your work, feel free to star the repository.

## Run the inference notebook

A standalone inference notebook is available at [`chilli_disease_inference.ipynb`](./chilli_disease_inference.ipynb). It loads `efficientNet_model.keras` from the same directory or downloads it from the Hugging Face Space when the model artifact is published there.

1. Open the notebook in Jupyter or Google Colab.
2. Install the packages with `pip install -r requirements.txt`.
3. Run the setup cell to load the model.
4. Set `IMAGE_PATH` to a chilli leaf image and run prediction, or run the final cell for an interactive upload widget.

The Gradio application source is [`app.py`](./app.py). The live demo and model hosting target is [Hugging Face Spaces](https://huggingface.co/spaces/NazmulNIR/chilli-leaf-disease-detector). The model file is hosted separately from GitHub because it is a large binary artifact.

The project files include `requirements.txt` with pinned inference dependencies. The dataset and trained model are not stored in this GitHub repository.