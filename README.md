# Chilli Leaf Disease Classifier

An image classification demo for screening chilli leaves across eight disease and healthy classes. The project includes a Jupyter inference notebook, a Gradio web app, and a model artifact hosted on Hugging Face.

**Live demo:** [Open the Chilli Leaf Disease Detector](https://huggingface.co/spaces/NazmulNIR/chilli-leaf-disease-detector)

## Project files

- [`chilli_disease_inference.ipynb`](./chilli_disease_inference.ipynb) — load the model, classify an image, and open a local upload widget from Jupyter or Colab.
- [`app.py`](./app.py) — Gradio interface used by the live Space.
- [`requirements.txt`](./requirements.txt) — pinned TensorFlow and Gradio versions plus the Hugging Face Spaces helper.
- [Model artifact](https://huggingface.co/spaces/NazmulNIR/chilli-leaf-disease-detector/blob/main/efficientNet_model.keras) — 21 MB Keras model hosted separately from GitHub.

## Run the notebook

Open `chilli_disease_inference.ipynb` in Jupyter or Google Colab. Install the dependencies with:

```bash
pip install -r requirements.txt
```

The first code cell loads `efficientNet_model.keras` from the notebook directory. If the file is not there, it downloads the public model artifact from the Hugging Face Space. Set `IMAGE_PATH` to a chilli leaf photo and run the next cell to get the top three class scores. The last cell starts an image upload widget in the notebook.

## Run the Gradio app locally

Download `efficientNet_model.keras` from the model link above into the same folder as `app.py`, then run:

```bash
pip install -r requirements.txt
python app.py
```

The app expects RGB images resized to 224 × 224 pixels and reports its top predictions with confidence scores.

## Output classes

The current inference label order is:

1. Whitefly
2. Yellowish
3. Anthracnose
4. Damping off
5. Leaf curl virus
6. Leaf spot
7. Veinal mottle virus
8. Healthy

## Limitations

This is an experimental screening demo. Prediction quality depends on the supplied training data and image quality. It does not confirm a diagnosis or recommend pesticide treatment. Consult an agricultural expert before acting on a prediction. No accuracy metrics are claimed in this repository.

**Author:** Sheikh Nazmul Islam · [GitHub profile](https://github.com/SheikhNazmul)
