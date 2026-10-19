import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import spaces
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), "efficientNet_model.keras")
CLASS_NAMES = [
    "Chilli __Whitefly",
    "Chilli __Yellowish",
    "Chilli__Anthracnos",
    "Chilli__Damping_Off",
    "Chilli__Leaf_Curl_Virus",
    "Chilli__Leaf_Spot",
    "Chilli__Veinal_Mottle_Virus",
    "Chilli___healthy",
]

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


@spaces.GPU
def predict_disease(image):
    if image is None:
        return {}, "Upload a clear chilli leaf image to get started."

    image = image.convert("RGB").resize((224, 224))
    pixels = tf.keras.utils.img_to_array(image)
    scores = model.predict(np.expand_dims(pixels, axis=0), verbose=0)[0]
    results = {CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))}
    best_idx = int(np.argmax(scores))
    confidence = float(scores[best_idx])
    label = CLASS_NAMES[best_idx]

    if confidence < 0.65:
        advice = (
            "⚠️ The model has low confidence. Try a well-lit, close-up image of a single chilli leaf. "
            "This demo is not a substitute for diagnosis by an agricultural expert."
        )
    elif "healthy" in label.lower():
        advice = "✅ The model classified this leaf as healthy. Continue regular crop monitoring."
    else:
        disease = label.replace("Chilli__", "").replace("Chilli __", "").replace("_", " ").strip()
        advice = (
            f"🚨 The model's top prediction is **{disease}** ({confidence:.1%}). "
            "Use this as an initial screening result and consult a local agricultural expert "
            "before choosing a treatment."
        )
    return results, advice


with gr.Blocks(theme=gr.themes.Soft(), title="Chilli Leaf Disease Detector") as demo:
    gr.Markdown("# 🌿 Chilli Leaf Disease Detector")
    gr.Markdown("Upload a chilli leaf photo to see the model's top disease predictions.")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Chilli leaf image", type="pil")
            analyze = gr.Button("Analyze leaf", variant="primary")
        with gr.Column():
            predictions = gr.Label(num_top_classes=3, label="Top predictions")
            advice = gr.Markdown()
    analyze.click(predict_disease, inputs=image_input, outputs=[predictions, advice])
    gr.Markdown(
        "---\n**Note:** This model provides an experimental screening result, not a confirmed diagnosis "
        "or pesticide recommendation. Prediction quality depends on image quality and training data."
    )

if __name__ == "__main__":
    demo.launch()



