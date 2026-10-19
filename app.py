import html
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import numpy as np
import spaces
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
DISPLAY_NAMES = {
    "Chilli __Whitefly": "Whitefly",
    "Chilli __Yellowish": "Yellowing",
    "Chilli__Anthracnos": "Anthracnose",
    "Chilli__Damping_Off": "Damping off",
    "Chilli__Leaf_Curl_Virus": "Leaf curl virus",
    "Chilli__Leaf_Spot": "Leaf spot",
    "Chilli__Veinal_Mottle_Virus": "Veinal mottle virus",
    "Chilli___healthy": "Healthy",
}

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def result_card(title, confidence, status, description):
    return f"""
    <section class="result-card">
      <div class="result-kicker"><span class="result-dot"></span>{status}</div>
      <div class="result-title">{html.escape(title)}</div>
      <div class="result-confidence">{confidence:.1%}<span> model confidence</span></div>
      <p>{html.escape(description)}</p>
    </section>
    """


@spaces.GPU
def predict_disease(image):
    if image is None:
        return {}, """
        <section class="result-card result-empty">
          <div class="result-kicker">YOUR SCREENING RESULT</div>
          <div class="result-title">Add a leaf photo to begin</div>
          <p>Choose a clear, well-lit image showing one chilli leaf.</p>
        </section>
        """

    image = image.convert("RGB").resize((224, 224))
    pixels = tf.keras.utils.img_to_array(image)
    scores = np.asarray(
        model.predict(np.expand_dims(pixels, axis=0), verbose=0)[0],
        dtype=np.float32,
    ).reshape(-1)

    if len(scores) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} model outputs, received {len(scores)}."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid scores. Please try another image.")

    # Keep the displayed values comparable even if the model output is not normalized.
    if np.any(scores < 0) or not np.isclose(float(scores.sum()), 1.0, atol=0.02):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        scores = exp_scores / exp_scores.sum()

    ranked = np.argsort(scores)[::-1][:3]
    results = {DISPLAY_NAMES[CLASS_NAMES[i]]: float(scores[i]) for i in ranked}
    best_idx = int(ranked[0])
    confidence = float(scores[best_idx])
    label = CLASS_NAMES[best_idx]
    title = DISPLAY_NAMES[label]

    if confidence < 0.65:
        status = "LOW CONFIDENCE · CHECK THE PHOTO"
        description = (
            "The model is not confident enough to support a useful screening result. "
            "Try a sharper close-up in natural light, with the leaf filling most of the frame."
        )
    elif "healthy" in label.lower():
        status = "TOP MATCH · HEALTHY"
        description = (
            "The model's strongest match is the healthy class. Keep monitoring the plant; "
            "this image-based result cannot rule out early or hidden problems."
        )
    else:
        status = "TOP MATCH · FURTHER CHECK RECOMMENDED"
        description = (
            f"The strongest match is {title}. Check other leaves and plant symptoms, "
            "then ask a local agricultural expert to confirm before taking action."
        )

    return results, result_card(title, confidence, status, description)


CUSTOM_CSS = """
:root { --ink: #f2f5ee; --muted: #aab8ad; --leaf: #a5d879; --deep: #0c1713; }
.gradio-container {
  max-width: 1180px !important; margin: 0 auto !important;
  background: radial-gradient(ellipse at 12% 0%, #203b2c 0, #111e18 38%, #0b1411 78%) !important;
  color: var(--ink) !important;
}
body, .gradio-container { font-family: Inter, ui-sans-serif, system-ui, sans-serif !important; }
#hero {
  padding: 34px 10px 22px; border-bottom: 1px solid rgba(190,220,190,.16);
  margin-bottom: 22px;
}
#hero .eyebrow { color: #bce49d; text-transform: uppercase; letter-spacing: .16em;
  font-size: 12px; font-weight: 750; }
#hero h1 { color: #f5f7f1; font-size: clamp(32px, 5vw, 48px); line-height: 1.06;
  margin: 10px 0 12px; letter-spacing: -.04em; }
#hero p { color: #c2cdc3; max-width: 710px; font-size: 16px; line-height: 1.7; }
.panel {
  background: rgba(26, 42, 34, .86) !important; border: 1px solid rgba(187,215,190,.16) !important;
  border-radius: 20px !important; box-shadow: 0 16px 48px rgba(0,0,0,.16) !important;
  padding: 20px !important;
}
.panel-heading { color: #f1f5ed; font-size: 19px; font-weight: 720; margin-bottom: 4px; }
.panel-copy { color: #aebdb1; font-size: 13px; line-height: 1.6; margin: 0 0 14px; }
#leaf-input { border-radius: 15px !important; overflow: hidden; }
#analyze button {
  min-height: 52px; border-radius: 13px !important; font-weight: 750 !important;
  background: #a5d879 !important; color: #112017 !important; border: 0 !important;
}
#clear button { min-height: 48px; border-radius: 13px !important; }
#result-card { min-height: 205px; }
.result-card {
  min-height: 174px; box-sizing: border-box; padding: 22px; border-radius: 16px;
  background: linear-gradient(135deg, rgba(160,211,122,.14), rgba(34,58,44,.62));
  border: 1px solid rgba(175,220,148,.23);
}
.result-empty { background: rgba(13,25,19,.55); border-color: rgba(187,215,190,.13); }
.result-kicker { color: #bee59e; font-size: 11px; font-weight: 800; letter-spacing: .1em; }
.result-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: #a5d879; margin-right: 9px; }
.result-title { color: #f6f8f3; font-size: 30px; font-weight: 760; letter-spacing: -.03em; margin-top: 16px; }
.result-confidence { color: #d7f0c4; font-size: 19px; font-weight: 720; margin-top: 5px; }
.result-confidence span { color: #a9b9ac; font-size: 12px; font-weight: 500; }
.result-card p { color: #c1cec3; font-size: 13px; line-height: 1.65; margin: 13px 0 0; }
#predictions { margin-top: 12px; }
#safety-note {
  margin: 24px 0 36px; padding: 16px 18px; border-radius: 14px;
  color: #bfcbc1; font-size: 13px; line-height: 1.65;
  background: rgba(8,16,12,.56); border: 1px solid rgba(187,215,190,.12);
}
#safety-note strong { color: #e2ebdf; }
footer { display: none !important; }
@media (max-width: 700px) {
  #hero { padding-top: 24px; }
  .panel { padding: 14px !important; }
  .result-title { font-size: 25px; }
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="lime",
        secondary_hue="green",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
    title="Chilli Leaf Disease Detector",
) as demo:
    gr.HTML(
        """
        <header id="hero">
          <div class="eyebrow">CHILLI CROP · IMAGE SCREENING</div>
          <h1>Look closer at your chilli leaves.</h1>
          <p>Upload a clear leaf photo to see which of eight classes best matches the image.
          Get a ranked result and sensible next steps in seconds.</p>
        </header>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=11, elem_classes="panel"):
            gr.HTML('<div class="panel-heading">01 &nbsp; Add a leaf photo</div>')
            gr.HTML(
                '<p class="panel-copy">Use a close, focused photo in natural light. '
                'Keep one leaf visible and avoid strong shadows.</p>'
            )
            image_input = gr.Image(
                label="Leaf image",
                type="pil",
                sources=["upload", "webcam"],
                height=350,
                elem_id="leaf-input",
            )
            with gr.Row():
                analyze = gr.Button("Analyze leaf", variant="primary", elem_id="analyze")
                clear = gr.Button("Clear", variant="secondary", elem_id="clear")

        with gr.Column(scale=10, elem_classes="panel"):
            gr.HTML('<div class="panel-heading">02 &nbsp; Understand the result</div>')
            gr.HTML(
                '<p class="panel-copy">The top match and its model confidence appear first. '
                'Compare it with the other likely classes below.</p>'
            )
            result = gr.HTML(
                """
                <section class="result-card result-empty">
                  <div class="result-kicker">YOUR SCREENING RESULT</div>
                  <div class="result-title">Ready when you are</div>
                  <p>Your result will appear here after you analyze a photo.</p>
                </section>
                """,
                elem_id="result-card",
            )
            predictions = gr.Label(
                num_top_classes=3,
                label="Other likely matches",
                elem_id="predictions",
            )

    with gr.Accordion("What this screening can—and cannot—tell you", open=False):
        gr.Markdown(
            """
            This demo compares a photo with eight learned classes: Whitefly, Yellowing,
            Anthracnose, Damping off, Leaf curl virus, Leaf spot, Veinal mottle virus,
            and Healthy. A high model confidence is **not** the same as verified field
            accuracy. Lighting, camera quality, leaf age, and symptoms outside the frame
            can change the result.

            Use the output as an initial screening prompt. Inspect the plant and consult
            a local agricultural expert before choosing a treatment. This tool does not
            diagnose a crop or recommend pesticides.
            """
        )

    gr.HTML(
        """
        <div id="safety-note">
          <strong>Experimental screening only.</strong> The result is an image-model
          prediction, not a confirmed diagnosis or pesticide recommendation. Model
          confidence is not a measure of clinical or field accuracy.
        </div>
        """
    )

    analyze.click(
        fn=predict_disease,
        inputs=image_input,
        outputs=[predictions, result],
    )
    clear.click(
        fn=lambda: (
            None,
            {},
            """<section class="result-card result-empty">
              <div class="result-kicker">YOUR SCREENING RESULT</div>
              <div class="result-title">Ready when you are</div>
              <p>Your result will appear here after you analyze a photo.</p>
            </section>""",
        ),
        outputs=[image_input, predictions, result],
    )


if __name__ == "__main__":
    demo.launch()
