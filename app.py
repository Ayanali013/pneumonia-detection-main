from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from image_preprocessing import MAX_UPLOAD_BYTES, prepare_image

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Custom CNN": PROJECT_ROOT / "pneumonia_detection_model.keras",
}


st.set_page_config(
    page_title="PneumoCatcher | X-ray Review",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #102a43; --muted: #627d98; --teal: #0f766e; --mint: #e8f5f2; --line: #d9e2ec; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    .stApp { background: #f5f8f7; }
    .block-container { max-width: 1180px; padding-top: 3rem; }
    .eyebrow { color: var(--teal); font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero { background: linear-gradient(120deg, #e8f5f2 0%, #f5f8f7 62%, #fff 100%); border: 1px solid var(--line); padding: 2rem 2.2rem; margin-bottom: 1.5rem; }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1.02; margin: .35rem 0 .8rem; }
    .hero p { color: var(--muted); font-size: 1.05rem; max-width: 700px; margin: 0; }
    .panel { background: white; border: 1px solid var(--line); padding: 1.35rem; min-height: 100%; }
    .result { border-left: 5px solid var(--teal); background: var(--mint); padding: 1.15rem 1.3rem; margin-top: 1rem; }
    .result h2 { margin: 0 0 .25rem; }
    .result p { color: var(--muted); margin: .2rem 0; }
    .note { color: var(--muted); font-size: .86rem; line-height: 1.5; }
    [data-testid="stFileUploader"] { border: 1px dashed #8fb9b1; padding: .45rem; background: #fbfefd; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


with st.sidebar:
    st.markdown("## Review setup")
    selected_model = st.selectbox("Model", list(MODEL_PATHS), index=0)
    st.markdown("---")
    st.markdown("**Labels**")
    st.markdown("`0`  Normal\n\n`1`  Pneumonia")
    st.markdown("---")
    st.markdown(
        '<p class="note">For research and demonstration only. This tool does not replace review by a qualified clinician.</p>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="hero"><div class="eyebrow">PneumoCatcher / Client Review</div>'
    '<h1>Chest X-ray review, made clear.</h1>'
    '<p>Upload one X-ray to receive a fast model estimate and a simple, presentation-ready result.</p></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.markdown('<div class="panel"><h3>1. Upload image</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray",
        type=["jpg", "jpeg", "png"],
        help=f"JPEG or PNG, up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
    )
    st.caption(f"Maximum file size: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    uploaded_image_data = None
    model_input = None
    if uploaded_file:
        uploaded_image_data = uploaded_file.getvalue()
        try:
            preview, model_input = prepare_image(uploaded_image_data, channels=1)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.image(preview, caption=uploaded_file.name, use_column_width=True)
    else:
        st.info("Add a JPG or PNG chest X-ray to begin the review.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><h3>2. Model result</h3>', unsafe_allow_html=True)
    if uploaded_file and model_input is not None:
        if st.button("Run assessment", type="primary", use_container_width=True):
            model_path = MODEL_PATHS[selected_model]
            if not model_path.exists():
                st.error(f"Model file not found: {model_path.name}")
            else:
                with st.spinner("Reviewing image..."):
                    try:
                        model = load_model(str(model_path))
                        if model.input_shape[-1] not in (1, 3):
                            raise ValueError(f"Unsupported model input channels: {model.input_shape[-1]}")
                        _, model_input = prepare_image(uploaded_image_data, model.input_shape[-1])
                        prediction = model.predict(model_input, verbose=0)
                        if prediction.shape != (1, 1):
                            raise ValueError(f"Unexpected model output shape: {prediction.shape}")
                        probability = float(prediction[0, 0])
                        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                            raise ValueError("The model returned an invalid probability.")
                    except (OSError, ValueError, tf.errors.OpError) as exc:
                        st.error(f"Assessment failed: {exc}")
                    else:
                        diagnosis = "Pneumonia pattern detected" if probability >= 0.5 else "No pneumonia pattern detected"
                        st.markdown(
                            f'<div class="result"><h2>{diagnosis}</h2>'
                            f'<p>Estimated pneumonia probability: <strong>{probability:.1%}</strong></p>'
                            f'<p>Model: {selected_model}</p></div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(probability, text="Pneumonia probability")
                        st.caption("This estimate is not a diagnosis. Confirm findings with a qualified clinician.")
    elif uploaded_file:
        st.info("Upload a valid chest X-ray image to run the assessment.")
    else:
        st.info("The assessment result will appear here after an image is uploaded.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### Workflow")
st.caption("Upload an image, choose the model in the sidebar, and run the assessment. The image is resized and converted to grayscale to match the training pipeline.")