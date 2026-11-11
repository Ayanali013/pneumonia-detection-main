# Project Audit

**Audit date:** 11 November 2026  
**Repository:** `Ayanali013/pneumonia-detection-main`  
**Audit scope:** Current local workspace and checked-out application. This is a static project audit plus a local HTTP smoke check; it is not a clinical validation or a full security assessment.

## 1. Project Overview

This repository contains **PneumoCatcher**, a small Streamlit application that accepts a chest X-ray image and displays an estimate from a saved CNN model. The project includes a local HDF5 image/label dataset and serialized Keras model artifacts.

The current application is a research/demo prototype, not a clinical product. It includes a disclaimer that its output does not replace review by a qualified clinician.

## 2. Technology Stack

| Area | Observed technology |
|---|---|
| User interface | Streamlit, rendered from Python in `app.py` |
| Application logic / inference | Python, TensorFlow/Keras |
| Image handling | Pillow |
| Numerical preprocessing | NumPy |
| Dataset format | HDF5 (`.h5`) |
| Database | None found |
| Authentication / authorization | None found |
| Application APIs | No project-owned REST or GraphQL API found; the Streamlit application handles page/session communication |
| Styling | Inline CSS in Streamlit Markdown; Google Fonts are imported from an external URL |
| Model artifact format | Keras `.keras`; Git LFS attributes are configured for `.keras` and `.h5` |

### Runtime versions observed

- Python: 3.11.9
- Streamlit: 1.39.1
- TensorFlow: 2.16.2
- NumPy: 1.26.4
- Pillow: 10.4.0
- h5py: 3.16.0
- The current CNN archive records Keras 3.15.1 in its saved metadata.

There is no tracked dependency manifest (`requirements.txt`, `pyproject.toml`, or equivalent). The versions above were read from the local `.venv311` environment and should not be treated as a reproducible installation specification.

## 3. Folder and File Structure

The following is the application-relevant structure. Git metadata and the contents of the two installed virtual environments are omitted.

```text
.
├── .gitattributes
├── .gitignore
├── app.py
├── pneumonia_detection_model.keras
├── pneumonia_detection_model.kerasga4bki47.part   # ignored partial-looking artifact
├── streamlit.stderr.log
├── streamlit.stdout.log
├── x_ray_full_dataset.h5
├── .venv/                                           # local virtual environment
├── .venv311/                                        # local virtual environment used for execution
└── report and presentation/
    ├── pneumonia_detection_model.keras
    └── x_ray_full_dataset.h5
```

At the audit snapshot, the two ResNet model files had been removed from the working tree following the user's earlier request; Git reports those tracked files as deleted both at the repository root and under `report and presentation/`.

No README, source training script, automated test suite, dependency manifest, documentation folder, or university guideline file was found in the application workspace.

## 4. Existing Features and Workflow

### Implemented

1. Presents a two-column Streamlit page for chest X-ray upload and result display.
2. Accepts `.jpg`, `.jpeg`, and `.png` uploads and displays an image preview.
3. Converts an uploaded image to grayscale, resizes it to 180 × 180 pixels, scales pixel values to `[0, 1]`, and adds the model channel dimension.
4. Loads the saved **Custom CNN** model on demand and caches the model resource for subsequent app interactions.
5. Runs model inference and interprets the first output value as a pneumonia probability, with a threshold of `0.5`.
6. Displays a result label, probability, progress bar, and a medical-use disclaimer.
7. Shows the label mapping in the sidebar: `0` Normal, `1` Pneumonia.

### Observed workflow

```text
Browser upload
  -> Streamlit receives the image
  -> Pillow converts/resizes it
  -> NumPy scales and shapes the input
  -> TensorFlow/Keras loads the saved CNN and predicts
  -> Streamlit renders the estimate and disclaimer
```

The HDF5 dataset is not accessed by the inference page. It is a local training/data artifact.

## 5. Dataset and Model Findings

### Dataset

The root `x_ray_full_dataset.h5` contains:

- `images`: shape `(5856, 180, 180)`, `uint8`
- `labels`: shape `(5856,)`, `uint8`
- Label counts observed: `0` = 1,583 and `1` = 4,273

The dataset is class-imbalanced. No dataset provenance, license, subject/patient grouping, split definition, or preprocessing/training documentation was found in the repository. The audit did not infer these details.

### Current CNN checkpoint

- The root checkpoint loads in the project's current Python environment.
- Input shape: `(None, 180, 180, 1)`; output shape: `(None, 1)`.
- Architecture: Keras `Sequential` model with three `Conv2D`/`MaxPooling2D` stages, followed by `Flatten`, `Dense`, `Dropout`, and a final `Dense` layer.
- Trainable parameters: 6,646,529.
- The saved optimizer state records iteration 4,470.
- The checkpoint was continued from iteration 2,640 for 10 epochs on all 5,856 images, batch size 32. The recorded final training accuracy was 97.49% and training loss was 0.0674.

**Metric limitation:** These are training-set metrics, not validation or test results. Because the complete dataset was used for training, they do not establish performance on unseen images. No independent evaluation split or clinical validation result was found.

The second CNN and dataset under `report and presentation/` are duplicate-named artifacts. Their byte sizes differ from the root artifacts; equivalence or provenance was not verified.

## 6. Database and API Architecture

### Database

No database, database driver, schema, migration, or persistence layer was found. The HDF5 file stores model-training arrays; it is not an application database. The app does not persist uploaded images, predictions, users, or review history.

### APIs

No project-owned REST, GraphQL, or other backend API was found. The UI, preprocessing, and model inference are implemented in the same Streamlit Python application. Streamlit supplies the application transport. Google Fonts are requested by the page's CSS import; no other external service integration was identified in the inspected source.

## 7. Bugs, Risks, and Gaps

### Functional and reliability gaps

- Invalid, corrupted, or unusually large images are not handled with a user-facing validation/error path around image decoding and preprocessing.
- The app assumes the model returns a single probability-like value and uses a fixed `0.5` threshold. Threshold selection and probability calibration are not documented or validated.
- No in-app upload size or image dimension limit is configured.
- The file upload, inference, and error paths have no automated tests in the repository.
- There is no source training script or recorded training configuration/results history to reproduce the current checkpoint's training process.
- The `.part` file is ignored by Git and was not validated; it is not referenced by the application.
- Large model and dataset artifacts are duplicated under `report and presentation/`.

### Security and privacy considerations

- There is no authentication or authorization. This may be acceptable for a local demonstration but is not appropriate for a remotely exposed, multi-user deployment without additional controls.
- Uploaded images are processed by the current application code and are not intentionally saved to an application database. Deployment, logging, temporary storage, and hosting behavior still need review before handling real patient data.
- There is no explicit upload-size limit, rate limit, or resource-usage control in the application source.
- Inline `unsafe_allow_html=True` is used for page styling and fixed result markup. The current inserted result/model strings come from fixed application choices, but future user-controlled content should not be inserted without escaping.
- The external Google Fonts import requires a network request and may disclose a browser request to that provider.
- No environment/configuration file or API secret was needed for the inspected implementation; no secret contents were read.

### Production readiness

**Assessment: prototype / academic demonstration only.** The app starts and serves its page locally, but the repository lacks reproducible dependency setup, automated tests, robust input/error handling, independent model evaluation, and deployment hardening. The medical-use disclaimer is appropriate but does not replace model validation or clinical review.

## 8. GitHub Repository and Working Tree

- Sanitized `origin` remote: `https://github.com/Ayanali013/pneumonia-detection-main.git`
- Current branch: `main`
- Latest commit observed: `ed13278b` — `Update Done`
- Git LFS attributes apply to `.keras` and `.h5` files.
- Git status before adding this audit report:
  - Modified: `app.py`
  - Modified: `pneumonia_detection_model.keras`
  - Deleted: root `v1_resnet50.keras` and `v3_resnet50.keras`
  - Deleted: `report and presentation/v1_resnet50.keras` and `report and presentation/v3_resnet50.keras`
- `git lfs status` showed these changes as unstaged; no staged changes or pending LFS pushes were reported.

These are existing local changes from earlier work in this conversation, not changes made during this audit. The `.part` file is ignored and does not appear in tracked Git files. No credentials were included in the remote URL displayed above.

## 9. Verification Performed

- Loaded the current Keras checkpoint and read its input/output shapes and saved optimizer state.
- Inspected HDF5 dataset shapes, dtypes, and label counts.
- Confirmed the local app root returned HTTP 200 at `http://127.0.0.1:8501/`.
- No automated test or lint command was available in the repository. The HTTP check verifies page availability only; it does not exercise image upload or inference end to end.

## 10. Recommended Improvements (Proposed, Not Yet Implemented)

Prioritized for discussion before implementation:

1. **Reproducibility:** Add a dependency manifest and a small training/evaluation entry point that records dataset source, split, random seed, preprocessing, epochs, and model artifact version.
2. **Model evaluation:** Define an independent patient-safe validation/test split or obtain an approved external test set; report confusion matrix, precision, recall, sensitivity, specificity, and ROC-AUC. Do not present training accuracy as generalization performance.
3. **Input robustness:** Add upload size/dimension checks and clear handling for invalid image data and inference failures.
4. **Medical UX:** Keep the non-diagnostic disclaimer prominent; explain the output as a model estimate and document the threshold and limitations.
5. **Maintainability:** Separate image preprocessing and inference from Streamlit presentation only if this improves testability without rewriting the working architecture.
6. **Testing:** Add focused tests for image preprocessing, model input/output compatibility, upload validation, and app-level inference behavior.
7. **Artifact management:** Document the dataset/model provenance and licensing; decide whether the duplicate report-folder artifacts and ignored partial file should remain. No files were removed as part of this audit.
8. **Deployment security:** If the app will be hosted for other users, define authentication, access control, transport security, logging/retention, and privacy requirements before deployment.
9. **Academic documentation:** After approval, create documentation from verified implementation facts and use placeholders for missing student/supervisor details. The Integral University 2026–2027 guideline/template file was not present in the inspected workspace; it will be needed to preserve official formatting.

## 11. Proposed Next Phase

After your approval, proceed in small, reviewable steps: agree on model evaluation and privacy requirements; add reproducible setup and focused tests; address input/error handling; then prepare the requested technical and university documentation from the verified implementation. No Phase 2 code changes or Phase 3–5 deliverables have been started.
