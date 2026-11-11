# Integral University B.Tech Final Year Project Report Draft

## Preliminary Pages

### 1. Front Page

**Title:** Pneumonia Detection from Chest X-ray Images using Convolutional Neural Network  
**Student Name:** [Insert Student Name]  
**Enrollment Number:** [Insert Enrollment Number]  
**Course:** B.Tech (Computer Science / IT / ECE / Related Discipline)  
**Semester:** VII  
**Academic Session:** 2026-2027  
**Department:** [Insert Department Name]  
**Institution:** Integral University, Lucknow  
**Supervisor:** [Insert Supervisor Name]  
**Date:** [Insert Date]

### 2. Certificate

This is to certify that the project titled **PneumoCatcher: Pneumonia Detection from Chest X-ray Images using Convolutional Neural Network** submitted by [Student Name] is a record of the work carried out by the student under my supervision and is approved for partial fulfillment of the requirements for the degree of Bachelor of Technology.

**Supervisor Name:** [Insert Name]  
**Designation:** [Insert Designation]  
**Department:** [Insert Department]  
**Signature:** [Insert Signature]  
**Date:** [Insert Date]

### 3. Declaration

I hereby declare that the project work entitled **PneumoCatcher: Pneumonia Detection from Chest X-ray Images using Convolutional Neural Network** submitted to Integral University, Lucknow, is my original work and has not been submitted elsewhere for award of any degree or diploma.

**Student Name:** [Insert Name]  
**Enrollment Number:** [Insert Enrollment Number]  
**Signature:** [Insert Signature]  
**Date:** [Insert Date]

### 4. Acknowledgement

I would like to express my sincere gratitude to [Supervisor Name], Department of [Department], Integral University, Lucknow, for guidance and support throughout the project.

I also acknowledge the support of the faculty, lab staff, and peers who contributed to the successful completion of this project.

### 5. Preface

This report presents the design, development, and evaluation of a chest X-ray pneumonia detection system using a CNN. The project was developed as a local software prototype to demonstrate the application of deep learning in medical image classification.

### 6. Abstract

Pneumonia is a major respiratory infection that can be life-threatening without timely treatment. Early detection is essential, especially in regions with limited radiology support. This project presents a CNN-based system for pneumonia detection from chest X-ray images. The system preprocesses uploaded images, loads a saved CNN model, and displays a pneumonia probability score. The application is implemented using Python, TensorFlow, and Streamlit and is designed as a local demonstration prototype. The trained model showed strong training-set accuracy, but it must be validated on independent data before clinical use.

### 7. Table of Contents

- Preliminary Pages
- Chapter 1: Introduction
- Chapter 2: Feasibility Study
- Chapter 3: Project Requirements / Proposed Methodology
- Chapter 4: Project Development
- Appendices (if any)

### 8. List of Figures

- Figure 1: System architecture diagram
- Figure 2: Use case diagram
- Figure 3: Activity diagram
- Figure 4: Sequence diagram
- Figure 5: DFD diagram
- Figure 6: Streamlit UI screenshot placeholder

### 9. List of Tables

- Table 1: Dataset summary
- Table 2: Software requirements
- Table 3: Hardware requirements
- Table 4: Training metrics summary

### 10. List of Abbreviations

- CNN: Convolutional Neural Network
- UI: User Interface
- HDF5: Hierarchical Data Format version 5
- X-ray: Radiographic image
- AI: Artificial Intelligence
- ML: Machine Learning

---

# Chapter 1: Introduction

## 1.1 Project Introduction

This project aims to design and implement a local chest X-ray pneumonia detection system using a CNN. It demonstrates how a machine learning model can classify medical images to assist in initial screening workflows.

## 1.2 Background

Chest X-rays are one of the most common diagnostic tools for respiratory conditions. Pneumonia is a serious disease requiring prompt diagnosis and treatment. The project uses medical imaging and deep learning to automate and support preliminary screening.

## 1.3 Problem Statement

Pneumonia detection from chest X-rays often requires expert interpretation and may be delayed due to limited access to radiologists. A lightweight AI-assisted tool can support early detection and prioritization, but any such tool must be carefully validated before being used in real medical settings.

## 1.4 Objectives

1. To build a chest X-ray classification system using CNN.
2. To preprocess medical images into a consistent model input format.
3. To provide a simple user interface for image upload and prediction.
4. To document the current implementation and identify future enhancements.

## 1.5 Existing System

The existing system is a single-page demo app built in Python using Streamlit. It allows the user to upload a chest X-ray image and obtain a pneumonia prediction using the saved model. It does not implement multi-user access, persistent storage, or deployment security.

## 1.6 Proposed System

The proposed system is a prototype that can accept chest X-ray inputs, preprocess them, run inference on the saved CNN model, and display the probability score along with a warning that the output is not a diagnosis.

## 1.7 Literature Survey

Deep learning, especially CNN-based architectures, has been widely used on image classification tasks, including medical image analysis. While model accuracy on training data is encouraging, strong generalization requires independent validation. The current project follows the same principle and demonstrates a feasible local prototype.

## 1.8 Research Gap

The project demonstrates prototype feasibility, but there is no formal medical validation dataset, no external benchmark, and no production deployment. This gap should be addressed for future clinical-grade use.

## 1.9 Feasibility Study

The project is technically feasible with Python, TensorFlow, and Streamlit. It is economically viable as a low-cost academic system. Operationally, it is useful for demonstration and educational purposes. Time-wise, it fits within a final-year project schedule.

## 1.10 Project Scope

The scope of this project includes:
- data understanding and preprocessing,
- building and testing a CNN,
- implementing local image inference,
- documenting the system and limitations,
- preparing the report draft for academic submission.

---

# Chapter 2: Feasibility Study

## 2.1 Technical Feasibility

The project uses open-source Python libraries and local hardware. The current implementation runs successfully in a local environment and loads the Keras model without errors. The required model input is clearly defined and processed before inference.

## 2.2 Economic Feasibility

The system has low development cost because it relies on open-source tools. Observed expenses are minimal for a prototype, but scaling to an actual hospital or production environment would demand additional validation, security controls, and infrastructure.

## 2.3 Operational Feasibility

The app is easy to install and run on a local machine. It is suitable for academic demonstration, but operational deployment to medical practice requires proper privacy review and clinical supervision.

## 2.4 Time Feasibility

This project is feasible within the semester schedule because the app is already functioning and the required reports and documentation can be prepared alongside implementation and testing work.

---

# Chapter 3: Project Requirements / Proposed Methodology

## 3.1 Hardware Requirements

- Minimum 8 GB RAM recommended
- Modern x64 processor
- Sufficient storage for model and dataset files
- GPU optional; CPU execution achieved successfully in the current environment

## 3.2 Software Requirements

- Python 3.11
- TensorFlow 2.16.2
- Keras 3.15.1
- Streamlit 1.39.1
- NumPy 1.26.4
- Pillow 10.4.0
- h5py 3.16.0

## 3.3 Functional Requirements

1. User can upload chest X-ray image.
2. Image validation must reject empty or unreadable files.
3. System preprocesses images to match model input shape.
4. System loads the saved CNN model and predicts pneumonia probability.
5. System displays a result and medical disclaimer.

## 3.4 Non-Functional Requirements

- Ease of use
- Low response time for local use
- Clear error handling
- Minimal dependencies for prototype deployment
- Use of local computation without external database

## 3.5 System Architecture

The system architecture includes a client-side Streamlit interface, preprocessing module, TensorFlow model, and result display. The app communicates in-process rather than via a separate backend API.

## 3.6 Proposed Methodology

The methodology follows a machine-learning prototype lifecycle:
1. Acquire chest X-ray dataset.
2. Validate dataset structure.
3. Preprocess images.
4. Train or continue training the CNN.
5. Save checkpoint and evaluate performance.
6. Build the user-facing app.
7. Document the project and limitations.

## 3.7 Database Requirements

The current prototype does not require a database. The training dataset is stored as a local HDF5 file, and the model is saved as a Keras file.

---

# Chapter 4: Project Development

## 4.1 Software Development Model

This project uses a lightweight iterative development model. Features were implemented in stages: user interface, preprocessing, model inference, and validation. This approach is appropriate for a short academic prototype.

## 4.2 UI/UX Design

The interface is intentionally simple and presentation-focused. It uses a clean two-panel Streamlit layout for upload and inference output. The UI uses a medical disclaimer to warn that the output is a model estimate, not a diagnosis.

## 4.3 Frontend Implementation

The frontend is implemented using Streamlit. It includes:
- sidebar with label explanation,
- file uploader,
- image preview,
- prediction result card,
- confidence progress bar,
- non-diagnostic result note.

## 4.4 Backend Implementation

The backend logic is encapsulated in Python modules. It validates upload input, preprocesses images, loads the model, and performs prediction. The model runs locally without a separate web service.

## 4.5 Database Implementation

No SQL or NoSQL database is implemented. The project uses local HDF5 files for data storage and Keras serialization for model storage. This is suitable for a local academic prototype but not for multi-user deployment.

## 4.6 Module Description

The project has the following modules:
- `app.py` for UI and inference orchestration
- `image_preprocessing.py` for validation and normalization
- `train_cnn.py` for training workflow
- `evaluate_model.py` for evaluation workflow
- `requirements.txt` for dependency listing

## 4.7 Coding

The coding phase includes image validation, model loading, safe preprocessing, and result rendering. The core logic is straightforward and aligned with the project objective.

## 4.8 Testing

The current project includes focused automated tests for preprocessing and evaluation functions. The application was also smoke-tested locally. These checks confirm that the app loads successfully and key helper functions work as expected.

### Known limitations

- Full clinical validation is not available.
- All dataset images were used for training; no independent test set was used.
- The system is not suitable for direct clinical use without additional validation.

---

## Appendix A: Placeholder Notes

- Replace bracketed placeholders with actual student, supervisor, and institutional details before final submission.
- Add actual screenshots for the final report after the app is run locally.
- Replace this draft with the final version approved by the department.
