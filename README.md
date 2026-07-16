A deep learning tool that classifies lung CT scans to assist in early cancer detection, built around a CNN model and exposed through a REST API for easy integration into other systems.

## Overview
### 🫁🩻👩🏻‍💻

This project uses a Convolutional Neural Network trained on labeled lung scan images to classify scans for signs of cancer. The focus was on getting a model that's both accurate and efficient enough to run in a lightweight, deployable service rather than requiring heavy infrastructure.

## Features

- Trained model achieves 90% accuracy on cancer classification through iterative training and hyperparameter tuning.
- Reduced computational overhead by 40% through model optimization and efficient image preprocessing (OpenCV-based pipeline)
- Modular API layer for submitting scans and retrieving predictions, built for scalable deployment


### Frontend
A Streamlit interface is used for user interaction, but the main API can be used along with other frameworks as well.



```
Tech Stack

Language: Python
Deep Learning: CNN
Image Processing: OpenCV
Pretrained Weights: ImageNet
Deployment: Heroku
```

### How It Works

- Input lung scan images are preprocessed (resizing, normalization, noise reduction) using OpenCV.
- The preprocessed image is passed through the trained CNN model.
- The model outputs a classification (cancerous / non-cancerous) with a confidence score
Results are returned via the REST API in JSON format.
