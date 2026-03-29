🚦 Traffic Sign Recognition System (Deep Learning)
Overview
Built an end-to-end computer vision system for traffic sign classification using a custom CNN in PyTorch, trained on the GTSRB dataset (43 classes).
The project covers the full ML lifecycle — from data preprocessing and model training to evaluation and real-time deployment via a web app.
Key Highlights
Developed a custom CNN architecture achieving high accuracy on 43-class classification
Solved class imbalance using weighted loss functions
Applied CLAHE-based preprocessing for better performance under varying lighting
Built full evaluation pipeline (confusion matrix + classification report)
Enabled real-time inference via webcam
Deployed an interactive Streamlit web app for user-friendly predictions
Model Details
Architecture: 3-layer CNN + Fully Connected layers
Input: 32×32 RGB images
Techniques used:
Batch Normalization
Dropout Regularization
Data Augmentation
Learning Rate Scheduling
Early Stopping
Tech Stack
Python, PyTorch, OpenCV
NumPy, Pandas, Scikit-learn
Matplotlib, Seaborn
Streamlit (Deployment UI)
Pipeline
Data preprocessing (CLAHE, resizing, normalization)
Data augmentation (rotation, affine transforms)
CNN model training with class weights
Validation + LR scheduling
Evaluation (accuracy, confusion matrix)
Deployment (web app + real-time inference)
