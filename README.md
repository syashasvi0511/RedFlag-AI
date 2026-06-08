# 🚩 RedFlag AI (Python)


📌 What This Project Does

This project is an AI-powered Anaemia Detection System built using Python. I created it with AI assistance to detect anaemia from blood smear microscopy images by analysing the shape, size, and colour of red blood cells (RBCs) under a microscope. 

Datasets were provided by the AneRBC-II benchmark and downloaded from kaggle.com:

Anemic individuals : RGB-segmented RBC microscopy images
     
Healthy individuals : RGB-segmented RBC microscopy images
     
Split used : 70% Train / 15% Validation / 15% Test
     

We used 1 deep learning algorithm with a 2 stage fine tuning approach:

MobileNetV2: 

This pre-trained deep learning model has 2 layers (the Base which contains hundreds of layers that learned general visual features like edges, textures, shapes from ImageNet and the Head which is the final classification layer for         the actual prediction). This model is a lightweight CNN (convolutional neural network) model which is trained on 1.4+ million images and understands general visual patterns. We fine-tuned it to specifically differentiate between           anemic and healthy images. We performed this in 2 stages:

1. Freezing the base and only training the new classification head.
2. Unfreezing the top 30 layers as these layers are responsible for high level semantic features and correspond to the last 4-5 bottlenecks in MobileNetV2.

📊 Results

Performance assessment was done using Sensitivity, Specificity, and Accuracy, and we utilized a weighted loss function (2× penalty for false negatives by changing the weight) to make sure that the model never missed an anaemia case.

Sensitivity (to identify positive cases correctly) - 96.3%  (It was priortized as missing an anemia case can be really dangerous)

Specificity (to identify negative cases correctly) - 51.3%

Accuracy - 51.3%


🛠️ Tech Stack

Model : MobileNetV2 (PyTorch)

Explainability : Grad-CAM heatmap (shows which cells triggered the prediction)

Training : Google Colab T4 GPU (~45 minutes)

Demo UI : Streamlit

API : FastAPI (not called since streamlit loaded the model itself using @st.cache_resource — it didn't need a separate API server. FastAPI would be needed if a seperate frontend app like React calling a backend over HTTP.)


🚀 How to Run

1. Download the dataset from kaggle. Link- https://www.kaggle.com/datasets/jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images
2. Setup kaggle credentials in google colab.
3. Run code 1 (It scans the folders for images and splits images into train, val, test. Then completes data augmentation (applied random jitters, rotations, flips to artificially increase variety). It then loads in MobileNetV2 and completes the stage 1 and stage 2 training with 10 epochs each. It calculates the accuracy, sensitivity, specificity. It finallys saves the model weights in .pt format (PyTorch)).
4. Run code 2 (Inference + Grad-cam) (Grad-cam is a helper code and not for training, it generates Heatmap to identify the cells which helped predict the result)
5. Rude code 3 (%%writefile app.py - Wrote the Streamlit UI code to disk as a file. Didn't run the app yet.)
6. Run code 4 (Launch started the Streamlit server. We used ngrok to create a temporary URL to access it.) (Google colab runs on google servers and not our local machine. Streamlit started on a port inside that server which cannot be accessed by us. So we used ngrok to create a temporary URL to access that port.)



⚠️NOTE

RedFlag AI is a screening aid only. It does not replace the clinical blood tests or professional diagnosis.
