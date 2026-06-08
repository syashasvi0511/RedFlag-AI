🚩 RedFlag AI (Python)
📌 What This Project Does
This project is an AI-powered Anaemia Detection System built using Python. I created it with AI assistance to detect anaemia from blood smear microscopy images by analysing the shape, size, and colour of red blood cells (RBCs) under a microscope. 

Datasets were provided by the AneRBC-II benchmark and downloaded from kaggle.com:
     Anemic individuals : RGB-segmented RBC microscopy images
     Healthy individuals : RGB-segmented RBC microscopy images
     Split used : 70% Train / 15% Validation / 15% Test

We used 1 deep learning algorithm with a 2 stage fine tuning approach:
     MobileNetV2: 
          This pre-trained deep learning model has 2 layers (the Base which contains hundreds of layers that learned general visual features like edges, textures, shapes from ImageNet and the Head which is the final classification layer for            the actual prediction). This model is a lightweight CNN (convolutional neural network) model which is trained on 1.4+ million images and understands general visual patterns. We fine-tuned it to specifically differentiate between              anemic and healthy images. We performed this in 2 stages:
                1. Freezing the base and only training the new classification head.
                2. Unfreezing the top 30 layers as these layers are responsible for high level semantic features and correspond to the last 4-5 bottlenecks in MobileNetV2.
