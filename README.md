# 🏥 Histopathologic Cancer Detection

## 📝 Problem Description
The challenge is a binary image classification problem aimed at identifying metastatic cancer in small image patches taken from larger digital pathology scans. This competition, hosted on Kaggle, involves creating models to accurately detect the presence of tumor tissue in 32x32 pixel image patches. Early and precise detection of metastatic cancer is crucial for effective patient treatment and management.

## 📊 Data Description
The dataset for this competition is a modified version of the PatchCamelyon (PCam) benchmark dataset. It contains 96x96 pixel image patches extracted from larger whole-slide images, with each patch labeled as either containing metastatic tissue (positive class) or not (negative class). The dataset is divided into training and test sets, where the training set includes both the images and their labels. The structure and size of the dataset facilitate the training of convolutional neural networks (CNNs) for image classification tasks. The dataset can be accessed from [Kaggle](https://www.kaggle.com/competitions/histopathologic-cancer-detection).

## 🔗 Reference
Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., & Welling, M. (2018). Rotation Equivariant CNNs for Digital Pathology. arXiv:1806.03962.