# Devnagari-Handwritten-Charactes-Classification
This handwritten Devanagari character classification project by Sheetal Munjewar utilizes TensorFlow and Keras to build a model capable of recognizing handwritten characters



# Devnagari Handwritten Character Recognition (CNN)  

## Project Overview  
This project focuses on recognizing handwritten Devnagari characters using a **Convolutional Neural Network (CNN)** built with **TensorFlow** and **Keras**. It follows a deep learning pipeline, including data preprocessing, augmentation, model training, and evaluation to achieve high accuracy in classification.  

## Author  
**Sheetal Munjewar**  
**Course**: DSC 680 - Applied Data Science  
**Institution**: Bellevue University  

## Dataset  
- **Source**: `DevanagariHandwrittenCharacterDataset`  
- **Train Path**: `Train/`  
- **Test Path**: `Test/`  
- **Image Dimensions**: 32x32 pixels (grayscale)  
- **Training Samples**: 78,200 (before augmentation)  
- **Test Samples**: 13,800  
- **Augmented Training Samples**: 156,400  

## Key Components  

### 1. **Preprocessing & Augmentation**  
- Images are loaded using **PIL** and converted into NumPy arrays.  
- Data augmentation is applied using **ImageDataGenerator**, with transformations such as:  
  - **Rotation** (10 degrees)  
  - **Width & Height Shift** (10%)  
  - **Shear Transformation**  
  - **Brightness Adjustment** (0.3 - 1.0)  
- Labels are converted into numerical values using **TensorFlow’s StringLookup**.  

### 2. **Model Architecture (CNN)**  
The CNN model consists of:  
- **Input Layer**: Rescaling (normalization of pixel values)  
- **Convolutional Layers** (16, 32, 64 filters) with **ReLU activation**  
- **Max-Pooling Layers** to reduce feature map size  
- **Flatten Layer** to convert features into a 1D vector  
- **Dense Layer** (256 neurons, ReLU)  
- **Dropout Layer** (0.5) to prevent overfitting  
- **Output Layer**: 47 classes for classification  

### 3. **Model Compilation & Training**  
- **Optimizer**: Adam  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Metric**: Accuracy  
- **Batch Size**: 32  
- **Epochs**: 30  
- **Early Stopping**: Applied with patience of 5 epochs  

### 4. **Results & Performance**  
- Achieved **98.2% accuracy** on the test dataset.  
- Training and validation accuracy/loss trends were plotted.  
- Predictions were visualized with confidence scores.  

## Dependencies  
- Python  
- TensorFlow & Keras  
- NumPy  
- Matplotlib  
- PIL (Pillow)  
- tqdm (progress bar for dataset loading)  

## How to Use  
1. Ensure dependencies are installed (`pip install -r requirements.txt`).  
2. Place the dataset in the `DevanagariHandwrittenCharacterDataset` directory.  
3. Run the preprocessing scripts to prepare the data.  
4. Train the CNN model using the provided training pipeline.  
5. Evaluate performance and visualize predictions.  

## Future Work  
- Experiment with deeper CNN architectures (e.g., ResNet, EfficientNet).  
- Deploy as a web-based application for real-time character recognition.  
- Improve data augmentation techniques for better generalization.  



# Devnagari Handwritten Character Classification (Scikit-Learn)  

## Project Overview  
This project focuses on classifying handwritten Devnagari characters using machine learning models implemented in Scikit-Learn. It involves data preprocessing, model training, hyperparameter tuning, and evaluation to achieve optimal classification performance.  

## Author  
Sheetal Munjewar  
Bellevue University  

## Dataset  
- **Source**: `DevanagariHandwrittenCharacterDataset`  
- **Train Path**: `Train/`  
- **Test Path**: `Test/`  
- **Data Format**: Images of size 32x32 pixels  
- **Total Training Samples**: 78,200  
- **Total Test Samples**: 13,800  

## Key Components  

### 1. **Preprocessing**  
- Image loading and conversion to NumPy arrays  
- Data normalization (scaling pixel values)  
- Label encoding for categorical classification  
- Data reshaping for compatibility with machine learning models  

### 2. **Feature Engineering**  
- Data transformation pipeline using `sklearn.pipeline`  
- Normalization using a custom `Normalizer` class  
- Encoding categorical labels using `LabelEncoder`  

### 3. **Model Training & Evaluation**  
- Models used:  
  - K-Nearest Neighbors (KNN)  
  - Decision Tree Classifier  
  - Random Forest Classifier  
  - LightGBM Classifier  
  - Support Vector Machine (SVM)  
- Cross-validation with `RepeatedStratifiedKFold`  
- Hyperparameter tuning using `GridSearchCV`  
- Performance metrics evaluation  

### 4. **Results & Best Models**  
- Model comparisons performed using bar plots for visualization  
- Random Forest showed strong performance with an F1-score of ~0.88  
- LightGBM and other models were optimized using different hyperparameters  

## Dependencies  
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Seaborn & Matplotlib (for visualization)  
- PIL & skimage (for image processing)  
- LightGBM, XGBoost (for advanced ML models)  

## How to Use  
1. Ensure all dependencies are installed (`pip install -r requirements.txt`).  
2. Place the dataset in the `DevanagariHandwrittenCharacterDataset` directory.  
3. Run the preprocessing scripts to prepare the data.  
4. Train models using the pipeline and evaluate performance.  
5. Visualize model comparisons using generated plots.  

## Future Work  
- Experiment with deep learning models (CNNs) for improved accuracy  
- Optimize hyperparameters further using Bayesian optimization  
- Deploy as a web app for real-time character recognition  

