# brain-tumor-mri-classification-tensorflow-cnn-CV-
The computer vision project aims to address the challenges associated with medical image classification, specifically in the domain of radiology. The focus is on leveraging state-of-the-art convolutional neural networks (CNNs) and transfer learning to enhance the accuracy and efficiency of image classification tasks.

1. Importing Libraries
The code begins by importing various Python libraries, including:

matplotlib.pyplot for creating visualizations.
NumPy for numerical operations.
pandas for data manipulation.
seaborn for statistical data visualization.
cv2 for computer vision tasks.
TensorFlow for building and training neural networks.
ImageDataGenerator from tensorflow.keras.preprocessing.image for data augmentation.
tqdm for displaying progress bars.
os for interacting with the operating system.
shuffle and train_test_split from sklearn.utils for shuffling and splitting data.
EfficientNetB0 from tensorflow.keras.applications for using a pre-trained EfficientNet model.
Callbacks such as EarlyStopping, ReduceLROnPlateau, TensorBoard, and ModelCheckpoint for training optimization.
classification_report and confusion_matrix from sklearn.metrics for model evaluation.
ipywidgets, io, Image, and display from IPython for creating interactive widgets.
warnings for managing warning messages.
2. Data Loading and Preprocessing
The code then proceeds to load and preprocess medical image data. It reads images from specified folders, resizes them, and stores them in numpy arrays (X_train and y_train). The labels include 'brain,' 'chest,' and 'lung'.

3. Data Visualization
A sample image from each label is visualized using matplotlib. The data is shuffled, and its shape is printed.

4. Data Distribution Analysis
The distribution of training and test images for each disease category is analyzed and visualized using bar charts and pie charts.

5. Label Encoding and One-Hot Encoding
The labels are encoded numerically, and one-hot encoding is applied to the target variables.

6. Model Construction
An EfficientNetB0 model is loaded with weights pre-trained on ImageNet. Additional layers are added to the model for classification.

7. Model Compilation and Training
The model is compiled with categorical cross-entropy loss and Adam optimizer. It is then trained on the preprocessed data using callbacks for logging, model checkpointing, and early stopping.

8. Training History Visualization
The training history (accuracy and loss) is visualized using matplotlib.

9. Model Evaluation
The trained model is evaluated on the test set, and a classification report and confusion matrix are displayed using seaborn.

10. Image Prediction Widget
A simple widget-based interface is provided for uploading an image and obtaining predictions from the trained model. It uses the FileUpload and Button widgets from ipywidgets.
