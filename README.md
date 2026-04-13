# Facial Landmark Detection with EfficientNetB0

⚪ Q1) What is this project about?

This project is a deep learning system for **facial landmark detection**.  
It predicts the positions of 5 important facial keypoints from a face image:

- Left eye
- Right eye
- Nose
- Left mouth corner
- Right mouth corner

The model takes a face image as input and returns 10 values, which represent the normalized (x, y) coordinates of these 5 landmarks.

---

⚪ Q2) What is the main goal of the project?
The goal of this project is to train a model that can understand facial structure and accurately locate key facial points automatically.

This can be useful in many computer vision tasks such as:

- Face alignment
- Face tracking
- Face preprocessing
- Emotion analysis preprocessing
- Facial analysis applications

---

⚪ Q3) Which dataset is used in this project?
This project uses the **CelebA dataset**, specifically:

- Aligned face images
- Landmark annotation file

The CSV file contains the coordinates of the 5 facial landmarks for each image.

---

⚪ Q4) Which landmarks does the model predict?
The model predicts the coordinates of the following 5 facial landmarks:

1. Left eye
2. Right eye
3. Nose
4. Left mouth corner
5. Right mouth corner

Since each landmark has an `x` and `y` coordinate, the final output contains:

- 5 landmarks × 2 coordinates = 10 output values

---

⚪ Q5) Which deep learning model is used?
This project uses **EfficientNetB0** as the backbone model.

EfficientNetB0 is used as a pretrained feature extractor with **ImageNet weights**, and then a custom regression head is added on top of it to predict facial landmark coordinates.

---

⚪  Q6) Why was EfficientNetB0 chosen?
EfficientNetB0 was chosen because it offers a strong balance between:

- Accuracy
- Speed
- Model size
- Transfer learning performance

It is lightweight compared to larger models, but still powerful enough to extract useful facial features.

---

⚪ Q7) Is this a classification project?
No.  
This is **not** a classification project.

It is a **regression project** because the model predicts continuous numeric values representing landmark coordinates.

---

⚪ Q8) Why does the model output 10 values?
Because the project predicts 5 landmark points, and each point has:

- 1 x-coordinate
- 1 y-coordinate

So the total output is:

- `5 × 2 = 10`

---

⚪ Q9) Why is the final activation function sigmoid?
The final layer uses **sigmoid** because the landmark coordinates are normalized to values between **0 and 1**.

So sigmoid is suitable because it keeps the model predictions within the expected normalized range.

---

⚪ Q10) How are the landmarks prepared before training?
Before training, the landmark coordinates are **normalized** using the original image dimensions.

For example:

- x coordinates are divided by image width
- y coordinates are divided by image height

This makes training more stable because the model learns values in a common range instead of raw pixel values.

---

⚪ Q11) What image size is used?
All images are resized to:

`224 × 224`

This matches the input size expected by the model.

---

⚪ Q12) Are the original image dimensions important?
Yes.

The original image dimensions are used to normalize the landmark coordinates before training.  
In this project, the original aligned CelebA image size is:

- Width = 178
- Height = 218

---

⚪ Q13) How is the dataset split?
The dataset is divided into:

- Training set = 80%
- Validation set = 10%
- Test set = 10%

This helps the project to:

- train the model
- validate performance during training
- test final performance on unseen data

---

⚪ Q14) What preprocessing is applied to the images?
The preprocessing steps include:

- Reading the image from disk
- Decoding JPEG images
- Resizing to `224 × 224`
- Converting image data to `float32`

The image values are **not divided by 255** in this version of the project.

---

⚪ Q15) Is data augmentation used?
Yes, lightweight data augmentation is used during training:

- Random brightness
- Random contrast

This helps the model generalize better and reduces overfitting.

---

⚪ Q16) Why is `tf.data.Dataset` used?
`tf.data.Dataset` is used because it provides an efficient and scalable input pipeline.

It helps with:

- loading data efficiently
- batching
- shuffling
- parallel preprocessing
- prefetching for faster training

---

⚪ Q17) What does `from_tensor_slices` do here?
It creates a TensorFlow dataset from:

- image paths
- corresponding landmark labels

This ensures that each image stays matched with its correct landmarks.

---

⚪ Q18) Why is shuffle used again in the dataset pipeline?
Even though the dataset was already split randomly, `shuffle()` is still useful during training because it changes the order of samples inside training batches.

This helps the model learn better across epochs and reduces bias from fixed ordering.

---

⚪ Q19) What is the model architecture after EfficientNetB0?
After the pretrained backbone, the model uses:

- GlobalAveragePooling2D
- BatchNormalization
- Dense(256, ReLU)
- Dropout(0.3)
- Dense(128, ReLU)
- Dense(10, sigmoid)

This final part is the regression head responsible for predicting the landmark coordinates.

---

⚪ Q20) Why is transfer learning used?
Transfer learning is used because EfficientNetB0 already learned useful visual features from ImageNet.

Instead of training everything from scratch, the project reuses these learned features and adapts them to the facial landmark task.

This usually leads to:

- faster training
- better accuracy
- less data requirement

---

⚪ Q21) What is the training strategy?
The project uses **two-stage training**.

### Stage 1
The EfficientNet backbone is frozen, and only the custom regression head is trained.

### Stage 2
Part of the EfficientNet backbone is unfrozen, and the model is fine-tuned with a smaller learning rate.

This approach improves stability and performance.

---

⚪ Q22) Why are two training stages used?
Two stages are used because training the full pretrained model immediately can damage the useful pretrained weights.

So first, the new head learns the task.  
Then, fine-tuning helps the deeper layers adapt gradually.

---

⚪ Q23) Which loss function is used?
The project uses **Huber loss**.

Huber loss is often a good choice for regression tasks because it is less sensitive to outliers than Mean Squared Error, while still giving stable learning.

---

⚪ Q24) Which metric is used during training?
The main metric used is:

- **MAE (Mean Absolute Error)**

MAE measures the average absolute difference between predicted landmark values and true landmark values.

---

⚪ Q25) What is Pixel MAE?
Pixel MAE is the average absolute error measured in **pixel units** instead of normalized values.

This makes the result easier to understand in practical terms.

For example, if Pixel MAE is low, it means the predicted landmark locations are close to the true points on the image.

---

⚪ Q26) Why is Pixel MAE useful?
Normalized MAE is useful for training, but Pixel MAE is easier to interpret.

For example:

- normalized MAE tells you the error in relative scale
- pixel MAE tells you approximately how many pixels the prediction is off

This gives a more realistic understanding of model performance.

---

⚪ Q27) How are predictions visualized?
After training, the model predicts landmarks on test images.

Then both are shown on the image:

- True landmarks
- Predicted landmarks

This visual comparison helps inspect model quality directly.

---

⚪ Q28) What callbacks are used?
The project uses the following callbacks:

- **EarlyStopping**
- **ReduceLROnPlateau**
- **ModelCheckpoint**

These help improve training efficiency and save the best model weights.

---

⚪ Q29) Why is EarlyStopping useful?
EarlyStopping stops training when validation performance stops improving.

This helps prevent overfitting and saves time.

---

⚪ Q30) Why is ReduceLROnPlateau useful?
When validation loss stops improving, ReduceLROnPlateau lowers the learning rate.

This helps the model make smaller and more precise updates during training.

---

⚪ Q31) What file is saved at the end?
The final trained model is saved as:

`celeba_landmarks_simple.keras`

This allows the model to be loaded later for testing, inference, or further improvement.

---

⚪ Q32) What are the main libraries used in this project?
The project uses:

- NumPy
- Pandas
- TensorFlow / Keras
- Matplotlib
- Scikit-learn
- Pathlib

---

⚪ Q33) What are the strengths of this project?
Some strong points of this project are:

- Clean and simple pipeline
- Transfer learning with EfficientNetB0
- Landmark normalization
- Two-stage training
- Pixel-level evaluation
- Visual prediction inspection
- Good structure for further improvement

---

⚪ Q34) What can be improved in the future?
Possible future improvements include:

- using more advanced augmentation
- trying EfficientNetV2 or MobileNet
- adding learning rate scheduling
- training for more epochs
- predicting more landmark points
- using heatmap-based landmark detection instead of direct regression
- adding test visualizations for more samples
- computing per-landmark error separately

---

⚪ Q35) What are the possible applications of this project?
This project can be extended to support:

- face alignment systems
- facial recognition preprocessing
- head pose estimation preprocessing
- face tracking pipelines
- research experiments in keypoint regression

Project link in Kaggle : https://www.kaggle.com/code/bassammoustafa/face-landmark-detection-regression
