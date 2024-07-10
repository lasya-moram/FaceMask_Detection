# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import math
from sklearn import metrics

# initialize learning rate, epochs, and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
#specify the path to the dataset directory and the categories (with and without masks)
DIRECTORY = r"C:\Users\Saanaa\Face-Mask-Detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]


print("[INFO] loading images...")

data = []
labels = []
# load the images and resize the image and  convert the image to a numpy array
## then preprocess the image using the MobileNetV2 preprocessing function
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# on the labels performing one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# convert the data and labels to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation by Rescaling, Rotating, Zooming, 
# #Shifting, Fliping and filling the missing pixels
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Extract features from the MobileNetV2 base model
##Use the pre-trained weights trained on the ImageNet dataset
###Exclude the top (classification) layers of the model
####Specify the input shape for the model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel) # Apply average pooling to reduce spatial dimensions
headModel = Flatten(name="flatten")(headModel) # Flatten the feature map into a 1D vector
headModel = Dense(128, activation="relu")(headModel) # Add a fully connected layer with 128 neurons and ReLU activation
headModel = Dropout(0.5)(headModel) # Apply dropout regularization to reduce overfitting
headModel = Dense(2, activation="softmax")(headModel) # Add a final fully connected layer with 2 neurons and softmax activation

# place the head FC model on top of the base model 
model = Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
	layer.trainable = False

# compile the model and define the Adam optimizer
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Train the head of the network using the augmented training data and the validation data
print("[INFO] training head...")
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
"""model.fit(
    trainX,
    steps_per_epoch=len(trainX),
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=val_steps,
    callbacks=[lr_scheduler]
)"""
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS) 

print(model.summary())
# predicting the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report and confusion matrix
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
Confusion_Matrix=(metrics.confusion_matrix(testY.argmax(axis=1), predIdxs))#,
#	target_names=lb.classes_))
print(Confusion_Matrix)
confusionmatrix_display=metrics.ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
confusionmatrix_display.plot()
plt.show()

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")
print("accuracy and loss plot and confusion matrix")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
plt.show()