# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large 
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

# Initialize the initial learning rate, number of epochs to train for and batch size

# Ideal values for the model till now
# INIT_LR = 1e-4
# EPOCHS = 20
# BS = 32

INIT_LR = 1e-22
EPOCHS = 22
BS = 38

# Dataset path
DIRECTORY = r"C:\Users\aksha\Desktop\NTCC\Mask Detection\dataset"
# Folders present inside dataset
CATEGORIES = ["with_mask", "without_mask"]

# Grab the list of images in our dataset directory, then initialize the list of data (i.e., images) and categorize
# Print a statement for loading of the images(dataset)
print("[INFO] loading images...")

# Images from the dataset
data = []
# Images labelled -> with mask or without mask
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)

    # List down all the images
    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        # Size of images taken -> 250,250
        image = load_img(img_path, target_size=(250, 250))

		#Convert the image to array for preprocessing -> Using MobileNet
        image = img_to_array(image)
        image = preprocess_input(image)

		# Adding images to the data array and labels with specific category
        data.append(image)

		# Labels are in text format
        labels.append(category)

# Perform One-Hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Comverting it to numpy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Using 20% of the dataset to train the model and rest 80% dataset to validate the model
# Random state can be changed to any number
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
												test_size=0.20, stratify=labels, random_state=38)

# Construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Load the MobileNetV2 network, ensuring the head FC layer sets are left off

# Create a base model and weights are initialized -> imagenet
# (250, 250, 3) -> size of image and the 3 channels are passed => RGB (coloured image is passed)
baseModel = MobileNetV3Large(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(250, 250, 3)))

# Construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)

# Go to activation function for non-linear use cases
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
			metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

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
