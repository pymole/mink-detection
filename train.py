import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
import pickle

AUTOTUNE = tf.data.experimental.AUTOTUNE

TRAIN_DIR = 'images/train1'
TEST_DIR = 'images/test'
BATCH_SIZE = 32
IMG_WIDTH = 224
IMG_HEIGHT = 224


def extract_resnet(X):
    resnet_model = ResNet50(weights='imagenet', include_top=False, classes=False)

    # Since top layer is the fc layer used for predictions
    features_array = resnet_model.predict(X)
    features_array = np.reshape(features_array, (-1, features_array.shape[1]*features_array.shape[2]*
                                                 features_array.shape[3]))
    print(features_array.shape)
    return features_array


image_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)


train = image_gen.flow_from_directory(
    TRAIN_DIR,
    batch_size=BATCH_SIZE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)
#
# test = image_gen.flow_from_directory(
#     TEST_DIR,
#     batch_size=BATCH_SIZE,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     shuffle=True,
#     seed=42,
#     interpolation="bilinear",
# )

train = extract_resnet(train)
# test = extract_resnet(test)

# Apply standard scaler to output from resnet50
ss = StandardScaler()
ss.fit(train)

train = ss.transform(train)
# test = ss.transform(test)

# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=512, whiten=True)
pca = pca.fit(train)

print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))


train = pca.transform(train)
test = pca.transform(test)

# Train classifier and obtain predictions for OC-SVM
oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)

# Obtained using grid search
if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)

oc_svm_clf.fit(train)
if_clf.fit(train)
oc_svm_preds = oc_svm_clf.predict(test)
if_preds = if_clf.predict(test)

print(oc_svm_preds)
print(if_preds)
