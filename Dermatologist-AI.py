from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from os import listdir
from keras.preprocessing import image
from tqdm import tqdm_notebook
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import csv
import subprocess

from keras import backend as K
K.tensorflow_backend._get_available_gpus()


def load_dataset(path):
    data = load_files(path)
    paths = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']))
    return paths, targets

train_files, train_targets = load_dataset(r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/data/train')
valid_files, valid_targets = load_dataset(r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/data/valid')
test_files, test_targets = load_dataset(r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/data/test')

######################################################################################################################
diseases = sorted(listdir('./data/train'))

# train 700
# valid 100
# test  200

print('There are {} classes: {}.'.format(len(diseases), ', '.join(diseases)))
print('There are {} training images.'.format(len(train_files)))
print('There are {} validation images.'.format(len(valid_files)))
print('There are {} testing images.'.format(len(test_files)))

def get_tensor(path):
    img = image.load_img(path, target_size=(224, 224))
    return np.expand_dims(image.img_to_array(img), axis=0)

def get_tensors(paths):
    return np.vstack([get_tensor(path) for path in tqdm_notebook(paths)])

######################################################################################################################
train_tensors = preprocess_input(get_tensors(train_files))
valid_tensors = preprocess_input(get_tensors(valid_files))
test_tensors = preprocess_input(get_tensors(test_files))

######################################################################################################################
resnet50 = ResNet50(include_top=False, input_shape=(224, 224, 3))

train_bottleneck = resnet50.predict(train_tensors)
valid_bottleneck = resnet50.predict(valid_tensors)
test_bottleneck = resnet50.predict(test_tensors)

######################################################################################################################
model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=train_bottleneck.shape[1:]))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
print(model.summary())

######################################################################################################################

# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)
# optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.005)
optimizer = optimizers.SGD(lr=0.001, decay=0.00, momentum=0.9, nesterov=True)

# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam) # Alternative

######################################################################################################################
checkpointer = ModelCheckpoint(filepath='resnet.from.bottleneck.hdf5', save_best_only=True)
history = model.fit(train_bottleneck, train_targets, epochs=100,
          validation_data=(valid_bottleneck, valid_targets),
          callbacks=[checkpointer], verbose=1, shuffle=True)

model.load_weights('resnet.from.bottleneck.hdf5')
print('\nTesting loss: {:.4f}\nTesting accuracy: {:.4f}'.format(*model.evaluate(test_bottleneck, test_targets)))

######################################################################################################################
print("Plotting...\n")

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

plt.plot(history.history['acc'], label='Training accuracy')
plt.plot(history.history['val_acc'], label='Validation accuracy')
plt.legend()
plt.show()

for i in range(len(diseases)):
    fpr, tpr, _ = roc_curve(test_targets[:,i], model.predict(test_bottleneck)[:,i])
    print('AUC for {}: {:.4f}'.format(diseases[i], auc(fpr, tpr)))
    plt.plot(fpr, tpr, label=diseases[i])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.title('Receiver operator characteristic curve')
plt.show()

cm = confusion_matrix(np.argmax(test_targets, axis=1),
                      np.argmax(model.predict(test_bottleneck), axis=1))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
indexes = np.arange(len(diseases))
for i in indexes:
    for j in indexes:
        plt.text(j, i, cm[i, j])
plt.xticks(indexes, diseases, rotation=90)
plt.xlabel('Predicted label')
plt.yticks(indexes, diseases)
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.show()

######################################################################################################################
# print("Exporting to csv...\n")
# with open('predictions.csv', 'w') as f:
#     csvwriter = csv.writer(f)
#     csvwriter.writerow(['Id', 'task_1', 'task_2'])
#     for path in tqdm_notebook(sorted(test_files)):
#         tensor = preprocess_input(get_tensor(path))
#         pred = model.predict(resnet50.predict(tensor))[0]
#         csvwriter.writerow([path, pred[0], pred[2]])
#
# print("Printing final results...\n")
# finalresult = ['python',r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/get_results.py',r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/predictions.csv']
# finalresultcomplete = subprocess.run(finalresult, shell=True)