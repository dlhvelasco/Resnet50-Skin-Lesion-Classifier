from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from os import listdir
import pickle as pkl
from keras.preprocessing import image
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, LeakyReLU
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import csv
import subprocess

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

######################################################################################################################
# def load_dataset(path):
#     data = load_files(path)
#     paths = np.array(data['filenames'])
#     targets = np_utils.to_categorical(np.array(data['target']))
#     return paths, targets
#
# train_files, train_targets = load_dataset(r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/data/train')
# valid_files, valid_targets = load_dataset(r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/data/valid')
# test_files, test_targets = load_dataset(r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/data/test')
#
# #####################################################################################################################
# print("\nPickling files & targets... ")
#
# with open('./pickledres/trn_files.pkl', 'wb') as f:
#     pkl.dump(train_files, f)
# with open('./pickledres/trn_labels.pkl', 'wb') as f:
#     pkl.dump(train_targets, f)
#
# with open('./pickledres/tst_files.pkl', 'wb') as f:
#     pkl.dump(test_files, f)
# with open('./pickledres/tst_labels.pkl', 'wb') as f:
#     pkl.dump(test_targets, f)
#
# with open('./pickledres/vld_files.pkl', 'wb') as f:
#     pkl.dump(valid_files, f)
# with open('./pickledres/vld_labels.pkl', 'wb') as f:
#     pkl.dump(valid_targets, f)
#
#######################################################################################################################
print("\nLoading files & targets... ")
with open('./pickledres/trn_files.pkl', 'rb') as f:
    train_files = pkl.load(f)
with open('./pickledres/trn_labels.pkl', 'rb') as f:
    train_targets = pkl.load(f)
with open('./pickledres/tst_files.pkl', 'rb') as f:
    test_files = pkl.load(f)
with open('./pickledres/tst_labels.pkl', 'rb') as f:
    test_targets = pkl.load(f)
with open('./pickledres/vld_files.pkl', 'rb') as f:
    valid_files = pkl.load(f)
with open('./pickledres/vld_labels.pkl', 'rb') as f:
    valid_targets = pkl.load(f)

######################################################################################################################
diseases = sorted(listdir('./data/train'))
# train 2000
# valid 150
# test  600

print('\nThere are {} classes: {}.'.format(len(diseases), ', '.join(diseases)))
print('There are {} training images.'.format(len(train_files)))
print('There are {} validation images.'.format(len(valid_files)))
print('There are {} testing images.'.format(len(test_files)))

def get_tensor(path):
    img = image.load_img(path, target_size=(224, 224))
    return np.expand_dims(image.img_to_array(img), axis=0)

def get_tensors(paths):
    return np.vstack([get_tensor(path) for path in tqdm(paths)])

# ######################################################################################################################
# print("\nPre-processing...")
#
# train_tensors = preprocess_input(get_tensors(train_files))
# valid_tensors = preprocess_input(get_tensors(valid_files))
# test_tensors = preprocess_input(get_tensors(test_files))
#
# ######################################################################################################################
# print("\nPickling Pre-processed data... ")
#
# with open('./pickledres/tens_trn.pkl', 'wb') as f:
#     pkl.dump(train_tensors, f)
# with open('./pickledres/tens_vld.pkl', 'wb') as f:
#     pkl.dump(valid_tensors, f)
# with open('./pickledres/tens_tst.pkl', 'wb') as f:
#     pkl.dump(test_tensors, f)

######################################################################################################################
print("\nLoading Pre-processed pickles... ")

with open('./pickledres/tens_trn.pkl', 'rb') as f:
    train_tensors = pkl.load(f)
with open('./pickledres/tens_vld.pkl', 'rb') as f:
    valid_tensors = pkl.load(f)
with open('./pickledres/tens_tst.pkl', 'rb') as f:
    test_tensors = pkl.load(f)

######################################################################################################################
resnet50 = ResNet50(include_top=False, input_shape=(224, 224, 3))

train_bottleneck = resnet50.predict(train_tensors)
valid_bottleneck = resnet50.predict(valid_tensors)
test_bottleneck = resnet50.predict(test_tensors)

######################################################################################################################
model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=train_bottleneck.shape[1:]))
# model.add(Dropout(0.6))
# model.add(Dense(1024, activation='relu'))
model.add(Dense(2048))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.7))
# model.add(Dense(1024, activation='relu'))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.7))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(3, activation='softmax'))

print(model.summary())

######################################################################################################################
def setup_to_transfer_learn(model):
    # """Freeze all pretrained layers and compile the model"""
    # for layer in model.layers:
    #     layer.trainable = False
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)
# optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.005)
# optimizer = optimizers.SGD(lr=0.0001, decay=0.00, momentum=0.9, nesterov=True)
# optimizer = optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6)

setup_to_transfer_learn(model)

######################################################################################################################
epochs = 200
batch_size = 100

checkpointer = ModelCheckpoint(filepath='resnet.from.bottleneck.hdf5', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='auto')

history = model.fit(train_bottleneck, train_targets, epochs=epochs, batch_size=batch_size,
          validation_data=(valid_bottleneck, valid_targets),
          callbacks=[checkpointer, early_stopping], verbose=1, shuffle=True)

model.load_weights('resnet.from.bottleneck.hdf5')
print('\nTesting loss: {:.4f}\nTesting accuracy: {:.4f}'.format(*model.evaluate(test_bottleneck, test_targets)))

######################################################################################################################
print("\nPlotting...")

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

#######################################################################################################################
user_input=input('\nWould you like to process?\n\n(Y/N)')
	if(user_input == 'N' or user_input == 'n'):
		continue
    else:
        print("Exporting to csv...\n")
        with open('predictions.csv', 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['Id', 'task_1', 'task_2'])
            for path in tqdm(sorted(test_files)):
                tensor = preprocess_input(get_tensor(path))
                pred = model.predict(resnet50.predict(tensor))[0]
                csvwriter.writerow([path, pred[0], pred[2]])

        print("Printing final results...\n")
        finalresult = ['python',r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/get_results.py',r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/predictions.csv']
        finalresultcomplete = subprocess.run(finalresult, shell=True)