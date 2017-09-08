import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import params

batch_size = params.batch_size
input_size = params.input_size
threshold = params.threshold

model = params.model
model.load_weights('weights/best_weights.hdf5')

df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

for start in tqdm(range(0, len(ids_train), batch_size)):
    x_batch = []
    end = min(start+batch_size, len(ids_train))
    ids_train_batch = ids_train[start:end]
    for id in ids_train_batch:
        img = cv2.imread('input/train/{}.jpg'.format(id))
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for pred, id in zip(preds, ids_train_batch):
        mask = pred > threshold
        mask = np.array(mask, np.float32) * 255
        cv2.imwrite('input/train_low/{}_low.jpg'.format(id), mask)
