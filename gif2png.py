import pandas as pd
from PIL import Image

df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

for i in ids_train.values:
    print('input/train_masks_gif/{}_mask.gif'.format(i) + '  -->  ' + 'input/train_masks/{}_mask.png'.format(i))
    im = Image.open('input/train_masks_gif/{}_mask.gif'.format(i))
    im.save('input/train_masks/{}_mask.png'.format(i))
