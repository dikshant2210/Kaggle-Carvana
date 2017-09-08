from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024
from model.custom_u_net import custom_unet_256

input_size = 256
mask_size = 1024

max_epochs = 30
batch_size = 8

orig_width = 1918
orig_height = 1280

threshold = 0.53

model = custom_unet_256()
# model = get_unet_256()
