import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import matplotlib.pyplot as plt
import os
from glob import glob
import deepgaze_pytorch
from tqdm import trange


DEVICE = 'cuda:0'



def normalize(data):
    return (data-np.min(data)) / (np.max(data) - np.min(data))


# you can use DeepGazeI or DeepGazeIIE
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

# load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
# you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
# alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
# centerbias_template = np.zeros((1024, 1024))
centerbias_template = np.load('centerbias_mit1003.npy')

# Directory to save logs and trained model
ROOT_DIR = os.path.abspath('')

DATASET_DIR = '/home/liaobuxin/Datasets'
CS18_DIR = os.path.join(DATASET_DIR, 'COCO-Search18')
TA_DIR = os.path.join(CS18_DIR, 'TA', 'coco_search18_images_TA')
TP_DIR = os.path.join(CS18_DIR, 'TP', 'images')
CS18_file_paths = glob(os.path.join(TP_DIR, "*", "*.jpg"))+glob(os.path.join(TA_DIR, "*", "*.jpg"))
image_names = map(os.path.basename, CS18_file_paths)
image_names = np.unique([image for image in image_names])
COCO_DIR = os.path.join(DATASET_DIR, 'MSCOCO2014')

map_name = ['deepgazeiie_map_bias', 'deepgazeiie_map_nobias']
use_biases = [True, False]

for i in range(2):

    use_bias = use_biases[i]
    MAP_DIR = os.path.join(ROOT_DIR, map_name[i])
    if not os.path.exists(MAP_DIR):
        os.mkdir(MAP_DIR)

    # to generate gaussian maps
    for image_name, idx in zip(image_names, trange(len(image_names))):
        file_path = glob(os.path.join(CS18_DIR, '*', '*', '*', image_name))[0]
        image = plt.imread(file_path)

        if use_bias:
            # rescale to match image size
            centerbias = zoom(centerbias_template, (
                image.shape[0] / centerbias_template.shape[0], image.shape[1] / centerbias_template.shape[1]), order=0,
                              mode='nearest')
        else:
            centerbias = np.zeros(image.shape[:2])

        # renormalize log density
        centerbias -= logsumexp(centerbias)
        if len(image.shape) == 2:
            image = np.transpose(np.array([image, image, image]), (1, 2, 0))
        image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
        centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

        log_density_prediction = model(image_tensor, centerbias_tensor).detach().cpu().numpy()[0, 0]
        np.savetxt(os.path.join(MAP_DIR, image_name[:-4] + '.csv'), log_density_prediction, fmt='%.4f')
