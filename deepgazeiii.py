import matplotlib.pyplot as plt
import numpy as np
# from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from pysaliency.models import sample_from_logdensity
import deepgaze_pytorch

DEVICE = 'cpu'

# you can use DeepGazeI or DeepGazeIIE
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)

image = plt.imread(r'samples/COCO_train2014_000000001455.jpg')

# location of previous scanpath fixations in x and y (pixel coordinates), starting with the initial fixation on the image.
fixation_history_x = np.array([1680 // 2, 1214, 1139, 957])
fixation_history_y = np.array([1050 // 2, 480, 171, 157])

# 1680 // 2, 1680 // 2, 1680 // 2, 1680 // 2, 1214, 1139, 957, 937, 268, 903
# 1050 // 2, 1050 // 2, 1050 // 2, 1050 // 2, 480, 171, 157, 115, 367, 196

# load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
# you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
# alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
centerbias_template = np.load('centerbias_mit1003.npy')
# rescale to match image size
centerbias = zoom(centerbias_template,
                  (image.shape[0] / centerbias_template.shape[0], image.shape[1] / centerbias_template.shape[1]),
                  order=0, mode='nearest')
# renormalize log density
centerbias -= logsumexp(centerbias)

image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

x_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
y_hist_tensor = torch.tensor([fixation_history_y[model.included_fixations]]).to(DEVICE)

log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)

f, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
axs[0].imshow(image)
axs[0].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
axs[0].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
axs[0].set_axis_off()
axs[1].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
axs[1].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
axs[1].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
axs[1].set_axis_off()
plt.show()


def get_fixation_history(fixation_coordinates, model):
    history = []
    for index in model.included_fixations:
        try:
            history.append(fixation_coordinates[index])
        except IndexError:
            history.append(np.nan)
    return history


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 12))

rst = np.random.RandomState(seed=23)

# for ax in axs.flatten():
fixations_x = [840]
fixations_y = [525]

for i in range(3):
    x_hist = get_fixation_history(fixations_x, model)
    y_hist = get_fixation_history(fixations_y, model)

    x_hist_tensor = torch.tensor([x_hist]).to(DEVICE)
    y_hist_tensor = torch.tensor([y_hist]).to(DEVICE)
    log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
    logD = log_density_prediction.detach().cpu().numpy()[0, 0]
    next_x, next_y = sample_from_logdensity(logD, rst=rst)

    fixations_x.append(next_x)
    fixations_y.append(next_y)

ax.imshow(image)
ax.plot(fixations_x, fixations_y, 'o-', color='red')
ax.set_axis_off()
plt.show()
