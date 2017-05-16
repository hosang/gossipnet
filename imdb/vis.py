
import matplotlib.pyplot as plt
from scipy.misc import imread


# visualisation helpers
def visualize_roidb(roidb):
    for roi in roidb:
        im = imread(roi['filename'])

        plt.figure()
        plt.imshow(im)
        plt.waitforbuttonpress()
        plt.close()