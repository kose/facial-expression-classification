# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np


def visualize(sx, sy, dataset):

    width = dataset.shape[3]
    height = dataset.shape[2]
    canvas = Image.new('L', (1 + (width + 1) * sx, 1 + (height + 1) * sy), (100))

    images = (dataset * 255).detach().numpy().copy().astype(np.uint8) # torch -> numpy
    images = images.reshape(-1, height, width)

    # タイリング
    for y in range(sy):
        for x in range(sx):
            i = y * sx + x
            if len(dataset) > i:
                image = Image.fromarray(images[i])
                canvas.paste(image, (1 + (width + 1) * x, 1 + ( height + 1) * y))


    canvas.show()
    

    # import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###

