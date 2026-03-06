import numpy as np

def world_to_image(coords, H):
    coords_hom = np.concatenate(
        [coords, np.ones((coords.shape[0],1))],
        axis=1
    )

    projected = H @ coords_hom.T
    projected = projected / projected[2,:]

    return projected[:2,:].T