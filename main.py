import matplotlib.pyplot as plt
import numpy as np
from preprocessing import ETHDataset
from utils import world_to_image

# path vers le dataset
data_path = "eth.txt"
dataset = ETHDataset(data_path)
sample = dataset[0]

obs = sample["obs_abs"].numpy()
pred = sample["pred_abs"].numpy()

plt.figure()
plt.plot(obs[:,0], obs[:,1], 'bo-', label="Observed")
plt.plot(pred[:,0], pred[:,1], 'ro-', label="Future")
plt.legend()
plt.title("Trajectory (World coordinates)")
plt.show()

obs_img = world_to_image(obs, H)
pred_img = world_to_image(pred, H)

# A changer avec les .txt homographique en fonction de la scène
# np.loadtxt("eth.txt")
H = np.eye(3)

obs_img = world_to_image(obs, H)
pred_img = world_to_image(pred, H)

# image noir pour le dessins
scene_image = np.zeros((600,800,3))

plt.imshow(scene_image)
plt.plot(obs_img[:,0], obs_img[:,1], 'bo-')
plt.plot(pred_img[:,0], pred_img[:,1], 'ro-')
plt.title("Projection test")
plt.show()

