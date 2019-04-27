from gan import CGAN
from data import y_train, label_encoder

import matplotlib.pyplot as plt

import numpy as np
import os

cgan = CGAN(100, 100, 3, len(label_encoder.classes_))
print(y_train)
y_train_new = y_train.reshape(-1, 1)
r, c = 1, max(y_train_new)[0] + 1
for j in range(0, 20):
    noise = np.random.normal(0, 1, (r * c, 100))
    sampled_labels = np.arange(0, c).reshape(-1, 1)
    gen_imgs = cgan.generator.predict([noise, sampled_labels])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    cnt = 0
    print(j)
    for i in range(c):
        folder = label_encoder.inverse_transform(y_train)[i]
        plt.imsave("generated/%s_%d.png" % (folder, j), gen_imgs[i,:,:,:])

        with open("gen.csv", "a") as f:
            f.write("%s_%d.png,%s\n" % (folder, j, folder))