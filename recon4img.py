import numpy as np
import struct
from functools import partial
from matplotlib import pyplot as plt

def extractIntensity(filepath):
    struct_fmt = 'B' * (512 * 512)
    # 'B' is for 1byte(8bit) unsigned integer which is same as uint8 in C++
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    # Loading
    with open(filepath, "rb") as file:
        datas = [struct_unpack(byte) for byte in iter(partial(file.read, struct_len), b'')]

    # Show up
    intense = np.asarray(datas)
    intense = np.reshape(intense, (382, 512, 512))
    # tmp /= 10. # Scaling (cm, mm)
    print(intense.shape)

    return intense

img_data = np.load(".../intensity.npy")
#img_data_2 = np.load(".../intensity_2.npy")

basedir = "..."

for i in range(0, 100):
    ex = img_data[i]

    fig, ax = plt.subplots()
    x = 222 / fig.dpi
    y = 222 / fig.dpi
    fig.set_figwidth(x)
    fig.set_figheight(y)

    plt.imshow(ex)
    plt.axis('off')
    file_name = basedir + "IMG_" + str(i + 4020) + ".jpg"
    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
