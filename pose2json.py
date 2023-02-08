import json
import struct
import numpy as np
from functools import partial
from collections import OrderedDict


def quaternion_rotation_matrix(Q):
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
    # print(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3) # check the normalization

    r00 = 1 - 2 * (q1 * q1 + q2 * q2)
    r01 = 2 * (q0 * q1 - q2 * q3)
    r02 = 2 * (q0 * q2 + q1 * q3)

    r10 = 2 * (q0 * q1 + q2 * q3)
    r11 = 1 - 2 * (q0 * q0 + q2 * q2)
    r12 = 2 * (q1 * q2 - q0 * q3)

    r20 = 2 * (q0 * q2 - q1 * q3)
    r21 = 2 * (q1 * q2 + q0 * q3)
    r22 = 1 - 2 * (q0 * q0 + q1 * q1)

    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def extractPose(filepath):
    struct_fmt = 'f' * 8

    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    # Loading
    with open(filepath, "rb") as file:
        datas = [struct_unpack(byte) for byte in iter(partial(file.read, struct_len), b'')]

    # Show up
    # poseinfo = np.asarray(datas)
    # print(poseinfo[1])

    return datas


imagefile = ".../poses"
imdata = extractPose(imagefile)

images = []
bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

for idx, im in enumerate(imdata):
    img_path = "data\\.../IMG_" + str(idx + 4020) + ".jpg"
    x, y, z = im[0:3]
    t = np.array([x, y, z])
    t = np.reshape(t, [3, 1])
    t = t / 50 # scaling
    Q = im[4:8] # 3x3 rotation matrix
    R = quaternion_rotation_matrix(Q)
    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0) # 4x4 translation matrix

    sharpness = 20.
    images.append({"file_path": img_path, "shaprness": sharpness, "transform_matrix": m.tolist()})
    # if idx == 49:
    #    break


file_data = OrderedDict()

# This follows instant NGP's input format
file_data["camera_anlge_x"] = 0.
file_data["camera_anlge_y"] = 0.
#file_data["scale"] = 0.05
file_data["fl_x"] = 7107.86474609
file_data["fl_y"] = 7107.86474609
file_data["k1"] = 0.
file_data["k2"] = 0.
file_data["k3"] = 0.
file_data["k4"] = 0.
file_data["p1"] = 0.
file_data["p2"] = 0.
file_data["is_fisheye"] = False
file_data["cx"] = -339.0484314
file_data["cy"] = 264.07528687 # 247.92471313
file_data["w"] = 512.
file_data["h"] = 512.
file_data["aabb_scale"] = 16
file_data["frames"] = images

# print(json.dumps(file_data, ensure_ascii=False, indent="\t"))

with open("transforms.json", 'w', encoding="utf-8") as make_file:
    json.dump(file_data, make_file, ensure_ascii=False, indent="\t")
