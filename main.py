import cv2
import numpy as np
import os
import glob
import time
import random
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

train_path = "dataset"
class_names = os.listdir(train_path)
print(class_names)

image_paths = []
image_classes = []


def img_list(path):
    return (os.path.join(path, f) for f in os.listdir(path))


for training_name in class_names:
    dir_ = os.path.join(train_path, training_name)
    class_path = img_list(dir_)
    image_paths += class_path

print(len(image_paths))

image_classes_0 = [0] * (len(image_paths) // 3)
image_classes_1 = [1] * (len(image_paths) // 3)
image_classes_2 = [2] * (len(image_paths) // 3)


image_classes = image_classes_0 + image_classes_1 + image_classes_2

D = []

for i in range(len(image_paths)):
    D.append((image_paths[i], image_classes[i]))
dataset = D
random.shuffle(dataset)
train = dataset[:70]
test = dataset[70:]

image_paths, y_train = zip(*train)
image_paths_test, y_test = zip(*test)

# ORB Feature
des_list = []
orb = cv2.ORB_create()
im = cv2.imread(image_paths[1])
#plt.imshow(im)

'''
# SIFT Feature
des_list = []
sift = cv2.SIFT_create()
im = cv2.imread(image_paths[1])
plt.imshow(im)
'''
'''
# SURF Feature
des_list = []
sift = cv2.SURF_create()
im = cv2.imread(image_paths[1])
plt.imshow(im)
'''


def draw_keypoints(vis, keypoints, color=(0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        plt.imshow(cv2.circle(vis, (int(x), int(y)), 2, color))


# Draw ORB feature on the example
#kp = orb.detect(im, None)
#kp, des = orb.compute(im, kp)
#draw_keypoints(im, kp)
'''
# Draw SIFT feature on the example
kp = sift.detect(im, None)
kp, des = sift.compute(im, kp)
draw_keypoints(im, kp)
'''

'''
# Draw SURF feature on the example
surf = cv2.SURF(400)
kp, des = surf.detectAndCompute(im, None)
draw_keypoints(im, kp)

'''
# Appending ORB descriptors of the training images in list
for image_pat in image_paths:
    im = cv2.imread(image_pat)
    kp = orb.detect(im, None)
    keypoints, descriptor = orb.compute(im, kp)
    des_list.append((image_pat, descriptor))

'''
# Appending SURF descriptors of the training images in list
for image_pat in image_paths:
    im = cv2.imread(image_pat)
    keypoints, descriptor = surf.detectAndCompute(im, None)
    des_list.append((image_pat, descriptor))

'''
'''

# Appending SIFT descriptors of the training images in list
for image_pat in image_paths:
    im = cv2.imread(image_pat)
    scale_percent = 80  # percent of original size
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    kp = sift.detect(im, None)
    keypoints, descriptor = sift.compute(im, kp)
    des_list.append((image_pat, descriptor))
'''
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
descriptors_float = descriptors.astype(float)


def calacc(k):
    print(k)
    start = time.time()
    voc, variance = kmeans(descriptors_float, k, 1)
    # Creating histogram of training image
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1
    stdslr = StandardScaler().fit(im_features)
    im_features = stdslr.transform(im_features)

    # Muliclass SVM

    clf = SVC()
    clf.fit(im_features, np.array(y_train))
    end = time.time()
    print('time:' + str(end - start))
    des_list_test = []

    for image_pat in image_paths_test:
        image = cv2.imread(image_pat)
        kp = orb.detect(image, None)
        keypoints_test, descriptor_test = orb.compute(image, kp)
        des_list_test.append((image_pat, descriptor_test))

    test_features = np.zeros((len(image_paths_test), k), "float32")
    for i in range(len(image_paths_test)):
        words, distance = vq(des_list_test[i][1], voc)
        for w in words:
            test_features[i][w] += 1

    test_features = stdslr.transform(test_features)

    true_classes = []
    a = list(y_test)
    for i in a:
        if i == 0:
            true_classes.append("cane")
        elif i == 1:
            true_classes.append("elefante")
        else:
            true_classes.append("gatto")

    predict_classes = []
    for i in clf.predict(test_features):
        if i == 0:
            predict_classes.append("cane")
        elif i == 1:
            predict_classes.append("elefante")
        else:
            predict_classes.append("gatto")

    #Show Accuracy
    accuracy = accuracy_score(true_classes, predict_classes)
    print('arr' + str(accuracy))
    return accuracy


out = []
for k in range(5, 100, 5):
    out += [calacc(k)]

plt.plot([i for i in range(5, 100, 5)], out)
