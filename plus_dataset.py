import pandas as pd
import os
import numpy as np
import cv2
file = 'train.csv'
path = 'D:\python\data\mnist_data/'

df = pd.read_csv(path+file,index_col=0, header = 0)
images = df.iloc[:,2:]
images = images.to_numpy()
images = images.reshape(-1,28,28)
print(images.shape)

for image_index in range(len(images)):
    image = images[image_index]
    cv2.imwrite('D:\python\data\mytest\%05d.png'%(image_index+50001), image)

#==================================================

letter = df.iloc[:,1]
print(letter)



# print(images.shape)

# for index in dirty_mnist2 :

#     image = index[3:]
#     print(image.shape)

# dataset_path = "..\data\pjt\VOCdevkit\VOC2012"

# IMAGE_FOLDER = "JPEGImages"
# ANNOTATIONS_FOLDER = "Annotations"

# ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
# img_root, amg_dir, img_files = next(os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

# for xml_file in ann_files:

#     # XML파일와 이미지파일은 이름이 같으므로, 확장자만 맞춰서 찾습니다.
#     img_name = img_files[img_files.index(".".join([xml_file.split(".")[0], "jpg"]))]

#     image = Image.open(os.path.join(img_root, img_name)).convert("RGB")
#     draw = ImageDraw.Draw(image)

#     xml = open(os.path.join(ann_root, xml_file), "r")
#     tree = Et.parse(xml)
#     root = tree.getroot()

#     size = root.find("size")

#     width = size.find("width").text
#     height = size.find("height").text
#     channels = size.find("depth").text

#     objects = root.findall("object")

#     for _object in objects:
#         name = _object.find("name").text
#         bndbox = _object.find("bndbox")
#         xmin = int(bndbox.find("xmin").text)
#         ymin = int(bndbox.find("ymin").text)
#         xmax = int(bndbox.find("xmax").text)
#         ymax = int(bndbox.find("ymax").text)

#         # Box를 그릴 때, 왼쪽 상단 점과, 오른쪽 하단 점의 좌표를 입력으로 주면 됩니다.
#         draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")
#         draw.text((xmin, ymin), name)

#     plt.figure(figsize=(25,20))
#     plt.imshow(image)
#     plt.show()
#     plt.close()