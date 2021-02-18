import os


# with open("../data/pjt/train_dataset.txt", "r") as f:
#     txt = f.readlines()
#     # print(txt)
#     annotations = []
#     for line in txt:
#         image_path = line.strip()

#         root, _ = os.path.splitext(image_path)
#         print(root)
#         with open(root + ".txt") as fd: #???????
#             boxes = fd.readlines()
#             print(boxes)
# annotations = []
# image_path = "..\data\pjt\VOCdevkit\VOC2012/JPEGImages\2008_000002.jpg 34,11,448,293,19"
# root, _ = os.path.splitext(image_path)
# root = "..\data\pjt\VOCdevkit\VOC2012/Annotations/2008_000002"
# with open(root + ".xml") as fd: #???????
#     boxes = fd.readlines()
#     string = ""
#     for box in boxes:
#         box = box.strip()
#         box = box.split()
#         print(box)
#         class_num = box[0]
#         print(class_num)
#         center_x = float(box[1])
#         center_y = float(box[2])
#         half_width = float(box[3]) / 2
#         half_height = float(box[4]) / 2
#         string += " {},{},{},{},{}".format(
#             center_x - half_width,
#             center_y - half_height,
#             center_x + half_width,
#             center_y + half_height,
#             class_num,
#         )
#     annotations.append(image_path + string)

# print(annotations)

# a = "../data/pjt/train_dataset.txt"
# with open(a, "r") as f:
#     txt = f.readlines()
#     annotations = [
#         line.strip()
#         for line in txt
#         if len(line.strip().split()[1:]) != 0
#     ]

# print(annotations)
