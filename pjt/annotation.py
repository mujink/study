import os
import argparse
import xml.etree.ElementTree as ET

def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':

    if os.path.exists("..\data\pjt\train_dataset.txt"):os.remove("..\data\pjt\train_dataset.txt")
    if os.path.exists("..\data\pjt\test_dataset.txt"):os.remove("..\data\pjt\test_dataset.txt")

    num1 = convert_voc_annotation(os.path.join('..\data\pjt\VOCdevkit\VOC2012/'), 'trainval', "..\\data\pjt\\train_dataset.txt", False)
    num2 = convert_voc_annotation(os.path.join('..\data\pjt\VOCdevkit\VOC2012/'), 'train', "..\\data\\pjt\\train_dataset.txt", False)
    num3 = convert_voc_annotation(os.path.join('..\data\pjt\VOCdevkit\VOC2012/'),  'val', "..\\data\\pjt\\test_dataset.txt", False)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' %(num1 + num2, num3))
