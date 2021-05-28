import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator( 
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # zca_whitening=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.05,
        rotation_range = 30)
        # validation_split=0.2)
    
def imgGenerator(img, label):

    x_set, y_set = [],[]
    for i in range(img.shape[0]): #2048
        num_aug = 0
        x = img[i]  #2048
        y = label[i]  #2048
        x_t = x.reshape((1,) + x.shape)
        # b = np.array(x)
        # print(b.shape)
        # b = b.reshape(28,28)
        # plt.imshow(b)
        # plt.show() 
        for x_aug in datagen.flow(x_t) : #20
            if num_aug >= 20:
                break
            x_set.append(x_aug) #2048*20
            y_set.append(y)
            # a = np.array(x_aug)
            # plt.imshow(a.reshape(28,28))
            # plt.show()
            # print(y)
            num_aug += 1

    x_set = np.array(x_set)
    y_set = np.array(y_set)

    x_set = x_set.reshape(-1, 28*28)
    return x_set, y_set

def image_preprocess(image, target_size, gt_boxes=None):
    image = np.array(image,dtype = 'uint8')

    image = image.reshape(4,4,28,28)

    ih, iw = target_size
    # _, h,  w = image.shape

    nw, nh  = int(iw/4), int(ih/4)
    image_paded = np.zeros(shape=(ih, iw))
    
    for i in range(4):
        for j in range(4):
            # print(type(image[i,j]))
            # print(image.shape)
            # plt.figure(figsize=(5,5))
            # plt.imshow(image[i,j])
            # plt.show()
            # image = image[i,j]

            image_resized = cv2.resize(image[i,j], (nw, nh))
            image_paded[nw*i:nh*(i+1), nw*j:nh*(j+1)] = image_resized
 
    # plt.figure(figsize=(5,5))
    # plt.imshow(image_paded)
    # plt.show()
    # plt.close()

    # 이미지로 저장할 때 정규화 해버리면 이미지가 검은색으로 보임
    # image_paded = image_paded / 255.
    return image_paded


# ============================== 이번 데이콘 mnist set 노이즈 제거 ===================================================

# for i in range(50000):
#     image_path = 'D:/python/data/train/%05d.png'%i
#     image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
#     image2 = np.where((image <= 249) & (image != 0), 0, image)#254보다 작은건 모조리 0으로 처리
#     image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
#     image4 = cv2.medianBlur(src=image3, ksize= 3)
#     image4 = np.where((image <= 254) & (image != 0), 0, image4)
#     cv2.imwrite('D:/python/data/dirty_mnist_2nd_noise_clean/%05d.png'%i, image4)


# for i in range(50000,55000):
#     image_path = 'D:/python/data/test/%05d.png'%i
#     image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
#     image2 = np.where((image <= 249) & (image != 0), 0, image)#254보다 작은건 모조리 0으로 처리
#     image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
#     image4 = cv2.medianBlur(src=image3, ksize= 3)
#     image4 = np.where((image <= 254) & (image != 0), 0, image4)
#     cv2.imwrite('D:/python/data/test_dirty_mnist_2nd_noise_clean/%05d.png'%i, image4)


# ============================== 저번 데이콘 낱 개 mnist set 4*4 그림으로 만들기 ===================================================

file = 'train.csv'
path = 'D:\python\data\mnist_data/'

df = pd.read_csv(path+file, header = 0)
images = df.iloc[:,3:]
label = df.iloc[:,2]


# 라벨 인코딩 =======================================
labels = np.zeros((len(label), len(label.unique())))
letter = ['A','B','C','D','E',
          'F','G','H','I','J',
          'K','L','M','N','O',
          'P','Q','R','S','T',
          'U','V','W','X','Y','Z']

for label_index, y in enumerate(label):
    for string_index, letteral in enumerate(letter) :
        if y == letteral :
            labels[label_index][string_index] = 1 
            


images = images.to_numpy()
images = images.reshape(-1,28,28,1)


label_1 = np.matrix(np.zeros(len(label.unique())))
clean_image = []
clean_label = []
image_16 = []

Random_index = np.array(range(len(images)))
i = 0
for i in list(range(0,500,1)):

    np.random.shuffle(Random_index)
    image_set, label_set = imgGenerator(images, labels)

    
    for index in Random_index:
        label_1 += np.matrix(label_set[index])
        image_16.append(image_set[index])

        if 2 in label_1 :
            # print("2가 포함되어있음 방금 추가한 레이블과 이미지를 리스트에서 삭제할거임",label_1)
            label_1 -= label_set[index]
            image_16 = image_16[:-1]

        elif (len(image_16) % 16) == 0 :
            image_16 = np.array(image_16)
            label_1 = np.array(label_1)
            clean_image.append(image_preprocess(image_16, target_size=(256,256), gt_boxes=None))
            clean_label.append(label_1)
            label_1 = np.matrix(np.zeros(len(label.unique())))
            image_16 = []

            # image = np.array(clean_image)
            # image = image.reshape(-1,256,256,1)
            # for i in range(len(image)):
            #     cv2.imwrite('D:/python/data/mytest2/%05d.png'%int(int(i)+50000), image[i])
                

    print(i,"번 째 진행중,", len(clean_image), "개 얻음")
    if len(clean_image) >= 5000:
        print(len(clean_image))
        break

clean_image = np.array(clean_image)
clean_image = clean_image.reshape(-1,256,256,1)

clean_label = np.array(clean_label)
clean_label = clean_label.reshape(-1,26)
df = pd.DataFrame(clean_label, columns=letter)

print(clean_image.shape) # 때마다 길이가 다름

# ============================== y 라벨 저장 위치 ============================================
df.to_csv('D:/python/data/mytest2/mnist_2nd_answer.csv', sep=',')

#  ============================ image 저장 =================================================
clean_image[0]
plt.figure(figsize=(5,5))
plt.imshow(clean_image[0])
plt.show()
plt.close()

for i in range(len(clean_image)):
    cv2.imwrite('D:/python/data/mytest2/%05d.png'%int(int(i)+50000), clean_image[i])

    # cv2.imwrite('D:/python/data/test_dirty_mnist_2nd_noise_clean/%05d.png'%i, image4)



from keras import backend as K

def loss(y_p,y_t):
    aaa = -((1/26)* K.mean(y_t*K.log((1+K.exp(-y_p))**-1)+(1-y_t)*K.log(K.exp(-y_p)/(1+K.exp(-y_p)))))
    return aaa
