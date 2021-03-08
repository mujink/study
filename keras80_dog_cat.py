# 이미지
# data/image/vgg/에 4개를 넣으시오
# 개, 고양이, 라이언, 슈트 jpg로 각각 1개씩
# 욜케 넣어 놓을 것
# 파일명 : 

from numpy.core.records import array
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.applications.vgg16 import VGG16, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_dog = load_img('../data/image/vgg/image/cat.jpg', target_size=(224,224))
img_cat = load_img('../data/image/vgg/image/dog.jpg', target_size=(224,224))
img_lion = load_img('../data/image/vgg/image/lion.jpg', target_size=(224,224))
img_suit = load_img('../data/image/vgg/image/suit.jpg', target_size=(224,224))


arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)

print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape)

from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
print(arr_dog)
print(arr_dog.shape) #(224,224,3)


arr_input = np.stack([arr_dog,arr_cat,arr_lion,arr_suit])

model = VGG16()
results = model.predict(arr_input)

print(results)
print('results.shape', results.shape)

# 이미지 결과 확인
results2 = decode_predictions(results)
print("=====================================")
print("results2 0 :", results2[0][0])
print("results2 1:", results2[1][0])
print("results2 2:", results2[2][0])
print("results2 3:", results2[3][0])
# results2 0 : ('n02123045', 'tabby', 0.28407922)
# results2 1: ('n02108551', 'Tibetan_mastiff', 0.9754567)
# results2 2: ('n03291819', 'envelope', 0.36190635) => 라이언은 학습되어있지 않음
# results2 3: ('n04350905', 'suit', 0.8135224)