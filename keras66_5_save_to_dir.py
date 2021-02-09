import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    # 받게될 이미지를 아래 내용기준으로 변환한다
    # 스케일 한다
    rescale=1./255,
    # 변환파라미터
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # 이동 후 빈자리를 nearest 모드를 한다.
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory(폴더명을 카테고리로 지정한 경우 폴더명이 Y라벨이됨)
# 이미지 플로트 하는 걸 과제로 낼거임

# test_datagen에 대해서 프롤우 프럼 디렉토리를 다음과 같이한다.
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train', # (160,256,256,3)
    target_size=(150,150), # (160,150,150,3)
    batch_size=30, # (전체 데이터/batch_size, batch_size,150,150,3 5차원 배열)
    class_mode='binary',
    save_to_dir='../data/image/brain_generator/train'
)

xy_test = test_datagen.flow_from_directory(
    # 첫번 째 인자는 디렉토리 폴더를 받음
    '../data/image/brain/test', # (120,256,256,3)
    target_size=(150,150), # (120,150,150,3)
    batch_size=5, # (전체 데이터/batch_size, batch_size,150,150,3 5차원 배열)
    class_mode='binary',
    save_to_dir='../data/image/brain_generator/test'
)

print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][0].shape)
# print(xy_test[0][0])
"""
>> print(xy_train)
<tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000021595418550>

>> print(xy_train[0])
xy_train[0] 은 x, y 값을 다 가지고있음
xy_train[0]의 첫번째 []의 범위는 => 전체데이터 길이/batch_size

>> print(xy_train[0][0])
xy_train[0][0]은 x 값만 가지고 있음
xy_train[0][0]의 두번째 []의 범위는 => x와 y 값 x=0, y=1

batch_size < 전체 데이터 일 때
ex)
batch_size = 5
x 값 쉐이프는 원래 데이터에서 차원이 하나 더 늘음
>> print(xy_train[0][0].shape)
(5, 150, 150, 3)

y 값 쉐이프는
>>print(xy_train[0][1].shape)
(베치사이즈,)
 
batch_size > 전체 데이터 일 때
ex)
batch_size = 1000000

x 값의 쉐이프는 4차원으로 나옴
>> print(xy_train[0][0].shape)
(전체데이터 길이, 150, 150, 3)

y 값의 쉐이프는
>>print(xy_train[0][1].shape)
(전체데이터 길이,)
"""