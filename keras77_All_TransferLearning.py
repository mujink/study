from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1

# model = VGG19()
model = VGG19(weights="imagenet", include_top= False, input_shape=(32,32,3))
"""
VGG16
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
32
32

VGG19
=================================================================
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
_________________________________________________________________
38
38

Xception
==================================================================================================
Total params: 22,910,480
Trainable params: 22,855,952
Non-trainable params: 54,528
__________________________________________________________________________________________________
236
156

ResNet50
==================================================================================================
Total params: 25,636,712
Trainable params: 25,583,592
Non-trainable params: 53,120
__________________________________________________________________________________________________
320
214
ResNet101
==================================================================================================
Total params: 44,707,176
Trainable params: 44,601,832
Non-trainable params: 105,344
__________________________________________________________________________________________________
626
418

InceptionV3
==================================================================================================
Total params: 23,851,784
Trainable params: 23,817,352
Non-trainable params: 34,432
__________________________________________________________________________________________________
378
190

InceptionResNetV2
==================================================================================================
Total params: 55,873,736
Trainable params: 55,813,192
Non-trainable params: 60,544
__________________________________________________________________________________________________
898
490

DenseNet121
==================================================================================================
Total params: 8,062,504
Trainable params: 7,978,856
Non-trainable params: 83,648
__________________________________________________________________________________________________
606
364

MobileNetV2
==================================================================================================
Total params: 3,538,984
Trainable params: 3,504,872
Non-trainable params: 34,112
__________________________________________________________________________________________________
262
158

NASNetMobile
==================================================================================================
Total params: 5,326,716
Trainable params: 5,289,978
Non-trainable params: 36,738
__________________________________________________________________________________________________
1126
742

EfficientNetB0
==================================================================================================
Total params: 5,330,571
Trainable params: 5,288,548
Non-trainable params: 42,023
__________________________________________________________________________________________________
314
213
"""

model.trainable = True

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))
