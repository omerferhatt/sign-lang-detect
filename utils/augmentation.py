import numpy as np
import cv2
import albumentations as A


train_transforms = A.Compose([
    A.Rotate((-20, 20), always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.IAAPiecewiseAffine(p=0.3),
    ], p=0.2),
    A.OneOf([
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
    A.ToFloat()
])


img = cv2.imread('../data/Train/A/A002.png', 1)
img = cv2.resize(img, (224, 224))

aug_data = train_transforms(image=img)
print(np.max(aug_data['image']))
