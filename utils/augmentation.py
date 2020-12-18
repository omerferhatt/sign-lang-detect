import cv2
import albumentations as A


transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Normalize(max_pixel_value=255.0),
    A.RandomBrightnessContrast(p=0.7),
    A.Rotate(limit=20, p=1., interpolation=cv2.INTER_LINEAR),
    A.Blur(blur_limit=3, p=0.3),
    A.Downscale(scale_min=0.7, scale_max=0.8, p=0.3)
])
