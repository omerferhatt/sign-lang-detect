import cv2
import albumentations as A


train_transforms = A.Compose([
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    A.RGBShift(always_apply=True, r_shift_limit=0.1, g_shift_limit=0.1, b_shift_limit=0.1),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=1., interpolation=cv2.INTER_LINEAR),
    A.ElasticTransform(alpha=1.0, sigma=20., alpha_affine=20.),
    A.Downscale(scale_min=0.7, scale_max=0.99),
    A.MotionBlur(blur_limit=15),
    A.GaussNoise(var_limit=0.05)
])

val_transforms = A.Compose([
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
])
