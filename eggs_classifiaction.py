from torchvision import datasets, models, transforms
import time
import timm
from torch import nn
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
global model

is_model_init = False
train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.CenterCrop(height=256, width=256),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def init_model():
   #device = torch.device("cpu")
   is_model_init = True
   if torch.cuda.is_available():
       map_location = 'cuda'
   else:
       map_location = 'cpu'

   model = timm.create_model('resnet18d', pretrained=True)
   model.fc = nn.Linear(512, 4)
   # model.to(device)
   model.load_state_dict(torch.load('models/eggs.pth', map_location=map_location))

def process_tensor(str_tensor):
    num_class=str_tensor[-3]
    if num_class=='0':
        return 'Яйца недожаренные('
    if num_class=='1':
        return 'Яйца разбитые('
    if num_class=='2':
        return 'Ваши яйца идеальные!'
    if num_class=='3':
        return 'Яйца пережаренные('

def process_img(path):
  #  if not is_model_init:
   #     init_model()
    if torch.cuda.is_available():
        map_location = 'cuda'
    else:
        map_location = 'cpu'

    model = timm.create_model('resnet18d', pretrained=True)
    model.fc = nn.Linear(512, 4)
  # model.to(device)
    model.load_state_dict(torch.load(os.path.join('models/eggs.pth'), map_location=map_location))

    test_image = cv2.imread(path)
    model.zero_grad()
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    # test_image = np.array(test_image).reshape( -1, 256, 256, 1)
    test_image = test_transforms(image=test_image)["image"]
    test_image = test_image.unsqueeze(0)
  #  test_image = test_image.to(device)
    prediction = model(test_image)
    os.remove(path)
    return process_tensor(str(torch.argmax(prediction, dim=-1)))

