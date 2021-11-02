import os
import numpy as np

import cv2
import paddle
from paddle.vision.transforms import ColorJitter, RandomCrop,  RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, Compose

from configs import IMAGE_DIR, LABEL_DIR, TEST_DIR, INPUT_SIZE, BLANK_SAMPLES

def read_img(path):
    img = cv2.imread(path)
    img[:,:,::-1] = img
    return img

def save_img(path, img):
    if len(img.shape)==3:
        img = img[:,:,::-1]
    cv2.imwrite(path, img)

def get_ids(remove_blank=True):
    ids = [f[:-4] for f in os.listdir(LABEL_DIR)]
    if remove_blank:
        ids = [i for i in ids if i not in BLANK_SAMPLES]
    return ids

def split_ids(ids, percent=0.8):
    percent = np.clip(percent, 0.01, 0.99)
    n = len(ids)
    n1 = int(n*percent)
    np.random.shuffle(ids)
    return ids[:n1], ids[n1:]

class MyDataset(paddle.io.Dataset):
    def __init__(self, ids, img_dir=IMAGE_DIR, label_dir=LABEL_DIR, argument=False):
        self.ids = ids
        self.argument = argument
        self.img_dir = img_dir
        self.label_dir = label_dir
        if argument:
            self.jitter = ColorJitter(brightness=0.5, contrast=0.4, saturation=0.1, hue=0.1)
            self.transformer = Compose([
                RandomRotation(degrees=(-90.,90.), fill=(0,0,0,255)),
                RandomCrop(size=INPUT_SIZE), 
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = self.ids[i]
        img = read_img(os.path.join(self.img_dir, fid+'.jpg'))
        label = read_img(os.path.join(self.label_dir, fid+'.png'))[:,:,:1] if self.label_dir is not None else None
        if self.argument:
            img, label = self.argu(img, label)
        img, label = self.pre_process(img, label)
        if label is None:
            return img
        else:
            return img, label

    def argu(self, img, label=None):
        img = self.jitter(img)

        all_data = np.concatenate([img, label], axis=2) if label is not None else img
        all_data = self.transformer(all_data)

        img = all_data[:,:,:3]
        if label is not None:
            label = all_data[:,:,3:]
        return img, label

    def pre_process(self, img, label=None):
        img = img.transpose([2,0,1]).astype(np.float32)/255-0.5
        if label is not None:
            label = label.squeeze().astype(np.int32)
        return img, label


if __name__=='__main__':
    print(paddle.vision.get_image_backend())
    ids = get_ids()
    train_ids, valid_ids = split_ids(ids)
    print(len(train_ids), len(valid_ids))
    dataset = MyDataset(train_ids, argument=False)
    print(len(dataset))
    img, label = dataset[0]
    print(img.shape, img.dtype, label.shape, label.dtype)

    test_ids = [f[:-4] for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    test_dataset = MyDataset(test_ids, TEST_DIR, None, argument=False)
    print(len(test_dataset))
    img = test_dataset[0]
    print(img.shape, img.dtype)

    # loader = paddle.io.DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)
    # img, label = next(iter(loader))
    # print(img.shape, label.shape)



