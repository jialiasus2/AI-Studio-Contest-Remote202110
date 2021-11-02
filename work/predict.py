import os
import cv2
from tqdm import tqdm
import numpy as np
import paddle

from my_dataset import MyDataset, save_img
from my_model import make_model
from configs import MODEL_PATH, TEST_DIR, SAVE_DIR, BATCH_SIZE

def post_process(result):
    '''
    result: 2*H*W
    '''
    if result.shape[0]<=3:
        res = cv2.GaussianBlur(result.transpose([1,2,0]),(5,5),1).transpose([2, 0, 1])
    else:
        res = cv2.GaussianBlur(result,(5,5),1)
    return res


if __name__=='__main__':
    test_ids = [f[:-4] for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    test_dataset = MyDataset(test_ids, TEST_DIR, None, argument=False)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    print(len(test_dataset))

    model = make_model()
    params = paddle.load(os.path.join(MODEL_PATH, 'model.pdparams'))
    print('Load model')
    model.set_state_dict(params)

    with paddle.no_grad():
        model.eval()
        fi = 0
        for X in tqdm(test_loader):
            Y = model(X)[0]
            Y = paddle.argmax(Y, axis=1).numpy().astype(np.uint8)
            for y in Y:
                save_path = os.path.join(SAVE_DIR, test_ids[fi]+'.png')
                save_img(save_path, y)
                fi += 1
    # os.system('zip -qr result.zip result/')
    pass
