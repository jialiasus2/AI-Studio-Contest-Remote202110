import paddle
import paddleseg

def make_model():
    backbone = paddleseg.models.backbones.HRNet_W48(pretrained='https://bj.bcebos.com/paddleseg/dygraph/hrnet_w48_ssld.tar.gz', has_se=False)
    # backbone = paddleseg.models.backbones.HRNet_W64(has_se=True)
    model = paddleseg.models.OCRNet(num_classes=4,backbone=backbone,backbone_indices=[-1],ocr_mid_channels=512,ocr_key_channels=256, pretrained='https://bj.bcebos.com/paddleseg/dygraph/ccf/fcn_hrnetw48_rs_256x256_160k/model.pdparams')
    return model

class MyLoss(paddle.nn.Layer):
    def __init__(self):
        super(MyLoss, self).__init__()
        # self.dice = paddleseg.models.losses.DiceLoss()
        self.lovasz = paddleseg.models.losses.LovaszSoftmaxLoss()
        self.ce = paddleseg.models.losses.CrossEntropyLoss()

    def forward(self, yp, yt):
        # dice_coef = 0.3
        ce_coef = 1.0
        lovasz_coef = 0.3
        main_loss = lovasz_coef*self.lovasz(yp[0], yt)+ce_coef*self.ce(yp[0], yt)
        soft_loss = lovasz_coef*self.lovasz(yp[1], yt)+ce_coef*self.ce(yp[1], yt)
        return 1.0*main_loss+0.4*soft_loss

if __name__=='__main__':
    # from my_dataset import MyDataset
    import numpy as np
    model = make_model()
    loss_func = MyLoss()
    x = paddle.to_tensor(np.random.uniform(-0.5, 0.5, (8,3,128,128)).astype(np.float32))
    yt = np.random.randint(0, 4, (8,128,128)).astype(np.int32)
    yt[:,[0,127],:] = 255
    yt[:,:,[0,127]] = 255
    yt = paddle.to_tensor(yt)
    model.train()
    yp = model(x)
    print(len(yp), yp[0].shape, yp[1].shape)
    loss = loss_func(yp, yt)
    print(loss)


