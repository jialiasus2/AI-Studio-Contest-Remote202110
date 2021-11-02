import os
import sys
import time
from tqdm import tqdm
import numpy as np
import paddle
import visualdl

from my_dataset import MyDataset, get_ids, split_ids
from my_model import MyLoss, make_model
from utils import LogWriter
from configs import BATCH_SIZE, LR, WARMUP_EPOCH, TRAIN_EPOCHS, EVAL_EPOCH, MODEL_PATH

@paddle.no_grad()
def predict_eval(model, loader, cls_th=0.5):
    '''逐batch预测结果，并将结果还原到原始序列长度'''
    model.eval()
    cnt = np.zeros([4,4],dtype=int)
    sys.stdout.flush()
    time.sleep(1)
    for x, yt in tqdm(loader):
        yp = model(x)[0]
        yp = yp.numpy()
        yp = np.argmax(yp, axis=1)
        yt = yt.numpy()
        for i in range(4):
            for j in range(4):
                cnt[i,j] += np.sum((yp==i)*(yt==j))
    miou = 0
    for i in range(4):
        miou += cnt[i,i]/(np.sum(cnt[i,:])+np.sum(cnt[:,i])-cnt[i,i])
    return miou/4


def train(model, opt, loss_fn, train_loader, valid_loader, epochs, save_path=MODEL_PATH, eval_epoch=EVAL_EPOCH):
    print('start train.')
    start_time = time.time()
    best_total = -1
    best_epoch = 0
    log = LogWriter('train.log', clear_pre_content=False)
    vdl_path = os.path.join(save_path, 'vdl_log')
    if os.path.exists(vdl_path):
        os.system('rm -rf %s'%vdl_path)
    vdl_writer = visualdl.LogWriter(vdl_path)
    iters = 0
    for epoch in range(epochs):
        model.train()
        sys.stdout.flush()
        time.sleep(1)
        # lr = 0
        for X, Yt in tqdm(train_loader):
            iters += 1
            Yp = model(X)
            loss = loss_fn(Yp, Yt)
            loss_scalar = loss.numpy()[0]
            if loss_scalar>1:
                loss_scalar = 1+(loss_scalar-1)/10
            vdl_writer.add_scalar('train/loss', loss_scalar, iters)
            vdl_writer.add_scalar('train/learning_rate', opt.get_lr(), iters)
            loss.backward()
            opt.step()
            if isinstance(opt._learning_rate,
                        paddle.optimizer.lr.LRScheduler):
                opt._learning_rate.step()
            opt.clear_grad()
        f = os.path.join(save_path, 'model.pdparams')
        paddle.save(model.state_dict(), f)
        # paddle.save(opt.state_dict(), save_path+'opt.pdopt')
        log('Save to '+f)
        if (epoch+1)%eval_epoch == 0 or epoch+1==epochs:
            miou = predict_eval(model, valid_loader)
            vdl_writer.add_scalar('valid/miou', miou, epoch)
            if miou>best_total:
                f = os.path.join(save_path, 'model_best.pdparams')
                paddle.save(model.state_dict(), f)
                # paddle.save(opt.state_dict(), save_path+'opt.pdopt')
                log('Save to %s with miou = %g'%(f, miou))
                best_total = miou
                best_epoch = epoch+1
            log('epoch_%d, %04.0fs: miou=%g, best f1 is %g at epoch_%d' \
                %(epoch+1, time.time()-start_time, miou, best_total, best_epoch))






if __name__=='__main__':
    np.random.seed(2021)
    paddle.seed(2021)
    model = make_model()

    ids = get_ids()
    train_ids, valid_ids = split_ids(ids)
    print(len(train_ids), len(valid_ids))
    train_dataset = MyDataset(train_ids, argument=True)
    valid_dataset = MyDataset(valid_ids, argument=False)

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)

    batch_per_epoch = len(train_dataset)//BATCH_SIZE
    iters_all = TRAIN_EPOCHS*batch_per_epoch
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(LR, power=0.9, decay_steps=batch_per_epoch*TRAIN_EPOCHS, end_lr=0)
    lr_warmup = paddle.optimizer.lr.LinearWarmup(lr_scheduler, batch_per_epoch*WARMUP_EPOCH, 0, LR)
    opt = paddle.optimizer.Momentum(learning_rate=lr_warmup, momentum=0.9, weight_decay=4e-5, parameters=model.parameters())
    loss_func = MyLoss()

    train(model, opt, loss_func, train_loader, valid_loader, WARMUP_EPOCH+TRAIN_EPOCHS)

