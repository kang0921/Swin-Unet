import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume, BinaryDiceLoss, JointLoss


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # 設置種子以實現可重復性
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    # 創建用於訓練數據集的 DataLoader
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
    #                          worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # 定義損失函數
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    # --- new add 定義損失函數 ---
    # bce_loss = nn.BCEWithLogitsLoss()
    # dice_loss = nn.BCEWithLogitsLoss()
    # # dice_loss = BinaryDiceLoss()
    # loss_fn = JointLoss(first=dice_loss, second=bce_loss, first_weight=0.5, second_weight=0.5).cuda()
    # ------

    # 優化器
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-3)

    ''' learning rate scheduler
    # learning rate scheduler之後可以加，可以稍微改進
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0 = 2,        # 初始restart的epoch數目
        T_mult = 2,     # 重啟之後因子，也就是每個restart後，T_0 = T_0 * T_mult
        eta_min = 1e-6  # 最低學習率
    )
    '''

    # TensorBoard 寫入器
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs  # 最大訓練周期
    max_iterations = args.max_epochs * len(trainloader)   # 總迭代次數  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0  # 變數用於存儲最佳性能指標
    iterator = tqdm(range(max_epoch), ncols=70) # 使用 tqdm 創建進度條
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print("image_batch.shape", image_batch.shape)
            # print("label_batch.shape", label_batch.shape)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            # print("")
            # print("outputs.shape:", outputs.shape)
            # print("label_batch.shape:", label_batch.shape)
            # print("")
            # 計算損失(這裡的ce_loss = CrossEntropyLoss()常用於多分類，換成BCELoss)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            # outputs = torch.squeeze(outputs)    # squeeze移除張量中尺寸為 1 的維度，用於簡化張量的形狀

            # loss = loss_fn(outputs, label_batch)
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 學習率調整
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            # TensorBoard 日誌
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # 記錄信息
            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            # 將圖片記錄到 TensorBoard
            # if iter_num % 20 == 0:                
            #     image = image_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 10  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            # 在指定的間隔保存模型
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            # 保存最終模型
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"