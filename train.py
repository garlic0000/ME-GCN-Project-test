import torch
import torch.nn as nn
import os
import numpy as np
from datasets import LOSO_DATASET
from model import AUwGCN
from torch.utils.tensorboard import SummaryWriter
from utils.train_utils import configure_optimizers
from utils.loss_func import _probability_loss, MultiCEFocalLoss_New
from functools import partial
import argparse
import yaml
import multiprocessing
import warnings


def same_seeds(seed):
    """
    为模型的训练环境设置随机种子，以确保结果的可复现性
    """
    torch.manual_seed(seed)  # 设置 PyTorch 的全局随机种子，以确保在 CPU 上生成的随机数具有一致性，从而在相同的种子下能得到相同的结果
    if torch.cuda.is_available():  # 检查是否有可用的 GPU，如果有，继续设置 GPU 的随机种子
        torch.cuda.manual_seed(seed)  # 设置当前 GPU 的随机种子
        torch.cuda.manual_seed_all(seed)  # 如果有多张 GPU 可用，设置所有 GPU 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子，保证使用 NumPy 生成的随机数在每次运行时一致
    torch.backends.cudnn.enabled = True  # 启用 cuDNN 库（适用于 NVIDIA 的 GPU），以加速深度学习任务中的卷积运算
    torch.backends.cudnn.benchmark = True  # 启用 cuDNN 的 benchmark 模式，通常在输入数据尺寸一致的情况下，可以提供更高的速度
    torch.backends.cudnn.deterministic = True  # 设置 cuDNN 的确定性模式，以保证在相同的输入和种子下产生相同的结果。需要注意的是，将 deterministic 设为 True 可能会降低训练速度。


same_seeds(1)  # 调用 same_seeds(1) 设置种子为 1，以便保证实验的可重复性

loss_list = []  # 用于保存训练过程中的损失值，以便后续进行分析或绘图


# keep track of statistics
class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    def avg(self):
        return self.sum / self.count


def train(opt, data_loader, model, optimizer, epoch, device, writer):
    # 训练模式
    model.train()
    # 损失率积累器
    loss_am = AverageMeter()

    # 用于二分类的损失函数
    bi_loss_apex = partial(_probability_loss, gamma=opt["abfcm_apex_gamma"],
                           alpha=opt["abfcm_apex_alpha"],
                           lb_smooth=opt["abfcm_label_smooth"])

    bi_loss_action = partial(_probability_loss,
                             gamma=opt["abfcm_action_gamma"],
                             alpha=opt["abfcm_action_alpha"],
                             lb_smooth=opt["abfcm_label_smooth"])

    # 用于三分类的损失函数
    _tmp_alpha = opt["abfcm_start_end_alpha"]
    cls_loss_func = MultiCEFocalLoss_New(
        class_num=3,
        alpha=torch.tensor(
            [_tmp_alpha / 2, _tmp_alpha / 2, 1 - _tmp_alpha],
            dtype=torch.float32),
        gamma=opt["abfcm_start_end_gama"],
        # lb_smooth=0.06,
    )

    # 循环训练
    for batch_idx, (feature, micro_apex_score, macro_apex_score,
                    micro_action_score, macro_action_score,
                    micro_start_end_label, macro_start_end_label
                    ) in enumerate(data_loader):
        # forward pass
        b, t, n, c = feature.shape
        feature = feature.to(device)

        micro_apex_score = micro_apex_score.to(device)
        macro_apex_score = macro_apex_score.to(device)
        micro_action_score = micro_action_score.to(device)
        macro_action_score = macro_action_score.to(device)
        micro_start_end_label = micro_start_end_label.to(device)
        macro_start_end_label = macro_start_end_label.to(device)

        STEP = int(opt["RECEPTIVE_FILED"] // 2)

        # 获取模型预测
        output_probability = model(feature)
        output_probability = output_probability[:, :, STEP:-STEP]

        output_micro_apex = output_probability[:, 6, :]
        output_macro_apex = output_probability[:, 7, :]
        output_micro_action = output_probability[:, 8, :]
        output_macro_action = output_probability[:, 9, :]

        output_micro_start_end = output_probability[:, 0: 0 + 3, :]
        output_macro_start_end = output_probability[:, 3: 3 + 3, :]

        # 计算损失 二分类损失
        loss_micro_apex = bi_loss_apex(output_micro_apex,
                                       micro_apex_score)

        loss_macro_apex = bi_loss_apex(output_macro_apex,
                                       macro_apex_score)
        loss_micro_action = bi_loss_action(output_micro_action,
                                           micro_action_score)
        loss_macro_action = bi_loss_action(output_macro_action,
                                           macro_action_score)
        # 计算损失 三分类损失

        loss_micro_start_end = cls_loss_func(
            output_micro_start_end.permute(0, 2, 1).contiguous(),
            micro_start_end_label)
        loss_macro_start_end = cls_loss_func(
            output_macro_start_end.permute(0, 2, 1).contiguous(),
            macro_start_end_label)

        # 加权的聚合损失
        loss = (1.8 * loss_micro_apex
                + 1.0 * loss_micro_start_end
                + 0.1 * loss_micro_action
                + opt['macro_ration'] * (
                        1.0 * loss_macro_apex
                        + 1.0 * loss_macro_start_end
                        + 0.1 * loss_macro_action
                ))

        # update step
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新损失
        loss_am.update(loss.detach())
        writer.add_scalar("Loss/train", loss, epoch)
    current_lr = optimizer.param_groups[0]['lr']
    results = "[Epoch {0:03d}/{1:03d}]\tLoss {2:.5f}(train)\tCurrent Learning rate {3:.5f}\n".format(
        epoch, opt["epochs"], loss_am.avg(), current_lr)

    print(results)

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}

    ckpt_dir = opt["model_save_root"]

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    weight_file = os.path.join(
        ckpt_dir,
        "checkpoint_epoch_" + str(epoch).zfill(3) + ".pth.tar")

    # save state_dict every x epochs to save memory
    if (epoch + 1) % opt['save_intervals'] == 0:
        torch.save(state, weight_file)
    print("weight file save in {0}/checkpoint_epoch_{1}.pth.tar\n".format(ckpt_dir, str(epoch).zfill(3)))


if __name__ == '__main__':
    from pprint import pprint
    import opts
    # # /opt/conda/lib/python3.10/multiprocessing/popen_fork.py: 66: RuntimeWarning: os.fork()
    # # was called.os.fork() is incompatible
    # # with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
    # # self.pid = os.fork()
    #
    # # 使用spawn而不是fork: Python的multiprocessing库默认使用fork来创建子进程
    # # 但在多线程应用中可能会出现问题。你可以通过设置multiprocessing的启动方式为spawn来避免这种问题。
    # multiprocessing.set_start_method("spawn")
    # 使用spawn太慢了
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork() was called.")
    args = opts.parse_args()

    # prep output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # load config & params.
    with open("/kaggle/working/ME-GCN-Project/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        if args.dataset is not None:
            dataset = args.dataset
        else:
            dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
        opt['dataset'] = dataset
    subject = args.subject

    # update opt. according to args.
    opt['output_dir_name'] = os.path.join(args.output, subject)
    opt['model_save_root'] = os.path.join(opt['output_dir_name'], 'models')

    # tensorboard writer
    writer_dir = os.path.join(opt['output_dir_name'], 'logs')
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    tb_writer = SummaryWriter(writer_dir)

    # save the current config
    with open(os.path.join(writer_dir, 'config.txt'), 'w') as fid:
        pprint(opt, stream=fid)
        fid.flush()

    # prep model
    device = opt['device'] if torch.cuda.is_available() else 'cpu'
    model = AUwGCN(opt)
    model = model.to(device)
    print("Starting training...\n")
    print("Using GPU: {} \n".format(device))

    # define dataset and dataloader
    train_dataset = LOSO_DATASET(opt, "train", subject)
    # 训练数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt['batch_size'],
                                               shuffle=True,
                                               num_workers=opt['num_workers'])

    # # define optimizer and scheduler
    optimizer = configure_optimizers(model, opt["abfcm_training_lr"],
                                     opt["abfcm_weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, opt['abfcm_lr_scheduler'])

    for epoch in range(opt['epochs']):
        train(opt, train_loader, model, optimizer, epoch, device, tb_writer)
        scheduler.step()

    tb_writer.close()
    print("Finish training!\n")

