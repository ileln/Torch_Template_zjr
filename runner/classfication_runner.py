import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import datetime
import wandb

from tools import import_class, lr_decay

# 运行器
class Runner():
    def __init__(self, args, **dic):
        super(Runner, self).__init__()

        # 参数
        self.args = args
        self.start_epoch = 1
        self.best_accuracy = 0.3
        self.output_n = self.args.output_n
        Model = import_class(self.args.model) # 调用配置文件里设置的模型
        self.model = Model(**vars(self.args)) # 模型初始化

        if not self.args.device == None:
            self.model.cuda()
        
        print(">>> total params: {:.2f}M\n".format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))

        # optimizer
        self.lr = self.args.lr
        Optimizer = import_class(self.args.optimizer) # 调用配置文件里设置的优化器
        self.optimizer = Optimizer(self.model.parameters(), lr = self.lr) # 将模型参数放入优化器中
        Train_loss = import_class(self.args.train_loss)
        self.train_loss = Train_loss().cuda()
        Test_loss = import_class(self.args.test_loss)
        self.test_loss = Test_loss().cuda()

        # train_data
        DataFeeder = import_class(self.args.feeder)
        self.train_dataset = DataFeeder(**vars(self.args), mode_name="train")
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)

        # test_data
        self.test_dataset = DataFeeder(**vars(self.args), mode_name="test")
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.args.test_batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        
        # tensorboard
        self.summary = SummaryWriter(self.args.work_dir)
    
    def save(self, checkpoint_path, best_err, curr_err):
        state = {
            "lr": self.lr,
            "best_err": best_err,
            "curr_err": curr_err,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)
    

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.args.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr = state["lr"]
        best_err = state['best_err']
        curr_err = state["curr_err"]
        print("load lr {}, curr_avg {}, best_avg {}.".format(state["lr"], curr_err, best_err))

    def train(self, epoch):
        self.model.train()
        average_loss = 0
        acc_value = []
        for batch_idx, (data, label, index) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=40):
            b, t, v, c = data.shape
            # skip the last batch if only have one sample for batch_norm layers
            if b == 1:
                continue

            self.global_step = (epoch - 1) * len(self.train_loader) + batch_idx + 1

            
            data = data.float().cuda(non_blocking=True, device=self.args.device)
            label = label.cuda(non_blocking=True, device=self.args.device)

            output = self.model(data) # B * 66 * 35

            # loss
            loss_curr = self.train_loss.forward(output, label)
            self.summary.add_scalar(f"Loss", loss_curr, self.global_step)
            wandb.log({f"Loss_train": loss_curr, "global_step": self.global_step}) # 添加wandb.log

            self.optimizer.zero_grad()
            loss_curr.backward()
            self.optimizer.step()

            average_loss += loss_curr.cpu().data.numpy()
            
            value, predict_label = torch.max(output, 1)
            acc = torch.mean((predict_label == label).float())
            acc_value.append(acc.data.item())
            self.summary.add_scalar('acc_epoch_train', acc, self.global_step)
            wandb.log({f"acc_epoch_train": acc}) # 添加wandb.log

        
        average_loss /= (batch_idx + 1)
        average_acc = np.mean(acc_value)
        return average_loss, average_acc
    
    def test(self, epoch=0):
        self.model.train()
        average_loss = 0
        acc_value=[]
        for batch_idx, (data, label, index) in tqdm(enumerate(self.test_loader), total=len(self.test_loader), ncols=40):
            b, t, v, c = data.shape
            # skip the last batch if only have one sample for batch_norm layers
            if b == 1:
                continue

            self.global_step = (epoch - 1) * len(self.train_loader) + batch_idx + 1

            data = data.float().cuda(non_blocking=True, device=self.args.device)
            label = label.cuda(non_blocking=True, device=self.args.device)

            output = self.model(data)

            # loss
            loss_curr = self.test_loss.forward(output, label)
            self.summary.add_scalar(f"Loss", loss_curr, self.global_step)
            wandb.log({f"Loss_test": loss_curr}) # 添加wandb.log

            average_loss += loss_curr.cpu().data.numpy()
            
            value, predict_label = torch.max(output, 1)
            acc = torch.mean((predict_label == label).float())
            acc_value.append(acc.data.item())
            self.summary.add_scalar('acc_epoch_test', acc, self.global_step)
            wandb.log({f"acc_epoch_test": acc}) # 添加wandb.log

        average_loss /= (batch_idx + 1)
        average_acc = np.mean(acc_value)
        return average_loss, average_acc
    
    def run(self):
        for epoch in range(self.start_epoch, self.args.n_epoch + 1):

            if epoch % 2 == 0:
                self.lr = lr_decay(self.optimizer, self.lr, self.args.lr_decay)
            self.summary.add_scalar("LR", self.lr, epoch)
            wandb.log({"LR": self.lr, "epoch": epoch})

            average_train_loss, average_train_acc = self.train(epoch)
            wandb.log({"average_train_loss":average_train_loss, "average_train_acc":average_train_acc})
            print('TRAIN: Epoch: {},  LR: {}, average_train_loss: {}, average_test_acc: {}'.format(epoch, self.lr, average_train_loss, average_train_acc))

            if average_train_loss > self.best_accuracy:
                self.best_accuracy = average_train_loss
                if not os.path.exists(os.path.join(self.args.work_dir, "models")):
                    os.makedirs(os.path.join(self.args.work_dir, "models"))
                self.save(os.path.join(self.args.work_dir, "models",
                                 '{}_in{}out{}dctn{}_best_epoch{}_err{:.4f}.pth'.format(self.args.exp_name, self.args.input_n, self.args.output_n, self.args.dct_n, epoch, average_train_loss)), self.best_accuracy, average_train_loss)

            self.save(os.path.join(self.args.work_dir, "models", '{}_in{}out{}dctn{}_last.pth'.format(self.args.exp_name, self.args.input_n, self.args.output_n, self.args.dct_n)), self.best_accuracy, average_train_loss)

            if epoch % 1 == 0:
                average_test_loss, average_test_acc = self.test(epoch)
                wandb.log({"average_test_loss":average_test_loss, "average_test_acc":average_test_acc})
                print('TEST: Epoch: {},  LR: {}, average_test_loss: {}, average_test_acc: {}'.format(epoch, self.lr, average_test_loss, average_test_acc))
