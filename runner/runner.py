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
from feeder.MSRGCN.datas import H36MMotionDataset, get_dct_matrix, reverse_dct_torch, define_actions,draw_pic_gt_pred
# from losses.loss import L2NormLoss_train, L2NormLoss_test

# 运行器
class Runner():
    def __init__(self, args, **dic):
        super(Runner, self).__init__()

        # 参数
        self.args = args
        self.start_epoch = 1
        self.best_accuracy = 1e15
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
        self.train_loss = Train_loss()
        Test_loss = import_class(self.args.test_loss)
        self.test_loss = Test_loss()

        # MSRGCN中的DCT变换矩阵
        dct_m, i_dct_m = get_dct_matrix(self.args.seq_len)
        self.dct_m = torch.from_numpy(dct_m).float()
        self.i_dct_m = torch.from_numpy(i_dct_m).float()
        if self.args.device != "cpu":
            self.dct_m = self.dct_m.cuda(non_blocking=True)
            self.i_dct_m = self.i_dct_m.cuda(non_blocking=True)

        # train_data
        DataFeeder = import_class(self.args.feeder)
        self.train_dataset = DataFeeder(**vars(self.args), mode_name="train", action=None)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        self.global_max = self.train_dataset.global_max # MSRGCN加载器下的全局最大最小值
        self.global_min = self.train_dataset.global_min # MSRGCN加载器下的全局最大最小值

        # test_data
        self.test_loader = dict()
        for act in define_actions(self.args.test_manner):
            self.test_dataset = DataFeeder(**vars(self.args), mode_name="test", action=act, global_max=self.global_max, global_min=self.global_min)
            self.test_loader[act] = DataLoader(dataset=self.test_dataset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=
            True)
        
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
        for i, (inputs, gts) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            b, cv, t_len = inputs[list(inputs.keys())[0]].shape
            # skip the last batch if only have one sample for batch_norm layers
            if b == 1:
                continue

            self.global_step = (epoch - 1) * len(self.train_loader) + i + 1

            for k in inputs:
                inputs[k] = inputs[k].float().cuda(non_blocking=True, device=self.args.device)
                gts[k] = gts[k].float().cuda(non_blocking=True, device=self.args.device)

            outputs = self.model(inputs) # B * 66 * 35

            losses = None
            for k in outputs:

                # MSRGCN的数据后续处理
                # 反 Norm
                outputs[k] = (outputs[k] + 1) / 2
                outputs[k] = outputs[k] * (self.global_max - self.global_min) + self.global_min
                # 回转空间
                outputs[k] = reverse_dct_torch(outputs[k], self.i_dct_m, self.args.seq_len)

                # loss
                loss_curr = self.train_loss.loss(gts[k], outputs[k])
                if losses is None:
                    losses = loss_curr
                else:
                    losses = losses + loss_curr
                self.summary.add_scalar(f"Loss/{k}", loss_curr, self.global_step)
                wandb.log({f"Loss/{k}": loss_curr, "global_step": self.global_step}) # 添加wandb.log

            self.optimizer.zero_grad()
            losses.backward()
            # nn.utils.clip_grad_value_(self.model.parameters(), clip_value=100) # 梯度裁剪，看看能不能有效 前向传播有问题 看起来并不能解决问题
            self.optimizer.step()

            average_loss += losses.cpu().data.numpy()

        average_loss /= (i + 1)
        return average_loss
    
    def test(self, epoch=0):
        self.model.eval()

        frame_ids = self.args.frame_ids
        if self.args.output_n == 10:
            frame_ids = frame_ids[0 : 4]
        total_loss = np.zeros((len(define_actions(self.args.test_manner)), len(frame_ids)))

        for act_idx, act in enumerate(define_actions(self.args.test_manner)):
            count = 0

            for i, (inputs, gts) in enumerate(self.test_loader[act]):
                b, cv, t_len = inputs[list(inputs.keys())[0]].shape
                for k in inputs:
                    inputs[k] = inputs[k].float().cuda(non_blocking=True, device=self.args.device)
                    gts[k] = gts[k].float().cuda(non_blocking=True, device=self.args.device)
                with torch.no_grad():
                    outputs = self.model(inputs)

                    # MSRGCN的数据后续处理
                    # 反 Norm
                    for k in outputs:
                        outputs[k] = (outputs[k] + 1) / 2
                        outputs[k] = outputs[k] * (self.global_max - self.global_min) + self.global_min

                        # 回转空间
                        outputs[k] = reverse_dct_torch(outputs[k], self.i_dct_m, self.args.seq_len)

                    # 用原始32个点的数据计算损失
                    mygt = gts['p32'].view(-1, self.args.origin_noden, 3, self.args.seq_len).clone()
                    myout = outputs['p22'].view(-1, self.args.final_out_noden, 3, self.args.seq_len)
                    mygt[:, self.args.dim_used_3d, :, :] = myout
                    mygt[:, self.args.dim_repeat_32, :, :] = myout[:, self.args.dim_repeat_22, :, :]
                    mygt = mygt.view(-1, self.args.origin_noden*3, self.args.seq_len)

                    loss = self.test_loss.loss(gts['p32'][:, :, self.args.input_n:], mygt[:, :, self.args.input_n:], frame_ids)
                    total_loss[act_idx] += loss
                    # count += 1
                    count += mygt.shape[0]
                    # ************ 画图
                    if act_idx == 0 and i == 0:
                        pred_seq = outputs['p22'].cpu().data.numpy()[0].reshape(self.args.final_out_noden, 3, self.args.seq_len)
                        gt_seq = gts['p22'].cpu().data.numpy()[0].reshape(self.args.final_out_noden, 3, self.args.seq_len)
                        if not os.path.exists(os.path.join(self.args.work_dir, "images")):
                            os.makedirs(os.path.join(self.args.work_dir, "images"))
                        for t in range(self.args.seq_len):
                            draw_pic_gt_pred(gt_seq[:, :, t], pred_seq[:, :, t], np.array(self.args.I22_plot), np.array(self.args.J22_plot), np.array(self.args.LR22_plot), os.path.join(self.args.work_dir, "images", f"{epoch}_{act}_{t}.png"))

            total_loss[act_idx] /= count
            for fidx, frame in enumerate(frame_ids):
                self.summary.add_scalar(f"Test/{act}/{frame}", total_loss[act_idx][fidx], epoch)
                wandb.log({f"Test/{act}_{frame}": total_loss[act_idx][fidx]})

        self.summary.add_scalar("Test/average", np.mean(total_loss), epoch)
        wandb.log({"Test_average/total": np.mean(total_loss)})
        for fidx, frame in enumerate(frame_ids):
            self.summary.add_scalar(f"Test/avg{frame}", np.mean(total_loss[:, fidx]), epoch)
            wandb.log({f"Test_average/avg{frame}": np.mean(total_loss[:, fidx])})
        return total_loss
    
    def run(self):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        wandb.init(project=self.args.wandb_project, config=vars(self.args), name=nowtime, save_code=True)
        for epoch in range(self.start_epoch, self.args.n_epoch + 1):

            if epoch % 2 == 0:
                self.lr = lr_decay(self.optimizer, self.lr, self.args.lr_decay)
            self.summary.add_scalar("LR", self.lr, epoch)
            wandb.log({"LR": self.lr, "epoch": epoch})

            average_train_loss = self.train(epoch)
            # # num = 1
            # parameter_file = open("/home/zhangjinrong/program/MSRGCN_zjr/test_parameter/"+str(epoch)+"parameter.txt", 'w')
            # for name, parameters in self.model.named_parameters(): # 打印模型参数
            #     # print(name, ':', parameters)
            #     parameter_file.write(name + ":" + str(parameters) + "\r\n\r\n")
            #     # num += 1
            # parameter_file.close()

            if average_train_loss < self.best_accuracy:
                self.best_accuracy = average_train_loss
                if not os.path.exists(os.path.join(self.args.work_dir, "models")):
                    os.makedirs(os.path.join(self.args.work_dir, "models"))
                self.save(os.path.join(self.args.work_dir, "models",
                                 '{}_in{}out{}dctn{}_best_epoch{}_err{:.4f}.pth'.format(self.args.exp_name, self.args.input_n, self.args.output_n, self.args.dct_n, epoch, average_train_loss)), self.best_accuracy, average_train_loss)

            self.save(os.path.join(self.args.work_dir, "models", '{}_in{}out{}dctn{}_last.pth'.format(self.args.exp_name, self.args.input_n, self.args.output_n, self.args.dct_n)), self.best_accuracy, average_train_loss)

            if epoch % 1 == 0:
                loss_l2_test = self.test(epoch)

                print('Epoch: {},  LR: {}, Current err test avg: {}'.format(epoch, self.lr, np.mean(loss_l2_test)))
        wandb.finish()