import numpy as np
import torch



class L2NormLoss_test():
    '''
    gt: B, 66, 25
    '''
    def loss(self, gt, out, frame_ids):  # (batch size,feature dim, seq len)
        t_3d = np.zeros(len(frame_ids))

        batch_size, features, seq_len = gt.shape
        gt = gt.permute(0, 2, 1).contiguous().view(batch_size, seq_len, -1, 3) # B, 25, 22, 3
        out = out.permute(0, 2, 1).contiguous().view(batch_size, seq_len, -1, 3) # B, 25, 22, 3
        for k in np.arange(0, len(frame_ids)):
            j = frame_ids[k]
            t_3d[k] = torch.mean(torch.norm(gt[:, j, :, :].contiguous().view(-1, 3) - out[:, j, :, :].contiguous().view(-1, 3), 2, 1)).cpu().data.numpy() * batch_size
        return t_3d

# def L2NormLoss_train(gt, out):
#     '''
#     # (batch size,feature dim, seq len)
#     等同于 mpjpe_error_p3d()
#     '''
#     batch_size, _, seq_len = gt.shape
#     gt = gt.view(batch_size, -1, 3, seq_len).permute(0, 3, 1, 2).contiguous()
#     out = out.view(batch_size, -1, 3, seq_len).permute(0, 3, 1, 2).contiguous()
#     # x = torch.norm(gt - out, 2, dim=-1)
#     # print(x.shape) # 想看一下torch的二范数损失是怎么做的
#     loss = torch.mean(torch.norm(gt - out, 2, dim=-1)) # 原本是2范数损失。调成1范数试试，等下再试试3范数，一个三维
#     return loss

class L2NormLoss_train():

    def loss(self, gt, out):
        batch_size, _, seq_len = gt.shape
        gt = gt.view(batch_size, -1, 3, seq_len).permute(0, 3, 1, 2).contiguous()
        out = out.view(batch_size, -1, 3, seq_len).permute(0, 3, 1, 2).contiguous()
        loss = torch.mean(torch.norm(gt - out, 2, dim=-1))
        return loss


