import torch
import torch.nn as nn
from .layers import SingleLeftLinear, SingleRightLinear, PreGCN, GC_Block, PostGCN
from feeder.MSRGCN.datas.dct import reverse_dct_torch


class STDecoderLayer(nn.Module):
    def __init__(self, p_dropout=0, leaky_c=0.2, final_out_noden=22, input_feature=35, t_features=256, **dic):
        super(STDecoderLayer, self).__init__()
        self.p_dropout = p_dropout
        self.leaky_c = leaky_c
        self.final_out_noden = final_out_noden
        self.input_feature = input_feature
        self.t_features = t_features
        
        self.feedforward = nn.Sequential(
            GC_Block(in_features=self.final_out_noden*3, p_dropout=self.p_dropout, node_n=t_features, leaky_c=self.leaky_c),
            
        )
        self.attention = nn.MultiheadAttention(embed_dim=self.t_features, num_heads=4, batch_first=True)
        self.first_norm = nn.BatchNorm1d(self.final_out_noden*3)
        self.second_norm = nn.BatchNorm1d(self.final_out_noden*3)
        
    def forward(self, input):
        attention_data, _ = self.attention(input, input, input, need_weights=False, average_attn_weights=False)
        attention_data = attention_data + input
        normed_attention_data = self.first_norm(attention_data)
        normed_attention_data = normed_attention_data.permute(0, 2, 1)
        forward_data = self.feedforward(normed_attention_data)
        forward_data = forward_data + normed_attention_data
        forward_data = forward_data.permute(0, 2, 1)
        normed_forward_data = self.second_norm(forward_data)
        
        return normed_forward_data

class STDecoder(nn.Module):
    def __init__(self, num_layers=6, num_node=22, t_features=256):
        super(STDecoder, self).__init__()
        self.num_layers = num_layers
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            self.decoders.append(
                STDecoderLayer(final_out_noden=num_node, t_features=t_features)
            )
        self.decoders.append(nn.Linear(256, 64))
    
    def forward(self, input):
        output = input
        # print("output.shape", output.shape)
        for i in range(self.num_layers+1):
            output = self.decoders[i](output)
        
        return output



class MSRGCN(nn.Module):
    def __init__(self, p_dropout, leaky_c=0.2, final_out_noden=22, input_feature=35, t_features=256, **dic):
        super(MSRGCN, self).__init__()
        # 左半部分
        self.first_enhance = PreGCN(input_feature=input_feature, hidden_feature=t_features, node_n=final_out_noden * 3,
                                    p_dropout=p_dropout, leaky_c=leaky_c)  # 35, 64, 66, 0.5
        self.first_decoder = STDecoder(num_layers=6, t_features=t_features)
        self.first_left = nn.Sequential(
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),  # 64, 0.5, 66
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
        )

        self.first_down = nn.Sequential(
            SingleLeftLinear(input_feature=final_out_noden * 3, out_features=36, seq_len=64, p_dropout=p_dropout,
                             leaky_c=leaky_c),  # 66, 128, 64
        )

        self.second_enhance = PreGCN(input_feature=64, hidden_feature=128, node_n=36, p_dropout=p_dropout,
                                     leaky_c=leaky_c)
        self.second_left = nn.Sequential(
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
        )

        self.second_down = nn.Sequential(
            SingleLeftLinear(input_feature=36, out_features=21, seq_len=128, p_dropout=p_dropout, leaky_c=leaky_c),
            # 66, 36, 64
        )

        self.third_enhance = PreGCN(input_feature=128, hidden_feature=256, node_n=21, p_dropout=p_dropout,
                                    leaky_c=leaky_c)
        self.third_left = nn.Sequential(
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
        )

        self.third_down = nn.Sequential(
            SingleLeftLinear(input_feature=21, out_features=12, seq_len=256, p_dropout=p_dropout, leaky_c=leaky_c),
            # 66, 36, 64
        )

        self.fourth_enhance = PreGCN(input_feature=256, hidden_feature=512, node_n=12, p_dropout=p_dropout,
                                     leaky_c=leaky_c)
        self.fourth_left = nn.Sequential(
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),  # 64, 0.5, 66
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
        )

        # 右半部分
        self.fourth_right = nn.Sequential(
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
        )
        self.fourth_up = nn.Sequential(
            SingleLeftLinear(input_feature=12, out_features=21, seq_len=512, p_dropout=p_dropout, leaky_c=leaky_c),
            SingleRightLinear(input_feature=512, out_features=256, node_n=21, p_dropout=p_dropout, leaky_c=leaky_c),
        )

        self.third_right_crop = nn.Sequential(
            SingleLeftLinear(input_feature=42, out_features=21, seq_len=256, p_dropout=p_dropout, leaky_c=leaky_c),
        )
        self.third_right = nn.Sequential(
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
        )
        self.third_up = nn.Sequential(
            SingleLeftLinear(input_feature=21, out_features=36, seq_len=256, p_dropout=p_dropout, leaky_c=leaky_c),
            SingleRightLinear(input_feature=256, out_features=128, node_n=36, p_dropout=p_dropout, leaky_c=leaky_c)
        )

        self.second_right_crop = nn.Sequential(
            SingleLeftLinear(input_feature=72, out_features=36, seq_len=128, p_dropout=p_dropout, leaky_c=leaky_c),
        )
        self.second_right = nn.Sequential(
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
        )
        self.second_up = nn.Sequential(
            SingleLeftLinear(input_feature=36, out_features=final_out_noden * 3, seq_len=128, p_dropout=p_dropout,
                             leaky_c=leaky_c),
            SingleRightLinear(input_feature=128, out_features=64, node_n=final_out_noden * 3, p_dropout=p_dropout,
                              leaky_c=leaky_c)
        )

        self.first_right_crop = nn.Sequential(
            SingleLeftLinear(input_feature=final_out_noden * 3 * 2, out_features=final_out_noden * 3, seq_len=64,
                             p_dropout=p_dropout, leaky_c=leaky_c),
        )
        self.first_right = nn.Sequential(
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),  # 64, 0.5, 66
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
        )

        # 右边出口部分
        self.first_extra = nn.Sequential(
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
            GC_Block(in_features=64, p_dropout=p_dropout, node_n=final_out_noden * 3, leaky_c=leaky_c),
        )
        self.first_out = PostGCN(input_feature=64, hidden_feature=input_feature, node_n=final_out_noden * 3)

        self.second_extra = nn.Sequential(
            GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            # GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            # GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            # GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            # GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
            # GC_Block(in_features=128, p_dropout=p_dropout, node_n=36, leaky_c=leaky_c),
        )
        self.second_out = PostGCN(input_feature=128, hidden_feature=input_feature, node_n=36)

        self.third_extra = nn.Sequential(
            GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            # GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            # GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            # GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            # GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
            # GC_Block(in_features=256, p_dropout=p_dropout, node_n=21, leaky_c=leaky_c),
        )
        self.third_out = PostGCN(input_feature=256, hidden_feature=input_feature, node_n=21)

        self.fourth_extra = nn.Sequential(
            GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            # GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            # GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            # GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            # GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
            # GC_Block(in_features=512, p_dropout=p_dropout, node_n=12, leaky_c=leaky_c),
        )
        self.fourth_out = PostGCN(input_feature=512, hidden_feature=input_feature, node_n=12)

    def forward(self, inputs):
        '''

        :param x: B, 66, 35
        :return:
        '''
        x_p22 = inputs['p22']
        # print("x_p22", x_p22.shape) # 看一下x_p22的维度
        x_p12 = inputs['p12']
        x_p7 = inputs['p7']
        x_p4 = inputs['p4']

        # 左半部分
        enhance_first_left = self.first_enhance(x_p22)  # B, 66, 64
        # print("enhance_first_left", enhance_first_left.shape)
        decoder_first = self.first_decoder(enhance_first_left) # zjr自己写的注意力变形
        # print("decoder_first", decoder_first.shape)
        out_first_left = self.first_left(decoder_first) + decoder_first  # 残差连接
        second_left = self.first_down(out_first_left)  # 8, 36, 64

        enhance_second_left = self.second_enhance(second_left)  # 8, 36, 128
        out_second_left = self.second_left(enhance_second_left) + enhance_second_left  # 残差连接
        third_left = self.second_down(out_second_left)

        enhance_third_left = self.third_enhance(third_left)  # 8, 21, 256
        out_third_left = self.third_left(enhance_third_left) + enhance_third_left  # 残差连接
        fourth_left = self.third_down(out_third_left)

        enhance_bottom = self.fourth_enhance(fourth_left)  # 8, 12, 512
        bottom = self.fourth_left(enhance_bottom) + enhance_bottom  # 残差连接

        # 右半部分
        bottom_right = self.fourth_right(bottom) + bottom  # 残差连接

        in_third_right = self.fourth_up(bottom_right)
        cat_third = torch.cat((out_third_left, in_third_right), dim=-2)
        crop_third_right = self.third_right_crop(cat_third)
        third_right = self.third_right(crop_third_right) + crop_third_right  # 残差连接

        in_second_right = self.third_up(third_right)
        cat_second = torch.cat((out_second_left, in_second_right), dim=-2)
        crop_second_right = self.second_right_crop(cat_second)
        second_right = self.second_right(crop_second_right) + crop_second_right  # 残差连接

        in_first_right = self.second_up(second_right)
        # print("out_first_left", out_first_left.shape)
        cat_first = torch.cat((out_first_left, in_first_right), dim=-2)
        # print("cat_first", cat_first.shape)
        crop_first_right = self.first_right_crop(cat_first)
        first_right = self.first_right(crop_first_right) + crop_first_right  # 残差连接

        # 出口部分
        fusion_first = self.first_extra(first_right) + first_right  # 残差连接
        pred_first = self.first_out(fusion_first) + x_p22  # 大残差连接

        fusion_second = self.second_extra(second_right) + second_right  # 残差连接
        pred_second = self.second_out(fusion_second) + x_p12  # 大残差连接

        fusion_third = self.third_extra(third_right) + third_right  # 两重残差连接
        pred_third = self.third_out(fusion_third) + x_p7  # 大残差连接

        fusion_fourth = self.fourth_extra(bottom_right) + bottom_right  # 残差连接
        pred_fourth = self.fourth_out(fusion_fourth) + x_p4  # 大残差连接

        # outputs = {"p22": pred_first, "p12": pred_second, "p7": pred_third, "p4": pred_fourth}


        return {
            "p22": pred_first, "p12": pred_second, "p7": pred_third, "p4": pred_fourth
            # "out_p22": pred_first

        }

# 判断一下batch中的动作样本是高速还是低速
class speed_class(nn.Module):
    def __init__(self, original_speed = 0.0007):
        super(speed_class, self).__init__()
        # 设置判定用的速度，肉眼判断中线为0.0007左右
        self.original_speed = original_speed
    
    def speed_getting(self, x):
        # batch_size, skeleton_num, frames = x.shape
        x_speed = x[:, :, 1:10] - x[:, :, 0:9]
        return x_speed

    def forward(self, inputs):
        x_p22 = inputs['p22'][:, :, 0:10] # B, V*N(66), 10(frames)
        x_p12 = inputs['p12'][:, :, 0:10]
        x_p7 = inputs['p7'][:, :, 0:10]
        x_p4 = inputs['p4'][:, :, 0:10]
        # x = torch.stack([x_p22, x_p12, x_p7, x_p4], dim=0) # 将矩阵拼接在一起方便后面矩阵计算提高速度，用不了，因为这几个维度对不齐

        # 求小批量样本中的每个样本的高低速判定
        x_p22_speed = self.speed_getting(x_p22) # B, V*N(66), 9(frames)
        x_p22_mean_speed = torch.mean(torch.mean(x_p22_speed, dim = 2, keepdim=True), dim = 1, keepdim=True) # 是一个有B个数值的1维数组，记录每一个样本的平均速度（骨骼点和时间都平均）,根据之前的统计结果，用绝对值0.0007来做高速低速的界限
        fast_ls = x_p22_mean_speed >= self.original_speed
        slow_ls = x_p22_mean_speed < self.original_speed
        
        x_p22_fast = x_p22 * fast_ls
        x_p22_slow = x_p22 * slow_ls
        x_p12_fast = x_p12 * fast_ls
        x_p12_slow = x_p12 * slow_ls
        x_p7_fast = x_p7 * fast_ls
        x_p7_slow = x_p7 * slow_ls
        x_p4_fast = x_p4 * fast_ls
        x_p4_slow = x_p4 * slow_ls

        # 用高速和低速做限制去mark，用不了，因为维度对不齐
        # x_fast = x * fast_ls
        # x_slow = x * slow_ls

        # 将mark后的数据高速低速数据分别存入字典中
        fast = {"p22":x_p22_fast, "p12":x_p12_fast, "p7":x_p7_fast, "p4":x_p4_fast}
        slow = {"p22":x_p22_slow, "p12":x_p12_slow, "p7":x_p7_slow, "p4":x_p4_slow}
        # fast = {"p22":x_fast[0], "p12":x_fast[1], "p7":x_fast[2], "p4":x_fast[3]}
        # slow = {"p22":x_slow[0], "p12":x_slow[1], "p7":x_slow[2], "p4":x_slow[3]}

        return fast, slow

# 创建两个MSRGCN流，一个走高速一个走低速，使用speed_class中mark过后的数据进行两个流的训练
class speed_flow(nn.Module):
    def __init__(self, original_speed = 0.0007):
        super(speed_flow, self).__init__()
        self.fast_GCN = MSRGCN(p_dropout = 0.05, leaky_c=0.2, final_out_noden=22, input_feature=10)
        self.slow_GCN = MSRGCN(p_dropout = 0.05, leaky_c=0.2, final_out_noden=22, input_feature=10)
        self.speed_class = speed_class(original_speed)
        
    def forward(self, inputs):
        input_fast, input_slow = self.speed_class(inputs)
        fast_x = self.fast_GCN(input_fast)
        slow_x = self.slow_GCN(input_slow)
        for key, value in fast_x.items():
            fast_x[key] += slow_x[key]
        
        return fast_x


if __name__ == "__main__":
    m = speed_flow(0.0007).cuda()
    print(">>> total params: {:.2f}M\n".format(sum(p.numel() for p in m.parameters()) / 1000000.0))
    pass