import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        ca = self.ca(x)
        out = x * ca
        sa = self.sa(out)
        out = out * sa
        return out + residual, ca, sa


#in this code, we will use gcn and global
# 生成邻接矩阵
class GATENet(nn.Module):
    def __init__(self, inc, reduction_ratio=128):
        super(GATENet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(inc, inc // reduction_ratio, bias=False),
                                nn.ELU(inplace=False),
                                nn.Linear(inc // reduction_ratio, inc, bias=False),
                                nn.Tanh(),
                                nn.ReLU(inplace=False))

    def forward(self, x):
        y = self.fc(x)
        return y


class resGCN(nn.Module):
    def __init__(self, inc, outc, band_num):
        super(resGCN, self).__init__()
        self.GConv1 = nn.Conv2d(in_channels=inc,
                                out_channels=outc,
                                kernel_size=(1, 3),
                                stride=(1, 1),
                                padding=(0, 0),
                                groups=band_num,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.GConv2 = nn.Conv2d(in_channels=outc,
                                out_channels=outc,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 1),
                                groups=band_num,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.ELU = nn.ELU(inplace=False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_p, L):
        x = self.bn2(self.GConv2(self.ELU(self.bn1(self.GConv1(x)))))
        y = torch.einsum('bijk,kp->bijp', (x, L))
        y = self.ELU(torch.add(y, x_p))
        return y


class HGCN(nn.Module):
    def __init__(self, dim, chan_num, band_num):
        super(HGCN, self).__init__()
        self.chan_num = chan_num
        self.dim = dim
        self.resGCN = resGCN(inc=dim * band_num,
                             outc=dim * band_num, band_num=band_num)
        self.ELU = nn.ELU(inplace=False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for j in m:
                    if isinstance(j, nn.Linear):
                        nn.init.xavier_uniform_(j.weight, gain=1)

    def forward(self, x, A_ds):
        L = torch.einsum('ik,kp->ip', (A_ds, torch.diag(torch.reciprocal(sum(A_ds)))))
        G = self.resGCN(x, x, L).contiguous()
        return G

class MHGCN(nn.Module):
    def __init__(self, layers, dim, chan_num, band_num, hidden_1, hidden_2):
        super(MHGCN, self).__init__()
        self.chan_num = chan_num
        self.band_num = band_num
        self.A = torch.rand((1, self.chan_num * self.chan_num), dtype=torch.float32, requires_grad=False)
        self.GATENet = GATENet(self.chan_num * self.chan_num, reduction_ratio=128)
        self.HGCN_layers = nn.ModuleList()
        for i in range(layers):
            self.HGCN_layers.append(HGCN(dim=1, chan_num=self.chan_num, band_num=self.band_num))

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for j in m:
                    if isinstance(j, nn.Linear):
                        nn.init.xavier_uniform_(j.weight, gain=1)

    def forward(self, x):
        self.A = self.A.to(x.device)
        A_ds = self.GATENet(self.A)
        A_ds = A_ds.reshape(self.chan_num, self.chan_num)
        output = []
        output.append(x)
        for i in range(len(self.HGCN_layers)):
            input = x
            output.append(self.HGCN_layers[i](input, A_ds))
            x = output[-1]
        out = torch.cat(output, dim=1)
        return out, A_ds


class Encoder(nn.Module):
    def __init__(self, in_planes=[5, 62], layers=2, hidden_1=256, hidden_2=64, class_nums=3):
        super(Encoder, self).__init__()
        self.chan_num = in_planes[1]
        self.band_num = in_planes[0]
        self.GGCN = MHGCN(layers=layers, dim=1, chan_num=self.chan_num, band_num=self.band_num, hidden_1=hidden_1,
                          hidden_2=hidden_2)

        self.CBAM =CBAMBlock(channel=(layers + 1) * self.band_num, reduction=4, kernel_size=3)
        self.fc1 = nn.Linear(self.chan_num * (layers + 1) * self.band_num, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.reshape(x.size(0), 5, 62)
        x = x.unsqueeze(2)
        g_feat, g_adj = self.GGCN(x)
        g_feat, ca, sa = self.CBAM(g_feat)
        out = self.fc1(g_feat.reshape(g_feat.size(0), -1))
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        return out, [g_adj, ca, sa]



class ClassClassifier(nn.Module):
    def __init__(self, hidden_2, num_cls):
        super(ClassClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_2, num_cls)

    def forward(self, x):
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_1):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Domain_adaption_model(nn.Module):
    def __init__(self, in_planes=[5, 62], layers=2, hidden_1=256, hidden_2=64, num_of_class=3, device='cuda:1', source_num=3944):
        super(Domain_adaption_model, self).__init__()
        self.encoder = Encoder(in_planes=in_planes, layers=layers, hidden_1=hidden_1, hidden_2=hidden_2, class_nums=num_of_class )
        self.cls_classifier = ClassClassifier(hidden_2=hidden_2, num_cls=num_of_class)
        self.source_f_bank = torch.randn(source_num, hidden_2)
        self.source_score_bank = torch.randn(source_num, num_of_class).to(device)
        self.num_of_class = num_of_class
        self.ema_factor = 0.8

    def forward(self, source, target, source_label, source_index):
        source_f, [self.src_adj, self.src_sa, self.src_ca] = self.encoder(source)
        target_f, [self.tar_adj, self.tar_sa, self.tar_ca] = self.encoder(target)

        source_predict = self.cls_classifier(source_f)
        target_predict = self.cls_classifier(target_f)

        source_label_feature = torch.nn.functional.softmax(source_predict, dim=1)
        target_label_feature = torch.nn.functional.softmax(target_predict, dim=1)

        target_label = self.get_target_labels(source_f, source_label_feature, source_index, target_f)
        return source_predict, source_f, target_predict, target_f, [self.src_adj, self.src_sa, self.src_ca], [self.tar_adj, self.tar_sa, self.tar_ca], target_label



    def get_target_labels(self, feature_source_f, source_label_feature, source_index, feature_target_f):
        self.eval()
        output_f = torch.nn.functional.normalize(feature_source_f)
        self.source_f_bank[source_index] = output_f.detach().clone().cpu()
        self.source_score_bank[source_index] = source_label_feature.detach().clone()

        output_f_ = torch.nn.functional.normalize(feature_target_f).cpu().detach().clone()
        distance = output_f_ @ self.source_f_bank.T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=7)
        score_near = self.source_score_bank[idx_near]  # batch x K x num_class
        score_near_weight = self.get_weight(score_near)
        score_near_sum_weight = torch.einsum('ijk,ij->ik', score_near, score_near_weight)
        # score_near_sum_weight = torch.mean(score_near, dim=1)  # batch x num_class
        target_predict = torch.nn.functional.softmax(score_near_sum_weight, dim=1)
        return target_predict

    def get_init_banks(self, source, source_index):
        self.eval()
        source_f, source_att = self.encoder(source)

        source_predict = self.cls_classifier(source_f)
        source_label_feature = torch.nn.functional.softmax(source_predict, dim=1)

        self.source_f_bank[source_index] = torch.nn.functional.normalize(source_f).detach().clone().cpu()
        self.source_score_bank[source_index] = source_label_feature.detach().clone()

    def target_predict(self, feature_target):
        self.eval()
        target_f, _ = self.encoder(feature_target)
        target_predict = self.cls_classifier(target_f)
        target_label_feature = torch.nn.functional.softmax(target_predict, dim=1)
        return target_label_feature

    def domain_discrepancy(self, out1, out2, loss_type):
        def huber_loss(e, d=1):
            t = torch.abs(e)
            ret = torch.where(t < d, 0.5 * t ** 2, d * (t - 0.5 * d))
            return torch.mean(ret)

        diff = out1 - out2
        if loss_type == 'L1':
            loss = torch.mean(torch.abs(diff))
        elif loss_type == 'Huber':
            loss = huber_loss(diff)
        else:
            loss = torch.mean(diff * diff)
        return loss

    def get_weight(self, score_near):
        epsilon = 1e-5
        entropy = -(1/score_near.size(1))*torch.sum(score_near*torch.log(score_near+ epsilon), dim=2)
        g = 1 - entropy
        score_near_weight = g / torch.tile(torch.sum(g, dim=1).view(-1, 1), (1, score_near.size(1)))
        return score_near_weight


    def Entropy(self, input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy