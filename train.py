import torch, argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from network_rawwwwww import Discriminator, Domain_adaption_model
from Adversarial import DAANLoss
import utils
from utils import create_logger
import numpy as np
from torch.optim.optimizer import Optimizer
from typing import Optional
import random
import math
import os
from sklearn.metrics import confusion_matrix


def set_seed(seed=20):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)


class StepwiseLR_GRL:
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75, max_iter: Optional[float] = 1000):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


def test(test_loader, model, criterion, cuda):
    model.eval()
    correct = 0
    confusion_matrixs =0
    for _, (test_input, label) in enumerate(test_loader):
        test_input, label = test_input.to(args.device), label.to(args.device)
        test_input, label = Variable(test_input), Variable(label)
        output = model.target_predict(test_input)
        loss = criterion(output, label.view(-1))
        _, pred = torch.max(output, dim=1)
        correct += pred.eq(label.data.view_as(pred)).sum()
        confusion_matrixs += confusion_matrix(label.data.squeeze().cpu(), pred.cpu())

    accuracy = float(correct) / len(test_loader.dataset)
    return loss, accuracy, confusion_matrixs


def getInit(train_loader, model):
    model.eval()
    correct = 0
    for _, (tran_input, tran_indx, _ ) in enumerate(train_loader):
        tran_input, tran_indx = tran_input.to(args.device), tran_indx.to(args.device)
        tran_input, tran_indx = Variable(tran_input), Variable(tran_indx)
        model.get_init_banks(tran_input, tran_indx)


class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=3, epsilon=0.05, ):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input, target):
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss



def main(test_id, writer, args):
    set_seed(args.seed)
    # get data1
    target_set, source_set = get_dataset(args.dataset, test_id, args.session)
    souece_sample_num = source_set['feature'].shape[0]
    torch_dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(source_set['feature']),
                                                         torch.arange(souece_sample_num).long(),
                                                         torch.from_numpy(source_set['label']))
    torch_dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(target_set['feature']),
                                                        torch.from_numpy(target_set['label']))
    source_features, source_labels = torch.from_numpy(source_set['feature']), torch.from_numpy(source_set['label'])

    source_loader = torch.utils.data.DataLoader(torch_dataset_train, batch_size=args.batch_size, shuffle=True,
                                                num_workers=2, pin_memory=True)
    target_loader = torch.utils.data.DataLoader(torch_dataset_test, batch_size=args.batch_size, shuffle=True,
                                                num_workers=2, pin_memory=True)
    data_loader_dict = {"source_loader": source_loader, "target_loader": target_loader, "test_loader": target_loader}

    # Create the model
    model = Domain_adaption_model(args.in_planes, args.layers, args.hidden_1, args.hidden_2, args.cls, args.device, souece_sample_num)
    domain_discriminator = Discriminator(args.hidden_2)

    # loss criterion
    criterion = nn.CrossEntropyLoss()
    # Use GPU

    model = model.to(args.device)
    domain_discriminator = domain_discriminator.to(args.device)
    criterion = criterion.to(args.device)
    dann_loss = DAANLoss(domain_discriminator, num_class=3).to(args.device)


    # Optimizer
    optimizer = torch.optim.RMSprop(
        list(model.parameters()) + list(domain_discriminator.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = StepwiseLR_GRL(optimizer, init_lr=args.lr, gamma=10, decay_rate=0.75, max_iter=args.epochs)
    interval = 10
    best_acc = 0

    logger.info("----------Starting training the model----------")
    # Begin training
    model.eval()
    source_index = torch.arange(souece_sample_num).long()
    getInit(data_loader_dict["source_loader"],model)
    del source_features, target_set, torch_dataset_train, torch_dataset_test
    d_g, d_l = 0, 0
    dynamic_factor = 0.8
    for epoch in range(args.epochs):
        model.train()
        dann_loss.train()
        correct = 0
        count = 0

        if epoch % interval == 0:
            test_loss, acc, confusion_matrixs = test(data_loader_dict["test_loader"], model, criterion, args)
            if acc > best_acc:
                best_acc = acc
                os.makedirs(args.output_model_dir+'/'+args.dataset, exist_ok=True)
                torch.save(model, args.output_model_dir+'/'+args.dataset +'/CrossSub_no_nsal_' + str(test_id) + '.pth')
            logger.info("Testing, Epoch: %d, accuracy: %f, best accuracy: %f" % (epoch, acc, best_acc))
            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("test/Accuracy", acc, epoch)
            writer.add_scalar("test/Best_Acc", best_acc, epoch)

        src_examples_iter, tar_examples_iter = enumerate(data_loader_dict["source_loader"]), enumerate(data_loader_dict["target_loader"])
        T = len(data_loader_dict["target_loader"].dataset) // args.batch_size
        for i in range(T):
            _, src_examples = next(src_examples_iter)
            _, tar_examples = next(tar_examples_iter)
            src_data, src_index, src_label_cls = src_examples
            tar_data, _ = tar_examples


            src_data, src_index, src_label_cls = src_data.to(args.device), src_index.to(args.device), src_label_cls.to(args.device).view(-1)
            tar_data = tar_data.to(args.device)

            src_data, src_index, src_label_cls = Variable(src_data), Variable(src_index), Variable(src_label_cls)
            tar_data = Variable(tar_data)

            # encoder model forward
            src_output_cls, src_feature, tar_output_cls, tar_feature, source_att, target_att, tar_label = model(src_data, tar_data, src_label_cls, src_index)
            cls_loss = criterion(src_output_cls, src_label_cls)

            tar_label = torch.argmax(tar_label, dim=1)
            target_loss = criterion(tar_output_cls, tar_label)
            global_transfer_loss= dann_loss(src_feature + 0.005 * torch.randn((src_feature.shape[0], (args.hidden_2 ))).to(args.device),
                                      tar_feature + 0.005 * torch.randn((tar_feature.shape[0], (args.hidden_2 ))).to(args.device),
                                      src_output_cls, tar_output_cls)
            boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-1 * epoch / 1000)) - 1)
            # update joint loss function
            optimizer.zero_grad()
            # d_g = d_g + 2 * (1 - 2 * global_transfer_loss.cpu().item())
            # d_l = d_l + 2 * (1 - 2 * (local_transfer_loss / 3).cpu().item())
            # dynamic_factor = 1 - boost_factor * d_g / (d_g + d_l)
            loss = cls_loss + global_transfer_loss + boost_factor * (target_loss)
            loss.backward()
            optimizer.step()

            # calculate the correct
            _, pred = torch.max(src_output_cls, dim=1)
            correct += pred.eq(src_label_cls.data.view_as(pred)).sum()
            count += pred.size(0)
        # lr_scheduler.step()
        accuracy = float(correct) / count

        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("train/accuracy", accuracy, epoch)
        writer.add_scalar("train/class-loss", cls_loss, epoch)
    return best_acc, source_att, target_att, confusion_matrixs


def get_dataset(dataset, test_id, session):  ## dataloading function, you should modify this function according to your environment setting.
    data, label = utils.load_data(dataset)
    data_session, label_session = np.array(data[session]), np.array(label[session])
    target_feature, target_label = data_session[test_id], label_session[test_id]
    train_idxs = list(range(15))
    del train_idxs[test_id]
    source_feature, source_label = np.vstack(data_session[train_idxs]), np.vstack(label_session[train_idxs])

    target_set = {'feature': target_feature, 'label': target_label}
    source_set = {'feature': source_feature, 'label': source_label}
    return target_set, source_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--dataset', type=str, nargs='?', default='seed3', help='select the dataset')
    parser.add_argument('--session', type=int, nargs='?', default='0', help='select the session')
    parser.add_argument('--cls', type=int, nargs='?', default=3, help="emotion classification")
    parser.add_argument('--in_planes', type=int, nargs='?', default=[5, 62], help="the size of input plane")
    parser.add_argument('--layers', type=int, nargs='?', default=2, help="DIAM squeeze ratio")
    parser.add_argument('--hidden_1', type=int, nargs='?', default=256, help="the size of hidden 1")
    parser.add_argument('--hidden_2', type=int, nargs='?', default=64, help="the size of hidden 2")
    parser.add_argument('--k', type=int, nargs='?', default=12, help="the size of k")

    parser.add_argument('--batch_size', type=int, nargs='?', default='48', help="batch_size")
    parser.add_argument('--epochs', type=int, nargs='?', default='1000', help="epochs")
    parser.add_argument('--lr', type=float, nargs='?', default='0.001', help="learning rate")
    parser.add_argument('--weight_decay', type=float, nargs='?', default='0.001', help="weight decay")
    parser.add_argument('--seed', type=int, nargs='?', default='200', help="random seed")
    parser.add_argument('--device', type=str, default=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
                        help='cuda or not')

    parser.add_argument('--output_log_dir', default='./train_log', type=str,
                        help='output path, subdir under output_root')
    parser.add_argument('--output_model_dir', default='./model', type=str,
                        help='output path, subdir under output_root')
    args = parser.parse_args()
    logger = create_logger(args)
    logger.info(args)

    sub_acc_max = []
    source_adj = []
    target_att =[]
    sub_confusion_matrixs =[]
    # Start
    logger.info("----------Starting training the model----------")
    for test_id in range(15):
        source_id = [i for i in range(15)]
        source_id.remove(test_id)
        logger.info("The source domain: {} \nthe target domain: {}".format(source_id, test_id))
        writer = SummaryWriter("data/tensorboard/experiment_"+str(args.dataset)+"/sesion_" + str(args.session) + "_C3DA/" )
        best_acc, sub_source_att, sub_target_att,confusion_matrixs = main(test_id, writer, args)
        writer.close()
        logger.info("The test id: {} \nthe max accuracy: {} \nthe confusion_matrixs: {}".format(test_id, best_acc,confusion_matrixs))
        sub_acc_max.append(best_acc)
        source_adj.append(sub_source_att)
        target_att.append(sub_target_att)
        sub_confusion_matrixs.append(confusion_matrixs)
    sub_acc_max = np.reshape(sub_acc_max, (-1, 1))
    logger.info("The mean accuracy: {}".format(np.mean(sub_acc_max)))
    logger.info("The confusion_matrixs: {}".format(np.sum(sub_confusion_matrixs,axis=0)))
