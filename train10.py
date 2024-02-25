import torch
from datasets import TrainDataset, EvalDataset
import os
import argparse
from torch.utils.data import DataLoader
from model10_1_1_2 import SRWaveMLP
import time
import copy
from tqdm import tqdm
from utils import AverageMeter
from utils import  calc_psnr , calc_ssim



class Trainer:
    record = {"train_loss": [], "train_psnr": [], "val_loss": [], "val_psnr": []}
    x_epoch = []

    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        transitions = [True, True, True, True]
        # layers = [2,2,2,2,2]
        # mlp_ratios = [4,4,4,4,4]
        # embed_dims = [64,64,64,64,64]
        # layers = [2,2,4,2]
        # mlp_ratios = [4,4,4,4]
        # embed_dims = [64,128,320,512] m
        layers = [3,3]
        mlp_ratios = [4,4]
        embed_dims = [64,64]
            
        self.net = SRWaveMLP(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        mlp_ratios=mlp_ratios,mode='depthwise',scale = 4)
        batch = self.args.batch
        # print(self.net)
        self.train_dataset = TrainDataset(args.train_file)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=batch, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(EvalDataset(args.eval_file),
                                     batch_size=1, shuffle=False, drop_last=True)
        self.criterion = torch.nn.L1Loss()
        self.epoch = 0
        self.lr = 0.01
        self.best_psnr = 0.
        if self.args.resume:
            if not os.path.exists(self.args.save_path):
                print("No params, start training...")
            else:
                param_dict = torch.load(self.args.save_path)
                self.epoch = param_dict["epoch"]
                self.lr = param_dict["lr"]
                self.net.load_state_dict(param_dict["net_dict"])
                self.best_psnr = param_dict["best_psnr"]
                print("Loaded params from {}\n[Epoch]: {}   [lr]: {}    [best_psnr]: {}".format(self.args.save_path,
                                                                                                self.epoch, self.lr,
                                                                                                self.best_psnr))
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam([ 
            {'params': self.net.parameters()},
        ], lr=self.lr)

    @staticmethod
    def calculate_psnr(img1, img2):
        return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

    def train(self, epoch):
        self.net.train()
        train_loss = 0.
        train_loss_all = 0.
        psnr = 0.
        total = 0
        start = time.time()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(self.train_dataset) - len(self.train_dataset) %self.args.batch)) as t:
            t.set_description('epoch:{}/{}'.format(epoch, self.args.num_epochs - 1))
            for i, (img, label) in enumerate(self.train_loader):
                img = img.to(self.device)
                label = label.to(self.device)
                pre = self.net(img)
                loss = self.criterion(pre, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_loss_all += loss.item()
                epoch_losses.update(loss.item(), len(img))
                psnr += self.calculate_psnr(pre, label).item()
                total += 1
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(img))

                if (i+1) % self.args.interval == 0:
                    # end = time.time()
                    # print("[Epoch]: {}[Progress: {:.1f}%]time:{:.2f} loss:{:.5f} psnr:{:.4f}".format(
                    #     epoch, (i+1)*100/len(self.train_loader), end-start, train_loss/self.args.interval,
                    #     psnr/total
                    # ))
                    train_loss = 0.
        #     print("Save params to {}".format(self.args.save_path))
        #     param_dict = copy.deepcopy(self.net.state_dict())

        # torch.save(param_dict, self.args.save_path)
        return train_loss_all/len(self.train_loader), psnr/total

    def val(self, epoch):
        self.net.eval()
        val_loss = 0.
        psnr = 0.
        total = 0
        ssim = 0.
        start = time.time()
        with torch.no_grad():
            for i, (img, label) in enumerate(self.val_loader):
                img = img.to(self.device)
                label = label.to(self.device)
                pre = self.net(img).clamp(0.0, 1.0)
                loss = self.criterion(pre, label)
                val_loss += loss.item()
                psnr += self.calculate_psnr(pre, label).item()
                total += 1

            mpsnr = psnr / total
            mssim = ssim / total
            end = time.time()
            print("[Epoch]: {} time:{:.2f} loss:{:.5f} psnr:{:.4f} ".format(
                epoch, end - start, val_loss / len(self.val_loader), mpsnr
            ))
            if mpsnr > self.best_psnr:
                self.best_psnr = mpsnr
        
                print("Save params to {}".format(self.args.save_path1))
                param_dict = copy.deepcopy(self.net.state_dict())
                torch.save(param_dict, self.args.save_path1)
        return val_loss/len(self.val_loader), mpsnr

    # def draw_curve(self, fig, epoch, train_loss, train_psnr, val_loss, val_psnr):
    #     ax0 = fig.add_subplot(121, title="loss")
    #     ax1 = fig.add_subplot(122, title="psnr")
    #     self.record["train_loss"].append(train_loss)
    #     self.record["train_psnr"].append(train_psnr)
    #     self.record["val_loss"].append(val_loss)
    #     self.record["val_psnr"].append(val_psnr)
    #     self.x_epoch.append(epoch)
    #     ax0.plot(self.x_epoch, self.record["train_loss"], "bo-", label="train")
    #     ax0.plot(self.x_epoch, self.record["val_loss"], "ro-", label="val")
    #     ax1.plot(self.x_epoch, self.record["train_psnr"], "bo-", label="train")
    #     ax1.plot(self.x_epoch, self.record["val_psnr"], "ro-", label="val")
    #     if epoch == 0:
    #         ax0.legend()
    #         ax1.legend()
    #     fig.savefig(r"./train_fig/train_{}.jpg".format(epoch))

    def lr_update(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr * 0.8
        self.lr = self.optimizer.param_groups[0]["lr"]
        print("===============================================")
        print("Learning rate has adjusted to {}".format(self.lr))


def main(args):
    t = Trainer(args)
    best_psnr = 0.0
    for epoch in range(t.epoch, t.epoch + args.num_epochs):  
        train_loss, train_psnr = t.train(epoch)
        val_loss, val_psnr = t.val(epoch)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
        
    print("=========================")
    print("best psnr =====")
    print(best_psnr)
    print("=========================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training ESPCN with celebA")
    parser.add_argument('--train-file', type=str,default="/home/dell/code/mlp/datasets/DF2K_4.hdf5")
    parser.add_argument('--eval-file', type=str, default="/home/dell/code/mlp/datasets/BSDS100_4.hdf5")                                        
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--num_epochs", default=400, type=int)
    parser.add_argument("--save_path", default=r"/home/dell/code/mlp/outputs/weight00.pth", type=str)
    parser.add_argument("--save_path1", default=r"/home/dell/code/mlp/outputs/weight_2.25.pth", type=str)
    parser.add_argument("--interval", default=20, type=int)
    parser.add_argument("--batch", default=8, type=int)
    args1 = parser.parse_args()
    main(args1)

 