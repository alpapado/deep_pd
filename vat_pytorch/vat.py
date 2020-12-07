import time
from functools import partial
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from vat_pytorch.torch_utils import _disable_tracking_bn_stats, kl_div_with_logit, l2_normalize, entropy_loss

print = partial(print, flush=True)


class Model1M(torch.nn.Module):
    val_loader = None
    xentropy = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, model, logdir=None,
                 class_weights=None, lr=1e-3, decision_th=0.5):
        super(Model1M, self).__init__()
        self.logit = model
        self.global_step = 0
        self.to(self.device)
        self.decision_th = 0.5
        #self.writer = SummaryWriter(logdir)
        self.writer = None
        if class_weights is not None:
            self.xentropy = nn.CrossEntropyLoss(
                weight=torch.from_numpy(class_weights))

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, amsgrad=False)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

    def forward(self, x):
        pass

    def train_supervised(self, train_loader):
        """ Train 1 iteration with supervised loss """
        running_loss = 0.0
        i = 1

        for i, (x, y, x_mask, sid) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            x_mask = x_mask.to(self.device)

            _, logit_x = self.logit(x, x_mask)
            loss = self.xentropy(logit_x, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            self.global_step += 1

        i += 1

        return running_loss/i

    def fit(self, train_loader, test_loader=None,
            mode='vat', num_epochs=20):
        acc = None

        loss = 0.0

        for epoch in range(num_epochs):
            start = time.time()

            if mode == "cr":
                loss = self.train_supervised(train_loader)

            end = time.time()

            if (epoch+1) % num_epochs == 0:  # and False:
                print('[%d/%d] - %.3f secs - loss: %.5f' %
                      (epoch + 1, num_epochs, end-start, loss))

                if test_loader is not None:
                    acc = self.validate(test_loader)

        model_path = "model.pt"
        torch.save({
            # 'epoch': epoch,
            'model_state_dict': self.logit.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)

        return acc, model_path

    def validate(self, val_loader):
        self.eval()
        correct = 0
        total = 0

        total_loss = 0

        i = 0

        with torch.no_grad():
            for i, (x, y, x_mask, x_sid) in enumerate(val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                x_mask = x_mask.to(self.device)

                _, logit_x = self.logit(x, x_mask)
                total_loss += self.xentropy(logit_x, y).item()
                prob_x = torch.softmax(logit_x, dim=1)

                _, predicted = torch.max(prob_x.data, 1)

                total += y.size(0)
                correct += (predicted == y).sum().item()

                x_sid = np.array(x_sid)
                error_idx = (predicted != y).cpu().numpy().astype(bool)
                error_sid = x_sid[error_idx]
                if len(error_sid) > 0:
                    print(error_sid)

        if i == 0:
            i += 1

        accuracy = 100*correct/total
        #print(prob_pd[:, 1])
        print("Training set accuracy = ", accuracy)
        #acc_tensor = torch.from_numpy(np.array(accuracy)).to(self.device)
        #self.writer.add_scalar('val/total_loss', total_loss, self.global_step)
        #self.writer.add_scalar('val/cr_loss', cr_loss, self.global_step)
        #self.writer.add_scalar('val/vat_loss', vat_loss, self.global_step)
        #self.writer.add_scalar('val/accuracy', acc_tensor, self.global_step)

        self.train()

        return accuracy

    def predict(self, val_loader):
        self.eval()
        probs = []
        serials = []

        with torch.no_grad():
            for x, y, x_mask, sid in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                x_mask = x_mask.to(self.device)

                _, logit_x = self.logit(x, x_mask)
                prob_x = torch.softmax(logit_x, dim=1)

                probs.append(prob_x[:, 1].cpu().detach().numpy())
                serials.append(list(sid))

        probs = np.concatenate(probs)
        serials = np.concatenate(serials)

        self.train()

        return probs, serials

class MultiLabelModel2M(torch.nn.Module):
    val_loader = None
    xentropy = nn.CrossEntropyLoss()
    # Multiple supervision
    bce = nn.BCEWithLogitsLoss(reduction='none')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, model, logdir=None,
                 class_weights=None, lr=1e-3, decision_th=0.5):
        super(MultiLabelModel2M, self).__init__()
        self.logit = model
        self.global_step = 0
        self.to(self.device)
        self.decision_th = 0.5
        #self.writer = SummaryWriter(logdir)
        self.writer = None
        if class_weights is not None:
            self.xentropy = nn.CrossEntropyLoss(
                weight=torch.from_numpy(class_weights))

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, amsgrad=False)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

    def forward(self, x):
        pass

    def train_supervised(self, train_loader):
        """ Train 1 iteration with supervised loss """
        pd_loss = 0.0
        tr_loss = 0.0
        tap_loss = 0.0
        i = 1

        for i, (x1, x2, y, x_mask1, x_mask2, sid) in enumerate(train_loader):
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            y = y.to(self.device)
            x_mask1 = x_mask1.to(self.device)
            x_mask2 = x_mask2.to(self.device)

            _, logit_x = self.logit(x1, x2, x_mask1, x_mask2)
            loss = self.bce(logit_x, y).mean(dim=0)

            tr_loss += loss[0].item()
            tap_loss += loss[1].item()
            pd_loss += loss[2].item()

            loss = loss.sum()
            #loss = loss[0] + loss[1] + loss[2]*5

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1

        i += 1

        return tr_loss/i, tap_loss/i, pd_loss/i

    def fit(self, train_loader, test_loader=None,
            mode='cr', num_epochs=20):
        acc = None

        loss = 0.0

        for epoch in range(num_epochs):
            start = time.time()

            if mode == "cr":
                tr_loss, tap_loss, pd_loss = self.train_supervised(train_loader)

            end = time.time()

            if (epoch+1) % num_epochs == 0:  # and False:
                print('[%d/%d] - %.3f secs - Tremor loss: %.5f - Tapping loss: %.5f - PD loss: %.5f ' %
                      (epoch + 1, num_epochs, end-start, tr_loss, tap_loss, pd_loss))

                if test_loader is not None:
                    acc_tr, acc_tap, acc_pd = self.validate(test_loader)

        model_path = "model.pt"
        torch.save({
            # 'epoch': epoch,
            'model_state_dict': self.logit.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)

        return acc_pd, model_path

    def validate(self, val_loader):
        self.eval()
        correct_pd = 0
        correct_tr = 0
        correct_tap = 0
        total = 0

        pd_loss = 0.0
        tr_loss = 0.0
        tap_loss = 0.0

        i = 0

        error_sid_tr = []
        error_sid_tap = []
        error_sid_pd = []

        with torch.no_grad():
            for i, (x1, x2, y, x_mask1, x_mask2, x_sid) in enumerate(val_loader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                y = y.to(self.device)
                x_mask1 = x_mask1.to(self.device)
                x_mask2 = x_mask2.to(self.device)

                _, logit_x = self.logit(x1, x2, x_mask1, x_mask2)
                loss = self.bce(logit_x, y).mean(dim=0)

                tr_loss += loss[0].item()
                tap_loss += loss[1].item()
                pd_loss += loss[2].item()

                loss = loss.sum()

                prob_all = torch.sigmoid(logit_x)

                prob_tr = prob_all[:, 0]
                prob_tap = prob_all[:, 1]
                prob_pd = prob_all[:, -1]

                predicted = (prob_all >= self.decision_th).int()

                total += y.size(0)
                correct_tr += (predicted[:, 0] == y[:, 0].int()).sum().item()
                correct_tap += (predicted[:, 1] == y[:, 1].int()).sum().item()
                correct_pd += (predicted[:, -1] == y[:, -1].int()).sum().item()

                x_sid = np.array(x_sid)
                error_idx_tr = (
                    predicted[:, 0] != y[:, 0].int()).cpu().numpy().astype(bool)
                error_idx_tap = (
                    predicted[:, 1] != y[:, 1].int()).cpu().numpy().astype(bool)
                error_idx_pd = (
                    predicted[:, -1] != y[:, -1].int()).cpu().numpy().astype(bool)

                error_sid_tr.extend(x_sid[error_idx_tr])
                error_sid_tap.extend(x_sid[error_idx_tap])
                error_sid_pd.extend(x_sid[error_idx_pd])

        if i == 0:
            i += 1

        accuracy_tr = 100 * correct_tr / total
        accuracy_tap = 100 * correct_tap / total
        accuracy_pd = 100 * correct_pd / total
        print("Training set tremor accuracy = ", accuracy_tr)
        print("Tremor errors ", error_sid_tr)
        print("Training set tapping accuracy = ", accuracy_tap)
        print("Tapping errors ", error_sid_tap)
        print("Training set PD accuracy = ", accuracy_pd)
        print("PD errors ", error_sid_pd)
        #acc_tensor = torch.from_numpy(np.array(accuracy)).to(self.device)
        #self.writer.add_scalar('val/total_loss', total_loss, self.global_step)
        #self.writer.add_scalar('val/cr_loss', cr_loss, self.global_step)
        #self.writer.add_scalar('val/vat_loss', vat_loss, self.global_step)
        #self.writer.add_scalar('val/accuracy', acc_tensor, self.global_step)

        self.train()

        return accuracy_tr, accuracy_tap, accuracy_pd

    def predict(self, val_loader):
        self.eval()

        serials = []
        probs = []

        with torch.no_grad():
            for x1, x2, y, x_mask1, x_mask2, sid in val_loader:
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                y = y.to(self.device)
                x_mask1 = x_mask1.to(self.device)
                x_mask2 = x_mask2.to(self.device)

                _, logit_x = self.logit(x1, x2, x_mask1, x_mask2)
                prob_all = torch.sigmoid(logit_x)
                #prob_pd = torch.sigmoid(logit_x[:, -1])

                probs.append(prob_all.cpu().detach().numpy())
                serials.append(list(sid))

        probs = np.concatenate(probs)
        serials = np.concatenate(serials)

        self.train()

        return probs, serials
