import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class AttentionMILTremorFrequency(nn.Module):
    def __init__(self, L, M, K, pooling):
        """
            M: Embedding dimension
            L: Attention dimension
            K: Bag length (with zero padding)

        """
        super(AttentionMILTremorFrequency, self).__init__()
        self.max_bag_length = K
        self.pooling = pooling

        self.embedding = nn.Sequential(
            nn.Linear(in_features=76, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=M),
            #nn.LeakyReLU(0.2),
        )

        self.V = nn.Linear(M, L, bias=True)
        self.U = nn.Linear(M, L, bias=True)
        self.w = nn.Linear(L, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        self.clf = nn.Sequential(
                nn.Linear(M, 32),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
                nn.Linear(16, 2),
        )

    def attention_pooling(self, h_k, x_mask):
        a_k = self.w(torch.tanh(self.V(h_k))).squeeze(dim=-1)
        a_k[x_mask == 0] = -1e30
        a_k = self.softmax(a_k).unsqueeze(-1)
        z = torch.sum(a_k * h_k, dim=1)
        return a_k, z

    def gated_attention_pooling(self, h_k, x_mask):
        a_k = self.w(torch.tanh(
            self.V(h_k)) * torch.sigmoid(self.U(h_k))).squeeze(dim=-1)
        a_k[x_mask == 0] = -1e30
        #a_k = a_k + (x_mask * -1e30)
        a_k = self.softmax(a_k).unsqueeze(-1)
        z = torch.sum(a_k * h_k, dim=1)
        return a_k, z

    def g(self, x, x_mask):
        x = x.sum(dim=2)
        h_k = self.embedding(x)
        if self.pooling == 'attention':
            a_k, z = self.attention_pooling(h_k, x_mask)
        elif self.pooling == 'gated_attention':
            a_k, z = self.gated_attention_pooling(h_k, x_mask)

        return a_k, z

    def forward(self, x, x_mask):
        """
            x: Input with shape (B, K_i, C, Ws)
            where:
            B is batch_size
            K_i is length of the bag if instances i (num of sessions for subject i)
            C num channels
            Ws window size

            If no padding is used, x will be a list of tensors (since K_i will
            be of variable length)

            If padding is used, K_i will be equal to max_bag_length for all i, and
            x will be a proper tensor

            x_mask: Which elements of x to consider for computations

        """
        a_k, z = self.g(x, x_mask)
        logit = self.clf(z)

        return z, logit

class AttentionMILTremorTime(nn.Module):
    def __init__(self, L, M, K, pooling):
        """
            M: Embedding dimension
            L: Attention dimension
            K: Bag length (with zero padding)

        """
        super(AttentionMILTremorTime, self).__init__()
        self.max_bag_length = K
        self.pooling = pooling
        self.L = L
        self.M = M
        self.predict = False

        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=8, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            #nn.Dropout(0.5),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            #nn.Dropout(0.5),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=16, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=16, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
        )

        self.fc = nn.Linear(16*20, M)
        #self.lrelu = nn.LeakyReLU(0.2)

        self.V = nn.Linear(M, L, bias=True)
        self.U = nn.Linear(M, L, bias=True)
        self.w = nn.Linear(L, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        self.clf = nn.Sequential(
            nn.Linear(M, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(16, 2),
        )

    def attention_pooling(self, h_k, x_mask):
        a_k = self.w(torch.tanh(self.V(h_k))).squeeze(dim=-1)
        a_k[x_mask == 0] = -1e30
        a_k = self.softmax(a_k).unsqueeze(-1)
        z = torch.sum(a_k * h_k, dim=1)
        return a_k, z

    def gated_attention_pooling(self, h_k, x_mask):
        a_k = self.w(torch.tanh(
            self.V(h_k)) * torch.sigmoid(self.U(h_k))).squeeze(dim=-1)
        a_k[x_mask == 0] = -1e30
        #a_k = a_k + (x_mask * -1e30)
        a_k = self.softmax(a_k).unsqueeze(-1)
        z = torch.sum(a_k * h_k, dim=1)
        return a_k, z

    def g(self, x, x_mask):
        B, K, C, Ws = x.size()
        x = x.view(B*K, C, Ws)
        h_k = self.embedding(x)
        #_, C, Ws = h_k.size()
        h_k = h_k.view(B*K, -1)
        h_k = self.fc(h_k)
        #h_k = self.lrelu(h_k)
        h_k = h_k.view(B, K, self.M)
        if self.pooling == 'attention':
            a_k, z = self.attention_pooling(h_k, x_mask)
        elif self.pooling == 'gated_attention':
            a_k, z = self.gated_attention_pooling(h_k, x_mask)

        return a_k, z

    def forward(self, x, x_mask):
        """
            x: Input with shape (B, K_i, C, Ws)
            where:
            B is batch_size
            K_i is length of the bag if instances i (num of sessions for subject i)
            C num channels
            Ws window size

            If no padding is used, x will be a list of tensors (since K_i will
            be of variable length)

            If padding is used, K_i will be equal to max_bag_length for all i, and
            x will be a proper tensor

            x_mask: Which elements of x to consider for computations

        """
        a_k, z = self.g(x, x_mask)
        logit = self.clf(z)

        return z, logit

class AttentionMILRigidity(nn.Module):
    def __init__(self, input_dim, L, M, K):
        """
            M: Embedding dimension
            L: Attention dimension
            K: Bag length (with zero padding)

        """
        super(AttentionMILRigidity, self).__init__()
        self.max_bag_length = K
        self.embedding_dim = M

        self.embedding = nn.Sequential(
            #nn.Linear(102, 100),
            #nn.Linear(502, 100),
            nn.Linear(input_dim, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(100, 50),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(50, M))

        self.V = nn.Linear(M, L, bias=True)
        self.U = nn.Linear(M, L, bias=True)
        self.w = nn.Linear(L, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        self.clf = nn.Sequential(
            nn.Linear(M, 30),
            nn.LeakyReLU(0.2),
            nn.Linear(30, 10),
            nn.LeakyReLU(0.2),
            nn.Linear(10, 2),
        )

    def attention_pooling(self, h_k, x_mask):
        a_k = self.w(torch.tanh(self.V(h_k))).squeeze(dim=-1)
        a_k[x_mask == 0] = -1e30
        a_k = self.softmax(a_k).unsqueeze(-1)

        ## DEBUG ##
        #a_k = torch.ones_like(h_k) / h_k.size(1)

        z = torch.sum(a_k * h_k, dim=1)
        return a_k, z

    def gated_attention_pooling(self, h_k, x_mask):
        a_k = self.w(torch.tanh(
            self.V(h_k)) * torch.sigmoid(self.U(h_k))).squeeze(dim=-1)
        a_k[x_mask == 0] = -1e30
        #a_k = a_k + (x_mask * -1e30)
        a_k = self.softmax(a_k).unsqueeze(-1)
        z = torch.sum(a_k * h_k, dim=1)
        return a_k, z

    def g(self, x, x_mask):
        B, K, F = x.size()
        h_k = self.embedding(x)
        a_k, z = self.attention_pooling(h_k, x_mask)
        return a_k, z

    def forward(self, x, x_mask):
        """
            x: Input with shape (B, K_i, Ws)
            where:
            B is batch_size
            K_i is length of the bag if instances i (num of sessions for subject i)
            Ws window size

            If no padding is used, x will be a list of tensors (since K_i will
            be of variable length)

            If padding is used, K_i will be equal to max_bag_length for all i, and
            x will be a proper tensor

            x_mask: Which elements of x to consider for computations

        """
        a_k, z = self.g(x, x_mask)
        logit = self.clf(z)

        return z, logit

class ModelFusion(nn.Module):
    def __init__(self, embedding1, embedding2, M):
        super(ModelFusion, self).__init__()
        self.embedding1 = embedding1
        self.embedding2 = embedding2

        self.clf = nn.Sequential(
            nn.Linear(M, 32),
            #nn.Linear(2*M, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x1, x2, x_mask1, x_mask2):
        _, z1 = self.embedding1.g(x1, x_mask1)
        _, z2 = self.embedding2.g(x2, x_mask2)
        z = z1 + z2
        #z = torch.cat((z1, z2), dim=1)
        logit_x = self.clf(z)
        return z, logit_x

class ModelFusionMultiLabel(nn.Module):
    def __init__(self, embedding1, embedding2, M):
        super(ModelFusionMultiLabel, self).__init__()
        self.embedding1 = embedding1
        self.embedding2 = embedding2

        self.clf = nn.Sequential(
            nn.Linear(M, 32),
            #nn.Linear(2*M, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(16, 3),
        )

    def forward(self, x1, x2, x_mask1, x_mask2):
        _, z1 = self.embedding1(x1, x_mask1)
        _, z2 = self.embedding2(x2, x_mask2)
        z = z1 + z2
        #z = torch.cat((z1, z2), dim=1)
        logit_x = self.clf(z)
        return z, logit_x
