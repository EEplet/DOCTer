import torch.nn as nn
import torch
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding, Conv1d
from torch.nn import functional as F
from torch.autograd import Variable
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class CnnLstmEmbedding(nn.Module):
    def __init__(self, input_sample = (1, 62, 250), emb_size = 124, drop_rate = 0.5):

        super(CnnLstmEmbedding, self).__init__() 
        self.drop_rate = drop_rate
        ch_num = input_sample[1]
        hidden = [62, 62, 62*2]  
        self.conv = nn.Sequential(
            nn.Conv1d(ch_num, hidden[0], kernel_size=25, stride=2), 
            nn.BatchNorm1d(hidden[0]),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),               
            nn.Dropout(drop_rate),

            nn.Conv1d(hidden[0], hidden[1], kernel_size=11, stride=1),  
            nn.BatchNorm1d(hidden[1]),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),                                
            nn.Dropout(drop_rate),

            nn.Conv1d(hidden[1], hidden[2], kernel_size=9, stride=1),   
            nn.BatchNorm1d(hidden[2]),
            nn.ELU(),
            
            nn.Dropout(drop_rate)
        )
        self.projection = nn.Sequential(
            nn.Conv1d(emb_size, emb_size, 1, 1)  
        )

    def forward(self, x):  
        x = self.conv(x)          
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=2,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, input_sample, n_classes):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(1860, 128),  # 2s->440
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out

class Attention_Module(Module):
    def __init__(self, in_dim):
        super(Attention_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input eeg feature maps( B x C x T)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, L = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, L)

        out = self.gamma*out + x
        return out

class DOCTer(nn.Module):
    def __init__(self, drop_rate=0.5, input_sample = (1, 62, 250), depth = 4):

        super(DOCTer, self).__init__()
        self.drop_rate = drop_rate
        emb_size = 124
        hidden = [62, 62, 124]
        self.eegembedding = CnnLstmEmbedding(input_sample = input_sample, emb_size = emb_size, drop_rate = drop_rate) # 15, 124
        self.sc = Attention_Module(emb_size)
   
        self.positions = nn.Parameter(torch.randn(15, emb_size))  # 2s: 15  8s: 109
        self.transformer = TransformerEncoder(depth = depth, emb_size = emb_size)
        
        self.fc1 = nn.Sequential(
            nn.Linear(1860, 128),  # 2s->1860
            nn.ELU(),
            nn.Dropout(drop_rate),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
        self.relu = nn.ReLU()
        self.fc_microstate1 = nn.Linear(250, 128)
        self.microstate_dim = 32
        self.fc_microstate2 = nn.Linear(128, self.microstate_dim)
        self.fusion_linear_eeg_microstate_bybc = nn.Linear((128 + 128 + self.microstate_dim), 128)
        self.cause_embedding = nn.Embedding(num_embeddings=3, embedding_dim=128)
        self.bc_embedding = nn.Embedding(num_embeddings=12, embedding_dim=128)
        self.fusion_linear_bybc_1 = nn.Linear(128 * 2, 128)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, out):  
        # feature extraction
        x, micro_sequence, cause_number, bc_number = out
        cause_embed = self.cause_embedding(cause_number)  # (N, 128)
        bc_embed = self.bc_embedding(bc_number)
        cause_f, bc_f = cause_embed, bc_embed  
        fusion_bybc = torch.cat((cause_f, bc_f), dim=1)  
        fusion_bybc = self.fusion_linear_bybc_1(fusion_bybc)  
        fusion_bybc = self.elu(fusion_bybc) 
        x = self.eegembedding(x)  
        x = x.permute(0, 2, 1)  
        x += self.positions
        x = self.transformer(x) 
        x = self.sc(x) 
        x = x.permute(0, 2, 1)  
        x = torch.flatten(x, 1)  
        x = self.fc1(x)  
        micro_sequence = self.fc_microstate2(self.relu(self.fc_microstate1(micro_sequence)))  
        x = torch.cat((x, fusion_bybc, micro_sequence), dim=1)  
        x = self.fusion_linear_eeg_microstate_bybc(x)  
        x = self.elu(x) 
        x = self.fc2(x)  
        return x


if __name__ == "__main__":
    DEVICE = torch.device('cpu')
    tlen = 250
    ch_num = 50
    model = DOCTer(input_sample = (1, ch_num, tlen)).to(DEVICE) 
    batch = torch.rand(4, ch_num, tlen).to(DEVICE)
    micro = torch.rand(4, tlen).to(DEVICE)
    cause = torch.tensor(np.array([1, 1, 1, 1])).to(DEVICE)
    bc = torch.tensor(np.array([2, 2, 2, 2])).to(DEVICE)
    input_ = (batch, micro, cause, bc)
    print("bathcshape:", batch.shape)
    # summary(model,(ch_num, tlen),device = "cpu")
    print(model(input_).shape)
