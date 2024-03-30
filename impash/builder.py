# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

class IMPaSh(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(IMPaSh, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        assert mlp == True
        self.q1_mlp=nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )
        self.k1_mlp=nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )
        self.q2_mlp=nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )
        self.k2_mlp=nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue1", torch.randn(dim, K))#　128x65536
        self.queue1 = nn.functional.normalize(self.queue1, dim=0)# 每个128维的负样本的L2范数都是1
        #added
        self.register_buffer("queue2", torch.randn(dim, K))#　128x65536
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)# 每个128维的负样本的L2范数都是1

        self.register_buffer("queue_ptr1", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, k1, k2):
        # gather keys before updating queue
        k1 = concat_all_gather(k1)# 收集所有GPU上的key
        k2 = concat_all_gather(k2)

        batch_size = k1.shape[0]# 总的batch_size,256

        ptr1 = int(self.queue_ptr1)#两者的ptr应该是在一个位置
        ptr2 = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity 65536%256 
        #added
        assert ptr1 == ptr2
        
        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, ptr : ptr + batch_size] = k1.T # 必然是整块替换，不会头尾两截
        self.queue2[:, ptr : ptr + batch_size] = k2.T
        
        ptr1 = (ptr1 + batch_size) % self.K  # move pointer
        ptr2 = (ptr2 + batch_size) % self.K
      
        self.queue_ptr1[0] = ptr1
        self.queue_ptr2[0] = ptr2

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]# 获取当前GPU的索引，并从 idx_shuffle 中选择与当前GPU对应的索引。

        return x_gather[idx_this], idx_unshuffle# 返回的应该是当前gpu分到的数据

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q1, im_k1, im_q2, im_k2 ):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        print("**enter_forward**")
        q1 = self.encoder_q(im_q1)  # queries: NxC
        q1 = self.q1_mlp(q1)
        q1 = nn.functional.normalize(q1, dim=1) #沿着张量的第一个维度（通常是特征维度的方向）进行归一化
        q2 = self.encoder_q(im_q2)  # queries: NxC
        q2 = self.q2_mlp(q2)
        q2 = nn.functional.normalize(q2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k1, idx_unshuffle = self._batch_shuffle_ddp(im_k1)
            im_k2, idx_unshuffle = self._batch_shuffle_ddp(im_k2)

            k1 = self.encoder_k(im_k1)  # keys: NxC
            k2 = self.encoder_k(im_k2)
        k1 = self.k1_mlp(k1)
        k1 = nn.functional.normalize(k1, dim=1)
        k2 = self.k2_mlp(k2)
        k2 = nn.functional.normalize(k2, dim=1)

        # undo shuffle
        k1 = self._batch_unshuffle_ddp(k1, idx_unshuffle)
        k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos1 = torch.einsum("nc,nc->n", [q1, k1]).unsqueeze(-1)#做点积，得到n*1
        # negative logits: NxK
        l_neg1 = torch.einsum("nc,ck->nk", [q1, self.queue1.clone().detach()]) #做矩阵乘，得到n*k

        

        #same
        l_pos2 = torch.einsum("nc,nc->n", [q2, k2]).unsqueeze(-1)
        l_neg2 = torch.einsum("nc,ck->nk", [q2, self.queue2.clone().detach()])
        
        # logits: Nx(1+K)
        logits1 = torch.cat([l_pos1, l_neg1], dim=1)
        logits2 = torch.cat([l_pos2, l_neg2], dim=1)
        logits3 = torch.cat([l_pos1, l_neg2], dim=1)
        logits4 = torch.cat([l_pos2, l_neg1], dim=1)
        
        # apply temperature
        logits1 /= self.T
        logits2 /= self.T
        logits3 /= self.T
        logits4 /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()# 将全0标签移动到GPU上，形状是n*1的全0向量

        # dequeue and enqueue
        self._dequeue_and_enqueue(k1,k2)# k is NxC(batch_size x Dim)

        return logits1, logits2, logits3, logits4, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # 将tensors_gather列表中的所有张量沿着维度0连接起来，形成一个单一的张量output
    output = torch.cat(tensors_gather, dim=0)
    return output
