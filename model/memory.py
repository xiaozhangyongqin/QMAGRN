import torch
import torch.nn as nn

class MemoryAugmented(nn.Module):
    def __init__(self, num_nodes=170, mem_num=40, mem_dim=64, loop_times=2, fusion_factor=0.7):
        super(MemoryAugmented, self).__init__()
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.loop_times = loop_times

        #self.fusion_factor = nn.Parameter(torch.rand(1), requires_grad=True)
        self.fusion_factor = fusion_factor

        self.memory = nn.ParameterDict({
            'Memory': nn.Parameter(torch.randn(mem_num, mem_dim), requires_grad=True),
            'We1': nn.Parameter(torch.randn(num_nodes, mem_num), requires_grad=True),
            'We2': nn.Parameter(torch.randn(num_nodes, mem_num), requires_grad=True)
        })

        self._init_weights()

    def _init_weights(self):
        for param in self.memory.values():
            nn.init.xavier_normal_(param)

    def query_memory(self, h_t):
        query = h_t
        value_list = [query]
        att_score_list = []
        for i in range(self.loop_times):
            att_score = torch.softmax(torch.matmul(value_list[i], self.memory['Memory'].T), dim=-1)
            value = torch.matmul(att_score, self.memory['Memory'])
            value_list.append(value)
            att_score_list.append(att_score)

        _, ind = torch.topk(att_score_list[-1], k=2, dim=-1)
        pos = self.memory['Memory'][ind[..., 0]]
        neg1 = self.memory['Memory'][ind[..., 1]]
        neg2 = self.memory['Memory'][ind[..., 1]]

        x_aug = self.fusion_factor * value_list[1] + (1 - self.fusion_factor) * value_list[2]
        # x_aug = value_list[1]

        return x_aug, query, pos, neg1, neg2

    def forward(self, x):
        # print(self.fusion_factor)
        return self.query_memory(x)
