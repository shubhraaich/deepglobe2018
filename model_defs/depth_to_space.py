import torch
import torch.nn as nn
import math


class DepthToSpace(nn.Module):
    def __init__(self, num_split):
        super(DepthToSpace, self).__init__()
        self.num_split = num_split


    def forward(self, x):
        x = torch.split(x, split_size=x.size(1)/self.num_split, dim=1); # along depth
        size_org = x[0].size();
        assert (size_org[2] == size_org[3]);
        batch_size = size_org[0];
        cell_size = int(math.sqrt(size_org[1])); # in pixels (be careful about float)
        block_size = cell_size * size_org[2]; # in pixels
        block_size_big = block_size/cell_size;

        x = [t.permute(0,2,3,1).contiguous() for t in x];
        x = [t.view(t.size(0), -1, cell_size, cell_size) for t in x];

        # cuda variable
        output_list = [torch.autograd.Variable(torch.zeros(batch_size, block_size, block_size)).cuda() for i in range(len(x))];

        for k in range(self.num_split) :
            count = 0;
            for i in range(0, block_size, cell_size) :
                for j in range(0, block_size, cell_size) :
                    output_list[k][:, i:i+cell_size, j:j+cell_size] = x[k][:, count];
                    count += 1;

        output = output_list[0].unsqueeze(1);
        for t in output_list[1:]:
            output = torch.cat((output, t.unsqueeze(1)), dim=1)

        return output;
