from torch import nn
import torch
import torch.nn as nn
import numpy as np
import onnx

class model(nn.Module):
    def __init__(self, in_channel=1):
        super().__init__()
        self.A = np.ones(shape=(3, 15, 15))
        self.A = torch.tensor(self.A, dtype=torch.float32, requires_grad=False)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        x = torch.einsum('nctkv,kvw->nctw', x, self.A)
        return x

input_tensor = torch.ones(size=[1, 3, 1, 1, 15])
Model = model()
Model.cuda()
out = Model(input_tensor)

input_name = ['input1']  # , 'input2']
output_name = ['output1']  # , 'output2']  # 必须要有输入输出
torch.onnx.export(Model,
                  input_tensor,
                  './gcn.onnx',
                  input_names=input_name, output_names=output_name,
                  verbose=True,
                  opset_version=12
                  )
model = onnx.load('./gcn.onnx')
print(onnx.checker.check_model(model))
print(out)
print(out.size())
print(out.sum())

