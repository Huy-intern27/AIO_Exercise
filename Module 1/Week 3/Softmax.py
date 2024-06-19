import torch # type: ignore
import torch.nn as nn # type: ignore

class SoftmaxBase(nn.Module):
    def __int__(self):
        super(SoftmaxBase, self).__init__()
        self.data = None

    def forward(self, data):
        self.data = data
        return self.calculate()
    def __str__(self):
        return f'{self.calculate()}'
    
class Softmax(SoftmaxBase):
    def calculate(self):
        sum_exp = torch.sum(torch.exp(self.data))
        data_exp = torch.exp(self.data)
        result = data_exp / sum_exp
        return result

class SoftmaxStable(SoftmaxBase):
    def calculate(self):
        max_element = max(self.data)
        data_exp = torch.exp(self.data - max_element)
        sum_exp = torch.sum(torch.exp(self.data - max_element))
        result = data_exp / sum_exp
        return result

if __name__=="__main__":
    data = torch.Tensor([1, 2, 3])

    softmax = Softmax()
    output1 = softmax(data)
    print(output1)

    softmax_stable = SoftmaxStable()
    output2 = softmax_stable(data)
    print(output2)
