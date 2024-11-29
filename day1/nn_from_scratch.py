#idea: Start with linear layers and simple activation, implement back prop, gradient descent
#Work in progress, taking lunch break
class Tensor1D():
    def __init__(self,width,init_method):
        self.width = width
        if self.init_method=='zero':
            self.data = [0 for i in range(width)]

class Tensor2D():
    def __init__(self,width,height,init_method):
        self.width = width
        self.height = height
        if init_method == 'zero':
            self.data = [Tensor1D(height) for j in range(width)]
    def __matmul__(self,B):
        return [self.data[i][j]*]

class LinearLayer():
    def __init__(self,n_channels,input_size,output_size,init_method):
        self.n_channels = n_channels
        self.input_size = input_size
        self.output_size = output_size
        if init_method == 'zero':
            self.layers = [Tensor2D(input_size,output_size) for _ in range(n_channels)]
    def forward(self, x):
        outputs = []
        for layer in self.layers:
