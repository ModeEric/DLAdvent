#idea: Start with linear layers and simple activation, implement back prop, gradient descent
#Work in progress, taking lunch break
class Tensor1D():
    def __init__(self,width,init_method='zero'):
        self.width = width
        self.init_method = init_method
        if self.init_method=='zero':
            self.data = [0 for i in range(width)]
    def __multiply__(self,B):
        return sum([self.getData(i)*B.getData(i) for i in range(len(self))])
    def __len__(self):
        return self.width
    def getData(self,i):
        return self.data[i]
    def setData(self,i,val):
        self.data[i] = val

class Tensor2D():
    def __init__(self,width,height,init_method='zero'):
        self.init_method = init_method
        self.width = width
        self.height = height
        if init_method == 'zero':
            self.data = [Tensor1D(width) for j in range(height)]
    def transpose(self):
        newtensordata = []
        for j in range(self.width):
            tmp = Tensor1D(self.height)
            for i in range(self.height):
                tmp.setData(i,self.data[i][j])
            newtensordata.append(tmp)
        self.data = newtensordata
        tmp = self.width
        self.width = self.height
        self.height = tmp
    def __matmul__(self,B):
        return [[sum(self.getData(i)*B.transpose().getData(j)) for j in range(len(B.tranpose()))] for i in range(len(self))]
    def __len__(self):
        return self.height
    def getData(self,i,j):
        if j is None:
            return self.data[i]
        else:
            return self.data[i][j]
    def setData(self,i,j,val):
        if j is None:
            self.data[i] = val
        else:
            self.data[i][j] = val
    def getWidth(self):
        return self.width
    def getHeight(self):
        return self.height




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
            outputs.append(self.layers @ x)
        return outputs
    
def relu(x: Tensor2D):
    for i in range(x.getHeight()):
        for j in range(x.getWidth()):
            elem = x.getData(i,j)
            val = 0 if elem<0 else elem
            x.setData(i,j,val)



def gradient_descent(model,data,labels, lr,epochs):
    for _ in range(epochs):
        for k in range(len(data)):
            outputs = model(data[k])
            error = outputs - labels
            model = model - lr * error


