#idea: Start with linear layers and simple activation, implement back prop, gradient descent
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
    def __add__(self,B):
        for i in range(len(B)):
            for j in range(len(B.getData(i))):
                self.setData(i,j,self.getData(i,j)+B.getData(i,j))
    def __multiply__(self,scalar):
        for i in range(len(self)):
            for j in range(len(self.getData(i))):
                self.setData(i,j,self.getData(i,j)*scalar)
    def __subtract__(self,B):
        self = self + B * -1



class LinearLayer():
    def __init__(self,n_channels,input_size,output_size,init_method,copy_weights, copy_biases):
        self.input_size = input_size
        self.output_size = output_size
        self.name="fclinear"
        if init_method == 'zero':
            self.weights = Tensor2D(input_size,output_size)
            self.biases = 0
        elif init_method == "copy":
            self.weights = copy_weights
            self.biases = copy_biases
    def forward(self, x):
        outputs = self.weights @ x+self.biases
        return outputs

class relu():
    def __init__(self):
        self.name = "relu"
    def forward(x: Tensor2D):
        for i in range(x.getHeight()):
            for j in range(x.getWidth()):
                elem = x.getData(i,j)
                val = 0 if elem<0 else elem
                x.setData(i,j,val)

class Sequential():
    def __init__(self,arr=[]):
        self.layers = arr
    def add_layer(self,layer):
        self.layers.append(layer)
    def forward(self,x):
        tmp = x
        for layer in self.layers:
            tmp = layer(tmp)
        return tmp

def gradient_descent_backprop(model,inputs,labels,lr): # \nabla Cost w.r.t weights and biases
    intermediates = [inputs]
    tmp = inputs
    for layer in model.layers:
        tmp = layer(inputs)
        intermediates.append(tmp)
    del intermediates[-1]
    outputs = tmp
    error = (outputs-labels)
    updates = []
    for intermediate,layer in reversed(zip(intermediates,model.layers)):
        if layer.name=="fclinear":
            error_m = intermediate @ error.transpose()
            error_b = error
            updates.append({
                "fclinear": {
                    error_m:error_m,
                    error_b:error_b  }}) #Struct?
        elif layer.name=="relu":
            errors = []
            for i in range(len(intermediate)):
                if intermediate[i]<0:
                    errors.append(0)
                else:
                    errors.append(error[i])
            updates.append({"relu": {
                            errors }})
    newmodel = Sequential()
    for layer,update in zip(model.layers,updates):
        if layer.name=="fclinear":
            newmodel.add_layer(LinearLayer(layer.weights - lr * update["fclinear"].error_m,layer.biases-lr*update["fclinear"].error_b))
        else:
            newmodel.add_layer(relu)
    return outputs,errors,newmodel
        

    

        

            

    

def gradient_descent(model,data,labels, lr,epochs):
    for i in range(epochs):
        print(f"Starting epoch {i}")
        for k in range(len(data)):
            outputs, errors, model = gradient_descent_backprop(model,data[k],labels[k],lr)
            print(outputs)
            print(errors)
    print("DONE")


#Going to gym, will come back and clean up