import torch
import torch.nn as nn
import torch.nn.functional as F

class driver_model(nn.Module):
	"""
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """

	def __init__(self):
		super(driver_model, self).__init__()
		#Normalization layer
		self.conv1 = nn.Conv2d(1,24,5,2)
		self.conv2 = nn.Conv2d(24,36,5,2)
		self.conv3 = nn.Conv2d(36,48,5,2)
		self.conv4 = nn.Conv2d(48,64,3)
		self.conv5 = nn.Conv2d(64,64,3)
		self.fc1 = nn.Linear(64*3*3,100,100)
		self.fc2 = nn.Linear(100,50)
		self.fc3 = nn.Linear(50,10)
		self.fc4 = nn.Linear(10,1)

	def forward(self,x):
		x = F.elu(self.conv1(x))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.dropout(F.elu(self.conv4(x)))
		x = x.view(-1, self.num_flat_features(x))
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = F.elu(self.fc3(x))
		return x

	def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":

	net = Net()
	print(net)


		


