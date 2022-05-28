import torch
import torch.nn as nn

SOURCE_IMG_SIZE = 48
IMG_SIZE = 96

CLASSES = ['neutral', 'happy', 'surprised', 'sad', 'angry'] #, 'disgusted', 'afraid'
NUM_CLASSES = len(CLASSES)

class Facial_Expression_Network_AlexNet(nn.Module):
    def __init__(self):
        super(Facial_Expression_Network_AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, IMG_SIZE, kernel_size=(3,3), stride=(1, 1), padding=(2, 2)), 
            nn.ReLU(inplace=True), # Rectified Linear Unit activation function
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False), # pooling layer for reducing dimensions
            nn.Conv2d(IMG_SIZE, 192, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6)) # Applies a 2D adaptive average pooling over an input composed of several input planes.


        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False), # Dropout layer for setting 50% of the activations to 0, fording the network to not rely on any 1 node
            nn.Linear(in_features=9216, out_features=4096, bias=True), # Linear layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=NUM_CLASSES, bias=True),
        ) # ---- Change out_features if change in amount of classes ----


        self.softmax = nn.Softmax(dim=1) # last activation function for the network, normalizing the output
        self.softmax_result = 0 

    def forward(self, x, verbose=False):
        x = self.features(x)
        x = self.avgpool(x)

        # The data needs to be flattened after the AdaptiveAvgPool2d as its output is H x W
        # This is because the classifier's first layer is a Linear layer
        x = torch.flatten(x, 1)

        x = self.classifier(x) 
        self.softmax_result = self.softmax(x)
        return x