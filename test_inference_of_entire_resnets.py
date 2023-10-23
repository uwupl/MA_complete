import torch
import torchvision.models as models
import time
import os
import torch.nn as nn

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# from efficientnet
def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )
        

# Load the models
resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
pdn_s = get_pdn_small()
pdn_m = get_pdn_medium()
wide_resnet50 = models.wide_resnet50_2(pretrained=True)

# Set the models to evaluation mode
resnet18.eval()
resnet34.eval()
wide_resnet50.eval()
resnet50.eval()
pdn_s.eval()
pdn_m.eval()
image = torch.randn(1, 3, 224, 224)
if torch.cuda.is_available():
    print('hey')
    resnet18 = resnet18.cuda()
    resnet34 = resnet34.cuda()
    resnet50 = resnet50.cuda()
    wide_resnet50 = wide_resnet50.cuda()
    pdn_m = pdn_m.cuda()
    pdn_s = pdn_s.cuda()

# Load a sample image
    image = image.cuda()

# Set the number of repetitions
num_reps = 100
# warm up
for i in range(num_reps):
    output = resnet18(image)
    output = resnet34(image)
    output = resnet50(image)
    output = wide_resnet50(image)
    output = pdn_s(image)
    output = pdn_m(image)
# Test the inference time of each model
start_time = time.perf_counter()
for i in range(num_reps):
    output = resnet18(image)
end_time = time.perf_counter()
print("ResNet18 inference time: ", 1e3*(end_time - start_time) / num_reps)

start_time = time.perf_counter()
for i in range(num_reps):
    output = resnet34(image)
end_time = time.perf_counter()
print("ResNet34 inference time: ", 1e3*(end_time - start_time) / num_reps)

start_time = time.perf_counter()
for i in range(num_reps):
    output = wide_resnet50(image)
end_time = time.perf_counter()
print("WideResNet50 inference time: ", 1e3*(end_time - start_time) / num_reps)

start_time = time.perf_counter()
for i in range(num_reps):
    output = resnet50(image)
end_time = time.perf_counter()
print("ResNet50 inference time: ", 1e3*(end_time - start_time) / num_reps)

start_time = time.perf_counter()
for i in range(num_reps):
    output = pdn_s(image)
end_time = time.perf_counter()
print("pdn_small inference time: ", 1e3*(end_time - start_time) / num_reps)

start_time = time.perf_counter()
for i in range(num_reps):
    output = pdn_m(image)
end_time = time.perf_counter()
print("pdn_medium inference time: ", 1e3*(end_time - start_time) / num_reps)