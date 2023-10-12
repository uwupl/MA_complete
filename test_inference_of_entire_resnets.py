import torch
import torchvision.models as models
import time
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the models
resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
wide_resnet50 = models.wide_resnet50_2(pretrained=True)

# Set the models to evaluation mode
resnet18.eval()
resnet34.eval()
wide_resnet50.eval()
image = torch.randn(1, 3, 224, 224)
if torch.cuda.is_available():
    print('hey')
    resnet18 = resnet18.cuda()
    resnet34 = resnet34.cuda()
    wide_resnet50 = wide_resnet50.cuda()
    

# Load a sample image
    image = image.cuda()

# Set the number of repetitions
num_reps = 100
# warm up
for i in range(num_reps):
    output = resnet18(image)
    output = resnet34(image)
    output = wide_resnet50(image)    
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