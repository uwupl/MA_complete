import torch
import torchvision.models as models
import time

# Load the models
resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
wide_resnet50 = models.wide_resnet50_2(pretrained=True)

# Set the models to evaluation mode
resnet18.eval()
resnet34.eval()
wide_resnet50.eval()

# Load a sample image
image = torch.randn(1, 3, 224, 224)

# Set the number of repetitions
num_reps = 10

# Test the inference time of each model
start_time = time.perf_counter()
for i in range(num_reps):
    output = resnet18(image)
end_time = time.perf_counter()
print("ResNet18 inference time: ", (end_time - start_time) / num_reps)

start_time = time.perf_counter()
for i in range(num_reps):
    output = resnet34(image)
end_time = time.perf_counter()
print("ResNet34 inference time: ", (end_time - start_time) / num_reps)

start_time = time.perf_counter()
for i in range(num_reps):
    output = wide_resnet50(image)
end_time = time.perf_counter()
print("WideResNet50 inference time: ", (end_time - start_time) / num_reps)