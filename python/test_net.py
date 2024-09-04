import numpy as np
import PIL.Image as Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import time

model = models.resnet18(pretrained=True)
model.eval().cuda()

tr = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
)

img = Image.open("../images/cat.ppm")

tens = tr(img).unsqueeze(0)
batch_size = 16
iters = 1
tens = torch.ones(size=(batch_size, 3, 224, 224)).float()

t = 0
for i in range(iters):
    s = time.time()

    tens = tens.cuda()
    out = model(tens)
    out.cpu()

    predicted, index  = torch.max(out1, 1)


    e = time.time()
    t += e - s

print("FPS: ", batch_size*iters/t)
