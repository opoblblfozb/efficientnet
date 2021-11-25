from efficientnet_pytorch import EfficientNet
from torch import nn, optim
import torch

model = EfficientNet.from_pretrained("efficientnet-b0")
model.set_swish(memory_efficient=True)
out_features = 2
model = nn.Sequential(model, nn.Linear(1000, 1))
model.train()
device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_num = 200
opt = optim.Adam(model.parameters())
criterion = nn.MSELoss()
for _ in range(100):
    image = torch.randn([batch_num, 3, 224, 224]).to(device)
    label = torch.randn([1]).to(device)

    opt.zero_grad()
    pred = model(image)
    loss: torch.Tensor = criterion(pred, label)
    loss.backward()

    print("backwarded")
