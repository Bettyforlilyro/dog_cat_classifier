import torch.nn
import torchvision
from torchvision.models import ResNet50_Weights

# 加载预训练的ResNet50模型
model_pretrained = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

num_classes = 2
# 替换全连接层输出维度
model_pretrained.fc = torch.nn.Linear(model_pretrained.fc.in_features, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_pretrained.to(device)
total_epochs = 20
learning_rate = 1e-4
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_pretrained.parameters(), lr=learning_rate)

