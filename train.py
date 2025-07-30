import swanlab
import torch

import model
import data_process

swanlab.init(
    # 实验名
    experiment_name='ResNet50-binary-classification',
    description='Pretrained-ResNet50-fc2-binary-classification: dogs and cats',
    # 记录超参数
    config={
        "model": "resnet50",
        "optimizer": model.optimizer,
        "lr": model.learning_rate,
        "batch_size": data_process.batch_size,
        "epochs": model.total_epochs,
        "device": model.device,
        "num_classes": model.num_classes,
    },
    mode="local"
)
model.model_pretrained.train()
for epoch in range(model.total_epochs):
    for idx, (inputs, targets) in enumerate(data_process.train_dataloader):
        # print(targets.shape)
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        outputs = model.model_pretrained(inputs)
        model.optimizer.zero_grad()
        loss = model.loss_fn(outputs, targets)
        loss.backward()
        model.optimizer.step()
        print(f'Epoch {epoch + 1}/{model.total_epochs}: Loss {loss.item():.4f}')
        swanlab.log({'loss': loss.item()})

model.model_pretrained.eval()
correct, total = 0, 0
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(data_process.val_dataloader):
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        outputs = model.model_pretrained(inputs)
        _, predicted = outputs.max(1)   # outputs.max(1) 返回(值，索引)的张量
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100 * correct / total
    print(f'Accuracy: {acc:.2f}%')
    swanlab.log({'val accuracy': acc})

if acc > 0.9:
    torch.save(model.model_pretrained.state_dict(), './saved_models/classifier_model.pth')
