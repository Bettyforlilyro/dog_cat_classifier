import torch, torchvision
from torchvision import transforms
import model
import gradio as gr

model_save_path = './saved_models/classifier_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

validate_model = torchvision.models.resnet50(weights=None).to(device)
validate_model.fc = torch.nn.Linear(validate_model.fc.in_features, model.num_classes).to(device)
validate_model.load_state_dict(torch.load(model_save_path, map_location=device))
validate_model.eval()


def preprocess_image(image_path):
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return image_transform(image_path).unsqueeze(0).to(device)


def predict(image):
    image_tensor = preprocess_image(image)
    classes = {'0': 'cat', '1': 'dog'}
    with torch.no_grad():
        output = validate_model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1).squeeze()
    # 正确计算概率
    pred_probs = {classes[str(i)]: float(probs[i]) for i in range(len(classes))}
    return pred_probs


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2)
)

if __name__ == '__main__':
    iface.launch()