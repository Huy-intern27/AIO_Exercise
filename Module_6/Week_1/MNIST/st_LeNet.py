import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from PIL import Image # type: ignore
import streamlit as st # type: ignore
from torchvision import transforms # type: ignore

class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LeNetClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding='same')
        self.avg1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avg1(outputs)
        outputs = F.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.avg2(outputs)
        outputs = F.relu(outputs)

        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)

        return outputs

@st.cache_resource
def load_model(model_path, num_classes=10):
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    lenet_model.eval()

    return lenet_model

model = load_model('data/lenet_model.pt')

def inference(image, model):
    w, h = image.size
    if w != h:
        crop = transforms.CenterCrop(min(w, h))
        image = crop(image)
        img_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])

        img_new = img_transforms(image)
        img_new = img_new.expand(1, 1, 28, 28)
        with torch.no_grad():
            predictions = model(img_new)
        preds = nn.Softmax(dim=1)(predictions)
        p_max, y_hat = torch.max(preds.data, 1)

        return p_max.item() * 100, y_hat.item()

def main():
    st.title('Digit Recognition')
    st.subheader('Model: Lenet. Dataset: MNIST')
    options = st.selectbox('How would you like to give the input ?', ('Upload Image', 'Run Example Image'))
    if options == 'Upload Image':
        file = st.file_uploader('Please upload an image of a digit', type=['jpg', 'png'])
        if file is not None:
            image = Image.open(file)
            p, label = inference(image, model)
            st.image(image)
            st.success(f'The uploaded image of the digit {label} with {p:.2f}% probability.')

    elif options == 'Run Example Image':
        image = Image.open('data/demo_8.png')
        p, label = inference(image, model)
        st.image(image)
        st.success(f'The image of the digit {label} with {p:2.f}% probability.')

if __name__ == '__main__':
    main()