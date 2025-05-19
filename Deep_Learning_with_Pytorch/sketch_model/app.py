import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# -------------------------------
# モデル定義
# -------------------------------
norm_layer = nn.InstanceNorm2d

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super().__init__()

        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True),
        )

        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features *= 2
        self.model1 = nn.Sequential(*model1)

        model2 = [ResidualBlock(in_features) for _ in range(n_residual_blocks)]
        self.model2 = nn.Sequential(*model2)

        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                norm_layer(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features //= 2
        self.model3 = nn.Sequential(*model3)

        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        x = self.model0(x)
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        return x

# -------------------------------
# モデルの読み込み
# -------------------------------
model1 = Generator(3, 1, 3)
model1.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model1.eval()

model2 = Generator(3, 1, 3)
model2.load_state_dict(torch.load("model2.pth", map_location=torch.device("cpu")))
model2.eval()

# -------------------------------
# 画像推論関数
# -------------------------------
def darken_pixel(pixel):
    constant = 2.0
    return pixel / constant if pixel < 200 else pixel

def predict(input_img: Image.Image, ver: str) -> Image.Image:
    transform = transforms.Compose([
        transforms.Resize((1080, 1080), Image.BICUBIC),
        transforms.ToTensor(),
    ])
    input_tensor = transform(input_img).unsqueeze(0)

    with torch.no_grad():
        if ver == "Simple Lines":
            output = model2(input_tensor)[0]
        else:
            output = model1(input_tensor)[0]

    output_img = transforms.ToPILImage()(output)
    output_img = output_img.point(darken_pixel)
    return output_img

# -------------------------------
# Gradio インターフェース設定
# -------------------------------
title = "Image to Line Drawings - Complex and Simple Portraits and Landscapes"

examples = [
    ["01.jpg", "Complex Lines"],
    ["02.jpg", "Simple Lines"],
    ["03.jpg", "Simple Lines"],
    ["04.jpg", "Simple Lines"],
    ["05.jpg", "Simple Lines"],
]

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Radio(
            choices=["Complex Lines", "Simple Lines"],
            value="Simple Lines",
            label="Version",
        ),
    ],
    outputs=gr.Image(type="pil", label="Line Drawing"),
    title=title,
    #examples=examples,
)

iface.launch()