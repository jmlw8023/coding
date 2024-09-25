


import torch
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import PIL
import IPython
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题



def crop(pic, box):
    img = to_pil_image(pic.cpu())
    cropped_img = img.crop(box)
    return pil_to_tensor(cropped_img).to(pic.device) / 255.

def display(img):
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.show()

# img = torch.ones(3, 64, 64)
# img *= torch.linspace(0, 1, steps=64) * torch.linspace(0, 1, steps=64).unsqueeze(-1)
# display(img)



# cropped_img = crop(img, (10, 10, 50, 50))
# display(cropped_img)


import numpy as np

@torch.library.custom_op("mylib::numpy_sin", mutates_args={"output"}, device_types="cpu")
def numpy_sin(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.device == output.device
    assert input.device.type == "cpu"
    input_np = input.numpy()
    output_np = output.numpy()
    np.sin(input_np, out=output_np)


@torch.compile(fullgraph=True)
def f(x):
    out = torch.empty(3)
    numpy_sin(x, out)
    return out

x = torch.randn(3)
y = f(x)
assert torch.allclose(y, x.sin())












