# train_equation_symbol_classifier.py
# Tiny CNN to classify single characters: digits 0-9 and operators + - / √
# Exports ONNX (FP32 + FP16, optional INT8). Aimed for < 1 MB model size.
# ---------------------------------------------------------------
# pip install torch torchvision onnx onnxconverter-common onnxruntime pillow
# (optional) pip install onnxruntime-tools

import json, math, os, random, io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms

import onnx
from onnxconverter_common import float16

# Optional INT8 quantization
try:
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
    ORT_QUANT = True
except Exception:
    ORT_QUANT = False

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Labels
# ---------------------------
DIGITS = [str(i) for i in range(10)]
OPS = ['+', '-', '/', '√']           # divide as '/', square root as '√'
CLASSES = DIGITS + OPS               # total 14
LABEL2ID = {c:i for i,c in enumerate(CLASSES)}
ID2LABEL = {i:c for c,i in LABEL2ID.items()}
Path("labels.json").write_text(json.dumps(ID2LABEL, ensure_ascii=False, indent=2))
print("Labels:", ID2LABEL)

# ---------------------------
# Tiny depthwise-separable CNN (very small)
# ---------------------------
class TinyDSCNN(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.dw1 = nn.Conv2d(1, 1, 3, padding=1, groups=1, bias=False)
        self.pw1 = nn.Conv2d(1, 8, 1, bias=False)
        self.dw2 = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False)
        self.pw2 = nn.Conv2d(8, 16, 1, bias=False)
        self.pool = nn.MaxPool2d(2,2)
        self.dw3 = nn.Conv2d(16,16,3,padding=1,groups=16,bias=False)
        self.pw3 = nn.Conv2d(16,32,1,bias=False)
        self.head = nn.Linear(32*7*7, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.pw1(self.dw1(x))))   # 8x14x14
        x = self.pool(F.relu(self.pw2(self.dw2(x))))   # 16x7x7
        x = F.relu(self.pw3(self.dw3(x)))              # 32x7x7
        x = x.reshape(x.size(0), -1)
        return self.head(x)

# ---------------------------
# Synthetic operators dataset (on-the-fly)
# ---------------------------
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

def try_load_font(size):
    # Try a few common fonts; fall back to PIL default
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",          # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # Linux
        "C:\\Windows\\Fonts\\arial.ttf"                          # Windows
    ]
    for p in candidates:
        if Path(p).exists():
            try: return ImageFont.truetype(p, size=size)
            except Exception: pass
    return ImageFont.load_default()

class OperatorSynth(Dataset):
    """
    Generates operator glyphs (+ - / √) as 28x28 white-on-black images,
    with small jitter/rotation/thickness to mimic handwriting.
    """
    def __init__(self, length=24000, train=True):
        self.length = length
        self.train = train
        self.ops = OPS
        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
    def __len__(self): return self.length
    def _rand_params(self):
        rot = random.uniform(-25, 25) if self.train else random.uniform(-10,10)
        thick = random.randint(2, 5) if self.train else 3
        scale = random.uniform(0.75, 1.05)
        dx = random.randint(-3,3) if self.train else random.randint(-1,1)
        dy = random.randint(-3,3) if self.train else random.randint(-1,1)
        return rot, thick, scale, dx, dy
    def _draw_op(self, op):
        img = Image.new("L", (28,28), 0)  # black bg
        draw = ImageDraw.Draw(img)
        rot, thick, scale, dx, dy = self._rand_params()
        # draw using primitives for crisp lines
        w,h = img.size
        if op == '+':
            draw.line((4,14,24,14), fill=255, width=thick)
            draw.line((14,4,14,24), fill=255, width=thick)
        elif op == '-':
            draw.line((5,14,23,14), fill=255, width=thick)
        elif op == '/':
            draw.line((6,24,22,4), fill=255, width=thick)
        elif op == '√':
            # simple root shape using polyline
            pts = [(6,16),(11,22),(20,6)]
            draw.line(pts, fill=255, width=thick, joint="curve")
        # scale & translate
        if scale != 1.0:
            new = int(w*scale), int(h*scale)
            img = ImageOps.contain(img, new)
            bg = Image.new("L",(28,28),0)
            bg.paste(img, ((28-img.size[0])//2, (28-img.size[1])//2))
            img = bg
        if dx or dy:
            bg = Image.new("L",(28,28),0); bg.paste(img,(dx,dy))
            img = bg
        # rotate
        img = img.rotate(rot, resample=Image.BILINEAR, fillcolor=0)
        # tiny blur sometimes to mimic pen
        if self.train and random.random()<0.3:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0,0.6)))
        return img
    def __getitem__(self, idx):
        op = random.choice(self.ops)
        label = LABEL2ID[op]
        img = self._draw_op(op)
        x = self.tfm(img)  # tensor normalized as MNIST
        return x, label

# ---------------------------
# Data loaders (MNIST + operators)
# ---------------------------
def get_loaders(batch_train=256, batch_test=512, num_workers=2):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    mnist_test  = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    # Wrap MNIST labels 0-9 already correct
    ops_train = OperatorSynth(length=24000, train=True)
    ops_test  = OperatorSynth(length=4000, train=False)

    train = ConcatDataset([mnist_train, ops_train])
    test  = ConcatDataset([mnist_test, ops_test])

    train_loader = DataLoader(train, batch_size=batch_train, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=batch_test, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

# ---------------------------
# Train / Eval
# ---------------------------
def train_and_eval(epochs=6, lr=2e-3, device="cpu", save_path="eqsym_tiny.pt"):
    train_loader, test_loader = get_loaders()
    model = TinyDSCNN(num_classes=len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()

        # eval
        model.eval(); correct=total=0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        acc = correct/total
        print(f"Epoch {epoch}: test acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), save_path)
    print("Best accuracy:", best)
    return best

# ---------------------------
# Export ONNX
# ---------------------------
def export_onnx(fp32_path="eqsym_tiny_fp32.onnx", fp16_path="eqsym_tiny_fp16.onnx"):
    model = TinyDSCNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load("eqsym_tiny.pt", map_location="cpu"))
    model.eval()
    dummy = torch.zeros(1,1,28,28, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, fp32_path,
        input_names=["input"], output_names=["logits"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}}
    )
    print("Exported:", fp32_path)
    m32 = onnx.load(fp32_path)
    m16 = float16.convert_float_to_float16(m32, keep_io_types=True)
    onnx.save(m16, fp16_path)
    print("Saved FP16:", fp16_path)

# Optional INT8 (calibration on-the-fly using MNIST + ops)
class CalibReader(CalibrationDataReader):
    def __init__(self, n=400, batch=50):
        self.input_name = "input"
        tfm = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
        self.mnist = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=tfm),
                                batch_size=batch, shuffle=True)
        self.ops   = DataLoader(OperatorSynth(length=4000, train=False),
                                batch_size=batch, shuffle=True)
        self.iters = iter(self._mix())
        self.total = n
    def _mix(self):
        for (x,_),(xo,_) in zip(self.mnist, self.ops):
            yield x
            yield xo
    def get_next(self):
        if self.total <= 0: return None
        try:
            x = next(self.iters)
        except StopIteration:
            return None
        self.total -= x.size(0)
        return {self.input_name: x.numpy().astype("float32")}

def quantize_int8(fp32="eqsym_tiny_fp32.onnx", out="eqsym_tiny_int8.onnx"):
    if not ORT_QUANT:
        print("onnxruntime.quantization not available; skipping INT8.")
        return
    calib = CalibReader(n=400, batch=50)
    quantize_static(
        model_input=fp32,
        model_output=out,
        calibration_data_reader=calib,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8
    )
    print("Saved INT8:", out)

def show_sizes(paths):
    for p in paths:
        if Path(p).exists():
            kb = Path(p).stat().st_size/1024
            print(f"{p}: {kb:.1f} KB")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    best = train_and_eval(epochs=6, lr=2e-3, device=device, save_path="eqsym_tiny.pt")
    export_onnx("eqsym_tiny_fp32.onnx", "eqsym_tiny_fp16.onnx")
    quantize_int8("eqsym_tiny_fp32.onnx", "eqsym_tiny_int8.onnx")
    show_sizes(["eqsym_tiny_fp32.onnx","eqsym_tiny_fp16.onnx","eqsym_tiny_int8.onnx","labels.json"])

