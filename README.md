# ✍️ Handwritten Equation → Calculator (ONNX Runtime Web)

Write a whole math expression by hand (e.g., `3+4`, `12-5/7`, `√9+2`) on a canvas and get the answer instantly — all **in your browser**.  
A tiny PyTorch CNN is exported to **ONNX** and runs with **onnxruntime-web** (WASM/WebGPU). No servers, no backends.

**Live demo:** https://zarah-irani.github.io/mnist-calculator/

> ⚡ Apple rolled out “Math Notes” like it was brand new — this project does the idea **fully client-side** with models under ~1 MB, no iCloud required. 😉

---

## ✨ Features
- **Freehand input:** draw full expressions on one canvas
- **Segmentation + recognition:** connected components → per-glyph ONNX inference
- **Safe evaluation:** parses tokens and computes result in-browser
- **Symbols supported:** digits `0–9`, operators `+ - / √`
- **Static hosting:** works on GitHub Pages

> Note: Parentheses are **not** included in the current model classes; examples above avoid `()` by design.

---

## 📦 Project Structure

