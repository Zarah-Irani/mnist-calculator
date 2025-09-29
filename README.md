<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MNIST Calculator – README</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f1220;
      color: #eef1f7;
      margin: 0 auto;
      max-width: 900px;
      padding: 40px;
      line-height: 1.6;
    }
    h1, h2, h3 { color: #7c9cf5; }
    a { color: #58a6ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    code {
      background: #171a2b;
      color: #eaf0ff;
      padding: 2px 6px;
      border-radius: 6px;
    }
    pre {
      background: #171a2b;
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
    }
    .highlight { color: #ffcc66; font-weight: bold; }
    .box {
      border: 1px solid rgba(255,255,255,0.15);
      padding: 16px;
      border-radius: 8px;
      margin: 20px 0;
      background: rgba(255,255,255,0.05);
    }
  </style>
</head>
<body>

<h1>✍️ Handwritten Equation → Calculator (ONNX Runtime Web)</h1>

<p>
An interactive browser demo where you can <span class="highlight">write math expressions by hand</span> 
(e.g., <code>3+4</code>, <code>12-5/7</code>, <code>√9+2</code>) on a canvas and instantly compute the result.  
It uses a tiny PyTorch CNN trained on digits and math symbols, exported to ONNX, and runs entirely 
client-side with <a href="https://onnxruntime.ai/">onnxruntime-web</a> (WebGPU/WASM).
</p>

<p>
👉 <strong>Live Demo:</strong> <a href="https://zarah-irani.github.io/mnist-calculator/">https://zarah-irani.github.io/mnist-calculator/</a>
</p>

<div class="box">
⚡ Apple announced “Math Notes” in 2023 like it was revolutionary — meanwhile, this repo runs the <em>same idea</em> fully in your browser with no iCloud, no locked ecosystem, and under <strong>1&nbsp;MB</strong> of models. 🙃
</div>

<h2>✨ Features</h2>
<ul>
  <li>🖊️ <strong>Freehand Equation Input</strong> – Write multi-character math expressions directly on canvas.</li>
  <li>🔍 <strong>Expression Parsing</strong> – Converts recognized digits/symbols into a full equation string.</li>
  <li>⚡ <strong>Instant Calculation</strong> – Evaluates the expression safely in-browser.</li>
  <li>🧮 <strong>Supported Symbols</strong> – Digits 0–9, operators + - / √.</li>
  <li>🌐 <strong>Static Hosting</strong> – Works fully offline or on GitHub Pages (no server needed).</li>
</ul>

<h2>📂 Project Structure</h2>
<pre><code>.
├── index.html              # Web UI + inference
├── train.py                # Training & ONNX export script
├── eqsym_tiny.pt           # Trained PyTorch weights
├── eqsym_tiny_fp32.onnx    # ONNX model (FP32)
├── eqsym_tiny_fp16.onnx    # ONNX model (FP16)
├── eqsym_tiny_int8.onnx    # Quantized ONNX (INT8, tiny size)
├── labels.json             # Label mapping (digits + operators)
└── README.html             # This file
</code></pre>

<h2>🚀 How to Run</h2>
<h3>Frontend (Inference)</h3>
<ol>
  <li>Open <code>index.html</code> in your browser (or deploy via GitHub Pages).</li>
  <li>Draw an equation on the canvas.</li>
  <li>Click <strong>Compute</strong> → See recognized expression + calculated answer.</li>
</ol>

<h3>Training (PyTorch → ONNX)</h3>
<ol>
  <li>Install requirements:</li>
</ol>
<pre><code>pip install torch torchvision onnx onnxruntime onnxconverter-common onnxruntime-tools pillow
</code></pre>
<ol start="2">
  <li>Train & export:</li>
</ol>
<pre><code>python train.py
</code></pre>
<ul>
  <li>Saves PyTorch weights → <code>eqsym_tiny.pt</code></li>
  <li>Exports ONNX → FP32, FP16, INT8</li>
  <li>Generates <code>labels.json</code></li>
</ul>

<h2>📈 Extensions</h2>
<ul>
  <li>Add multiplication <code>*</code> support.</li>
  <li>Train on more handwriting data (real user samples).</li>
  <li>Add LaTeX-style rendering for neat math display.</li>
  <li>Mobile-friendly optimized canvas for touchscreens.</li>
</ul>

<h2>📜 License</h2>
<p>MIT License – free to use, fork, and modify.</p>

<p>
👩‍💻 Built as a portfolio project to showcase <strong>handwriting recognition + in-browser ML with ONNX Runtime Web</strong>.  
(And yes, it does what Apple bragged about — only without needing a $1500 laptop 😉).
</p>

</body>
</html>
