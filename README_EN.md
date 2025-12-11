# Brainy ML

> Fast AI/ML framework for Bun/Node.js with GPU/CPU support, automatic differentiation and PyTorch-like API

[![npm version](https://img.shields.io/npm/v/brainy-ml.svg)](https://www.npmjs.com/package/brainy-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)

## Features

- **PyTorch-like API** - Familiar interface for PyTorch users
- **Automatic Differentiation** - Full autograd with backward() for any computation graph
- **GPU Support** - WebGPU acceleration for massive speedups
- **Pre-built Models** - GPT, Transformers, VAE, TRM and more
- **Hugging Face Integration** - Download models directly from HF Hub
- **Type-Safe** - Full TypeScript support with comprehensive type definitions
- **Lightweight** - Zero heavy dependencies, works in Bun and Node.js

## Installation

```bash
# Using Bun (recommended)
bun add brainy-ml

# Using npm
npm install brainy-ml

# Using yarn
yarn add brainy-ml
```

## Quick Start

### Basic Tensor Operations

```typescript
import { tensor, zeros, ones, randn } from 'brainy-ml';

// Create tensors
const a = tensor([[1, 2], [3, 4]]);
const b = zeros([2, 2]);
const c = randn([3, 3]);

// Operations
const sum = a.add(b);
const product = a.matmul(a.T);  // Matrix multiplication
const mean = c.mean();

console.log('Shape:', sum.shape);      // [2, 2]
console.log('Product:', product.toArray());
console.log('Mean:', mean.item());
```

### Automatic Differentiation

```typescript
import { tensor } from 'brainy-ml';

// Create tensor with gradient tracking
const x = tensor([[2.0]], { requiresGrad: true });

// y = x^2 + 3x + 1
const y = x.pow(2).add(x.mul(3)).add(1);

// Compute gradients
y.backward();

// dy/dx = 2x + 3 = 2*2 + 3 = 7
console.log('x:', x.item());           // 2
console.log('y:', y.item());           // 11
console.log('dy/dx:', x.grad.item());  // 7
```

### Building Neural Networks

```typescript
import {
  Sequential, Linear, ReLU, Sigmoid,
  MSELoss, Adam, tensor
} from 'brainy-ml';

// Build a model
const model = new Sequential(
  new Linear(10, 64),
  new ReLU(),
  new Linear(64, 32),
  new ReLU(),
  new Linear(32, 1),
  new Sigmoid()
);

// Setup training
const optimizer = new Adam(model.parameters(), 0.001);
const criterion = new MSELoss();

// Training loop
for (let epoch = 0; epoch < 100; epoch++) {
  const x = tensor([[/* your data */]]);
  const y = tensor([[/* your labels */]]);

  const pred = model.forward(x);
  const loss = criterion.forward(pred, y);

  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();

  if (epoch % 10 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${loss.item()}`);
  }
}
```

### XOR Problem Example

```typescript
import {
  Sequential, Linear, ReLU, Sigmoid,
  MSELoss, Adam, tensor
} from 'brainy-ml';

// XOR data
const X = tensor([[0,0], [0,1], [1,0], [1,1]]);
const Y = tensor([[0], [1], [1], [0]]);

// Simple network
const model = new Sequential(
  new Linear(2, 8),
  new ReLU(),
  new Linear(8, 1),
  new Sigmoid()
);

const optimizer = new Adam(model.parameters(), 0.1);
const criterion = new MSELoss();

// Train
for (let i = 0; i < 1000; i++) {
  const pred = model.forward(X);
  const loss = criterion.forward(pred, Y);

  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();
}

// Test
console.log('Predictions:', model.forward(X).toArray());
// Expected: [[~0], [~1], [~1], [~0]]
```

### Loading Models from Hugging Face

```typescript
import { HuggingFaceHub } from 'brainy-ml';

const hub = new HuggingFaceHub();

// Get model info
const info = await hub.getModelInfo('Qwen/Qwen2.5-0.5B');
console.log('Model:', info.id);

// Download configuration
const config = await hub.downloadConfig('Qwen/Qwen2.5-0.5B');
console.log('Hidden size:', config.hidden_size);

// Download weights (safetensors format)
const weights = await hub.downloadWeights('Qwen/Qwen2.5-0.5B');
console.log('Loaded', weights.size, 'tensors');
```

## API Reference

### Core

| Function | Description |
|----------|-------------|
| `tensor(data, options?)` | Create tensor from nested array |
| `zeros(shape, options?)` | Create tensor filled with zeros |
| `ones(shape, options?)` | Create tensor filled with ones |
| `rand(shape, options?)` | Uniform random tensor [0, 1) |
| `randn(shape, mean?, std?, options?)` | Normal random tensor |
| `eye(n, options?)` | Identity matrix |
| `linspace(start, end, steps, options?)` | Evenly spaced values |
| `arange(start, end?, step?, options?)` | Range of values |

### Tensor Operations

| Method | Description |
|--------|-------------|
| `.add(other)` | Element-wise addition |
| `.sub(other)` | Element-wise subtraction |
| `.mul(other)` | Element-wise multiplication |
| `.div(other)` | Element-wise division |
| `.matmul(other)` | Matrix multiplication |
| `.pow(exp)` | Power |
| `.sqrt()` | Square root |
| `.exp()` | Exponential |
| `.log()` | Natural logarithm |
| `.sum(dim?, keepdim?)` | Sum reduction |
| `.mean(dim?, keepdim?)` | Mean reduction |
| `.max(dim?, keepdim?)` | Maximum |
| `.min(dim?, keepdim?)` | Minimum |
| `.reshape(...shape)` | Reshape tensor |
| `.transpose(dim0, dim1)` | Swap dimensions |
| `.T` | Shortcut for transpose(0, 1) |
| `.backward()` | Compute gradients |

### Neural Network Layers

| Class | Description |
|-------|-------------|
| `Linear(in, out, bias?)` | Fully connected layer |
| `Conv2d(in, out, kernel, stride?, padding?)` | 2D convolution |
| `MaxPool2d(kernel, stride?, padding?)` | Max pooling |
| `AvgPool2d(kernel, stride?, padding?)` | Average pooling |
| `BatchNorm1d(features)` | 1D batch normalization |
| `BatchNorm2d(features)` | 2D batch normalization |
| `LayerNorm(shape)` | Layer normalization |
| `Dropout(p)` | Dropout regularization |
| `Embedding(num, dim)` | Embedding layer |
| `Flatten(startDim?, endDim?)` | Flatten dimensions |

### Activations

| Class | Description |
|-------|-------------|
| `ReLU()` | Rectified Linear Unit |
| `LeakyReLU(slope?)` | Leaky ReLU |
| `GELU()` | Gaussian Error Linear Unit |
| `SiLU()` / `Swish()` | Sigmoid Linear Unit |
| `Sigmoid()` | Sigmoid activation |
| `Tanh()` | Hyperbolic tangent |
| `Softmax(dim)` | Softmax activation |
| `LogSoftmax(dim)` | Log softmax |

### Loss Functions

| Class | Description |
|-------|-------------|
| `MSELoss()` | Mean Squared Error |
| `L1Loss()` | Mean Absolute Error |
| `CrossEntropyLoss()` | Cross-entropy for classification |
| `BCELoss()` | Binary cross-entropy |
| `BCEWithLogitsLoss()` | BCE with sigmoid |
| `NLLLoss()` | Negative log likelihood |
| `HingeLoss()` | Hinge loss for SVM |
| `KLDivLoss()` | KL Divergence |

### Optimizers

| Class | Description |
|-------|-------------|
| `SGD(params, lr, momentum?)` | Stochastic Gradient Descent |
| `Adam(params, lr, betas?, eps?)` | Adam optimizer |
| `AdamW(params, lr, betas?, weightDecay?)` | Adam with weight decay |
| `RMSprop(params, lr, alpha?)` | RMSprop optimizer |
| `Adagrad(params, lr)` | Adagrad optimizer |

### Learning Rate Schedulers

| Class | Description |
|-------|-------------|
| `StepLR(optimizer, stepSize, gamma?)` | Step decay |
| `ExponentialLR(optimizer, gamma)` | Exponential decay |
| `CosineAnnealingLR(optimizer, tMax)` | Cosine annealing |
| `ReduceLROnPlateau(optimizer, mode?, factor?)` | Reduce on plateau |

### Pre-built Models

| Class | Description |
|-------|-------------|
| `GPT(config)` | GPT-style transformer |
| `VAE(config)` | Variational Autoencoder |
| `ConvVAE(config)` | Convolutional VAE |
| `TRM*(config)` | TRM model variants |
| `TransformerEncoder` | Transformer encoder |
| `TransformerDecoder` | Transformer decoder |

### Utilities

| Function/Class | Description |
|----------------|-------------|
| `saveModel(model, path)` | Save model to file |
| `loadModel(model, path)` | Load model from file |
| `HuggingFaceHub` | Hugging Face Hub integration |
| `Quantizer` | Model quantization |
| `LoRALayer` | Low-Rank Adaptation |
| `DataLoader` | Data loading utilities |
| `Tokenizer` | Text tokenization |

## GPU Support

Brainy ML supports WebGPU for GPU acceleration:

```typescript
import { isWebGPUSupported, createDevice } from 'brainy-ml';

if (await isWebGPUSupported()) {
  const device = await createDevice('gpu');
  console.log('GPU enabled!');
}
```

## Examples

Check the `examples/` directory for more examples:

- `01-basic-tensors.ts` - Tensor operations
- `02-autograd.ts` - Automatic differentiation
- `03-linear-regression.ts` - Linear regression
- `04-xor-neural-network.ts` - XOR problem
- `05-mnist-classification.ts` - MNIST classification
- `07-text-generation.ts` - Text generation with GPT
- `26-huggingface-example.ts` - Hugging Face integration
- `27-qwen3-example.ts` - Qwen model loading

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- [Documentation](https://nopass0.github.io/brainy)
- [GitHub Repository](https://github.com/Nopass0/brainy)
- [npm Package](https://www.npmjs.com/package/brainy-ml)
