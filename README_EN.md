# Brainy ML

> Fast AI/ML framework for Bun/Node.js with GPU/CPU support, automatic differentiation and PyTorch-like API

[![npm version](https://img.shields.io/npm/v/brainy-ml.svg)](https://www.npmjs.com/package/brainy-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)

[Russian Documentation](README.md) | [Online Documentation](https://nopass0.github.io/brainy)

## Features

- **PyTorch-like API** - Familiar interface for PyTorch users
- **Automatic Differentiation** - Full autograd with backward() for any computation graph
- **GPU Support** - WebGPU acceleration for massive speedups
- **Pre-built Models** - GPT, Transformers, VAE, TRM and more
- **Hugging Face Integration** - Download models directly from HF Hub
- **Reinforcement Learning** - DQN, Policy Gradient, Actor-Critic agents
- **Quantization & Fine-tuning** - LoRA, Adapters, QAT
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
import { tensor, zeros, ones, randn, eye, linspace, arange } from 'brainy-ml';

// Create tensors
const a = tensor([[1, 2], [3, 4]]);
const b = zeros([2, 2]);
const c = randn([3, 3]);
const identity = eye(3);
const range = linspace(0, 10, 5);  // [0, 2.5, 5, 7.5, 10]

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
import { tensor, noGrad } from 'brainy-ml';

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

// Disable gradients for inference
noGrad(() => {
  const result = x.mul(2);  // No gradient tracking
});
```

### Building Neural Networks

```typescript
import {
  Sequential, Linear, ReLU, Sigmoid, Dropout, BatchNorm1d,
  MSELoss, Adam, tensor
} from 'brainy-ml';

// Create model
const model = new Sequential(
  new Linear(10, 64),
  new BatchNorm1d(64),
  new ReLU(),
  new Dropout(0.2),
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

// Switch between train/eval modes
model.train();  // Enables Dropout, BatchNorm in training mode
model.eval();   // Disables Dropout, BatchNorm in inference mode
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

### Transformers

```typescript
import {
  TransformerEncoder, TransformerDecoder,
  MultiHeadAttention, PositionalEncoding, RotaryEmbedding,
  createCausalMask, tensor
} from 'brainy-ml';

// Multi-Head Attention
const mha = new MultiHeadAttention(512, 8);  // d_model=512, heads=8
const query = tensor(/* [batch, seq, 512] */);
const key = tensor(/* [batch, seq, 512] */);
const value = tensor(/* [batch, seq, 512] */);
const output = mha.forward(query, key, value);

// Positional Encoding
const posEnc = new PositionalEncoding(512, 1000);  // d_model, max_len
const withPos = posEnc.forward(tensor(/* [batch, seq, 512] */));

// Rotary Embeddings (RoPE)
const rope = new RotaryEmbedding(64);  // head_dim
const rotated = rope.forward(query, 0);  // with position offset

// Causal mask for autoregression
const mask = createCausalMask(100);  // [100, 100] mask

// Full Transformer Encoder
const encoder = new TransformerEncoder({
  d_model: 512,
  nhead: 8,
  num_layers: 6,
  dim_feedforward: 2048,
  dropout: 0.1
});

const encoded = encoder.forward(tensor(/* input */));
```

### Loading Models from Hugging Face

```typescript
import { HuggingFaceHub, loadWeightsIntoModel } from 'brainy-ml';

const hub = new HuggingFaceHub();

// Get model info
const info = await hub.getModelInfo('Qwen/Qwen2.5-0.5B');
console.log('Model:', info.id);

// List files in repository
const files = await hub.listFiles('Qwen/Qwen2.5-0.5B');
console.log('Files:', files);

// Download configuration
const config = await hub.downloadConfig('Qwen/Qwen2.5-0.5B');
console.log('Hidden size:', config.hidden_size);

// Download tokenizer
const tokenizer = await hub.downloadTokenizer('Qwen/Qwen2.5-0.5B');

// Download weights (safetensors format)
const weights = await hub.downloadWeights('Qwen/Qwen2.5-0.5B');
console.log('Loaded', weights.size, 'tensors');

// Load weights into model
const result = loadWeightsIntoModel(model, weights);
console.log('Loaded:', result.loaded);
console.log('Missing:', result.missing);
```

### Working with Data

```typescript
import {
  DataLoader, TensorDataset, ArrayDataset,
  trainTestSplit, StreamingDataset, loadJson, loadCsv
} from 'brainy-ml';

// Create dataset from tensors
const X = tensor([[1,2], [3,4], [5,6], [7,8]]);
const Y = tensor([[0], [1], [0], [1]]);
const dataset = new TensorDataset(X, Y);

// DataLoader for batching
const loader = new DataLoader(dataset, {
  batchSize: 2,
  shuffle: true
});

for (const batch of loader) {
  const [inputs, targets] = batch;
  console.log('Batch shape:', inputs.shape);
}

// Train/test split
const [trainData, testData] = trainTestSplit(dataset, 0.8);

// Load data from files
const jsonData = await loadJson('data.json');
const csvData = await loadCsv('data.csv');

// Streaming for large data
const stream = new StreamingDataset('large_data.jsonl', {
  batchSize: 32
});
```

### Text Tokenization

```typescript
import {
  BPETokenizer, WordPieceTokenizer, CharTokenizer,
  loadTokenizer
} from 'brainy-ml';

// BPE Tokenizer
const bpe = new BPETokenizer();
await bpe.train(['Hello world', 'How are you?'], { vocabSize: 1000 });
const encoded = bpe.encode('Hello world');
const decoded = bpe.decode(encoded.ids);

// WordPiece Tokenizer
const wp = new WordPieceTokenizer();
const tokens = wp.encode('Running quickly');

// Load tokenizer from HuggingFace
const tokenizer = await loadTokenizer('bert-base-uncased');
```

### Reinforcement Learning

```typescript
import {
  DQNAgent, PolicyGradientAgent, ActorCriticAgent,
  ReplayBuffer, CartPoleEnv, trainDQN
} from 'brainy-ml';

// DQN Agent
const agent = new DQNAgent({
  stateSize: 4,
  actionSize: 2,
  hiddenSize: 64,
  learningRate: 0.001,
  gamma: 0.99,
  epsilon: 1.0,
  epsilonDecay: 0.995,
  epsilonMin: 0.01
});

// CartPole Environment
const env = new CartPoleEnv();

// Training
for (let episode = 0; episode < 500; episode++) {
  let state = env.reset();
  let totalReward = 0;

  while (true) {
    const action = agent.act(state);
    const { nextState, reward, done } = env.step(action);
    agent.remember(state, action, reward, nextState, done);
    agent.replay(32);  // batch size

    state = nextState;
    totalReward += reward;
    if (done) break;
  }

  console.log(`Episode ${episode}: ${totalReward}`);
}
```

### Fine-tuning with LoRA

```typescript
import {
  createLoRAModel, LoRALayer, AdapterLayer,
  FineTuneTrainer, fineTune
} from 'brainy-ml';

// Create LoRA model
const loraModel = createLoRAModel(baseModel, {
  rank: 8,
  alpha: 16,
  targetModules: ['query', 'value']
});

// Fine-tuning
const trainer = new FineTuneTrainer(loraModel, {
  learningRate: 1e-4,
  epochs: 3,
  warmupSteps: 100
});

await trainer.train(trainDataset, valDataset);

// Or quick fine-tune
await fineTune(model, dataset, {
  strategy: 'lora',
  rank: 4
});
```

### Quantization

```typescript
import {
  Quantizer, dynamicQuantize, prepareQAT, convertQAT,
  QuantizationMode, getModelSize
} from 'brainy-ml';

// Dynamic quantization
const quantizedModel = dynamicQuantize(model, {
  bits: 8,
  mode: QuantizationMode.DYNAMIC
});

// QAT (Quantization-Aware Training)
const qatModel = prepareQAT(model);
// ... training ...
const finalModel = convertQAT(qatModel);

// Check model size
const sizeBefore = getModelSize(model);
const sizeAfter = getModelSize(quantizedModel);
console.log(`Compression: ${sizeBefore / sizeAfter}x`);
```

### Model Visualization

```typescript
import { visualize, summary, printModel } from 'brainy-ml';

// ASCII visualization of architecture
visualize(model);

// Detailed summary
summary(model, [1, 10]);  // input shape

// Print parameters
printModel(model);
```

### Saving and Loading

```typescript
import {
  saveModel, loadModel, saveStateDict, loadStateDict,
  saveCheckpoint, loadCheckpoint, exportModel
} from 'brainy-ml';

// Simple save/load
await saveModel(model, 'model.bin');
await loadModel(model, 'model.bin');

// State dict
const state = model.stateDict();
await saveStateDict(state, 'weights.bin');
const loadedState = await loadStateDict('weights.bin');
model.loadStateDict(loadedState);

// Checkpoints with metadata
await saveCheckpoint({
  model: model.stateDict(),
  optimizer: optimizer.stateDict(),
  epoch: 10,
  loss: 0.05
}, 'checkpoint.bin');

const checkpoint = await loadCheckpoint('checkpoint.bin');
model.loadStateDict(checkpoint.model);
optimizer.loadStateDict(checkpoint.optimizer);

// Export for inference
await exportModel(model, 'model_inference.bin', {
  compress: true
});
```

## API Reference

### Tensor Creation

| Function | Description |
|----------|-------------|
| `tensor(data, options?)` | Create tensor from nested array |
| `scalar(value, options?)` | Scalar tensor |
| `zeros(shape, options?)` | Tensor filled with zeros |
| `ones(shape, options?)` | Tensor filled with ones |
| `full(shape, value, options?)` | Tensor filled with value |
| `rand(shape, options?)` | Uniform random [0, 1) |
| `randn(shape, mean?, std?, options?)` | Normal distribution |
| `eye(n, options?)` | Identity matrix |
| `linspace(start, end, steps, options?)` | Evenly spaced values |
| `arange(start, end?, step?, options?)` | Range of values |
| `cat(tensors, dim?)` | Concatenate tensors |
| `stack(tensors, dim?)` | Stack tensors |

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
| `.abs()` | Absolute value |
| `.neg()` | Negation |
| `.sin()`, `.cos()`, `.tan()` | Trigonometry |
| `.sum(dim?, keepdim?)` | Sum reduction |
| `.mean(dim?, keepdim?)` | Mean reduction |
| `.max(dim?, keepdim?)` | Maximum |
| `.min(dim?, keepdim?)` | Minimum |
| `.argmax(dim?)` | Index of maximum |
| `.argmin(dim?)` | Index of minimum |
| `.reshape(...shape)` | Reshape tensor |
| `.view(...shape)` | View reshape |
| `.transpose(dim0, dim1)` | Swap dimensions |
| `.permute(...dims)` | Permute dimensions |
| `.squeeze(dim?)` | Remove size-1 dimensions |
| `.unsqueeze(dim)` | Add dimension |
| `.T` | Shortcut for transpose(0, 1) |
| `.backward()` | Compute gradients |
| `.detach()` | Detach from gradient graph |
| `.clone()` | Deep copy |

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
| `ELU(alpha?)` | Exponential Linear Unit |
| `SELU()` | Scaled ELU |
| `PReLU(numParams?)` | Parametric ReLU |
| `GELU()` | Gaussian Error Linear Unit |
| `SiLU()` / `Swish()` | Sigmoid Linear Unit |
| `Mish()` | Mish activation |
| `Hardswish()` | Hard Swish |
| `Sigmoid()` | Sigmoid activation |
| `Tanh()` | Hyperbolic tangent |
| `Softmax(dim)` | Softmax |
| `LogSoftmax(dim)` | Log softmax |

### Loss Functions

| Class | Description |
|-------|-------------|
| `MSELoss()` | Mean Squared Error |
| `L1Loss()` | Mean Absolute Error |
| `SmoothL1Loss(beta?)` | Smooth L1 (Huber Loss) |
| `CrossEntropyLoss()` | Cross-entropy for classification |
| `BCELoss()` | Binary cross-entropy |
| `BCEWithLogitsLoss()` | BCE with sigmoid |
| `NLLLoss()` | Negative log likelihood |
| `HingeLoss()` | Hinge loss for SVM |
| `KLDivLoss()` | KL Divergence |
| `CosineEmbeddingLoss()` | Cosine embedding loss |

### Optimizers

| Class | Description |
|-------|-------------|
| `SGD(params, lr, momentum?, weightDecay?)` | Stochastic Gradient Descent |
| `Adam(params, lr, betas?, eps?)` | Adam optimizer |
| `AdamW(params, lr, betas?, weightDecay?)` | Adam with weight decay |
| `RMSprop(params, lr, alpha?, eps?)` | RMSprop optimizer |
| `Adagrad(params, lr, eps?)` | Adagrad optimizer |

### Learning Rate Schedulers

| Class | Description |
|-------|-------------|
| `StepLR(optimizer, stepSize, gamma?)` | Step decay |
| `ExponentialLR(optimizer, gamma)` | Exponential decay |
| `CosineAnnealingLR(optimizer, tMax, etaMin?)` | Cosine annealing |
| `ReduceLROnPlateau(optimizer, mode?, factor?, patience?)` | Reduce on plateau |

### Weight Initialization

| Function | Description |
|----------|-------------|
| `xavierUniform(tensor, gain?)` | Xavier uniform |
| `xavierNormal(tensor, gain?)` | Xavier normal |
| `kaimingUniform(tensor, a?, mode?)` | Kaiming uniform |
| `kaimingNormal(tensor, a?, mode?)` | Kaiming normal |
| `uniform(tensor, a, b)` | Uniform distribution |
| `normal(tensor, mean, std)` | Normal distribution |
| `constant(tensor, value)` | Constant |
| `zeros_(tensor)` | Fill with zeros |
| `ones_(tensor)` | Fill with ones |
| `orthogonal(tensor, gain?)` | Orthogonal initialization |

### Transformers

| Class | Description |
|-------|-------------|
| `MultiHeadAttention(d_model, nhead)` | Multi-head attention |
| `ScaledDotProductAttention()` | Scaled dot-product attention |
| `FeedForward(d_model, dim_ff)` | Feed-forward network |
| `TransformerEncoderBlock(config)` | Encoder block |
| `TransformerDecoderBlock(config)` | Decoder block |
| `TransformerEncoder(config)` | Full encoder |
| `TransformerDecoder(config)` | Full decoder |
| `PositionalEncoding(d_model, max_len)` | Sinusoidal positional encoding |
| `LearnedPositionalEmbedding(max_len, d)` | Learned positions |
| `RotaryEmbedding(dim)` | Rotary Position Embedding (RoPE) |
| `createCausalMask(size)` | Causal mask for autoregression |
| `createPaddingMask(lengths, max_len)` | Padding mask |

### Pre-built Models

| Class | Description |
|-------|-------------|
| `GPT(config)` | GPT-style transformer |
| `createSmallGPT()` | Small GPT model |
| `createMediumGPT()` | Medium GPT model |
| `VAE(config)` | Variational Autoencoder |
| `ConvVAE(config)` | Convolutional VAE |
| `createMNISTVAE()` | VAE for MNIST |
| `TRM(config)` | TRM model |
| `TRMv2(config)` | TRM v2 with attention and MoE |
| `TRMUltra(config)` | TRM Ultra (stable) |
| `TRMFinal(config)` | TRM Final (optimized) |
| `TRMX(config)` | TRM-X (extreme performance) |
| `TRMLite(config)` | TRM-Lite (simple) |
| `TRMPro(config)` | TRM-Pro (98%+ accuracy) |
| `TRMSupreme(config)` | TRM-Supreme with residual scaling |
| `MultimodalFewShot(config)` | Multimodal model |

### Utilities

| Function/Class | Description |
|----------------|-------------|
| `saveModel(model, path)` | Save model |
| `loadModel(model, path)` | Load model |
| `saveCheckpoint(data, path)` | Save checkpoint |
| `loadCheckpoint(path)` | Load checkpoint |
| `HuggingFaceHub` | Hugging Face integration |
| `Quantizer` | Model quantization |
| `LoRALayer` | Low-Rank Adaptation |
| `AdapterLayer` | Adapter layers |
| `DataLoader` | Data loading with batching |
| `Tokenizer` | Text tokenization |
| `visualize(model)` | Architecture visualization |
| `summary(model, inputShape)` | Model summary |
| `manualSeed(seed)` | Set seed for reproducibility |

### Compute

| Function/Class | Description |
|----------------|-------------|
| `isWebGPUSupported()` | Check WebGPU support |
| `createDevice(type)` | Create device (cpu/gpu) |
| `DeviceManager` | Device management |
| `GPUBackend` | GPU backend |
| `WorkerPool` | Worker pool for CPU |
| `HybridEngine` | Hybrid CPU/GPU engine |
| `asyncOps` | Async operations |

## GPU Support

Brainy ML supports WebGPU for GPU acceleration:

```typescript
import { isWebGPUSupported, createDevice, GPUBackend } from 'brainy-ml';

if (await isWebGPUSupported()) {
  const device = await createDevice('gpu');
  console.log('GPU enabled!');

  // GPU backend for tensor operations
  const gpu = await GPUBackend.create();
  const result = await gpu.matmul(a, b);
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
- `08-transformer.ts` - Transformer architecture
- `09-vae.ts` - Variational Autoencoder
- `10-reinforcement-learning.ts` - DQN agent
- `26-huggingface-example.ts` - Hugging Face integration
- `27-qwen3-example.ts` - Qwen model loading

## Running

```bash
# Run example
bun run examples/01-basic-tensors.ts

# Run tests
bun test

# Run specific test
bun test tests/tensor.test.ts
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- [Documentation](https://nopass0.github.io/brainy)
- [GitHub Repository](https://github.com/Nopass0/brainy)
- [npm Package](https://www.npmjs.com/package/brainy-ml)
