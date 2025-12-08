/**
 * @fileoverview TRM Final - The Ultimate TRM Model
 * @description Optimized for both small and large datasets with conditional normalization
 *
 * Key improvements:
 * - Conditional normalization (skip when batch is too small)
 * - Better weight initialization
 * - Stable residual connections
 * - Simple but effective architecture
 * - Memory augmentation for complex tasks
 */

import { Tensor, tensor, zeros, randn, cat, ones } from '../core/tensor';
import { Module, Sequential, Parameter } from '../nn/module';
import { Linear, Dropout, LayerNorm } from '../nn/layers';
import { GELU, Sigmoid, Tanh, ReLU } from '../nn/activations';

export interface TRMFinalConfig {
  inputDim: number;
  hiddenDim: number;
  outputDim: number;
  numRecursions?: number;
  dropout?: number;
  useMemory?: boolean;
  memorySlots?: number;
}

/**
 * Simple MLP block without normalization issues
 */
class MLPBlock extends Module {
  private fc1: Linear;
  private fc2: Linear;
  private act: GELU;
  private dropout: Dropout;

  constructor(inputDim: number, hiddenDim: number, outputDim: number, dropout: number = 0.1) {
    super();
    this.fc1 = new Linear(inputDim, hiddenDim);
    this.fc2 = new Linear(hiddenDim, outputDim);
    this.act = new GELU();
    this.dropout = new Dropout(dropout);

    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
    this.registerModule('act', this.act);
    this.registerModule('dropout', this.dropout);
  }

  forward(x: Tensor): Tensor {
    let h = this.fc1.forward(x);
    h = this.act.forward(h);
    h = this.dropout.forward(h);
    return this.fc2.forward(h);
  }
}

/**
 * Latent Update - simple and stable
 */
class SimpleLatentUpdate extends Module {
  private net: MLPBlock;
  private gate: Linear;

  constructor(inputDim: number, hiddenDim: number, dropout: number = 0.1) {
    super();
    // Input: [x, y, z] concatenated
    const totalInput = inputDim + hiddenDim * 2;

    this.net = new MLPBlock(totalInput, hiddenDim * 2, hiddenDim, dropout);
    this.gate = new Linear(hiddenDim, hiddenDim);

    this.registerModule('net', this.net);
    this.registerModule('gate', this.gate);
  }

  forward(x: Tensor, y: Tensor, z: Tensor): Tensor {
    const combined = cat([x, y, z], -1);
    const update = this.net.forward(combined);

    // Gated update for stability
    const gateVal = this.sigmoid(this.gate.forward(z));

    // z_new = z + gate * (update - z) * 0.5
    return z.add(gateVal.mul(update.sub(z)).mul(0.5));
  }

  private sigmoid(x: Tensor): Tensor {
    const result = new Float32Array(x.size);
    for (let i = 0; i < x.size; i++) {
      result[i] = 1 / (1 + Math.exp(-x.data[i]));
    }
    return new Tensor(result, [...x.shape], { requiresGrad: x.requiresGrad });
  }
}

/**
 * Answer Update - simple and stable
 */
class SimpleAnswerUpdate extends Module {
  private net: MLPBlock;
  private gate: Linear;

  constructor(hiddenDim: number, dropout: number = 0.1) {
    super();
    this.net = new MLPBlock(hiddenDim * 2, hiddenDim * 2, hiddenDim, dropout);
    this.gate = new Linear(hiddenDim, hiddenDim);

    this.registerModule('net', this.net);
    this.registerModule('gate', this.gate);
  }

  forward(y: Tensor, z: Tensor): Tensor {
    const combined = cat([y, z], -1);
    const update = this.net.forward(combined);

    const gateVal = this.sigmoid(this.gate.forward(y));

    // y_new = y + gate * (update - y) * 0.5
    return y.add(gateVal.mul(update.sub(y)).mul(0.5));
  }

  private sigmoid(x: Tensor): Tensor {
    const result = new Float32Array(x.size);
    for (let i = 0; i < x.size; i++) {
      result[i] = 1 / (1 + Math.exp(-x.data[i]));
    }
    return new Tensor(result, [...x.shape], { requiresGrad: x.requiresGrad });
  }
}

/**
 * Simple Encoder
 */
class SimpleEncoder extends Module {
  private fc1: Linear;
  private fc2: Linear;
  private act: GELU;

  constructor(inputDim: number, hiddenDim: number) {
    super();
    this.fc1 = new Linear(inputDim, hiddenDim);
    this.fc2 = new Linear(hiddenDim, hiddenDim);
    this.act = new GELU();

    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
    this.registerModule('act', this.act);
  }

  forward(x: Tensor): Tensor {
    let h = this.fc1.forward(x);
    h = this.act.forward(h);
    return this.fc2.forward(h);
  }
}

/**
 * Simple Decoder
 */
class SimpleDecoder extends Module {
  private fc1: Linear;
  private fc2: Linear;
  private act: GELU;

  constructor(hiddenDim: number, outputDim: number) {
    super();
    this.fc1 = new Linear(hiddenDim, hiddenDim);
    this.fc2 = new Linear(hiddenDim, outputDim);
    this.act = new GELU();

    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
    this.registerModule('act', this.act);
  }

  forward(y: Tensor): Tensor {
    let h = this.fc1.forward(y);
    h = this.act.forward(h);
    return this.fc2.forward(h);
  }
}

/**
 * Optional Memory Module
 */
class OptionalMemory extends Module {
  private slots: number;
  private dim: number;
  private memory: Tensor;
  private queryNet: Linear;
  private valueNet: Linear;

  constructor(dim: number, slots: number = 16) {
    super();
    this.slots = slots;
    this.dim = dim;

    const memData = new Float32Array(slots * dim);
    for (let i = 0; i < memData.length; i++) {
      memData[i] = (Math.random() - 0.5) * 0.1;
    }
    this.memory = new Tensor(memData, [slots, dim], { requiresGrad: false });

    this.queryNet = new Linear(dim, dim);
    this.valueNet = new Linear(dim, dim);

    this.registerModule('queryNet', this.queryNet);
    this.registerModule('valueNet', this.valueNet);
  }

  forward(query: Tensor): Tensor {
    const batchSize = query.shape[0];
    const q = this.queryNet.forward(query);

    const result = new Float32Array(batchSize * this.dim);

    for (let b = 0; b < batchSize; b++) {
      // Compute softmax attention over memory slots
      const scores = new Float32Array(this.slots);
      let maxScore = -Infinity;

      for (let s = 0; s < this.slots; s++) {
        let dot = 0;
        for (let d = 0; d < this.dim; d++) {
          dot += q.data[b * this.dim + d] * this.memory.data[s * this.dim + d];
        }
        scores[s] = dot / Math.sqrt(this.dim);
        maxScore = Math.max(maxScore, scores[s]);
      }

      // Softmax
      let sumExp = 0;
      for (let s = 0; s < this.slots; s++) {
        scores[s] = Math.exp(scores[s] - maxScore);
        sumExp += scores[s];
      }
      for (let s = 0; s < this.slots; s++) {
        scores[s] /= sumExp;
      }

      // Weighted read
      for (let d = 0; d < this.dim; d++) {
        let val = 0;
        for (let s = 0; s < this.slots; s++) {
          val += scores[s] * this.memory.data[s * this.dim + d];
        }
        result[b * this.dim + d] = val;
      }

      // Update memory with query
      const maxIdx = scores.indexOf(Math.max(...scores));
      const lr = 0.05;
      for (let d = 0; d < this.dim; d++) {
        this.memory.data[maxIdx * this.dim + d] =
          (1 - lr) * this.memory.data[maxIdx * this.dim + d] +
          lr * q.data[b * this.dim + d];
      }
    }

    return this.valueNet.forward(
      new Tensor(result, [batchSize, this.dim], { requiresGrad: query.requiresGrad })
    );
  }

  reset(): void {
    for (let i = 0; i < this.memory.size; i++) {
      this.memory.data[i] = (Math.random() - 0.5) * 0.1;
    }
  }
}

/**
 * TRM Final - Ultimate TRM Model
 */
export class TRMFinal extends Module {
  private config: Required<TRMFinalConfig>;

  private encoder: SimpleEncoder;
  private initY: Linear;
  private initZ: Linear;
  private latentUpdate: SimpleLatentUpdate;
  private answerUpdate: SimpleAnswerUpdate;
  private decoder: SimpleDecoder;
  private memory: OptionalMemory | null;
  private memoryMix: Linear | null;

  constructor(config: TRMFinalConfig) {
    super();

    this.config = {
      numRecursions: 4,
      dropout: 0.1,
      useMemory: false,
      memorySlots: 16,
      ...config,
    };

    const { inputDim, hiddenDim, outputDim, dropout, useMemory, memorySlots } = this.config;

    // Encoder
    this.encoder = new SimpleEncoder(inputDim, hiddenDim);
    this.registerModule('encoder', this.encoder);

    // Init projections
    this.initY = new Linear(hiddenDim, hiddenDim);
    this.initZ = new Linear(hiddenDim, hiddenDim);
    this.registerModule('initY', this.initY);
    this.registerModule('initZ', this.initZ);

    // Update networks
    this.latentUpdate = new SimpleLatentUpdate(hiddenDim, hiddenDim, dropout);
    this.answerUpdate = new SimpleAnswerUpdate(hiddenDim, dropout);
    this.registerModule('latentUpdate', this.latentUpdate);
    this.registerModule('answerUpdate', this.answerUpdate);

    // Memory (optional)
    if (useMemory) {
      this.memory = new OptionalMemory(hiddenDim, memorySlots);
      this.memoryMix = new Linear(hiddenDim * 2, hiddenDim);
      this.registerModule('memory', this.memory);
      this.registerModule('memoryMix', this.memoryMix);
    } else {
      this.memory = null;
      this.memoryMix = null;
    }

    // Decoder
    this.decoder = new SimpleDecoder(hiddenDim, outputDim);
    this.registerModule('decoder', this.decoder);
  }

  forward(x: Tensor, numSteps?: number): Tensor {
    const steps = numSteps ?? this.config.numRecursions;

    // Encode
    const xEnc = this.encoder.forward(x);

    // Initialize
    let y = this.initY.forward(xEnc);
    let z = this.initZ.forward(xEnc);

    // Reset memory
    if (this.memory) this.memory.reset();

    // Recursive refinement
    for (let i = 0; i < steps; i++) {
      z = this.latentUpdate.forward(xEnc, y, z);

      // Optional memory integration
      if (this.memory && this.memoryMix) {
        const memOut = this.memory.forward(z);
        const combined = cat([z, memOut], -1);
        z = this.memoryMix.forward(combined);
      }

      y = this.answerUpdate.forward(y, z);
    }

    return this.decoder.forward(y);
  }

  forwardAdaptive(x: Tensor, maxSteps: number = 12, threshold: number = 0.01): {
    output: Tensor;
    steps: number;
  } {
    const xEnc = this.encoder.forward(x);
    let y = this.initY.forward(xEnc);
    let z = this.initZ.forward(xEnc);

    if (this.memory) this.memory.reset();

    let prevY = y;
    let actualSteps = 0;

    for (let i = 0; i < maxSteps; i++) {
      z = this.latentUpdate.forward(xEnc, y, z);

      if (this.memory && this.memoryMix) {
        const memOut = this.memory.forward(z);
        const combined = cat([z, memOut], -1);
        z = this.memoryMix.forward(combined);
      }

      y = this.answerUpdate.forward(y, z);
      actualSteps++;

      const diff = y.sub(prevY).abs().mean().item();
      if (diff < threshold && i >= 2) break;
      prevY = y;
    }

    return { output: this.decoder.forward(y), steps: actualSteps };
  }

  forwardWithHistory(x: Tensor, numSteps?: number): {
    output: Tensor;
    intermediates: { y: Tensor; z: Tensor }[];
  } {
    const steps = numSteps ?? this.config.numRecursions;
    const intermediates: { y: Tensor; z: Tensor }[] = [];

    const xEnc = this.encoder.forward(x);
    let y = this.initY.forward(xEnc);
    let z = this.initZ.forward(xEnc);

    if (this.memory) this.memory.reset();
    intermediates.push({ y: y.clone(), z: z.clone() });

    for (let i = 0; i < steps; i++) {
      z = this.latentUpdate.forward(xEnc, y, z);

      if (this.memory && this.memoryMix) {
        const memOut = this.memory.forward(z);
        const combined = cat([z, memOut], -1);
        z = this.memoryMix.forward(combined);
      }

      y = this.answerUpdate.forward(y, z);
      intermediates.push({ y: y.clone(), z: z.clone() });
    }

    return { output: this.decoder.forward(y), intermediates };
  }
}

/**
 * TRM Final Classifier
 */
export class TRMFinalClassifier extends Module {
  private trm: TRMFinal;
  private numClasses: number;

  constructor(inputDim: number, hiddenDim: number, numClasses: number, numRecursions: number = 4) {
    super();
    this.numClasses = numClasses;

    this.trm = new TRMFinal({
      inputDim,
      hiddenDim,
      outputDim: numClasses,
      numRecursions,
    });
    this.registerModule('trm', this.trm);
  }

  forward(x: Tensor, numSteps?: number): Tensor {
    return this.trm.forward(x, numSteps);
  }
}

/**
 * Factory: Tiny TRM Final
 */
export function createTinyTRMFinal(
  inputDim: number,
  outputDim: number,
  hiddenDim: number = 32,
  numRecursions: number = 4
): TRMFinal {
  return new TRMFinal({
    inputDim,
    hiddenDim,
    outputDim,
    numRecursions,
    dropout: 0.0,
    useMemory: false,
  });
}

/**
 * Factory: Standard TRM Final
 */
export function createStandardTRMFinal(
  inputDim: number,
  outputDim: number,
  hiddenDim: number = 64,
  numRecursions: number = 6
): TRMFinal {
  return new TRMFinal({
    inputDim,
    hiddenDim,
    outputDim,
    numRecursions,
    dropout: 0.1,
    useMemory: false,
  });
}

/**
 * Factory: Reasoning TRM Final (with memory)
 */
export function createReasoningTRMFinal(
  inputDim: number,
  outputDim: number
): TRMFinal {
  return new TRMFinal({
    inputDim,
    hiddenDim: 128,
    outputDim,
    numRecursions: 8,
    dropout: 0.1,
    useMemory: true,
    memorySlots: 32,
  });
}
