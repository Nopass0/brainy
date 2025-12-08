/**
 * @fileoverview TRM-X - Extreme Performance TRM
 * @description Optimized TRM with proven training workflow
 *
 * Key features:
 * - Proper training workflow (forward -> loss -> zeroGrad -> backward -> step)
 * - Optional Sigmoid output for binary classification
 * - Memory-augmented reasoning
 * - Adaptive computation with confidence
 * - Proven architecture based on working examples
 */

import { Tensor, tensor, zeros, randn, cat, ones } from '../core/tensor';
import { Module, Sequential, Parameter } from '../nn/module';
import { Linear, Dropout } from '../nn/layers';
import { GELU, Sigmoid, ReLU, Tanh } from '../nn/activations';

export interface TRMXConfig {
  inputDim: number;
  hiddenDim: number;
  outputDim: number;
  numRecursions?: number;
  dropout?: number;
  useMemory?: boolean;
  memorySlots?: number;
  useSigmoidOutput?: boolean; // Important for binary classification
}

/**
 * Simple Memory for TRM-X
 */
class TRMMemory extends Module {
  private slots: number;
  private dim: number;
  private memory: Tensor;
  private queryNet: Linear;
  private outputNet: Linear;

  constructor(dim: number, slots: number = 16) {
    super();
    this.slots = slots;
    this.dim = dim;

    const data = new Float32Array(slots * dim);
    for (let i = 0; i < data.length; i++) {
      data[i] = (Math.random() - 0.5) * 0.1;
    }
    this.memory = new Tensor(data, [slots, dim], { requiresGrad: false });

    this.queryNet = new Linear(dim, dim);
    this.outputNet = new Linear(dim, dim);

    this.registerModule('queryNet', this.queryNet);
    this.registerModule('outputNet', this.outputNet);
  }

  forward(query: Tensor): Tensor {
    const bs = query.shape[0];
    const q = this.queryNet.forward(query);
    const result = new Float32Array(bs * this.dim);

    for (let b = 0; b < bs; b++) {
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

      let sumExp = 0;
      for (let s = 0; s < this.slots; s++) {
        scores[s] = Math.exp(scores[s] - maxScore);
        sumExp += scores[s];
      }
      for (let s = 0; s < this.slots; s++) {
        scores[s] /= sumExp;
      }

      for (let d = 0; d < this.dim; d++) {
        let val = 0;
        for (let s = 0; s < this.slots; s++) {
          val += scores[s] * this.memory.data[s * this.dim + d];
        }
        result[b * this.dim + d] = val;
      }

      // Update memory
      const maxIdx = scores.indexOf(Math.max(...scores));
      const lr = 0.05;
      for (let d = 0; d < this.dim; d++) {
        this.memory.data[maxIdx * this.dim + d] =
          (1 - lr) * this.memory.data[maxIdx * this.dim + d] +
          lr * q.data[b * this.dim + d];
      }
    }

    return this.outputNet.forward(
      new Tensor(result, [bs, this.dim], { requiresGrad: query.requiresGrad })
    );
  }

  reset(): void {
    for (let i = 0; i < this.memory.size; i++) {
      this.memory.data[i] = (Math.random() - 0.5) * 0.1;
    }
  }
}

/**
 * TRM-X Core Model
 */
export class TRMX extends Module {
  private config: Required<TRMXConfig>;

  // Encoder
  private encoder: Sequential;

  // Init projections
  private initY: Linear;
  private initZ: Linear;

  // Update networks
  private latentNet: Sequential;
  private answerNet: Sequential;

  // Memory
  private memory: TRMMemory | null;
  private memoryMix: Linear | null;

  // Decoder
  private decoder: Sequential;

  constructor(config: TRMXConfig) {
    super();

    this.config = {
      numRecursions: 4,
      dropout: 0.0, // No dropout for small datasets
      useMemory: false,
      memorySlots: 16,
      useSigmoidOutput: false,
      ...config,
    };

    const { inputDim, hiddenDim, outputDim, dropout, useMemory, memorySlots, useSigmoidOutput } = this.config;

    // Encoder: input -> hidden
    this.encoder = new Sequential(
      new Linear(inputDim, hiddenDim),
      new ReLU(),
      new Linear(hiddenDim, hiddenDim)
    );
    this.registerModule('encoder', this.encoder);

    // Init projections
    this.initY = new Linear(hiddenDim, hiddenDim);
    this.initZ = new Linear(hiddenDim, hiddenDim);
    this.registerModule('initY', this.initY);
    this.registerModule('initZ', this.initZ);

    // Latent update: [x, y, z] -> z_new
    const latentInput = hiddenDim * 3;
    this.latentNet = new Sequential(
      new Linear(latentInput, hiddenDim * 2),
      new ReLU(),
      new Dropout(dropout),
      new Linear(hiddenDim * 2, hiddenDim),
      new ReLU(),
      new Linear(hiddenDim, hiddenDim)
    );
    this.registerModule('latentNet', this.latentNet);

    // Answer update: [y, z] -> y_new
    this.answerNet = new Sequential(
      new Linear(hiddenDim * 2, hiddenDim * 2),
      new ReLU(),
      new Dropout(dropout),
      new Linear(hiddenDim * 2, hiddenDim)
    );
    this.registerModule('answerNet', this.answerNet);

    // Memory
    if (useMemory) {
      this.memory = new TRMMemory(hiddenDim, memorySlots);
      this.memoryMix = new Linear(hiddenDim * 2, hiddenDim);
      this.registerModule('memory', this.memory);
      this.registerModule('memoryMix', this.memoryMix);
    } else {
      this.memory = null;
      this.memoryMix = null;
    }

    // Decoder
    if (useSigmoidOutput) {
      this.decoder = new Sequential(
        new Linear(hiddenDim, hiddenDim),
        new ReLU(),
        new Linear(hiddenDim, outputDim),
        new Sigmoid()
      );
    } else {
      this.decoder = new Sequential(
        new Linear(hiddenDim, hiddenDim),
        new ReLU(),
        new Linear(hiddenDim, outputDim)
      );
    }
    this.registerModule('decoder', this.decoder);
  }

  forward(x: Tensor, numSteps?: number): Tensor {
    const steps = numSteps ?? this.config.numRecursions;

    // Encode
    const xEnc = this.encoder.forward(x);

    // Initialize y and z
    let y = this.initY.forward(xEnc);
    let z = this.initZ.forward(xEnc);

    // Reset memory
    if (this.memory) this.memory.reset();

    // Recursive refinement
    for (let i = 0; i < steps; i++) {
      // Update z
      const combined = cat([xEnc, y, z], -1);
      const zUpdate = this.latentNet.forward(combined);
      z = z.add(zUpdate.sub(z).mul(0.5)); // Residual with mixing

      // Memory integration
      if (this.memory && this.memoryMix) {
        const memOut = this.memory.forward(z);
        const memCombined = cat([z, memOut], -1);
        z = this.memoryMix.forward(memCombined);
      }

      // Update y
      const yzCombined = cat([y, z], -1);
      const yUpdate = this.answerNet.forward(yzCombined);
      y = y.add(yUpdate.sub(y).mul(0.5)); // Residual with mixing
    }

    // Decode
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
      const combined = cat([xEnc, y, z], -1);
      const zUpdate = this.latentNet.forward(combined);
      z = z.add(zUpdate.sub(z).mul(0.5));

      if (this.memory && this.memoryMix) {
        const memOut = this.memory.forward(z);
        const memCombined = cat([z, memOut], -1);
        z = this.memoryMix.forward(memCombined);
      }

      const yzCombined = cat([y, z], -1);
      const yUpdate = this.answerNet.forward(yzCombined);
      y = y.add(yUpdate.sub(y).mul(0.5));

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
      const combined = cat([xEnc, y, z], -1);
      const zUpdate = this.latentNet.forward(combined);
      z = z.add(zUpdate.sub(z).mul(0.5));

      if (this.memory && this.memoryMix) {
        const memOut = this.memory.forward(z);
        const memCombined = cat([z, memOut], -1);
        z = this.memoryMix.forward(memCombined);
      }

      const yzCombined = cat([y, z], -1);
      const yUpdate = this.answerNet.forward(yzCombined);
      y = y.add(yUpdate.sub(y).mul(0.5));

      intermediates.push({ y: y.clone(), z: z.clone() });
    }

    return { output: this.decoder.forward(y), intermediates };
  }

  getConfig(): Required<TRMXConfig> {
    return { ...this.config };
  }
}

/**
 * TRM-X Classifier with few-shot capability
 */
export class TRMXClassifier extends Module {
  private trm: TRMX;
  private numClasses: number;

  constructor(inputDim: number, hiddenDim: number, numClasses: number, numRecursions: number = 4) {
    super();
    this.numClasses = numClasses;

    this.trm = new TRMX({
      inputDim,
      hiddenDim,
      outputDim: numClasses,
      numRecursions,
      useSigmoidOutput: false, // Use softmax externally for multi-class
    });
    this.registerModule('trm', this.trm);
  }

  forward(x: Tensor, numSteps?: number): Tensor {
    return this.trm.forward(x, numSteps);
  }
}

/**
 * Factory: Tiny TRM-X for binary classification
 */
export function createTinyTRMX(
  inputDim: number,
  outputDim: number,
  hiddenDim: number = 8,
  numRecursions: number = 4,
  useSigmoid: boolean = true
): TRMX {
  return new TRMX({
    inputDim,
    hiddenDim,
    outputDim,
    numRecursions,
    dropout: 0.0,
    useMemory: false,
    useSigmoidOutput: useSigmoid,
  });
}

/**
 * Factory: Standard TRM-X
 */
export function createStandardTRMX(
  inputDim: number,
  outputDim: number,
  hiddenDim: number = 32,
  numRecursions: number = 6
): TRMX {
  return new TRMX({
    inputDim,
    hiddenDim,
    outputDim,
    numRecursions,
    dropout: 0.1,
    useMemory: false,
    useSigmoidOutput: false,
  });
}

/**
 * Factory: Reasoning TRM-X with memory
 */
export function createReasoningTRMX(
  inputDim: number,
  outputDim: number
): TRMX {
  return new TRMX({
    inputDim,
    hiddenDim: 64,
    outputDim,
    numRecursions: 8,
    dropout: 0.1,
    useMemory: true,
    memorySlots: 32,
    useSigmoidOutput: false,
  });
}
