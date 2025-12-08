/**
 * @fileoverview Tiny Recursion Model (TRM)
 * @description Implementation of TRM architecture from https://arxiv.org/html/2510.04871v1
 *
 * TRM uses recursive prediction refinement with three components:
 * - x: Input embedding
 * - y: Current answer (refined iteratively)
 * - z: Latent reasoning state
 */

import { Tensor, tensor, zeros, randn, cat, stack } from '../core/tensor';
import { Module, Sequential, Parameter } from '../nn/module';
import { Linear, Dropout, LayerNorm } from '../nn/layers';
import { ReLU, GELU, Softmax, Tanh, Sigmoid } from '../nn/activations';
import { noGrad } from '../core/tensor';

/**
 * TRM Configuration
 */
export interface TRMConfig {
  /** Input dimension */
  inputDim: number;
  /** Hidden/latent dimension */
  hiddenDim: number;
  /** Output dimension */
  outputDim: number;
  /** Number of recursion steps */
  numRecursions?: number;
  /** Dropout probability */
  dropout?: number;
  /** Use residual connections */
  useResidual?: boolean;
  /** Number of attention heads for cross-attention */
  numHeads?: number;
}

/**
 * Latent Update Network - Updates the reasoning state z
 */
class LatentUpdateNet extends Module {
  private net: Sequential;
  private norm: LayerNorm;

  constructor(inputDim: number, hiddenDim: number, dropout: number = 0.1) {
    super();

    // Takes concatenated [x, y, z] and produces new z
    const totalInput = inputDim + hiddenDim * 2;  // x + y + z

    this.net = new Sequential(
      new Linear(totalInput, hiddenDim * 2),
      new GELU(),
      new Dropout(dropout),
      new Linear(hiddenDim * 2, hiddenDim),
      new GELU(),
      new Linear(hiddenDim, hiddenDim)
    );

    this.norm = new LayerNorm(hiddenDim);

    this.registerModule('net', this.net);
    this.registerModule('norm', this.norm);
  }

  forward(x: Tensor, y: Tensor, z: Tensor): Tensor {
    // Concatenate inputs
    const combined = cat([x, y, z], -1);
    const out = this.net.forward(combined);
    // Residual connection with z
    return this.norm.forward(out.add(z));
  }
}

/**
 * Answer Update Network - Updates the answer y from latent z
 */
class AnswerUpdateNet extends Module {
  private net: Sequential;
  private norm: LayerNorm;

  constructor(hiddenDim: number, dropout: number = 0.1) {
    super();

    // Takes z and current y, produces refined y
    this.net = new Sequential(
      new Linear(hiddenDim * 2, hiddenDim * 2),
      new GELU(),
      new Dropout(dropout),
      new Linear(hiddenDim * 2, hiddenDim)
    );

    this.norm = new LayerNorm(hiddenDim);

    this.registerModule('net', this.net);
    this.registerModule('norm', this.norm);
  }

  forward(y: Tensor, z: Tensor): Tensor {
    const combined = cat([y, z], -1);
    const out = this.net.forward(combined);
    // Residual connection with y
    return this.norm.forward(out.add(y));
  }
}

/**
 * Input Encoder - Encodes raw input to latent space
 */
class InputEncoder extends Module {
  private net: Sequential;

  constructor(inputDim: number, hiddenDim: number) {
    super();

    this.net = new Sequential(
      new Linear(inputDim, hiddenDim),
      new GELU(),
      new Linear(hiddenDim, hiddenDim)
    );

    this.registerModule('net', this.net);
  }

  forward(x: Tensor): Tensor {
    return this.net.forward(x);
  }
}

/**
 * Output Decoder - Decodes latent answer to output space
 */
class OutputDecoder extends Module {
  private net: Sequential;

  constructor(hiddenDim: number, outputDim: number) {
    super();

    this.net = new Sequential(
      new Linear(hiddenDim, hiddenDim),
      new GELU(),
      new Linear(hiddenDim, outputDim)
    );

    this.registerModule('net', this.net);
  }

  forward(y: Tensor): Tensor {
    return this.net.forward(y);
  }
}

/**
 * Tiny Recursion Model (TRM)
 *
 * A recursive refinement model that iteratively improves predictions through
 * alternating updates of latent reasoning state (z) and answer (y).
 */
export class TRM extends Module {
  private config: Required<TRMConfig>;
  private encoder: InputEncoder;
  private latentUpdate: LatentUpdateNet;
  private answerUpdate: AnswerUpdateNet;
  private decoder: OutputDecoder;

  // Initial projections
  private initY: Linear;
  private initZ: Linear;

  constructor(config: TRMConfig) {
    super();

    this.config = {
      numRecursions: 4,
      dropout: 0.1,
      useResidual: true,
      numHeads: 4,
      ...config,
    };

    const { inputDim, hiddenDim, outputDim, dropout } = this.config;

    // Input encoder
    this.encoder = new InputEncoder(inputDim, hiddenDim);
    this.registerModule('encoder', this.encoder);

    // Initial projections for y and z
    this.initY = new Linear(hiddenDim, hiddenDim);
    this.initZ = new Linear(hiddenDim, hiddenDim);
    this.registerModule('initY', this.initY);
    this.registerModule('initZ', this.initZ);

    // Recursion networks
    this.latentUpdate = new LatentUpdateNet(hiddenDim, hiddenDim, dropout);
    this.answerUpdate = new AnswerUpdateNet(hiddenDim, dropout);
    this.registerModule('latentUpdate', this.latentUpdate);
    this.registerModule('answerUpdate', this.answerUpdate);

    // Output decoder
    this.decoder = new OutputDecoder(hiddenDim, outputDim);
    this.registerModule('decoder', this.decoder);
  }

  /**
   * Forward pass with recursive refinement
   * @param x Input tensor [batch, inputDim]
   * @param numSteps Override default recursion steps
   * @returns Output tensor [batch, outputDim]
   */
  forward(x: Tensor, numSteps?: number): Tensor {
    const steps = numSteps ?? this.config.numRecursions;

    // Encode input
    const xEnc = this.encoder.forward(x);

    // Initialize y and z from encoded input
    let y = this.initY.forward(xEnc);
    let z = this.initZ.forward(xEnc);

    // Recursive refinement
    for (let i = 0; i < steps; i++) {
      // Update latent reasoning state
      z = this.latentUpdate.forward(xEnc, y, z);

      // Update answer based on reasoning
      y = this.answerUpdate.forward(y, z);
    }

    // Decode final answer
    return this.decoder.forward(y);
  }

  /**
   * Forward with intermediate outputs (for visualization/analysis)
   */
  forwardWithHistory(x: Tensor, numSteps?: number): {
    output: Tensor;
    intermediates: { y: Tensor; z: Tensor }[];
  } {
    const steps = numSteps ?? this.config.numRecursions;
    const intermediates: { y: Tensor; z: Tensor }[] = [];

    const xEnc = this.encoder.forward(x);
    let y = this.initY.forward(xEnc);
    let z = this.initZ.forward(xEnc);

    intermediates.push({ y: y.clone(), z: z.clone() });

    for (let i = 0; i < steps; i++) {
      z = this.latentUpdate.forward(xEnc, y, z);
      y = this.answerUpdate.forward(y, z);
      intermediates.push({ y: y.clone(), z: z.clone() });
    }

    return {
      output: this.decoder.forward(y),
      intermediates,
    };
  }

  /**
   * Adaptive computation - runs until convergence or max steps
   */
  forwardAdaptive(x: Tensor, maxSteps: number = 16, threshold: number = 0.01): {
    output: Tensor;
    steps: number;
  } {
    const xEnc = this.encoder.forward(x);
    let y = this.initY.forward(xEnc);
    let z = this.initZ.forward(xEnc);

    let prevY = y;
    let actualSteps = 0;

    for (let i = 0; i < maxSteps; i++) {
      z = this.latentUpdate.forward(xEnc, y, z);
      y = this.answerUpdate.forward(y, z);
      actualSteps++;

      // Check convergence
      const diff = y.sub(prevY).abs().mean().item();
      if (diff < threshold) {
        break;
      }

      prevY = y;
    }

    return {
      output: this.decoder.forward(y),
      steps: actualSteps,
    };
  }
}

/**
 * TRM for sequence-to-sequence tasks
 */
export class TRMSeq2Seq extends Module {
  private trm: TRM;
  private embeddings: Linear;
  private vocabSize: number;

  constructor(
    vocabSize: number,
    embeddingDim: number,
    hiddenDim: number,
    outputVocabSize?: number,
    numRecursions: number = 4
  ) {
    super();

    this.vocabSize = vocabSize;

    this.embeddings = new Linear(vocabSize, embeddingDim, false);
    this.registerModule('embeddings', this.embeddings);

    this.trm = new TRM({
      inputDim: embeddingDim,
      hiddenDim,
      outputDim: outputVocabSize || vocabSize,
      numRecursions,
    });
    this.registerModule('trm', this.trm);
  }

  forward(x: Tensor, numSteps?: number): Tensor {
    // x: [batch, seqLen] of token indices → one-hot → embeddings
    // For simplicity, assume x is already one-hot or embedded
    const embedded = this.embeddings.forward(x);
    return this.trm.forward(embedded, numSteps);
  }
}

/**
 * TRM for classification with few-shot learning capability
 */
export class TRMClassifier extends Module {
  private trm: TRM;
  private numClasses: number;

  constructor(
    inputDim: number,
    hiddenDim: number,
    numClasses: number,
    numRecursions: number = 4
  ) {
    super();

    this.numClasses = numClasses;

    this.trm = new TRM({
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

  /**
   * Few-shot prediction: provide support examples and query
   */
  fewShotPredict(
    supportX: Tensor,
    supportY: number[],
    queryX: Tensor,
    numSteps?: number
  ): number[] {
    // Compute class prototypes from support set
    const prototypes: Float32Array[] = [];
    const classCounts: number[] = new Array(this.numClasses).fill(0);

    // Initialize prototypes
    for (let c = 0; c < this.numClasses; c++) {
      prototypes.push(new Float32Array(supportX.shape[1]).fill(0));
    }

    // Accumulate support examples by class
    for (let i = 0; i < supportY.length; i++) {
      const classIdx = supportY[i];
      classCounts[classIdx]++;
      for (let j = 0; j < supportX.shape[1]; j++) {
        prototypes[classIdx][j] += supportX.data[i * supportX.shape[1] + j];
      }
    }

    // Average prototypes
    for (let c = 0; c < this.numClasses; c++) {
      if (classCounts[c] > 0) {
        for (let j = 0; j < prototypes[c].length; j++) {
          prototypes[c][j] /= classCounts[c];
        }
      }
    }

    // Predict for query examples
    const predictions: number[] = [];
    const batchSize = queryX.shape[0];

    for (let i = 0; i < batchSize; i++) {
      // Get query features
      const queryData = new Float32Array(queryX.shape[1]);
      for (let j = 0; j < queryX.shape[1]; j++) {
        queryData[j] = queryX.data[i * queryX.shape[1] + j];
      }

      // Find nearest prototype (cosine similarity)
      let bestClass = 0;
      let bestSim = -Infinity;

      for (let c = 0; c < this.numClasses; c++) {
        if (classCounts[c] === 0) continue;

        let dot = 0, normQ = 0, normP = 0;
        for (let j = 0; j < queryData.length; j++) {
          dot += queryData[j] * prototypes[c][j];
          normQ += queryData[j] * queryData[j];
          normP += prototypes[c][j] * prototypes[c][j];
        }

        const sim = dot / (Math.sqrt(normQ) * Math.sqrt(normP) + 1e-8);
        if (sim > bestSim) {
          bestSim = sim;
          bestClass = c;
        }
      }

      predictions.push(bestClass);
    }

    return predictions;
  }
}

/**
 * Factory function for creating small TRM models
 */
export function createTinyTRM(
  inputDim: number,
  outputDim: number,
  hiddenDim: number = 64,
  numRecursions: number = 4
): TRM {
  return new TRM({
    inputDim,
    hiddenDim,
    outputDim,
    numRecursions,
    dropout: 0.1,
    useResidual: true,
  });
}

/**
 * Factory function for reasoning tasks
 */
export function createReasoningTRM(
  inputDim: number,
  outputDim: number
): TRM {
  return new TRM({
    inputDim,
    hiddenDim: 128,
    outputDim,
    numRecursions: 8,  // More recursions for complex reasoning
    dropout: 0.1,
    useResidual: true,
  });
}
