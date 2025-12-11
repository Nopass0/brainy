/**
 * @fileoverview TRM-Lite - Simplified TRM with proven stability
 * @description Minimal TRM without complex residual connections
 */

import { Tensor, tensor, zeros, randn, cat } from '../core/tensor';
import { Module, Sequential, Parameter } from '../nn/module';
import { Linear, Dropout } from '../nn/layers';
import { ReLU, Sigmoid } from '../nn/activations';

export interface TRMLiteConfig {
  inputDim: number;
  hiddenDim: number;
  outputDim: number;
  numRecursions?: number;
  useSigmoidOutput?: boolean;
}

/**
 * TRM-Lite - Simple and stable
 */
export class TRMLite extends Module {
  private config: Required<TRMLiteConfig>;

  private encoder: Sequential;
  private initY: Linear;
  private initZ: Linear;
  private latentNet: Sequential;
  private answerNet: Sequential;
  private decoder: Sequential;

  constructor(config: TRMLiteConfig) {
    super();

    this.config = {
      numRecursions: 4,
      useSigmoidOutput: false,
      ...config,
    };

    const { inputDim, hiddenDim, outputDim, useSigmoidOutput } = this.config;

    // Simple encoder
    this.encoder = new Sequential(
      new Linear(inputDim, hiddenDim),
      new ReLU()
    );
    this.registerModule('encoder', this.encoder);

    // Initialize y and z from encoded input (connected to computation graph!)
    this.initY = new Linear(hiddenDim, hiddenDim);
    this.initZ = new Linear(hiddenDim, hiddenDim);
    this.registerModule('initY', this.initY);
    this.registerModule('initZ', this.initZ);

    // Latent network: [x, y, z] -> z_new
    this.latentNet = new Sequential(
      new Linear(hiddenDim * 3, hiddenDim * 2),
      new ReLU(),
      new Linear(hiddenDim * 2, hiddenDim),
      new ReLU()
    );
    this.registerModule('latentNet', this.latentNet);

    // Answer network: [y, z] -> y_new
    this.answerNet = new Sequential(
      new Linear(hiddenDim * 2, hiddenDim * 2),
      new ReLU(),
      new Linear(hiddenDim * 2, hiddenDim),
      new ReLU()
    );
    this.registerModule('answerNet', this.answerNet);

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

    // Initialize y and z from xEnc (connected to computation graph)
    let y = this.initY.forward(xEnc);
    let z = this.initZ.forward(xEnc);

    // Recursive refinement
    for (let i = 0; i < steps; i++) {
      // Update z
      const combined = cat([xEnc, y, z], -1);
      z = this.latentNet.forward(combined);

      // Update y
      const yzCombined = cat([y, z], -1);
      y = this.answerNet.forward(yzCombined);
    }

    // Decode
    return this.decoder.forward(y);
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

    intermediates.push({ y: y.clone(), z: z.clone() });

    for (let i = 0; i < steps; i++) {
      const combined = cat([xEnc, y, z], -1);
      z = this.latentNet.forward(combined);

      const yzCombined = cat([y, z], -1);
      y = this.answerNet.forward(yzCombined);

      intermediates.push({ y: y.clone(), z: z.clone() });
    }

    return { output: this.decoder.forward(y), intermediates };
  }
}

/**
 * Factory: Create tiny TRM-Lite
 */
export function createTinyTRMLite(
  inputDim: number,
  outputDim: number,
  hiddenDim: number = 8,
  numRecursions: number = 4,
  useSigmoid: boolean = true
): TRMLite {
  return new TRMLite({
    inputDim,
    hiddenDim,
    outputDim,
    numRecursions,
    useSigmoidOutput: useSigmoid,
  });
}

/**
 * Factory: Create standard TRM-Lite
 */
export function createStandardTRMLite(
  inputDim: number,
  outputDim: number,
  hiddenDim: number = 32,
  numRecursions: number = 6
): TRMLite {
  return new TRMLite({
    inputDim,
    hiddenDim,
    outputDim,
    numRecursions,
    useSigmoidOutput: false,
  });
}
