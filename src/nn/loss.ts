/**
 * @fileoverview Функции потерь (Loss Functions)
 * @description Функции для вычисления ошибки между предсказаниями и целевыми значениями
 */

import { Tensor, scalar, zeros, ones } from '../core/tensor';
import { Module } from './module';
import { GradNode, GradContext, isNoGradEnabled } from '../core/autograd';

/**
 * Mean Squared Error Loss (Среднеквадратичная ошибка)
 * L = mean((y_pred - y_true)^2)
 * Аналог nn.MSELoss в PyTorch
 */
export class MSELoss extends Module {
  /** Метод редукции: 'mean', 'sum', или 'none' */
  readonly reduction: 'mean' | 'sum' | 'none';

  /**
   * Создаёт MSELoss
   * @param reduction - Способ агрегации: 'mean' (по умолчанию), 'sum', 'none'
   */
  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean') {
    super();
    this.reduction = reduction;
  }

  /**
   * Вычисляет MSE loss
   * @param input - Предсказания модели
   * @param target - Целевые значения
   * @returns Значение loss
   */
  forward(input: Tensor, target?: Tensor): Tensor {
    if (!target) {
      throw new Error('MSELoss requires target tensor');
    }

    const diff = input.sub(target);
    const squared = diff.pow(2);

    if (this.reduction === 'none') {
      return squared;
    } else if (this.reduction === 'sum') {
      return squared.sum();
    } else {
      return squared.mean();
    }
  }

  /**
   * Позволяет вызывать loss как функцию
   */
  call(input: Tensor, target: Tensor): Tensor {
    return this.forward(input, target);
  }

  toString(): string {
    return `MSELoss(reduction='${this.reduction}')`;
  }
}

/**
 * L1 Loss (Mean Absolute Error)
 * L = mean(|y_pred - y_true|)
 */
export class L1Loss extends Module {
  readonly reduction: 'mean' | 'sum' | 'none';

  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target?: Tensor): Tensor {
    if (!target) {
      throw new Error('L1Loss requires target tensor');
    }

    const diff = input.sub(target).abs();

    if (this.reduction === 'none') {
      return diff;
    } else if (this.reduction === 'sum') {
      return diff.sum();
    } else {
      return diff.mean();
    }
  }

  call(input: Tensor, target: Tensor): Tensor {
    return this.forward(input, target);
  }

  toString(): string {
    return `L1Loss(reduction='${this.reduction}')`;
  }
}

/**
 * Smooth L1 Loss (Huber Loss)
 * Комбинация L1 и L2: L2 для малых ошибок, L1 для больших
 */
export class SmoothL1Loss extends Module {
  readonly reduction: 'mean' | 'sum' | 'none';
  readonly beta: number;

  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean', beta: number = 1.0) {
    super();
    this.reduction = reduction;
    this.beta = beta;
  }

  forward(input: Tensor, target?: Tensor): Tensor {
    if (!target) {
      throw new Error('SmoothL1Loss requires target tensor');
    }

    const outputData = new Float32Array(input.size);

    for (let i = 0; i < input.size; i++) {
      const diff = Math.abs(input.data[i] - target.data[i]);
      if (diff < this.beta) {
        outputData[i] = 0.5 * diff * diff / this.beta;
      } else {
        outputData[i] = diff - 0.5 * this.beta;
      }
    }

    const loss = new Tensor(outputData, [...input.shape], { dtype: input.dtype });

    if (this.reduction === 'none') {
      return loss;
    } else if (this.reduction === 'sum') {
      return loss.sum();
    } else {
      return loss.mean();
    }
  }

  call(input: Tensor, target: Tensor): Tensor {
    return this.forward(input, target);
  }

  toString(): string {
    return `SmoothL1Loss(reduction='${this.reduction}', beta=${this.beta})`;
  }
}

/**
 * Binary Cross-Entropy Loss
 * L = -mean(y * log(p) + (1 - y) * log(1 - p))
 * Для бинарной классификации
 */
export class BCELoss extends Module {
  readonly reduction: 'mean' | 'sum' | 'none';
  readonly eps: number;

  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean', eps: number = 1e-7) {
    super();
    this.reduction = reduction;
    this.eps = eps;
  }

  forward(input: Tensor, target?: Tensor): Tensor {
    if (!target) {
      throw new Error('BCELoss requires target tensor');
    }

    const outputData = new Float32Array(input.size);

    for (let i = 0; i < input.size; i++) {
      const p = Math.max(this.eps, Math.min(1 - this.eps, input.data[i]));
      const t = target.data[i];
      outputData[i] = -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
    }

    const result = new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    if (input.requiresGrad && !isNoGradEnabled()) {
      result.gradNode = new GradNode(
        (gradOutput) => {
          const gradData = new Float32Array(input.size);
          for (let i = 0; i < input.size; i++) {
            const p = Math.max(this.eps, Math.min(1 - this.eps, input.data[i]));
            const t = target.data[i];
            gradData[i] = gradOutput.data[i] * (-t / p + (1 - t) / (1 - p));
          }
          return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
        },
        [input],
        new GradContext()
      );
    }

    if (this.reduction === 'none') {
      return result;
    } else if (this.reduction === 'sum') {
      return result.sum();
    } else {
      return result.mean();
    }
  }

  call(input: Tensor, target: Tensor): Tensor {
    return this.forward(input, target);
  }

  toString(): string {
    return `BCELoss(reduction='${this.reduction}')`;
  }
}

/**
 * BCE with Logits Loss
 * Комбинация Sigmoid + BCE для численной стабильности
 */
export class BCEWithLogitsLoss extends Module {
  readonly reduction: 'mean' | 'sum' | 'none';

  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target?: Tensor): Tensor {
    if (!target) {
      throw new Error('BCEWithLogitsLoss requires target tensor');
    }

    const outputData = new Float32Array(input.size);

    for (let i = 0; i < input.size; i++) {
      const x = input.data[i];
      const t = target.data[i];
      // Численно стабильная формула: max(x, 0) - x * t + log(1 + exp(-|x|))
      const maxXZero = Math.max(x, 0);
      outputData[i] = maxXZero - x * t + Math.log(1 + Math.exp(-Math.abs(x)));
    }

    const result = new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    if (input.requiresGrad && !isNoGradEnabled()) {
      result.gradNode = new GradNode(
        (gradOutput) => {
          const gradData = new Float32Array(input.size);
          for (let i = 0; i < input.size; i++) {
            const x = input.data[i];
            const t = target.data[i];
            const sigmoid = 1 / (1 + Math.exp(-x));
            gradData[i] = gradOutput.data[i] * (sigmoid - t);
          }
          return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
        },
        [input],
        new GradContext()
      );
    }

    if (this.reduction === 'none') {
      return result;
    } else if (this.reduction === 'sum') {
      return result.sum();
    } else {
      return result.mean();
    }
  }

  call(input: Tensor, target: Tensor): Tensor {
    return this.forward(input, target);
  }

  toString(): string {
    return `BCEWithLogitsLoss(reduction='${this.reduction}')`;
  }
}

/**
 * Cross-Entropy Loss
 * L = -sum(y * log(softmax(x)))
 * Для многоклассовой классификации
 */
export class CrossEntropyLoss extends Module {
  readonly reduction: 'mean' | 'sum' | 'none';

  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean') {
    super();
    this.reduction = reduction;
  }

  /**
   * Вычисляет Cross-Entropy Loss
   * @param input - Logits [batch, num_classes]
   * @param target - Индексы классов [batch] или one-hot [batch, num_classes]
   */
  forward(input: Tensor, target?: Tensor): Tensor {
    if (!target) {
      throw new Error('CrossEntropyLoss requires target tensor');
    }

    // Предполагаем input: [batch, classes], target: [batch] (индексы)
    const batch = input.shape[0];
    const classes = input.shape[1];

    const lossData = new Float32Array(batch);

    for (let b = 0; b < batch; b++) {
      // LogSoftmax для численной стабильности
      let maxVal = -Infinity;
      for (let c = 0; c < classes; c++) {
        maxVal = Math.max(maxVal, input.data[b * classes + c]);
      }

      let sumExp = 0;
      for (let c = 0; c < classes; c++) {
        sumExp += Math.exp(input.data[b * classes + c] - maxVal);
      }

      const logSumExp = maxVal + Math.log(sumExp);

      // target может быть индексом или one-hot
      if (target.ndim === 1) {
        const classIdx = Math.floor(target.data[b]);
        lossData[b] = -(input.data[b * classes + classIdx] - logSumExp);
      } else {
        // One-hot encoded
        let loss = 0;
        for (let c = 0; c < classes; c++) {
          if (target.data[b * classes + c] > 0) {
            loss -= target.data[b * classes + c] * (input.data[b * classes + c] - logSumExp);
          }
        }
        lossData[b] = loss;
      }
    }

    const result = new Tensor(lossData, [batch], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    // Autograd
    if (input.requiresGrad && !isNoGradEnabled()) {
      result.gradNode = new GradNode(
        (gradOutput) => {
          const gradData = new Float32Array(input.size);

          for (let b = 0; b < batch; b++) {
            // Вычисляем softmax
            let maxVal = -Infinity;
            for (let c = 0; c < classes; c++) {
              maxVal = Math.max(maxVal, input.data[b * classes + c]);
            }

            let sumExp = 0;
            const expVals = new Float32Array(classes);
            for (let c = 0; c < classes; c++) {
              expVals[c] = Math.exp(input.data[b * classes + c] - maxVal);
              sumExp += expVals[c];
            }

            const gradScale = this.reduction === 'mean' ? 1 / batch : 1;

            for (let c = 0; c < classes; c++) {
              const softmax = expVals[c] / sumExp;
              let targetVal: number;
              if (target.ndim === 1) {
                targetVal = c === Math.floor(target.data[b]) ? 1 : 0;
              } else {
                targetVal = target.data[b * classes + c];
              }
              gradData[b * classes + c] = gradScale * (softmax - targetVal);
            }
          }

          return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
        },
        [input],
        new GradContext()
      );
    }

    if (this.reduction === 'none') {
      return result;
    } else if (this.reduction === 'sum') {
      return result.sum();
    } else {
      return result.mean();
    }
  }

  call(input: Tensor, target: Tensor): Tensor {
    return this.forward(input, target);
  }

  toString(): string {
    return `CrossEntropyLoss(reduction='${this.reduction}')`;
  }
}

/**
 * Negative Log Likelihood Loss
 * Ожидает log-probabilities на входе (после LogSoftmax)
 */
export class NLLLoss extends Module {
  readonly reduction: 'mean' | 'sum' | 'none';

  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target?: Tensor): Tensor {
    if (!target) {
      throw new Error('NLLLoss requires target tensor');
    }

    const batch = input.shape[0];
    const classes = input.shape[1];
    const lossData = new Float32Array(batch);

    for (let b = 0; b < batch; b++) {
      const classIdx = Math.floor(target.data[b]);
      lossData[b] = -input.data[b * classes + classIdx];
    }

    const result = new Tensor(lossData, [batch], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    if (this.reduction === 'none') {
      return result;
    } else if (this.reduction === 'sum') {
      return result.sum();
    } else {
      return result.mean();
    }
  }

  call(input: Tensor, target: Tensor): Tensor {
    return this.forward(input, target);
  }

  toString(): string {
    return `NLLLoss(reduction='${this.reduction}')`;
  }
}

/**
 * Hinge Loss (для SVM)
 * L = max(0, 1 - y * x)
 */
export class HingeLoss extends Module {
  readonly reduction: 'mean' | 'sum' | 'none';
  readonly margin: number;

  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean', margin: number = 1.0) {
    super();
    this.reduction = reduction;
    this.margin = margin;
  }

  forward(input: Tensor, target?: Tensor): Tensor {
    if (!target) {
      throw new Error('HingeLoss requires target tensor');
    }

    const outputData = new Float32Array(input.size);

    for (let i = 0; i < input.size; i++) {
      outputData[i] = Math.max(0, this.margin - target.data[i] * input.data[i]);
    }

    const result = new Tensor(outputData, [...input.shape], { dtype: input.dtype });

    if (this.reduction === 'none') {
      return result;
    } else if (this.reduction === 'sum') {
      return result.sum();
    } else {
      return result.mean();
    }
  }

  call(input: Tensor, target: Tensor): Tensor {
    return this.forward(input, target);
  }

  toString(): string {
    return `HingeLoss(reduction='${this.reduction}', margin=${this.margin})`;
  }
}

/**
 * Cosine Embedding Loss
 * Для обучения similarity
 */
export class CosineEmbeddingLoss extends Module {
  readonly margin: number;
  readonly reduction: 'mean' | 'sum' | 'none';

  constructor(margin: number = 0.0, reduction: 'mean' | 'sum' | 'none' = 'mean') {
    super();
    this.margin = margin;
    this.reduction = reduction;
  }

  forward(input: Tensor, target?: Tensor): Tensor {
    // Реализация для пар (x1, x2) с labels y ∈ {-1, 1}
    throw new Error('CosineEmbeddingLoss requires two input tensors - use call(x1, x2, y)');
  }

  /**
   * Вычисляет Cosine Embedding Loss
   * @param x1 - Первый тензор embeddings
   * @param x2 - Второй тензор embeddings
   * @param y - Labels: 1 (similar) или -1 (dissimilar)
   */
  compute(x1: Tensor, x2: Tensor, y: Tensor): Tensor {
    const batch = x1.shape[0];
    const features = x1.shape[1];
    const lossData = new Float32Array(batch);

    for (let b = 0; b < batch; b++) {
      let dot = 0, norm1 = 0, norm2 = 0;
      for (let f = 0; f < features; f++) {
        const v1 = x1.data[b * features + f];
        const v2 = x2.data[b * features + f];
        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
      }
      const cosSim = dot / (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8);
      const label = y.data[b];

      if (label === 1) {
        lossData[b] = 1 - cosSim;
      } else {
        lossData[b] = Math.max(0, cosSim - this.margin);
      }
    }

    const result = new Tensor(lossData, [batch], { dtype: x1.dtype });

    if (this.reduction === 'none') {
      return result;
    } else if (this.reduction === 'sum') {
      return result.sum();
    } else {
      return result.mean();
    }
  }

  toString(): string {
    return `CosineEmbeddingLoss(margin=${this.margin}, reduction='${this.reduction}')`;
  }
}

/**
 * KL Divergence Loss
 * L = sum(p * log(p / q))
 */
export class KLDivLoss extends Module {
  readonly reduction: 'mean' | 'sum' | 'batchmean' | 'none';

  constructor(reduction: 'mean' | 'sum' | 'batchmean' | 'none' = 'mean') {
    super();
    this.reduction = reduction;
  }

  /**
   * Вычисляет KL Divergence
   * @param input - Log-probabilities (Q)
   * @param target - Probabilities (P)
   */
  forward(input: Tensor, target?: Tensor): Tensor {
    if (!target) {
      throw new Error('KLDivLoss requires target tensor');
    }

    const outputData = new Float32Array(input.size);

    for (let i = 0; i < input.size; i++) {
      // KL(P || Q) = sum(P * (log(P) - log(Q)))
      // input = log(Q), target = P
      if (target.data[i] > 0) {
        outputData[i] = target.data[i] * (Math.log(target.data[i]) - input.data[i]);
      }
    }

    const result = new Tensor(outputData, [...input.shape], { dtype: input.dtype });

    if (this.reduction === 'none') {
      return result;
    } else if (this.reduction === 'sum') {
      return result.sum();
    } else if (this.reduction === 'batchmean') {
      return result.sum().div(input.shape[0]);
    } else {
      return result.mean();
    }
  }

  call(input: Tensor, target: Tensor): Tensor {
    return this.forward(input, target);
  }

  toString(): string {
    return `KLDivLoss(reduction='${this.reduction}')`;
  }
}
