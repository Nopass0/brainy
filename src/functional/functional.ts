/**
 * @fileoverview Функциональный API для нейросетевых операций
 * @description Функции без состояния для активаций, операций и потерь
 */

import { Tensor, zeros, ones, scalar } from '../core/tensor';
import { GradNode, GradContext, isNoGradEnabled } from '../core/autograd';

// ============================================
// АКТИВАЦИИ (Functional)
// ============================================

/**
 * ReLU активация
 * @param input - Входной тензор
 * @returns max(0, x)
 */
export function relu(input: Tensor): Tensor {
  const outputData = new Float32Array(input.size);
  for (let i = 0; i < input.size; i++) {
    outputData[i] = Math.max(0, input.data[i]);
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
          gradData[i] = input.data[i] > 0 ? gradOutput.data[i] : 0;
        }
        return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
      },
      [input],
      new GradContext()
    );
  }

  return result;
}

/**
 * Leaky ReLU активация
 * @param input - Входной тензор
 * @param negativeSlope - Наклон для отрицательных значений
 * @returns max(negativeSlope * x, x)
 */
export function leakyRelu(input: Tensor, negativeSlope: number = 0.01): Tensor {
  const outputData = new Float32Array(input.size);
  for (let i = 0; i < input.size; i++) {
    outputData[i] = input.data[i] > 0 ? input.data[i] : input.data[i] * negativeSlope;
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
          gradData[i] = input.data[i] > 0 ? gradOutput.data[i] : gradOutput.data[i] * negativeSlope;
        }
        return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
      },
      [input],
      new GradContext()
    );
  }

  return result;
}

/**
 * Sigmoid активация
 * @param input - Входной тензор
 * @returns 1 / (1 + exp(-x))
 */
export function sigmoid(input: Tensor): Tensor {
  const outputData = new Float32Array(input.size);
  for (let i = 0; i < input.size; i++) {
    outputData[i] = 1 / (1 + Math.exp(-input.data[i]));
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
          const s = outputData[i];
          gradData[i] = gradOutput.data[i] * s * (1 - s);
        }
        return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
      },
      [input],
      new GradContext()
    );
  }

  return result;
}

/**
 * Tanh активация
 * @param input - Входной тензор
 * @returns tanh(x)
 */
export function tanh(input: Tensor): Tensor {
  const outputData = new Float32Array(input.size);
  for (let i = 0; i < input.size; i++) {
    outputData[i] = Math.tanh(input.data[i]);
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
          const t = outputData[i];
          gradData[i] = gradOutput.data[i] * (1 - t * t);
        }
        return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
      },
      [input],
      new GradContext()
    );
  }

  return result;
}

/**
 * Softmax активация
 * @param input - Входной тензор
 * @param dim - Размерность для softmax
 * @returns softmax(x)
 */
export function softmax(input: Tensor, dim: number = -1): Tensor {
  const actualDim = dim < 0 ? input.ndim + dim : dim;

  if (input.ndim === 1) {
    let maxVal = -Infinity;
    for (let i = 0; i < input.size; i++) {
      maxVal = Math.max(maxVal, input.data[i]);
    }

    let sumExp = 0;
    const expData = new Float32Array(input.size);
    for (let i = 0; i < input.size; i++) {
      expData[i] = Math.exp(input.data[i] - maxVal);
      sumExp += expData[i];
    }

    for (let i = 0; i < input.size; i++) {
      expData[i] /= sumExp;
    }

    return new Tensor(expData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });
  }

  // 2D случай
  const outputData = new Float32Array(input.size);
  const [batch, features] = input.shape;

  for (let b = 0; b < batch; b++) {
    let maxVal = -Infinity;
    for (let f = 0; f < features; f++) {
      maxVal = Math.max(maxVal, input.data[b * features + f]);
    }

    let sumExp = 0;
    for (let f = 0; f < features; f++) {
      sumExp += Math.exp(input.data[b * features + f] - maxVal);
    }

    for (let f = 0; f < features; f++) {
      outputData[b * features + f] = Math.exp(input.data[b * features + f] - maxVal) / sumExp;
    }
  }

  return new Tensor(outputData, [...input.shape], {
    dtype: input.dtype,
    requiresGrad: input.requiresGrad,
  });
}

/**
 * Log Softmax
 * @param input - Входной тензор  
 * @param dim - Размерность
 * @returns log(softmax(x))
 */
export function logSoftmax(input: Tensor, dim: number = -1): Tensor {
  if (input.ndim === 1) {
    let maxVal = -Infinity;
    for (let i = 0; i < input.size; i++) {
      maxVal = Math.max(maxVal, input.data[i]);
    }

    let sumExp = 0;
    for (let i = 0; i < input.size; i++) {
      sumExp += Math.exp(input.data[i] - maxVal);
    }

    const logSumExp = maxVal + Math.log(sumExp);
    const outputData = new Float32Array(input.size);
    for (let i = 0; i < input.size; i++) {
      outputData[i] = input.data[i] - logSumExp;
    }

    return new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });
  }

  const outputData = new Float32Array(input.size);
  const [batch, features] = input.shape;

  for (let b = 0; b < batch; b++) {
    let maxVal = -Infinity;
    for (let f = 0; f < features; f++) {
      maxVal = Math.max(maxVal, input.data[b * features + f]);
    }

    let sumExp = 0;
    for (let f = 0; f < features; f++) {
      sumExp += Math.exp(input.data[b * features + f] - maxVal);
    }

    const logSumExp = maxVal + Math.log(sumExp);
    for (let f = 0; f < features; f++) {
      outputData[b * features + f] = input.data[b * features + f] - logSumExp;
    }
  }

  return new Tensor(outputData, [...input.shape], {
    dtype: input.dtype,
    requiresGrad: input.requiresGrad,
  });
}

/**
 * GELU активация
 * @param input - Входной тензор
 * @returns GELU(x)
 */
export function gelu(input: Tensor): Tensor {
  const outputData = new Float32Array(input.size);

  for (let i = 0; i < input.size; i++) {
    const x = input.data[i];
    const inner = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x);
    outputData[i] = 0.5 * x * (1 + Math.tanh(inner));
  }

  return new Tensor(outputData, [...input.shape], {
    dtype: input.dtype,
    requiresGrad: input.requiresGrad,
  });
}

/**
 * SiLU / Swish активация
 * @param input - Входной тензор
 * @returns x * sigmoid(x)
 */
export function silu(input: Tensor): Tensor {
  const outputData = new Float32Array(input.size);

  for (let i = 0; i < input.size; i++) {
    const x = input.data[i];
    const sig = 1 / (1 + Math.exp(-x));
    outputData[i] = x * sig;
  }

  return new Tensor(outputData, [...input.shape], {
    dtype: input.dtype,
    requiresGrad: input.requiresGrad,
  });
}

// ============================================
// ЛИНЕЙНЫЕ ОПЕРАЦИИ
// ============================================

/**
 * Линейное преобразование: y = x @ W^T + b
 * @param input - Входной тензор
 * @param weight - Матрица весов
 * @param bias - Вектор смещений (опционально)
 * @returns Результат линейного преобразования
 */
export function linear(input: Tensor, weight: Tensor, bias?: Tensor): Tensor {
  let output = input.matmul(weight.T);
  if (bias) {
    output = output.add(bias);
  }
  return output;
}

/**
 * Dropout
 * @param input - Входной тензор
 * @param p - Вероятность dropout
 * @param training - Режим обучения
 * @returns Тензор с dropout
 */
export function dropout(input: Tensor, p: number = 0.5, training: boolean = true): Tensor {
  if (!training || p === 0) {
    return input;
  }

  const mask = new Float32Array(input.size);
  const scale = 1 / (1 - p);

  for (let i = 0; i < input.size; i++) {
    mask[i] = Math.random() > p ? scale : 0;
  }

  const maskTensor = new Tensor(mask, [...input.shape], { dtype: input.dtype });
  return input.mul(maskTensor);
}

// ============================================
// ФУНКЦИИ ПОТЕРЬ
// ============================================

/**
 * Mean Squared Error
 * @param input - Предсказания
 * @param target - Целевые значения
 * @param reduction - 'mean', 'sum', или 'none'
 * @returns MSE loss
 */
export function mseLoss(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const diff = input.sub(target);
  const squared = diff.pow(2);

  if (reduction === 'none') return squared;
  if (reduction === 'sum') return squared.sum();
  return squared.mean();
}

/**
 * Binary Cross-Entropy
 * @param input - Предсказания (вероятности)
 * @param target - Целевые значения
 * @param reduction - Способ агрегации
 * @returns BCE loss
 */
export function bceLoss(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const eps = 1e-7;
  const outputData = new Float32Array(input.size);

  for (let i = 0; i < input.size; i++) {
    const p = Math.max(eps, Math.min(1 - eps, input.data[i]));
    const t = target.data[i];
    outputData[i] = -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
  }

  const result = new Tensor(outputData, [...input.shape], { dtype: input.dtype });

  if (reduction === 'none') return result;
  if (reduction === 'sum') return result.sum();
  return result.mean();
}

/**
 * Binary Cross-Entropy with Logits
 * @param input - Logits
 * @param target - Целевые значения
 * @param reduction - Способ агрегации
 * @returns BCE with logits loss
 */
export function bceWithLogitsLoss(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const outputData = new Float32Array(input.size);

  for (let i = 0; i < input.size; i++) {
    const x = input.data[i];
    const t = target.data[i];
    const maxXZero = Math.max(x, 0);
    outputData[i] = maxXZero - x * t + Math.log(1 + Math.exp(-Math.abs(x)));
  }

  const result = new Tensor(outputData, [...input.shape], { dtype: input.dtype });

  if (reduction === 'none') return result;
  if (reduction === 'sum') return result.sum();
  return result.mean();
}

/**
 * Cross-Entropy Loss
 * @param input - Logits [batch, classes]
 * @param target - Индексы классов [batch]
 * @param reduction - Способ агрегации
 * @returns Cross-entropy loss
 */
export function crossEntropyLoss(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const batch = input.shape[0];
  const classes = input.shape[1];
  const lossData = new Float32Array(batch);

  for (let b = 0; b < batch; b++) {
    let maxVal = -Infinity;
    for (let c = 0; c < classes; c++) {
      maxVal = Math.max(maxVal, input.data[b * classes + c]);
    }

    let sumExp = 0;
    for (let c = 0; c < classes; c++) {
      sumExp += Math.exp(input.data[b * classes + c] - maxVal);
    }

    const logSumExp = maxVal + Math.log(sumExp);
    const classIdx = Math.floor(target.data[b]);
    lossData[b] = -(input.data[b * classes + classIdx] - logSumExp);
  }

  const result = new Tensor(lossData, [batch], { dtype: input.dtype });

  if (reduction === 'none') return result;
  if (reduction === 'sum') return result.sum();
  return result.mean();
}

/**
 * NLL Loss
 * @param input - Log-probabilities
 * @param target - Индексы классов
 * @param reduction - Способ агрегации
 * @returns NLL loss
 */
export function nllLoss(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const batch = input.shape[0];
  const classes = input.shape[1];
  const lossData = new Float32Array(batch);

  for (let b = 0; b < batch; b++) {
    const classIdx = Math.floor(target.data[b]);
    lossData[b] = -input.data[b * classes + classIdx];
  }

  const result = new Tensor(lossData, [batch], { dtype: input.dtype });

  if (reduction === 'none') return result;
  if (reduction === 'sum') return result.sum();
  return result.mean();
}

/**
 * L1 Loss
 * @param input - Предсказания
 * @param target - Целевые значения
 * @param reduction - Способ агрегации
 * @returns L1 loss
 */
export function l1Loss(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const diff = input.sub(target).abs();

  if (reduction === 'none') return diff;
  if (reduction === 'sum') return diff.sum();
  return diff.mean();
}

// Алиас
export { silu as swish };
