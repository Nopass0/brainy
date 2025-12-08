/**
 * @fileoverview Функции активации
 * @description Нелинейные функции активации для нейронных сетей
 */

import { Tensor, zeros } from '../core/tensor';
import { Module } from './module';
import { GradNode, GradContext, isNoGradEnabled } from '../core/autograd';

/**
 * ReLU (Rectified Linear Unit) активация
 * f(x) = max(0, x)
 * Аналог nn.ReLU в PyTorch
 */
export class ReLU extends Module {
  forward(input: Tensor): Tensor {
    const outputData = new Float32Array(input.size);
    for (let i = 0; i < input.size; i++) {
      outputData[i] = Math.max(0, input.data[i]);
    }

    const result = new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    if (input.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveTensors(input);
      result.gradNode = new GradNode(
        (gradOutput) => {
          const gradData = new Float32Array(input.size);
          for (let i = 0; i < input.size; i++) {
            gradData[i] = input.data[i] > 0 ? gradOutput.data[i] : 0;
          }
          return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
        },
        [input],
        ctx
      );
    }

    return result;
  }

  toString(): string {
    return 'ReLU()';
  }
}

/**
 * Leaky ReLU активация
 * f(x) = max(alpha * x, x)
 */
export class LeakyReLU extends Module {
  readonly negativeSlope: number;

  constructor(negativeSlope: number = 0.01) {
    super();
    this.negativeSlope = negativeSlope;
  }

  forward(input: Tensor): Tensor {
    const outputData = new Float32Array(input.size);
    for (let i = 0; i < input.size; i++) {
      outputData[i] = input.data[i] > 0 ? input.data[i] : input.data[i] * this.negativeSlope;
    }

    const result = new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    if (input.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveTensors(input);
      result.gradNode = new GradNode(
        (gradOutput) => {
          const gradData = new Float32Array(input.size);
          for (let i = 0; i < input.size; i++) {
            gradData[i] = input.data[i] > 0 ? gradOutput.data[i] : gradOutput.data[i] * this.negativeSlope;
          }
          return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
        },
        [input],
        ctx
      );
    }

    return result;
  }

  toString(): string {
    return `LeakyReLU(negative_slope=${this.negativeSlope})`;
  }
}

/**
 * ELU (Exponential Linear Unit) активация
 * f(x) = x if x > 0, else alpha * (exp(x) - 1)
 */
export class ELU extends Module {
  readonly alpha: number;

  constructor(alpha: number = 1.0) {
    super();
    this.alpha = alpha;
  }

  forward(input: Tensor): Tensor {
    const outputData = new Float32Array(input.size);
    for (let i = 0; i < input.size; i++) {
      outputData[i] = input.data[i] > 0 ? input.data[i] : this.alpha * (Math.exp(input.data[i]) - 1);
    }

    const result = new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    if (input.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveTensors(input);
      result.gradNode = new GradNode(
        (gradOutput) => {
          const gradData = new Float32Array(input.size);
          for (let i = 0; i < input.size; i++) {
            gradData[i] = input.data[i] > 0 
              ? gradOutput.data[i] 
              : gradOutput.data[i] * this.alpha * Math.exp(input.data[i]);
          }
          return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
        },
        [input],
        ctx
      );
    }

    return result;
  }

  toString(): string {
    return `ELU(alpha=${this.alpha})`;
  }
}

/**
 * SELU (Scaled Exponential Linear Unit) активация
 * Самонормализующаяся активация
 */
export class SELU extends Module {
  private readonly alpha = 1.6732632423543772;
  private readonly scale = 1.0507009873554805;

  forward(input: Tensor): Tensor {
    const outputData = new Float32Array(input.size);
    for (let i = 0; i < input.size; i++) {
      const x = input.data[i];
      outputData[i] = this.scale * (x > 0 ? x : this.alpha * (Math.exp(x) - 1));
    }

    const result = new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    if (input.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveTensors(input);
      result.gradNode = new GradNode(
        (gradOutput) => {
          const gradData = new Float32Array(input.size);
          for (let i = 0; i < input.size; i++) {
            const x = input.data[i];
            gradData[i] = gradOutput.data[i] * this.scale * (x > 0 ? 1 : this.alpha * Math.exp(x));
          }
          return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
        },
        [input],
        ctx
      );
    }

    return result;
  }

  toString(): string {
    return 'SELU()';
  }
}

/**
 * Sigmoid активация
 * f(x) = 1 / (1 + exp(-x))
 */
export class Sigmoid extends Module {
  forward(input: Tensor): Tensor {
    const outputData = new Float32Array(input.size);
    for (let i = 0; i < input.size; i++) {
      outputData[i] = 1 / (1 + Math.exp(-input.data[i]));
    }

    const result = new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    if (input.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      result.gradNode = new GradNode(
        (gradOutput) => {
          const gradData = new Float32Array(input.size);
          for (let i = 0; i < input.size; i++) {
            const sig = outputData[i];
            gradData[i] = gradOutput.data[i] * sig * (1 - sig);
          }
          return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
        },
        [input],
        ctx
      );
    }

    return result;
  }

  toString(): string {
    return 'Sigmoid()';
  }
}

/**
 * Tanh активация
 * f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 */
export class Tanh extends Module {
  forward(input: Tensor): Tensor {
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

  toString(): string {
    return 'Tanh()';
  }
}

/**
 * Softmax активация
 * Преобразует logits в вероятности
 */
export class Softmax extends Module {
  readonly dim: number;

  constructor(dim: number = -1) {
    super();
    this.dim = dim;
  }

  forward(input: Tensor): Tensor {
    // Для численной стабильности вычитаем максимум
    const dim = this.dim < 0 ? input.ndim + this.dim : this.dim;

    if (input.ndim === 1) {
      // Одномерный случай
      let maxVal = -Infinity;
      for (let i = 0; i < input.size; i++) {
        maxVal = Math.max(maxVal, input.data[i]);
      }

      const expData = new Float32Array(input.size);
      let sumExp = 0;
      for (let i = 0; i < input.size; i++) {
        expData[i] = Math.exp(input.data[i] - maxVal);
        sumExp += expData[i];
      }

      const outputData = new Float32Array(input.size);
      for (let i = 0; i < input.size; i++) {
        outputData[i] = expData[i] / sumExp;
      }

      return new Tensor(outputData, [...input.shape], {
        dtype: input.dtype,
        requiresGrad: input.requiresGrad,
      });
    }

    // 2D случай: softmax по последней размерности
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

  toString(): string {
    return `Softmax(dim=${this.dim})`;
  }
}

/**
 * LogSoftmax активация
 * log(softmax(x)) - более численно стабильная версия
 */
export class LogSoftmax extends Module {
  readonly dim: number;

  constructor(dim: number = -1) {
    super();
    this.dim = dim;
  }

  forward(input: Tensor): Tensor {
    const dim = this.dim < 0 ? input.ndim + this.dim : this.dim;

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

  toString(): string {
    return `LogSoftmax(dim=${this.dim})`;
  }
}

/**
 * GELU (Gaussian Error Linear Unit) активация
 * Используется в Transformers (BERT, GPT)
 */
export class GELU extends Module {
  forward(input: Tensor): Tensor {
    const outputData = new Float32Array(input.size);
    const sqrt2 = Math.sqrt(2);

    for (let i = 0; i < input.size; i++) {
      const x = input.data[i];
      // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
      // Аппроксимация: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
      const inner = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x);
      outputData[i] = 0.5 * x * (1 + Math.tanh(inner));
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
            const inner = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x);
            const tanhInner = Math.tanh(inner);
            const sech2 = 1 - tanhInner * tanhInner;
            const dInner = Math.sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * x * x);
            gradData[i] = gradOutput.data[i] * (0.5 * (1 + tanhInner) + 0.5 * x * sech2 * dInner);
          }
          return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
        },
        [input],
        new GradContext()
      );
    }

    return result;
  }

  toString(): string {
    return 'GELU()';
  }
}

/**
 * SiLU / Swish активация
 * f(x) = x * sigmoid(x)
 */
export class SiLU extends Module {
  forward(input: Tensor): Tensor {
    const outputData = new Float32Array(input.size);

    for (let i = 0; i < input.size; i++) {
      const x = input.data[i];
      const sigmoid = 1 / (1 + Math.exp(-x));
      outputData[i] = x * sigmoid;
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
            const sigmoid = 1 / (1 + Math.exp(-x));
            // d/dx (x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            gradData[i] = gradOutput.data[i] * (sigmoid + x * sigmoid * (1 - sigmoid));
          }
          return [new Tensor(gradData, [...input.shape], { dtype: input.dtype })];
        },
        [input],
        new GradContext()
      );
    }

    return result;
  }

  toString(): string {
    return 'SiLU()';
  }
}

// Алиас
export { SiLU as Swish };

/**
 * PReLU (Parametric ReLU) активация
 * f(x) = max(0, x) + a * min(0, x)
 * где a - обучаемый параметр
 */
export class PReLU extends Module {
  weight: Tensor;

  constructor(numParameters: number = 1, init: number = 0.25) {
    super();
    const data = new Float32Array(numParameters).fill(init);
    this.weight = new Tensor(data, [numParameters], { requiresGrad: true });
  }

  forward(input: Tensor): Tensor {
    const outputData = new Float32Array(input.size);
    const a = this.weight.data[0]; // Упрощённая версия с одним параметром

    for (let i = 0; i < input.size; i++) {
      outputData[i] = input.data[i] > 0 ? input.data[i] : a * input.data[i];
    }

    return new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });
  }

  toString(): string {
    return `PReLU(num_parameters=${this.weight.size})`;
  }
}

/**
 * Hardswish активация
 * Аппроксимация Swish с меньшими вычислительными затратами
 */
export class Hardswish extends Module {
  forward(input: Tensor): Tensor {
    const outputData = new Float32Array(input.size);

    for (let i = 0; i < input.size; i++) {
      const x = input.data[i];
      if (x <= -3) {
        outputData[i] = 0;
      } else if (x >= 3) {
        outputData[i] = x;
      } else {
        outputData[i] = x * (x + 3) / 6;
      }
    }

    return new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });
  }

  toString(): string {
    return 'Hardswish()';
  }
}

/**
 * Mish активация
 * f(x) = x * tanh(softplus(x))
 */
export class Mish extends Module {
  forward(input: Tensor): Tensor {
    const outputData = new Float32Array(input.size);

    for (let i = 0; i < input.size; i++) {
      const x = input.data[i];
      const softplus = Math.log(1 + Math.exp(x));
      outputData[i] = x * Math.tanh(softplus);
    }

    return new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });
  }

  toString(): string {
    return 'Mish()';
  }
}
