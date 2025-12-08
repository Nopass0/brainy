/**
 * @fileoverview Нейросетевые слои (layers)
 * @description Реализация основных слоёв: Linear, Conv2d, Pooling, BatchNorm, Dropout и др.
 */

import { Tensor, zeros, randn, ones } from '../core/tensor';
import { Module, Parameter } from './module';
import { computeSize } from '../core/shape';

/**
 * Полносвязный (линейный) слой
 * Выполняет операцию: y = x @ W^T + b
 * Аналог nn.Linear в PyTorch
 * 
 * @example
 * const layer = new Linear(10, 5);
 * const output = layer.forward(input); // [batch, 5]
 */
export class Linear extends Module {
  /** Матрица весов */
  weight: Parameter;
  /** Вектор смещений (если bias=true) */
  bias: Parameter | null;
  /** Количество входных признаков */
  readonly inFeatures: number;
  /** Количество выходных признаков */
  readonly outFeatures: number;

  /**
   * Создаёт линейный слой
   * @param inFeatures - Количество входных признаков
   * @param outFeatures - Количество выходных признаков
   * @param useBias - Использовать смещение (по умолчанию true)
   */
  constructor(inFeatures: number, outFeatures: number, useBias: boolean = true) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;

    // Инициализация весов (Xavier uniform)
    const k = Math.sqrt(1 / inFeatures);
    const weightData = new Float32Array(outFeatures * inFeatures);
    for (let i = 0; i < weightData.length; i++) {
      weightData[i] = (Math.random() * 2 - 1) * k;
    }
    this.weight = new Parameter(
      new Tensor(weightData, [outFeatures, inFeatures], { requiresGrad: true }),
      'weight'
    );
    this.registerParameter('weight', this.weight);

    if (useBias) {
      const biasData = new Float32Array(outFeatures);
      for (let i = 0; i < biasData.length; i++) {
        biasData[i] = (Math.random() * 2 - 1) * k;
      }
      this.bias = new Parameter(
        new Tensor(biasData, [outFeatures], { requiresGrad: true }),
        'bias'
      );
      this.registerParameter('bias', this.bias);
    } else {
      this.bias = null;
    }
  }

  /**
   * Прямой проход линейного слоя
   * @param input - Входной тензор [batch, inFeatures] или [inFeatures]
   * @returns Выходной тензор [batch, outFeatures] или [outFeatures]
   */
  forward(input: Tensor): Tensor {
    // input: [batch, in] или [in]
    // weight: [out, in]
    // output: [batch, out] или [out]

    let output: Tensor;

    if (input.ndim === 1) {
      // [in] @ [out, in]^T = [out]
      output = input.reshape(1, -1).matmul(this.weight.data.T).reshape(-1);
    } else {
      // [batch, in] @ [in, out] = [batch, out]
      output = input.matmul(this.weight.data.T);
    }

    if (this.bias) {
      output = output.add(this.bias.data);
    }

    return output;
  }

  toString(): string {
    return `Linear(in_features=${this.inFeatures}, out_features=${this.outFeatures}, bias=${this.bias !== null})`;
  }
}

/**
 * 2D свёрточный слой
 * Аналог nn.Conv2d в PyTorch
 * 
 * @example
 * const conv = new Conv2d(3, 64, 3, 1, 1);
 * const output = conv.forward(input); // [batch, 64, H, W]
 */
export class Conv2d extends Module {
  /** Ядра свёртки */
  weight: Parameter;
  /** Смещения */
  bias: Parameter | null;
  /** Количество входных каналов */
  readonly inChannels: number;
  /** Количество выходных каналов */
  readonly outChannels: number;
  /** Размер ядра */
  readonly kernelSize: [number, number];
  /** Шаг свёртки */
  readonly stride: [number, number];
  /** Паддинг */
  readonly padding: [number, number];

  /**
   * Создаёт свёрточный слой
   * @param inChannels - Количество входных каналов
   * @param outChannels - Количество выходных каналов (фильтров)
   * @param kernelSize - Размер ядра (число или кортеж)
   * @param stride - Шаг свёртки (по умолчанию 1)
   * @param padding - Паддинг (по умолчанию 0)
   * @param useBias - Использовать смещение
   */
  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number],
    stride: number | [number, number] = 1,
    padding: number | [number, number] = 0,
    useBias: boolean = true
  ) {
    super();
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
    this.stride = typeof stride === 'number' ? [stride, stride] : stride;
    this.padding = typeof padding === 'number' ? [padding, padding] : padding;

    // Инициализация весов (Kaiming uniform)
    const fanIn = inChannels * this.kernelSize[0] * this.kernelSize[1];
    const k = Math.sqrt(1 / fanIn);
    const weightSize = outChannels * inChannels * this.kernelSize[0] * this.kernelSize[1];
    const weightData = new Float32Array(weightSize);
    for (let i = 0; i < weightSize; i++) {
      weightData[i] = (Math.random() * 2 - 1) * k;
    }
    this.weight = new Parameter(
      new Tensor(weightData, [outChannels, inChannels, this.kernelSize[0], this.kernelSize[1]], { requiresGrad: true }),
      'weight'
    );
    this.registerParameter('weight', this.weight);

    if (useBias) {
      const biasData = new Float32Array(outChannels);
      for (let i = 0; i < outChannels; i++) {
        biasData[i] = (Math.random() * 2 - 1) * k;
      }
      this.bias = new Parameter(
        new Tensor(biasData, [outChannels], { requiresGrad: true }),
        'bias'
      );
      this.registerParameter('bias', this.bias);
    } else {
      this.bias = null;
    }
  }

  /**
   * Прямой проход свёрточного слоя
   * @param input - Входной тензор [batch, inChannels, H, W]
   * @returns Выходной тензор [batch, outChannels, H_out, W_out]
   */
  forward(input: Tensor): Tensor {
    const [batch, , inH, inW] = input.shape;
    const [kH, kW] = this.kernelSize;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;

    // Вычисляем выходные размеры
    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

    const output = zeros([batch, this.outChannels, outH, outW], { requiresGrad: input.requiresGrad });

    // Наивная реализация свёртки
    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < this.outChannels; oc++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let sum = this.bias ? this.bias.data.data[oc] : 0;

            for (let ic = 0; ic < this.inChannels; ic++) {
              for (let kh = 0; kh < kH; kh++) {
                for (let kw = 0; kw < kW; kw++) {
                  const ih = oh * sH - pH + kh;
                  const iw = ow * sW - pW + kw;

                  if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                    const inputIdx = b * (this.inChannels * inH * inW) + ic * (inH * inW) + ih * inW + iw;
                    const weightIdx = oc * (this.inChannels * kH * kW) + ic * (kH * kW) + kh * kW + kw;
                    sum += input.data[inputIdx] * this.weight.data.data[weightIdx];
                  }
                }
              }
            }

            const outputIdx = b * (this.outChannels * outH * outW) + oc * (outH * outW) + oh * outW + ow;
            (output.data as Float32Array)[outputIdx] = sum;
          }
        }
      }
    }

    return output;
  }

  toString(): string {
    return `Conv2d(${this.inChannels}, ${this.outChannels}, kernel_size=${this.kernelSize}, stride=${this.stride}, padding=${this.padding})`;
  }
}

/**
 * 2D Max Pooling слой
 * Аналог nn.MaxPool2d в PyTorch
 */
export class MaxPool2d extends Module {
  /** Размер окна */
  readonly kernelSize: [number, number];
  /** Шаг */
  readonly stride: [number, number];
  /** Паддинг */
  readonly padding: [number, number];

  /**
   * Создаёт MaxPool2d слой
   * @param kernelSize - Размер окна
   * @param stride - Шаг (по умолчанию = kernelSize)
   * @param padding - Паддинг
   */
  constructor(
    kernelSize: number | [number, number],
    stride?: number | [number, number],
    padding: number | [number, number] = 0
  ) {
    super();
    this.kernelSize = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
    this.stride = stride
      ? (typeof stride === 'number' ? [stride, stride] : stride)
      : this.kernelSize;
    this.padding = typeof padding === 'number' ? [padding, padding] : padding;
  }

  /**
   * Прямой проход MaxPool2d
   * @param input - [batch, channels, H, W]
   * @returns [batch, channels, H_out, W_out]
   */
  forward(input: Tensor): Tensor {
    const [batch, channels, inH, inW] = input.shape;
    const [kH, kW] = this.kernelSize;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;

    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

    const outputData = new Float32Array(batch * channels * outH * outW);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let maxVal = -Infinity;

            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * sH - pH + kh;
                const iw = ow * sW - pW + kw;

                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                  const idx = b * (channels * inH * inW) + c * (inH * inW) + ih * inW + iw;
                  maxVal = Math.max(maxVal, input.data[idx]);
                }
              }
            }

            const outIdx = b * (channels * outH * outW) + c * (outH * outW) + oh * outW + ow;
            outputData[outIdx] = maxVal;
          }
        }
      }
    }

    return new Tensor(outputData, [batch, channels, outH, outW], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });
  }

  toString(): string {
    return `MaxPool2d(kernel_size=${this.kernelSize}, stride=${this.stride}, padding=${this.padding})`;
  }
}

/**
 * 2D Average Pooling слой
 * Аналог nn.AvgPool2d в PyTorch
 */
export class AvgPool2d extends Module {
  readonly kernelSize: [number, number];
  readonly stride: [number, number];
  readonly padding: [number, number];

  constructor(
    kernelSize: number | [number, number],
    stride?: number | [number, number],
    padding: number | [number, number] = 0
  ) {
    super();
    this.kernelSize = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
    this.stride = stride
      ? (typeof stride === 'number' ? [stride, stride] : stride)
      : this.kernelSize;
    this.padding = typeof padding === 'number' ? [padding, padding] : padding;
  }

  forward(input: Tensor): Tensor {
    const [batch, channels, inH, inW] = input.shape;
    const [kH, kW] = this.kernelSize;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;

    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

    const outputData = new Float32Array(batch * channels * outH * outW);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let sum = 0;
            let count = 0;

            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * sH - pH + kh;
                const iw = ow * sW - pW + kw;

                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                  const idx = b * (channels * inH * inW) + c * (inH * inW) + ih * inW + iw;
                  sum += input.data[idx];
                  count++;
                }
              }
            }

            const outIdx = b * (channels * outH * outW) + c * (outH * outW) + oh * outW + ow;
            outputData[outIdx] = sum / count;
          }
        }
      }
    }

    return new Tensor(outputData, [batch, channels, outH, outW], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });
  }

  toString(): string {
    return `AvgPool2d(kernel_size=${this.kernelSize}, stride=${this.stride}, padding=${this.padding})`;
  }
}

/**
 * Слой Dropout для регуляризации
 * Случайно обнуляет элементы во время обучения
 * Аналог nn.Dropout в PyTorch
 */
export class Dropout extends Module {
  /** Вероятность dropout */
  readonly p: number;

  /**
   * Создаёт Dropout слой
   * @param p - Вероятность обнуления (по умолчанию 0.5)
   */
  constructor(p: number = 0.5) {
    super();
    this.p = p;
  }

  forward(input: Tensor): Tensor {
    if (!this.training || this.p === 0) {
      return input;
    }

    const mask = new Float32Array(input.size);
    const scale = 1 / (1 - this.p);

    for (let i = 0; i < input.size; i++) {
      mask[i] = Math.random() > this.p ? scale : 0;
    }

    const maskTensor = new Tensor(mask, [...input.shape], { dtype: input.dtype });
    return input.mul(maskTensor);
  }

  toString(): string {
    return `Dropout(p=${this.p})`;
  }
}

/**
 * Batch Normalization для 1D данных (полносвязные слои)
 * Аналог nn.BatchNorm1d в PyTorch
 */
export class BatchNorm1d extends Module {
  /** Количество признаков */
  readonly numFeatures: number;
  /** Epsilon для численной стабильности */
  readonly eps: number;
  /** Momentum для running stats */
  readonly momentum: number;
  /** Обучаемый масштаб */
  gamma: Parameter;
  /** Обучаемый сдвиг */
  beta: Parameter;
  /** Running mean */
  runningMean: Tensor;
  /** Running variance */
  runningVar: Tensor;

  constructor(numFeatures: number, eps: number = 1e-5, momentum: number = 0.1) {
    super();
    this.numFeatures = numFeatures;
    this.eps = eps;
    this.momentum = momentum;

    this.gamma = new Parameter(ones([numFeatures], { requiresGrad: true }), 'gamma');
    this.beta = new Parameter(zeros([numFeatures], { requiresGrad: true }), 'beta');
    this.registerParameter('weight', this.gamma);
    this.registerParameter('bias', this.beta);

    this.runningMean = zeros([numFeatures]);
    this.runningVar = ones([numFeatures]);
  }

  forward(input: Tensor): Tensor {
    if (this.training) {
      // Вычисляем статистики по batch
      const mean = input.mean(0);
      const variance = input.sub(mean).pow(2).mean(0);

      // Обновляем running stats
      this.runningMean = this.runningMean.mul(1 - this.momentum).add(mean.mul(this.momentum));
      this.runningVar = this.runningVar.mul(1 - this.momentum).add(variance.mul(this.momentum));

      // Нормализация
      const normalized = input.sub(mean).div(variance.add(this.eps).sqrt());
      return normalized.mul(this.gamma.data).add(this.beta.data);
    } else {
      // Используем running stats
      const normalized = input.sub(this.runningMean).div(this.runningVar.add(this.eps).sqrt());
      return normalized.mul(this.gamma.data).add(this.beta.data);
    }
  }

  toString(): string {
    return `BatchNorm1d(${this.numFeatures}, eps=${this.eps}, momentum=${this.momentum})`;
  }
}

/**
 * Batch Normalization для 2D данных (свёрточные слои)
 * Аналог nn.BatchNorm2d в PyTorch
 */
export class BatchNorm2d extends Module {
  readonly numFeatures: number;
  readonly eps: number;
  readonly momentum: number;
  gamma: Parameter;
  beta: Parameter;
  runningMean: Tensor;
  runningVar: Tensor;

  constructor(numFeatures: number, eps: number = 1e-5, momentum: number = 0.1) {
    super();
    this.numFeatures = numFeatures;
    this.eps = eps;
    this.momentum = momentum;

    this.gamma = new Parameter(ones([numFeatures], { requiresGrad: true }), 'gamma');
    this.beta = new Parameter(zeros([numFeatures], { requiresGrad: true }), 'beta');
    this.registerParameter('weight', this.gamma);
    this.registerParameter('bias', this.beta);

    this.runningMean = zeros([numFeatures]);
    this.runningVar = ones([numFeatures]);
  }

  forward(input: Tensor): Tensor {
    const [batch, channels, height, width] = input.shape;

    // Вычисляем mean и var по batch, height, width (для каждого канала)
    const outputData = new Float32Array(input.size);

    for (let c = 0; c < channels; c++) {
      let mean: number, variance: number;

      if (this.training) {
        // Вычисляем статистики
        let sum = 0;
        let sumSq = 0;
        const count = batch * height * width;

        for (let b = 0; b < batch; b++) {
          for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
              const idx = b * (channels * height * width) + c * (height * width) + h * width + w;
              sum += input.data[idx];
              sumSq += input.data[idx] * input.data[idx];
            }
          }
        }

        mean = sum / count;
        variance = sumSq / count - mean * mean;

        // Обновляем running stats
        (this.runningMean.data as Float32Array)[c] = 
          this.runningMean.data[c] * (1 - this.momentum) + mean * this.momentum;
        (this.runningVar.data as Float32Array)[c] = 
          this.runningVar.data[c] * (1 - this.momentum) + variance * this.momentum;
      } else {
        mean = this.runningMean.data[c];
        variance = this.runningVar.data[c];
      }

      const std = Math.sqrt(variance + this.eps);
      const gamma = this.gamma.data.data[c];
      const beta = this.beta.data.data[c];

      // Нормализация
      for (let b = 0; b < batch; b++) {
        for (let h = 0; h < height; h++) {
          for (let w = 0; w < width; w++) {
            const idx = b * (channels * height * width) + c * (height * width) + h * width + w;
            outputData[idx] = ((input.data[idx] - mean) / std) * gamma + beta;
          }
        }
      }
    }

    return new Tensor(outputData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });
  }

  toString(): string {
    return `BatchNorm2d(${this.numFeatures}, eps=${this.eps}, momentum=${this.momentum})`;
  }
}

/**
 * Слой Flatten - выравнивает тензор
 * Аналог nn.Flatten в PyTorch
 */
export class Flatten extends Module {
  readonly startDim: number;
  readonly endDim: number;

  /**
   * Создаёт Flatten слой
   * @param startDim - Начальная размерность для выравнивания
   * @param endDim - Конечная размерность
   */
  constructor(startDim: number = 1, endDim: number = -1) {
    super();
    this.startDim = startDim;
    this.endDim = endDim;
  }

  forward(input: Tensor): Tensor {
    return input.flatten(this.startDim, this.endDim);
  }

  toString(): string {
    return `Flatten(start_dim=${this.startDim}, end_dim=${this.endDim})`;
  }
}

/**
 * Embedding слой для работы с дискретными признаками
 * Аналог nn.Embedding в PyTorch
 */
export class Embedding extends Module {
  weight: Parameter;
  readonly numEmbeddings: number;
  readonly embeddingDim: number;

  /**
   * Создаёт Embedding слой
   * @param numEmbeddings - Размер словаря
   * @param embeddingDim - Размерность embedding
   */
  constructor(numEmbeddings: number, embeddingDim: number) {
    super();
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;

    this.weight = new Parameter(
      randn([numEmbeddings, embeddingDim], 0, 1, { requiresGrad: true }),
      'weight'
    );
    this.registerParameter('weight', this.weight);
  }

  forward(input: Tensor): Tensor {
    // input: [batch, seq_len] или [seq_len] - индексы
    // output: [batch, seq_len, embedding_dim] или [seq_len, embedding_dim]

    const flatInput = input.flatten();
    const outputData = new Float32Array(flatInput.size * this.embeddingDim);

    for (let i = 0; i < flatInput.size; i++) {
      const idx = Math.floor(flatInput.data[i]);
      for (let j = 0; j < this.embeddingDim; j++) {
        outputData[i * this.embeddingDim + j] = this.weight.data.data[idx * this.embeddingDim + j];
      }
    }

    const outputShape = [...input.shape, this.embeddingDim];
    return new Tensor(outputData, outputShape, {
      dtype: this.weight.data.dtype,
      requiresGrad: this.weight.data.requiresGrad,
    });
  }

  toString(): string {
    return `Embedding(${this.numEmbeddings}, ${this.embeddingDim})`;
  }
}

/**
 * Layer Normalization
 * Аналог nn.LayerNorm в PyTorch
 */
export class LayerNorm extends Module {
  readonly normalizedShape: number[];
  readonly eps: number;
  gamma: Parameter;
  beta: Parameter;

  constructor(normalizedShape: number | number[], eps: number = 1e-5) {
    super();
    this.normalizedShape = typeof normalizedShape === 'number' ? [normalizedShape] : normalizedShape;
    this.eps = eps;

    const size = computeSize(this.normalizedShape);
    this.gamma = new Parameter(ones(this.normalizedShape, { requiresGrad: true }), 'weight');
    this.beta = new Parameter(zeros(this.normalizedShape, { requiresGrad: true }), 'bias');
    this.registerParameter('weight', this.gamma);
    this.registerParameter('bias', this.beta);
  }

  forward(input: Tensor): Tensor {
    // Нормализация по последним dim
    const normalizedDims = this.normalizedShape.length;
    const batchDims = input.ndim - normalizedDims;

    // Вычисляем mean и var по нормализуемым размерностям
    let result = input;
    for (let i = input.ndim - 1; i >= batchDims; i--) {
      const mean = result.mean(i, true);
      const variance = result.sub(mean).pow(2).mean(i, true);
      result = result.sub(mean).div(variance.add(this.eps).sqrt());
    }

    return result.mul(this.gamma.data).add(this.beta.data);
  }

  toString(): string {
    return `LayerNorm(${this.normalizedShape}, eps=${this.eps})`;
  }
}
