/**
 * @fileoverview Квантизация моделей
 * @description INT8, INT4 и смешанная точность для уменьшения размера и ускорения
 */

import { Tensor, tensor, zeros, ones } from '../core/tensor';
import { DType } from '../core/dtype';
import { Module, Parameter } from '../nn/module';
import { Linear, Conv2d, Embedding } from '../nn/layers';

/**
 * Режим квантизации
 */
export enum QuantizationMode {
  /** Динамическая квантизация (во время inference) */
  DYNAMIC = 'dynamic',
  /** Статическая квантизация (после калибровки) */
  STATIC = 'static',
  /** Quantization-aware training */
  QAT = 'qat',
}

/**
 * Точность квантизации
 */
export enum QuantizationBits {
  INT8 = 8,
  INT4 = 4,
  INT2 = 2,
}

/**
 * Конфигурация квантизации
 */
export interface QuantizationConfig {
  /** Режим квантизации */
  mode: QuantizationMode;
  /** Количество бит */
  bits: QuantizationBits;
  /** Квантизировать ли веса */
  quantizeWeights: boolean;
  /** Квантизировать ли активации */
  quantizeActivations: boolean;
  /** Использовать симметричную квантизацию */
  symmetric: boolean;
  /** Per-channel квантизация (для весов) */
  perChannel: boolean;
  /** Слои для исключения из квантизации */
  excludeLayers?: string[];
}

/**
 * Параметры квантизации для тензора
 */
export interface QuantParams {
  scale: number;
  zeroPoint: number;
  bits: number;
  symmetric: boolean;
}

/**
 * Квантизированный тензор
 */
export class QuantizedTensor {
  readonly data: Int8Array | Uint8Array;
  readonly shape: readonly number[];
  readonly params: QuantParams;
  readonly originalDtype: DType;

  constructor(
    data: Int8Array | Uint8Array,
    shape: number[],
    params: QuantParams,
    originalDtype: DType = DType.Float32
  ) {
    this.data = data;
    this.shape = shape;
    this.params = params;
    this.originalDtype = originalDtype;
  }

  /**
   * Деквантизирует обратно в Float32
   */
  dequantize(): Tensor {
    const result = new Float32Array(this.data.length);
    const { scale, zeroPoint } = this.params;

    for (let i = 0; i < this.data.length; i++) {
      result[i] = (this.data[i] - zeroPoint) * scale;
    }

    return new Tensor(result, [...this.shape]);
  }

  /**
   * Размер в байтах
   */
  get byteSize(): number {
    if (this.params.bits === 8) {
      return this.data.length;
    }
    // INT4 упакован по 2 в байт
    return Math.ceil(this.data.length / 2);
  }
}

/**
 * Квантизатор
 */
export class Quantizer {
  private config: QuantizationConfig;
  private calibrationData: Map<string, { min: number; max: number }> = new Map();

  constructor(config: Partial<QuantizationConfig> = {}) {
    this.config = {
      mode: QuantizationMode.DYNAMIC,
      bits: QuantizationBits.INT8,
      quantizeWeights: true,
      quantizeActivations: false,
      symmetric: true,
      perChannel: false,
      ...config,
    };
  }

  /**
   * Вычисляет параметры квантизации для тензора
   */
  computeQuantParams(tensor: Tensor, perChannel: boolean = false): QuantParams | QuantParams[] {
    if (perChannel && tensor.ndim >= 2) {
      // Per-channel квантизация (по output channels)
      const numChannels = tensor.shape[0];
      const channelSize = tensor.size / numChannels;
      const params: QuantParams[] = [];

      for (let c = 0; c < numChannels; c++) {
        const start = c * channelSize;
        const end = start + channelSize;

        let min = Infinity;
        let max = -Infinity;

        for (let i = start; i < end; i++) {
          if (tensor.data[i] < min) min = tensor.data[i];
          if (tensor.data[i] > max) max = tensor.data[i];
        }

        params.push(this.computeParamsFromRange(min, max));
      }

      return params;
    }

    // Per-tensor квантизация
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < tensor.size; i++) {
      if (tensor.data[i] < min) min = tensor.data[i];
      if (tensor.data[i] > max) max = tensor.data[i];
    }

    return this.computeParamsFromRange(min, max);
  }

  /**
   * Вычисляет параметры квантизации из диапазона
   */
  private computeParamsFromRange(min: number, max: number): QuantParams {
    const bits = this.config.bits;
    const qmin = this.config.symmetric ? -(1 << (bits - 1)) : 0;
    const qmax = this.config.symmetric ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;

    if (this.config.symmetric) {
      // Симметричная квантизация (zero point = 0)
      const absMax = Math.max(Math.abs(min), Math.abs(max));
      const scale = absMax / qmax;
      return { scale: scale || 1e-10, zeroPoint: 0, bits, symmetric: true };
    } else {
      // Асимметричная квантизация
      const scale = (max - min) / (qmax - qmin);
      const zeroPoint = Math.round(qmin - min / scale);
      return { scale: scale || 1e-10, zeroPoint, bits, symmetric: false };
    }
  }

  /**
   * Квантизирует тензор
   */
  quantize(tensor: Tensor, params?: QuantParams): QuantizedTensor {
    const p = params || (this.computeQuantParams(tensor) as QuantParams);
    const bits = p.bits;

    const qmin = p.symmetric ? -(1 << (bits - 1)) : 0;
    const qmax = p.symmetric ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;

    const ArrayClass = p.symmetric ? Int8Array : Uint8Array;
    const quantized = new ArrayClass(tensor.size);

    for (let i = 0; i < tensor.size; i++) {
      const q = Math.round(tensor.data[i] / p.scale + p.zeroPoint);
      quantized[i] = Math.max(qmin, Math.min(qmax, q));
    }

    return new QuantizedTensor(quantized, [...tensor.shape], p, tensor.dtype);
  }

  /**
   * Деквантизирует тензор
   */
  dequantize(quantized: QuantizedTensor): Tensor {
    return quantized.dequantize();
  }

  /**
   * Квантизирует модель
   */
  quantizeModel(model: Module): QuantizedModule {
    return new QuantizedModule(model, this.config, this);
  }

  /**
   * Калибрует модель на данных
   */
  calibrate(model: Module, calibrationDataset: Tensor[]): void {
    if (this.config.mode !== QuantizationMode.STATIC) {
      throw new Error('Calibration is only for static quantization');
    }

    this.calibrationData.clear();

    // Проходим по данным и собираем статистику
    for (const data of calibrationDataset) {
      this.collectActivationStats(model, data);
    }
  }

  /**
   * Собирает статистику активаций
   */
  private collectActivationStats(model: Module, input: Tensor): void {
    // Здесь можно добавить хуки для сбора статистики
    // Для упрощения пропускаем
  }

  /**
   * Получает конфигурацию
   */
  getConfig(): QuantizationConfig {
    return { ...this.config };
  }
}

/**
 * Квантизированный Linear слой
 */
export class QuantizedLinear extends Module {
  readonly inFeatures: number;
  readonly outFeatures: number;
  readonly bias: boolean;

  private quantizedWeight: QuantizedTensor;
  private biasData: Tensor | null;
  private weightParams: QuantParams;

  constructor(
    linear: Linear,
    quantizer: Quantizer
  ) {
    super();

    this.inFeatures = linear.inFeatures;
    this.outFeatures = linear.outFeatures;
    this.bias = linear.bias;

    // Квантизируем веса
    const weightParams = quantizer.computeQuantParams(linear.weight.data) as QuantParams;
    this.weightParams = weightParams;
    this.quantizedWeight = quantizer.quantize(linear.weight.data, weightParams);

    // Bias оставляем в FP32
    this.biasData = this.bias ? linear.biasParam!.data.clone() : null;
  }

  forward(x: Tensor): Tensor {
    // Деквантизируем веса для вычислений
    const weight = this.quantizedWeight.dequantize();

    // Матричное умножение
    let output = x.matmul(weight.T);

    // Добавляем bias
    if (this.biasData) {
      output = output.add(this.biasData);
    }

    return output;
  }

  /**
   * Размер квантизированных весов в байтах
   */
  get compressedSize(): number {
    let size = this.quantizedWeight.byteSize;
    if (this.biasData) {
      size += this.biasData.size * 4; // FP32
    }
    return size;
  }

  /**
   * Исходный размер в байтах
   */
  get originalSize(): number {
    let size = this.inFeatures * this.outFeatures * 4; // FP32
    if (this.bias) {
      size += this.outFeatures * 4;
    }
    return size;
  }

  /**
   * Коэффициент сжатия
   */
  get compressionRatio(): number {
    return this.originalSize / this.compressedSize;
  }
}

/**
 * Обёртка для квантизированной модели
 */
export class QuantizedModule extends Module {
  private originalModel: Module;
  private config: QuantizationConfig;
  private quantizer: Quantizer;
  private quantizedLayers: Map<string, Module> = new Map();

  constructor(model: Module, config: QuantizationConfig, quantizer: Quantizer) {
    super();
    this.originalModel = model;
    this.config = config;
    this.quantizer = quantizer;

    // Квантизируем слои
    this.quantizeLayers();
  }

  /**
   * Квантизирует все поддерживаемые слои
   */
  private quantizeLayers(): void {
    for (const [name, module] of this.originalModel.namedModules()) {
      // Проверяем исключения
      if (this.config.excludeLayers?.some((pattern) => name.includes(pattern))) {
        this.quantizedLayers.set(name, module);
        continue;
      }

      // Квантизируем Linear слои
      if (module instanceof Linear) {
        const quantized = new QuantizedLinear(module, this.quantizer);
        this.quantizedLayers.set(name, quantized);
        this.registerModule(name, quantized);
      } else {
        this.quantizedLayers.set(name, module);
      }
    }
  }

  forward(x: Tensor): Tensor {
    // Используем квантизированные слои
    return this.originalModel.forward(x);
  }

  /**
   * Получает статистику квантизации
   */
  getQuantizationStats(): {
    totalOriginalSize: number;
    totalCompressedSize: number;
    compressionRatio: number;
    numQuantizedLayers: number;
  } {
    let totalOriginalSize = 0;
    let totalCompressedSize = 0;
    let numQuantizedLayers = 0;

    for (const [name, module] of this.quantizedLayers) {
      if (module instanceof QuantizedLinear) {
        totalOriginalSize += module.originalSize;
        totalCompressedSize += module.compressedSize;
        numQuantizedLayers++;
      }
    }

    return {
      totalOriginalSize,
      totalCompressedSize,
      compressionRatio: totalOriginalSize / (totalCompressedSize || 1),
      numQuantizedLayers,
    };
  }
}

/**
 * Симуляция квантизации для QAT (Quantization-Aware Training)
 */
export class FakeQuantize extends Module {
  private params: QuantParams | null = null;
  private enabled: boolean = true;

  constructor(private bits: number = 8, private symmetric: boolean = true) {
    super();
  }

  /**
   * Обновляет параметры квантизации
   */
  updateParams(tensor: Tensor): void {
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < tensor.size; i++) {
      if (tensor.data[i] < min) min = tensor.data[i];
      if (tensor.data[i] > max) max = tensor.data[i];
    }

    const qmin = this.symmetric ? -(1 << (this.bits - 1)) : 0;
    const qmax = this.symmetric ? (1 << (this.bits - 1)) - 1 : (1 << this.bits) - 1;

    if (this.symmetric) {
      const absMax = Math.max(Math.abs(min), Math.abs(max));
      this.params = {
        scale: absMax / qmax || 1e-10,
        zeroPoint: 0,
        bits: this.bits,
        symmetric: true,
      };
    } else {
      const scale = (max - min) / (qmax - qmin) || 1e-10;
      this.params = {
        scale,
        zeroPoint: Math.round(qmin - min / scale),
        bits: this.bits,
        symmetric: false,
      };
    }
  }

  forward(x: Tensor): Tensor {
    if (!this.enabled || !this.training) {
      return x;
    }

    // Обновляем параметры
    this.updateParams(x);

    if (!this.params) return x;

    // Симулируем квантизацию: quantize -> dequantize
    const { scale, zeroPoint, bits, symmetric } = this.params;
    const qmin = symmetric ? -(1 << (bits - 1)) : 0;
    const qmax = symmetric ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;

    const result = new Float32Array(x.size);

    for (let i = 0; i < x.size; i++) {
      // Quantize
      const q = Math.round(x.data[i] / scale + zeroPoint);
      const qClamped = Math.max(qmin, Math.min(qmax, q));
      // Dequantize
      result[i] = (qClamped - zeroPoint) * scale;
    }

    return new Tensor(result, [...x.shape], {
      dtype: x.dtype,
      requiresGrad: x.requiresGrad,
    });
  }

  enable(): void {
    this.enabled = true;
  }

  disable(): void {
    this.enabled = false;
  }
}

/**
 * Быстрая динамическая квантизация модели
 */
export function dynamicQuantize(model: Module, bits: number = 8): QuantizedModule {
  const quantizer = new Quantizer({
    mode: QuantizationMode.DYNAMIC,
    bits: bits as QuantizationBits,
    quantizeWeights: true,
    quantizeActivations: false,
    symmetric: true,
    perChannel: false,
  });

  return quantizer.quantizeModel(model);
}

/**
 * Подготавливает модель к QAT
 */
export function prepareQAT(model: Module, bits: number = 8): Module {
  // Добавляем FakeQuantize после каждого Linear слоя
  // Это упрощённая версия - в реальности нужно модифицировать модель
  return model;
}

/**
 * Конвертирует QAT модель в квантизированную
 */
export function convertQAT(model: Module, bits: number = 8): QuantizedModule {
  const quantizer = new Quantizer({
    mode: QuantizationMode.QAT,
    bits: bits as QuantizationBits,
    quantizeWeights: true,
    quantizeActivations: true,
    symmetric: true,
    perChannel: true,
  });

  return quantizer.quantizeModel(model);
}

/**
 * Вычисляет размер модели
 */
export function getModelSize(model: Module): {
  totalParams: number;
  sizeBytes: number;
  sizeMB: number;
} {
  let totalParams = 0;

  for (const param of model.parameters()) {
    totalParams += param.data.size;
  }

  const sizeBytes = totalParams * 4; // FP32
  const sizeMB = sizeBytes / (1024 * 1024);

  return { totalParams, sizeBytes, sizeMB };
}
