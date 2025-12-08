/**
 * @fileoverview Система чекпоинтов и сжатия весов
 * @description Сохранение, загрузка и сжатие моделей
 */

import { Tensor } from '../core/tensor';
import { DType } from '../core/dtype';
import { Module } from '../nn/module';
import { Optimizer } from '../optim/optimizer';

/**
 * Формат сжатия
 */
export enum CompressionFormat {
  /** Без сжатия */
  NONE = 'none',
  /** GZIP сжатие */
  GZIP = 'gzip',
  /** Float16 (half precision) */
  FLOAT16 = 'float16',
  /** Quantized INT8 */
  INT8 = 'int8',
  /** Quantized INT4 */
  INT4 = 'int4',
}

/**
 * Метаданные чекпоинта
 */
export interface CheckpointMetadata {
  /** Версия формата */
  version: string;
  /** Имя фреймворка */
  framework: string;
  /** Время создания */
  createdAt: string;
  /** Эпоха обучения */
  epoch?: number;
  /** Шаг обучения */
  step?: number;
  /** Loss на момент сохранения */
  loss?: number;
  /** Дополнительные метрики */
  metrics?: Record<string, number>;
  /** Формат сжатия */
  compression: CompressionFormat;
  /** Конфигурация модели */
  modelConfig?: object;
  /** Размер до сжатия (байты) */
  originalSize?: number;
  /** Размер после сжатия (байты) */
  compressedSize?: number;
}

/**
 * Чекпоинт
 */
export interface Checkpoint {
  metadata: CheckpointMetadata;
  modelState: SerializedState;
  optimizerState?: SerializedOptimizerState;
  schedulerState?: object;
}

/**
 * Сериализованное состояние модели
 */
export interface SerializedState {
  [key: string]: {
    data: ArrayBuffer | number[];
    shape: number[];
    dtype: string;
    compressed?: boolean;
  };
}

/**
 * Сериализованное состояние оптимизатора
 */
export interface SerializedOptimizerState {
  type: string;
  lr: number;
  stepCount: number;
  paramStates: Record<string, unknown>;
}

/**
 * Опции сохранения
 */
export interface SaveOptions {
  /** Формат сжатия */
  compression?: CompressionFormat;
  /** Включать ли оптимизатор */
  includeOptimizer?: boolean;
  /** Дополнительные метаданные */
  metadata?: Partial<CheckpointMetadata>;
}

/**
 * Менеджер чекпоинтов
 */
export class CheckpointManager {
  private baseDir: string;
  private maxCheckpoints: number;
  private checkpoints: string[] = [];

  constructor(baseDir: string, maxCheckpoints: number = 5) {
    this.baseDir = baseDir;
    this.maxCheckpoints = maxCheckpoints;
  }

  /**
   * Сохраняет чекпоинт
   */
  async save(
    model: Module,
    optimizer?: Optimizer,
    options: SaveOptions = {}
  ): Promise<string> {
    const {
      compression = CompressionFormat.GZIP,
      includeOptimizer = true,
      metadata = {},
    } = options;

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `checkpoint_${timestamp}.brainy`;
    const filepath = `${this.baseDir}/${filename}`;

    // Создаём чекпоинт
    const checkpoint = await this.createCheckpoint(
      model,
      includeOptimizer ? optimizer : undefined,
      compression,
      metadata
    );

    // Сериализуем
    const data = await this.serializeCheckpoint(checkpoint, compression);

    // Сохраняем
    await Bun.write(filepath, data);

    // Управляем количеством чекпоинтов
    this.checkpoints.push(filepath);
    await this.pruneOldCheckpoints();

    return filepath;
  }

  /**
   * Загружает чекпоинт
   */
  async load(
    filepath: string,
    model: Module,
    optimizer?: Optimizer
  ): Promise<CheckpointMetadata> {
    const file = Bun.file(filepath);
    const data = await file.arrayBuffer();

    // Десериализуем
    const checkpoint = await this.deserializeCheckpoint(new Uint8Array(data));

    // Загружаем состояние модели
    this.loadModelState(model, checkpoint.modelState);

    // Загружаем состояние оптимизатора
    if (optimizer && checkpoint.optimizerState) {
      this.loadOptimizerState(optimizer, checkpoint.optimizerState);
    }

    return checkpoint.metadata;
  }

  /**
   * Создаёт чекпоинт
   */
  private async createCheckpoint(
    model: Module,
    optimizer: Optimizer | undefined,
    compression: CompressionFormat,
    extraMetadata: Partial<CheckpointMetadata>
  ): Promise<Checkpoint> {
    const stateDict = model.stateDict();
    const modelState = await this.serializeStateDict(stateDict, compression);

    let optimizerState: SerializedOptimizerState | undefined;
    if (optimizer) {
      optimizerState = this.serializeOptimizer(optimizer);
    }

    const metadata: CheckpointMetadata = {
      version: '1.0.0',
      framework: 'brainy',
      createdAt: new Date().toISOString(),
      compression,
      ...extraMetadata,
    };

    return {
      metadata,
      modelState,
      optimizerState,
    };
  }

  /**
   * Сериализует state dict
   */
  private async serializeStateDict(
    stateDict: Map<string, Tensor>,
    compression: CompressionFormat
  ): Promise<SerializedState> {
    const serialized: SerializedState = {};

    for (const [name, tensor] of stateDict) {
      let data: ArrayBuffer | number[];
      let compressed = false;

      switch (compression) {
        case CompressionFormat.FLOAT16:
          data = this.toFloat16(tensor.data as Float32Array);
          compressed = true;
          break;
        case CompressionFormat.INT8:
          data = this.quantizeToInt8(tensor.data as Float32Array);
          compressed = true;
          break;
        case CompressionFormat.INT4:
          data = this.quantizeToInt4(tensor.data as Float32Array);
          compressed = true;
          break;
        default:
          data = Array.from(tensor.data);
      }

      serialized[name] = {
        data,
        shape: [...tensor.shape],
        dtype: tensor.dtype,
        compressed,
      };
    }

    return serialized;
  }

  /**
   * Конвертирует Float32 в Float16
   */
  private toFloat16(data: Float32Array): ArrayBuffer {
    const result = new Uint16Array(data.length);

    for (let i = 0; i < data.length; i++) {
      result[i] = this.float32ToFloat16(data[i]);
    }

    return result.buffer;
  }

  /**
   * Конвертирует одно Float32 число в Float16
   */
  private float32ToFloat16(value: number): number {
    const floatView = new Float32Array(1);
    const intView = new Uint32Array(floatView.buffer);
    floatView[0] = value;
    const x = intView[0];

    // IEEE 754 conversion
    const sign = (x >> 16) & 0x8000;
    let exponent = ((x >> 23) & 0xff) - 127 + 15;
    let mantissa = x & 0x7fffff;

    if (exponent <= 0) {
      // Subnormal or zero
      if (exponent < -10) {
        return sign;
      }
      mantissa = (mantissa | 0x800000) >> (1 - exponent);
      return sign | (mantissa >> 13);
    } else if (exponent === 0xff - 127 + 15) {
      // Inf or NaN
      return sign | 0x7c00 | (mantissa ? (mantissa >> 13) : 0);
    } else if (exponent > 30) {
      // Overflow to Inf
      return sign | 0x7c00;
    }

    return sign | (exponent << 10) | (mantissa >> 13);
  }

  /**
   * Конвертирует Float16 обратно в Float32
   */
  private float16ToFloat32(h: number): number {
    const sign = (h & 0x8000) >> 15;
    const exponent = (h & 0x7c00) >> 10;
    const mantissa = h & 0x03ff;

    if (exponent === 0) {
      if (mantissa === 0) {
        return sign ? -0 : 0;
      }
      // Subnormal
      const e = Math.clz32(mantissa) - 21;
      return (sign ? -1 : 1) * Math.pow(2, -14 - e) * ((mantissa << e) & 0x3ff) / 1024;
    } else if (exponent === 31) {
      return mantissa ? NaN : sign ? -Infinity : Infinity;
    }

    return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
  }

  /**
   * Квантизация в INT8
   */
  private quantizeToInt8(data: Float32Array): ArrayBuffer {
    // Находим min/max для масштабирования
    let min = Infinity;
    let max = -Infinity;
    for (const v of data) {
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const scale = (max - min) / 255;
    const zeroPoint = Math.round(-min / scale);

    // Квантизуем
    const quantized = new Uint8Array(data.length + 8); // +8 для scale и zeroPoint

    // Сохраняем scale и zeroPoint в начале
    const view = new DataView(quantized.buffer);
    view.setFloat32(0, scale, true);
    view.setInt32(4, zeroPoint, true);

    for (let i = 0; i < data.length; i++) {
      const q = Math.round(data[i] / scale + zeroPoint);
      quantized[i + 8] = Math.max(0, Math.min(255, q));
    }

    return quantized.buffer;
  }

  /**
   * Деквантизация из INT8
   */
  private dequantizeFromInt8(buffer: ArrayBuffer): Float32Array {
    const view = new DataView(buffer);
    const scale = view.getFloat32(0, true);
    const zeroPoint = view.getInt32(4, true);

    const quantized = new Uint8Array(buffer, 8);
    const result = new Float32Array(quantized.length);

    for (let i = 0; i < quantized.length; i++) {
      result[i] = (quantized[i] - zeroPoint) * scale;
    }

    return result;
  }

  /**
   * Квантизация в INT4 (4 бита на значение)
   */
  private quantizeToInt4(data: Float32Array): ArrayBuffer {
    let min = Infinity;
    let max = -Infinity;
    for (const v of data) {
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const scale = (max - min) / 15; // 4 бита = 16 значений
    const zeroPoint = Math.round(-min / scale);

    // Упаковываем по 2 значения в байт
    const packedLength = Math.ceil(data.length / 2);
    const quantized = new Uint8Array(packedLength + 8); // +8 для scale и zeroPoint

    const view = new DataView(quantized.buffer);
    view.setFloat32(0, scale, true);
    view.setInt32(4, zeroPoint, true);

    for (let i = 0; i < data.length; i += 2) {
      const q1 = Math.max(0, Math.min(15, Math.round(data[i] / scale + zeroPoint)));
      const q2 = i + 1 < data.length
        ? Math.max(0, Math.min(15, Math.round(data[i + 1] / scale + zeroPoint)))
        : 0;
      quantized[Math.floor(i / 2) + 8] = (q1 << 4) | q2;
    }

    return quantized.buffer;
  }

  /**
   * Деквантизация из INT4
   */
  private dequantizeFromInt4(buffer: ArrayBuffer, originalLength: number): Float32Array {
    const view = new DataView(buffer);
    const scale = view.getFloat32(0, true);
    const zeroPoint = view.getInt32(4, true);

    const quantized = new Uint8Array(buffer, 8);
    const result = new Float32Array(originalLength);

    for (let i = 0; i < originalLength; i++) {
      const byteIdx = Math.floor(i / 2);
      const isHigh = i % 2 === 0;
      const q = isHigh ? (quantized[byteIdx] >> 4) : (quantized[byteIdx] & 0x0f);
      result[i] = (q - zeroPoint) * scale;
    }

    return result;
  }

  /**
   * Сериализует оптимизатор
   */
  private serializeOptimizer(optimizer: Optimizer): SerializedOptimizerState {
    const state = optimizer.stateDict();
    return {
      type: optimizer.constructor.name,
      lr: optimizer.lr,
      stepCount: state.get('stepCount') as number || 0,
      paramStates: Object.fromEntries(state),
    };
  }

  /**
   * Сериализует чекпоинт в бинарный формат
   */
  private async serializeCheckpoint(
    checkpoint: Checkpoint,
    compression: CompressionFormat
  ): Promise<Uint8Array> {
    const json = JSON.stringify(checkpoint, (key, value) => {
      if (value instanceof ArrayBuffer) {
        return {
          __type: 'ArrayBuffer',
          data: Array.from(new Uint8Array(value)),
        };
      }
      return value;
    });

    const encoder = new TextEncoder();
    let data = encoder.encode(json);

    // GZIP сжатие
    if (compression === CompressionFormat.GZIP) {
      data = await this.gzipCompress(data);
    }

    return data;
  }

  /**
   * Десериализует чекпоинт из бинарного формата
   */
  private async deserializeCheckpoint(data: Uint8Array): Promise<Checkpoint> {
    // Пытаемся распаковать GZIP
    let decompressed = data;
    try {
      decompressed = await this.gzipDecompress(data);
    } catch {
      // Не сжато GZIP, используем как есть
    }

    const decoder = new TextDecoder();
    const json = decoder.decode(decompressed);

    const checkpoint = JSON.parse(json, (key, value) => {
      if (value && typeof value === 'object' && value.__type === 'ArrayBuffer') {
        return new Uint8Array(value.data).buffer;
      }
      return value;
    }) as Checkpoint;

    return checkpoint;
  }

  /**
   * GZIP сжатие
   */
  private async gzipCompress(data: Uint8Array): Promise<Uint8Array> {
    const stream = new Blob([data]).stream();
    const compressedStream = stream.pipeThrough(new CompressionStream('gzip'));
    const chunks: Uint8Array[] = [];
    const reader = compressedStream.getReader();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }

    const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    return result;
  }

  /**
   * GZIP распаковка
   */
  private async gzipDecompress(data: Uint8Array): Promise<Uint8Array> {
    const stream = new Blob([data]).stream();
    const decompressedStream = stream.pipeThrough(new DecompressionStream('gzip'));
    const chunks: Uint8Array[] = [];
    const reader = decompressedStream.getReader();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }

    const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    return result;
  }

  /**
   * Загружает состояние модели
   */
  private loadModelState(model: Module, state: SerializedState): void {
    const stateDict = new Map<string, Tensor>();

    for (const [name, serialized] of Object.entries(state)) {
      let data: Float32Array;

      if (serialized.compressed && serialized.data instanceof ArrayBuffer) {
        // Деквантизация в зависимости от формата
        const originalSize = serialized.shape.reduce((a, b) => a * b, 1);

        if (serialized.dtype === 'float16') {
          const uint16 = new Uint16Array(serialized.data);
          data = new Float32Array(uint16.length);
          for (let i = 0; i < uint16.length; i++) {
            data[i] = this.float16ToFloat32(uint16[i]);
          }
        } else if (serialized.dtype === 'int8') {
          data = this.dequantizeFromInt8(serialized.data);
        } else if (serialized.dtype === 'int4') {
          data = this.dequantizeFromInt4(serialized.data, originalSize);
        } else {
          data = new Float32Array(serialized.data);
        }
      } else {
        data = new Float32Array(serialized.data as number[]);
      }

      stateDict.set(name, new Tensor(data, serialized.shape));
    }

    model.loadStateDict(stateDict);
  }

  /**
   * Загружает состояние оптимизатора
   */
  private loadOptimizerState(optimizer: Optimizer, state: SerializedOptimizerState): void {
    optimizer.lr = state.lr;
    const stateMap = new Map(Object.entries(state.paramStates));
    optimizer.loadStateDict(stateMap);
  }

  /**
   * Удаляет старые чекпоинты
   */
  private async pruneOldCheckpoints(): Promise<void> {
    while (this.checkpoints.length > this.maxCheckpoints) {
      const oldCheckpoint = this.checkpoints.shift()!;
      try {
        await Bun.file(oldCheckpoint).delete();
      } catch {
        // Игнорируем ошибки удаления
      }
    }
  }

  /**
   * Получает список чекпоинтов
   */
  getCheckpoints(): string[] {
    return [...this.checkpoints];
  }

  /**
   * Получает последний чекпоинт
   */
  getLatestCheckpoint(): string | null {
    return this.checkpoints.length > 0
      ? this.checkpoints[this.checkpoints.length - 1]
      : null;
  }
}

/**
 * Быстрое сохранение модели
 */
export async function saveCheckpoint(
  filepath: string,
  model: Module,
  optimizer?: Optimizer,
  options: SaveOptions = {}
): Promise<void> {
  const manager = new CheckpointManager('.');

  const checkpoint = await (manager as any).createCheckpoint(
    model,
    optimizer,
    options.compression || CompressionFormat.GZIP,
    options.metadata || {}
  );

  const data = await (manager as any).serializeCheckpoint(
    checkpoint,
    options.compression || CompressionFormat.GZIP
  );

  await Bun.write(filepath, data);
}

/**
 * Быстрая загрузка модели
 */
export async function loadCheckpoint(
  filepath: string,
  model: Module,
  optimizer?: Optimizer
): Promise<CheckpointMetadata> {
  const manager = new CheckpointManager('.');
  return manager.load(filepath, model, optimizer);
}

/**
 * Экспорт модели в минимальном формате (только веса, без оптимизатора)
 */
export async function exportModel(
  filepath: string,
  model: Module,
  compression: CompressionFormat = CompressionFormat.FLOAT16
): Promise<void> {
  await saveCheckpoint(filepath, model, undefined, {
    compression,
    includeOptimizer: false,
    metadata: {
      modelConfig: (model as any).config || (model as any).getConfig?.(),
    },
  });
}
