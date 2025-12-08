/**
 * @fileoverview Гибридный вычислительный движок GPU + CPU
 * @description Автоматическое распределение вычислений между GPU и CPU
 */

import { Tensor } from '../core/tensor';
import { DeviceManager, DeviceType, DeviceConfig } from './device';
import { GPUBackend, createGPUBackend } from './gpu';
import { WorkerPool, getWorkerPool } from './cpu-workers';

/**
 * Конфигурация гибридных вычислений
 */
export interface HybridConfig {
  /** Минимальный размер тензора для GPU */
  gpuThreshold?: number;
  /** Приоритет GPU (0-1, где 1 = всегда GPU если возможно) */
  gpuPriority?: number;
  /** Включить автобалансировку нагрузки */
  autoBalance?: boolean;
  /** Профилирование для оптимизации */
  profiling?: boolean;
}

/**
 * Статистика операции
 */
interface OpStats {
  gpuTime: number;
  cpuTime: number;
  gpuCount: number;
  cpuCount: number;
}

/**
 * Гибридный вычислительный движок
 * Автоматически выбирает оптимальное устройство для каждой операции
 */
export class HybridEngine {
  private gpuBackend: GPUBackend | null = null;
  private workerPool: WorkerPool | null = null;
  private deviceManager: DeviceManager;
  private config: HybridConfig;
  private initialized: boolean = false;

  // Статистика для автобалансировки
  private opStats: Map<string, OpStats> = new Map();

  constructor(config: HybridConfig = {}) {
    this.config = {
      gpuThreshold: 1024,
      gpuPriority: 0.7,
      autoBalance: true,
      profiling: false,
      ...config,
    };
    this.deviceManager = DeviceManager.getInstance();
  }

  /**
   * Инициализирует движок
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    await this.deviceManager.initialize();

    // Инициализируем GPU backend если доступен
    if (this.deviceManager.isGPUAvailable()) {
      this.gpuBackend = createGPUBackend();
    }

    // Инициализируем пул воркеров
    try {
      this.workerPool = await getWorkerPool();
    } catch (e) {
      console.warn('Worker pool initialization failed, using single-threaded CPU');
    }

    this.initialized = true;
  }

  /**
   * Выбирает оптимальное устройство для операции
   */
  private selectDevice(opName: string, tensorSize: number): DeviceType {
    const deviceConfig = this.deviceManager.getConfig();

    // Если явно указан CPU
    if (deviceConfig.type === DeviceType.CPU) {
      return DeviceType.CPU;
    }

    // Если GPU недоступен
    if (!this.gpuBackend) {
      return DeviceType.CPU;
    }

    // Автобалансировка на основе статистики
    if (this.config.autoBalance && this.opStats.has(opName)) {
      const stats = this.opStats.get(opName)!;
      if (stats.gpuCount > 5 && stats.cpuCount > 5) {
        const avgGpuTime = stats.gpuTime / stats.gpuCount;
        const avgCpuTime = stats.cpuTime / stats.cpuCount;

        // Выбираем более быстрое устройство
        if (avgGpuTime < avgCpuTime * 0.8) {
          return DeviceType.GPU;
        } else if (avgCpuTime < avgGpuTime * 0.8) {
          return DeviceType.CPU;
        }
      }
    }

    // Используем порог размера и приоритет GPU
    if (tensorSize >= this.config.gpuThreshold!) {
      return Math.random() < this.config.gpuPriority! ? DeviceType.GPU : DeviceType.CPU;
    }

    // Для маленьких тензоров предпочитаем CPU (меньше overhead)
    return DeviceType.CPU;
  }

  /**
   * Записывает статистику операции
   */
  private recordStats(opName: string, device: DeviceType, timeMs: number): void {
    if (!this.config.profiling && !this.config.autoBalance) return;

    if (!this.opStats.has(opName)) {
      this.opStats.set(opName, { gpuTime: 0, cpuTime: 0, gpuCount: 0, cpuCount: 0 });
    }

    const stats = this.opStats.get(opName)!;
    if (device === DeviceType.GPU) {
      stats.gpuTime += timeMs;
      stats.gpuCount++;
    } else {
      stats.cpuTime += timeMs;
      stats.cpuCount++;
    }
  }

  /**
   * Поэлементное сложение
   */
  async add(a: Tensor, b: Tensor): Promise<Tensor> {
    const device = this.selectDevice('add', a.size);
    const startTime = performance.now();

    let result: Tensor;

    if (device === DeviceType.GPU && this.gpuBackend) {
      result = await this.gpuBackend.add(a, b);
    } else if (this.workerPool) {
      result = await this.workerPool.add(a, b);
    } else {
      // Fallback на синхронный CPU
      result = a.add(b);
    }

    this.recordStats('add', device, performance.now() - startTime);
    return result;
  }

  /**
   * Поэлементное умножение
   */
  async mul(a: Tensor, b: Tensor): Promise<Tensor> {
    const device = this.selectDevice('mul', a.size);
    const startTime = performance.now();

    let result: Tensor;

    if (device === DeviceType.GPU && this.gpuBackend) {
      result = await this.gpuBackend.mul(a, b);
    } else if (this.workerPool) {
      result = await this.workerPool.mul(a, b);
    } else {
      result = a.mul(b);
    }

    this.recordStats('mul', device, performance.now() - startTime);
    return result;
  }

  /**
   * Матричное умножение
   */
  async matmul(a: Tensor, b: Tensor): Promise<Tensor> {
    const device = this.selectDevice('matmul', a.size * b.size);
    const startTime = performance.now();

    let result: Tensor;

    if (device === DeviceType.GPU && this.gpuBackend && a.ndim === 2 && b.ndim === 2) {
      result = await this.gpuBackend.matmul(a, b);
    } else if (this.workerPool && a.ndim === 2 && b.ndim === 2) {
      result = await this.workerPool.matmul(a, b);
    } else {
      result = a.matmul(b);
    }

    this.recordStats('matmul', device, performance.now() - startTime);
    return result;
  }

  /**
   * ReLU активация
   */
  async relu(input: Tensor): Promise<Tensor> {
    const device = this.selectDevice('relu', input.size);
    const startTime = performance.now();

    let result: Tensor;

    if (device === DeviceType.GPU && this.gpuBackend) {
      result = await this.gpuBackend.relu(input);
    } else if (this.workerPool) {
      result = await this.workerPool.relu(input);
    } else {
      // Синхронная реализация
      const data = input.data.map((v) => Math.max(0, v));
      result = new Tensor(data as unknown as number[], [...input.shape], {
        dtype: input.dtype,
        requiresGrad: input.requiresGrad,
      });
    }

    this.recordStats('relu', device, performance.now() - startTime);
    return result;
  }

  /**
   * GELU активация
   */
  async gelu(input: Tensor): Promise<Tensor> {
    const device = this.selectDevice('gelu', input.size);
    const startTime = performance.now();

    let result: Tensor;

    if (device === DeviceType.GPU && this.gpuBackend) {
      result = await this.gpuBackend.gelu(input);
    } else if (this.workerPool) {
      result = await this.workerPool.gelu(input);
    } else {
      const SQRT_2_OVER_PI = 0.7978845608;
      const data = input.data.map((x) => {
        const inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1 + Math.tanh(inner));
      });
      result = new Tensor(data as unknown as number[], [...input.shape], {
        dtype: input.dtype,
        requiresGrad: input.requiresGrad,
      });
    }

    this.recordStats('gelu', device, performance.now() - startTime);
    return result;
  }

  /**
   * Sigmoid активация
   */
  async sigmoid(input: Tensor): Promise<Tensor> {
    const device = this.selectDevice('sigmoid', input.size);
    const startTime = performance.now();

    let result: Tensor;

    if (device === DeviceType.GPU && this.gpuBackend) {
      result = await this.gpuBackend.sigmoid(input);
    } else if (this.workerPool) {
      result = await this.workerPool.sigmoid(input);
    } else {
      const data = input.data.map((x) => 1 / (1 + Math.exp(-x)));
      result = new Tensor(data as unknown as number[], [...input.shape], {
        dtype: input.dtype,
        requiresGrad: input.requiresGrad,
      });
    }

    this.recordStats('sigmoid', device, performance.now() - startTime);
    return result;
  }

  /**
   * Softmax
   */
  async softmax(input: Tensor, dim: number = -1): Promise<Tensor> {
    const device = this.selectDevice('softmax', input.size);
    const startTime = performance.now();

    let result: Tensor;

    if (this.workerPool && input.ndim >= 2) {
      result = await this.workerPool.softmax(input, dim);
    } else {
      // Синхронная реализация
      if (dim === -1) dim = input.ndim - 1;

      // Для простоты - softmax по последнему измерению
      const batchSize = input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
      const seqLen = input.shape[input.ndim - 1];

      const resultData = new Float32Array(input.size);

      for (let b = 0; b < batchSize; b++) {
        const offset = b * seqLen;

        let maxVal = -Infinity;
        for (let i = 0; i < seqLen; i++) {
          if (input.data[offset + i] > maxVal) maxVal = input.data[offset + i];
        }

        let sumExp = 0;
        for (let i = 0; i < seqLen; i++) {
          resultData[offset + i] = Math.exp(input.data[offset + i] - maxVal);
          sumExp += resultData[offset + i];
        }

        for (let i = 0; i < seqLen; i++) {
          resultData[offset + i] /= sumExp;
        }
      }

      result = new Tensor(resultData, [...input.shape], {
        dtype: input.dtype,
        requiresGrad: input.requiresGrad,
      });
    }

    this.recordStats('softmax', device, performance.now() - startTime);
    return result;
  }

  /**
   * Layer Normalization
   */
  async layerNorm(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: number = 1e-5
  ): Promise<Tensor> {
    const device = this.selectDevice('layerNorm', input.size);
    const startTime = performance.now();

    let result: Tensor;

    if (this.workerPool && input.ndim >= 2) {
      result = await this.workerPool.layerNorm(input, gamma, beta, eps);
    } else {
      // Синхронная реализация
      const batchSize = input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
      const hiddenSize = input.shape[input.ndim - 1];

      const resultData = new Float32Array(input.size);

      for (let b = 0; b < batchSize; b++) {
        const offset = b * hiddenSize;

        // Среднее
        let mean = 0;
        for (let i = 0; i < hiddenSize; i++) {
          mean += input.data[offset + i];
        }
        mean /= hiddenSize;

        // Дисперсия
        let variance = 0;
        for (let i = 0; i < hiddenSize; i++) {
          const diff = input.data[offset + i] - mean;
          variance += diff * diff;
        }
        variance /= hiddenSize;

        // Нормализация
        const invStd = 1 / Math.sqrt(variance + eps);
        for (let i = 0; i < hiddenSize; i++) {
          const normalized = (input.data[offset + i] - mean) * invStd;
          resultData[offset + i] = normalized * gamma.data[i] + beta.data[i];
        }
      }

      result = new Tensor(resultData, [...input.shape], {
        dtype: input.dtype,
        requiresGrad: input.requiresGrad,
      });
    }

    this.recordStats('layerNorm', device, performance.now() - startTime);
    return result;
  }

  /**
   * Экспонента
   */
  async exp(input: Tensor): Promise<Tensor> {
    const device = this.selectDevice('exp', input.size);
    const startTime = performance.now();

    let result: Tensor;

    if (device === DeviceType.GPU && this.gpuBackend) {
      result = await this.gpuBackend.exp(input);
    } else if (this.workerPool) {
      result = await this.workerPool.exp(input);
    } else {
      result = input.exp();
    }

    this.recordStats('exp', device, performance.now() - startTime);
    return result;
  }

  /**
   * Получает статистику операций
   */
  getOpStats(): Map<string, OpStats> {
    return new Map(this.opStats);
  }

  /**
   * Сбрасывает статистику
   */
  resetStats(): void {
    this.opStats.clear();
  }

  /**
   * Получает информацию о доступных устройствах
   */
  getDeviceInfo(): { gpu: boolean; cpuWorkers: number } {
    return {
      gpu: this.gpuBackend !== null,
      cpuWorkers: this.workerPool?.getNumWorkers() || 1,
    };
  }

  /**
   * Проверяет инициализацию
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Освобождает ресурсы
   */
  dispose(): void {
    if (this.gpuBackend) {
      this.gpuBackend.clearCache();
    }
    this.workerPool = null;
    this.gpuBackend = null;
    this.initialized = false;
    this.opStats.clear();
  }
}

// Singleton движок
let hybridEngine: HybridEngine | null = null;

/**
 * Получает или создаёт гибридный движок
 */
export async function getHybridEngine(config?: HybridConfig): Promise<HybridEngine> {
  if (!hybridEngine) {
    hybridEngine = new HybridEngine(config);
    await hybridEngine.initialize();
  }
  return hybridEngine;
}

/**
 * Освобождает гибридный движок
 */
export function disposeHybridEngine(): void {
  if (hybridEngine) {
    hybridEngine.dispose();
    hybridEngine = null;
  }
}

/**
 * Вспомогательные функции для асинхронных операций
 */
export const asyncOps = {
  /**
   * Асинхронное сложение
   */
  async add(a: Tensor, b: Tensor): Promise<Tensor> {
    const engine = await getHybridEngine();
    return engine.add(a, b);
  },

  /**
   * Асинхронное умножение
   */
  async mul(a: Tensor, b: Tensor): Promise<Tensor> {
    const engine = await getHybridEngine();
    return engine.mul(a, b);
  },

  /**
   * Асинхронное матричное умножение
   */
  async matmul(a: Tensor, b: Tensor): Promise<Tensor> {
    const engine = await getHybridEngine();
    return engine.matmul(a, b);
  },

  /**
   * Асинхронный ReLU
   */
  async relu(input: Tensor): Promise<Tensor> {
    const engine = await getHybridEngine();
    return engine.relu(input);
  },

  /**
   * Асинхронный GELU
   */
  async gelu(input: Tensor): Promise<Tensor> {
    const engine = await getHybridEngine();
    return engine.gelu(input);
  },

  /**
   * Асинхронный Sigmoid
   */
  async sigmoid(input: Tensor): Promise<Tensor> {
    const engine = await getHybridEngine();
    return engine.sigmoid(input);
  },

  /**
   * Асинхронный Softmax
   */
  async softmax(input: Tensor, dim?: number): Promise<Tensor> {
    const engine = await getHybridEngine();
    return engine.softmax(input, dim);
  },

  /**
   * Асинхронный LayerNorm
   */
  async layerNorm(input: Tensor, gamma: Tensor, beta: Tensor, eps?: number): Promise<Tensor> {
    const engine = await getHybridEngine();
    return engine.layerNorm(input, gamma, beta, eps);
  },
};
