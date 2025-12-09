/**
 * @fileoverview Менеджер вычислительных устройств
 * @description Унифицированный интерфейс для GPU и CPU вычислений
 * @supports Browser WebGPU, Node.js WebGPU (via webgpu package)
 */

import { Tensor } from '../core/tensor';

// Environment variable to disable GPU
const DISABLE_GPU = process.env.BRAINY_DISABLE_GPU === '1' || process.env.BRAINY_DISABLE_GPU === 'true';

// WebGPU provider interface
interface WebGPUProvider {
  type: 'bun-webgpu' | 'webgpu' | 'browser';
  gpu: GPU;
}

// Try to import WebGPU (Bun or Node.js)
let webgpuProvider: WebGPUProvider | null = null;

if (!DISABLE_GPU) {
  // Try bun-webgpu first (for Bun runtime)
  try {
    const bunWebGPU = require('bun-webgpu');
    if (bunWebGPU && bunWebGPU.gpu) {
      webgpuProvider = { type: 'bun-webgpu', gpu: bunWebGPU.gpu };
    }
  } catch {
    // Not available
  }

  // Try Node.js webgpu (dawn) as fallback
  if (!webgpuProvider) {
    try {
      const nodeWebGPU = require('webgpu');
      if (nodeWebGPU && nodeWebGPU.create) {
        // Apply globals
        if (nodeWebGPU.globals) {
          Object.assign(globalThis, nodeWebGPU.globals);
        }
        const gpu = nodeWebGPU.create([]);
        webgpuProvider = { type: 'webgpu', gpu };
      }
    } catch {
      // Not available
    }
  }
}

/**
 * Типы вычислительных устройств
 */
export enum DeviceType {
  CPU = 'cpu',
  GPU = 'gpu',
  HYBRID = 'hybrid', // Совмещение GPU и CPU
}

/**
 * Конфигурация устройства
 */
export interface DeviceConfig {
  /** Тип устройства */
  type: DeviceType;
  /** Количество потоков для CPU (по умолчанию: количество ядер) */
  numThreads?: number;
  /** Предпочитать GPU для операций больше этого размера */
  gpuThreshold?: number;
  /** Включить профилирование */
  profiling?: boolean;
}

/**
 * Информация о GPU
 */
export interface GPUInfo {
  available: boolean;
  name?: string;
  vendor?: string;
  maxBufferSize?: number;
  maxComputeWorkgroups?: [number, number, number];
}

/**
 * Информация о CPU
 */
export interface CPUInfo {
  cores: number;
  threads: number;
  architecture: string;
}

/**
 * Статистика производительности
 */
export interface PerformanceStats {
  totalOps: number;
  gpuOps: number;
  cpuOps: number;
  totalTimeMs: number;
  gpuTimeMs: number;
  cpuTimeMs: number;
}

/**
 * Менеджер вычислительных устройств
 * Координирует работу GPU и CPU для оптимального распределения нагрузки
 */
export class DeviceManager {
  private static instance: DeviceManager | null = null;

  private config: DeviceConfig;
  private gpuDevice: GPUDevice | null = null;
  private gpuAdapter: GPUAdapter | null = null;
  private gpuAvailable: boolean = false;
  private initialized: boolean = false;

  private stats: PerformanceStats = {
    totalOps: 0,
    gpuOps: 0,
    cpuOps: 0,
    totalTimeMs: 0,
    gpuTimeMs: 0,
    cpuTimeMs: 0,
  };

  private constructor(config: DeviceConfig) {
    this.config = {
      numThreads: navigator?.hardwareConcurrency || 4,
      gpuThreshold: 1024, // Элементов
      profiling: false,
      ...config,
    };
  }

  /**
   * Получает singleton экземпляр DeviceManager
   */
  static getInstance(config?: DeviceConfig): DeviceManager {
    if (!DeviceManager.instance) {
      DeviceManager.instance = new DeviceManager(config || { type: DeviceType.CPU });
    }
    return DeviceManager.instance;
  }

  /**
   * Сбрасывает singleton (для тестов)
   */
  static reset(): void {
    if (DeviceManager.instance) {
      DeviceManager.instance.dispose();
      DeviceManager.instance = null;
    }
  }

  /**
   * Инициализирует устройства
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Пытаемся инициализировать GPU
    if (this.config.type === DeviceType.GPU || this.config.type === DeviceType.HYBRID) {
      try {
        await this.initGPU();
      } catch (error) {
        console.warn('GPU initialization failed:', error);
        if (this.config.type === DeviceType.GPU) {
          throw new Error('GPU requested but not available');
        }
      }
    }

    this.initialized = true;
  }

  /**
   * Инициализация WebGPU
   * Supports: bun-webgpu (Bun), webgpu/dawn (Node.js), browser WebGPU
   */
  private async initGPU(): Promise<void> {
    let gpu: GPU | null = null;
    let providerName = 'unknown';

    // Use pre-initialized provider if available
    if (webgpuProvider) {
      gpu = webgpuProvider.gpu;
      providerName = webgpuProvider.type;
    }

    // Fallback to browser WebGPU
    if (!gpu && typeof navigator !== 'undefined' && 'gpu' in navigator) {
      gpu = (navigator as Navigator & { gpu: GPU }).gpu;
      providerName = 'browser';
    }

    if (!gpu) {
      throw new Error(
        'WebGPU not supported.\n' +
        '  For Bun: npm install bun-webgpu\n' +
        '  For Node.js: npm install webgpu\n' +
        '  For Browser: Use Chrome/Edge with WebGPU enabled'
      );
    }

    console.log(`Using WebGPU provider: ${providerName}`);

    this.gpuAdapter = await gpu.requestAdapter({
      powerPreference: 'high-performance',
    });

    if (!this.gpuAdapter) {
      throw new Error('No GPU adapter found. Make sure you have a compatible GPU and drivers.');
    }

    // Log GPU info
    const adapterInfo = this.gpuAdapter.info;
    if (adapterInfo) {
      console.log(`GPU: ${adapterInfo.device || 'Unknown'} (${adapterInfo.vendor || 'Unknown vendor'})`);
    }

    this.gpuDevice = await this.gpuAdapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: this.gpuAdapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension: this.gpuAdapter.limits.maxComputeWorkgroupsPerDimension,
      },
    });

    this.gpuAvailable = true;
    console.log('GPU initialized successfully');

    // Обработка потери устройства
    this.gpuDevice.lost.then((info) => {
      console.error('GPU device lost:', info.message);
      this.gpuAvailable = false;
      this.gpuDevice = null;
    });
  }

  /**
   * Получает информацию о GPU
   */
  getGPUInfo(): GPUInfo {
    if (!this.gpuAdapter || !this.gpuDevice) {
      return { available: false };
    }

    return {
      available: true,
      name: this.gpuAdapter.info?.device || 'Unknown',
      vendor: this.gpuAdapter.info?.vendor || 'Unknown',
      maxBufferSize: this.gpuAdapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroups: [
        this.gpuAdapter.limits.maxComputeWorkgroupsPerDimension,
        this.gpuAdapter.limits.maxComputeWorkgroupsPerDimension,
        this.gpuAdapter.limits.maxComputeWorkgroupsPerDimension,
      ],
    };
  }

  /**
   * Получает информацию о CPU
   */
  getCPUInfo(): CPUInfo {
    const cores = navigator?.hardwareConcurrency || 4;
    return {
      cores,
      threads: this.config.numThreads || cores,
      architecture: typeof process !== 'undefined' ? process.arch : 'unknown',
    };
  }

  /**
   * Проверяет доступность GPU
   */
  isGPUAvailable(): boolean {
    return this.gpuAvailable;
  }

  /**
   * Получает GPU устройство
   */
  getGPUDevice(): GPUDevice | null {
    return this.gpuDevice;
  }

  /**
   * Определяет лучшее устройство для операции
   */
  selectDevice(tensorSize: number): DeviceType {
    if (this.config.type === DeviceType.CPU) {
      return DeviceType.CPU;
    }

    if (this.config.type === DeviceType.GPU && this.gpuAvailable) {
      return DeviceType.GPU;
    }

    // HYBRID режим - автоматический выбор
    if (this.config.type === DeviceType.HYBRID) {
      if (this.gpuAvailable && tensorSize >= (this.config.gpuThreshold || 1024)) {
        return DeviceType.GPU;
      }
      return DeviceType.CPU;
    }

    return DeviceType.CPU;
  }

  /**
   * Получает конфигурацию
   */
  getConfig(): DeviceConfig {
    return { ...this.config };
  }

  /**
   * Обновляет конфигурацию
   */
  setConfig(config: Partial<DeviceConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Получает статистику производительности
   */
  getStats(): PerformanceStats {
    return { ...this.stats };
  }

  /**
   * Сбрасывает статистику
   */
  resetStats(): void {
    this.stats = {
      totalOps: 0,
      gpuOps: 0,
      cpuOps: 0,
      totalTimeMs: 0,
      gpuTimeMs: 0,
      cpuTimeMs: 0,
    };
  }

  /**
   * Записывает статистику операции
   */
  recordOp(device: DeviceType, timeMs: number): void {
    this.stats.totalOps++;
    this.stats.totalTimeMs += timeMs;

    if (device === DeviceType.GPU) {
      this.stats.gpuOps++;
      this.stats.gpuTimeMs += timeMs;
    } else {
      this.stats.cpuOps++;
      this.stats.cpuTimeMs += timeMs;
    }
  }

  /**
   * Освобождает ресурсы
   */
  dispose(): void {
    if (this.gpuDevice) {
      this.gpuDevice.destroy();
      this.gpuDevice = null;
    }
    this.gpuAdapter = null;
    this.gpuAvailable = false;
    this.initialized = false;
  }
}

/**
 * Создаёт контекст устройства
 */
export function createDevice(config: DeviceConfig): Promise<DeviceManager> {
  const manager = DeviceManager.getInstance(config);
  return manager.initialize().then(() => manager);
}

/**
 * Получает текущее устройство
 */
export function getDevice(): DeviceManager {
  return DeviceManager.getInstance();
}

/**
 * Проверяет поддержку WebGPU (Bun, Node.js, or browser)
 */
export function isWebGPUSupported(): boolean {
  // Check if GPU is disabled
  if (DISABLE_GPU) {
    return false;
  }
  // Check if we have a WebGPU provider
  if (webgpuProvider) {
    return true;
  }
  // Check browser WebGPU
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

/**
 * Получает информацию о WebGPU провайдере
 */
export function getWebGPUProviderInfo(): { available: boolean; type: string } {
  if (DISABLE_GPU) {
    return { available: false, type: 'disabled' };
  }
  if (webgpuProvider) {
    return { available: true, type: webgpuProvider.type };
  }
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    return { available: true, type: 'browser' };
  }
  return { available: false, type: 'none' };
}

/**
 * Получает количество CPU ядер
 */
export function getCPUCores(): number {
  return navigator?.hardwareConcurrency || 4;
}
