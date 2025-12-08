/**
 * @fileoverview Модуль вычислений - GPU и CPU многопоточность
 * @description Экспорт всех компонентов вычислительного модуля
 */

export {
  DeviceType,
  DeviceManager,
  createDevice,
  getDevice,
  isWebGPUSupported,
  getCPUCores,
} from './device';
export type { DeviceConfig, GPUInfo, CPUInfo, PerformanceStats } from './device';

export { GPUBackend, createGPUBackend, isGPUBackendAvailable } from './gpu';

export { WorkerPool, WorkerOp, getWorkerPool, terminateWorkerPool } from './cpu-workers';

export {
  HybridEngine,
  getHybridEngine,
  disposeHybridEngine,
  asyncOps,
} from './hybrid';
export type { HybridConfig } from './hybrid';
