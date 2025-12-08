/**
 * @fileoverview CPU многопоточность через Worker threads
 * @description Параллельные вычисления на CPU с использованием Web Workers / Bun Workers
 */

import { Tensor, zeros } from '../core/tensor';
import { DType } from '../core/dtype';
import { computeSize, computeStrides } from '../core/shape';
import { DeviceManager, DeviceType } from './device';

/**
 * Типы операций для воркеров
 */
export enum WorkerOp {
  ADD = 'add',
  MUL = 'mul',
  MATMUL = 'matmul',
  RELU = 'relu',
  SIGMOID = 'sigmoid',
  TANH = 'tanh',
  EXP = 'exp',
  LOG = 'log',
  SUM = 'sum',
  MEAN = 'mean',
  CONV2D = 'conv2d',
  TRANSPOSE = 'transpose',
  SOFTMAX = 'softmax',
  GELU = 'gelu',
  LAYER_NORM = 'layerNorm',
}

/**
 * Сообщение для воркера
 */
interface WorkerMessage {
  id: number;
  op: WorkerOp;
  data: {
    a?: Float32Array;
    b?: Float32Array;
    shapeA?: number[];
    shapeB?: number[];
    params?: Record<string, unknown>;
    startIdx: number;
    endIdx: number;
  };
}

/**
 * Результат от воркера
 */
interface WorkerResult {
  id: number;
  result: Float32Array;
  error?: string;
}

/**
 * Код воркера (будет выполняться в отдельном потоке)
 */
const workerCode = /* javascript */ `
// Воркер для CPU вычислений

self.onmessage = function(e) {
  const { id, op, data } = e.data;
  const { a, b, shapeA, shapeB, params, startIdx, endIdx } = data;

  try {
    let result;

    switch (op) {
      case 'add':
        result = new Float32Array(endIdx - startIdx);
        for (let i = startIdx; i < endIdx; i++) {
          result[i - startIdx] = a[i] + b[i];
        }
        break;

      case 'mul':
        result = new Float32Array(endIdx - startIdx);
        for (let i = startIdx; i < endIdx; i++) {
          result[i - startIdx] = a[i] * b[i];
        }
        break;

      case 'relu':
        result = new Float32Array(endIdx - startIdx);
        for (let i = startIdx; i < endIdx; i++) {
          result[i - startIdx] = Math.max(0, a[i]);
        }
        break;

      case 'sigmoid':
        result = new Float32Array(endIdx - startIdx);
        for (let i = startIdx; i < endIdx; i++) {
          result[i - startIdx] = 1 / (1 + Math.exp(-a[i]));
        }
        break;

      case 'tanh':
        result = new Float32Array(endIdx - startIdx);
        for (let i = startIdx; i < endIdx; i++) {
          result[i - startIdx] = Math.tanh(a[i]);
        }
        break;

      case 'exp':
        result = new Float32Array(endIdx - startIdx);
        for (let i = startIdx; i < endIdx; i++) {
          result[i - startIdx] = Math.exp(a[i]);
        }
        break;

      case 'log':
        result = new Float32Array(endIdx - startIdx);
        for (let i = startIdx; i < endIdx; i++) {
          result[i - startIdx] = Math.log(a[i]);
        }
        break;

      case 'gelu':
        result = new Float32Array(endIdx - startIdx);
        const SQRT_2_OVER_PI = 0.7978845608;
        for (let i = startIdx; i < endIdx; i++) {
          const x = a[i];
          const inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
          result[i - startIdx] = 0.5 * x * (1 + Math.tanh(inner));
        }
        break;

      case 'matmul':
        // Матричное умножение для части строк
        const M = shapeA[0];
        const K = shapeA[1];
        const N = shapeB[1];

        // startIdx и endIdx здесь - номера строк результата
        const numRows = endIdx - startIdx;
        result = new Float32Array(numRows * N);

        for (let i = 0; i < numRows; i++) {
          const rowIdx = startIdx + i;
          for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let k = 0; k < K; k++) {
              sum += a[rowIdx * K + k] * b[k * N + j];
            }
            result[i * N + j] = sum;
          }
        }
        break;

      case 'sum':
        let sum = 0;
        for (let i = startIdx; i < endIdx; i++) {
          sum += a[i];
        }
        result = new Float32Array([sum]);
        break;

      case 'mean':
        let mean = 0;
        const count = endIdx - startIdx;
        for (let i = startIdx; i < endIdx; i++) {
          mean += a[i];
        }
        result = new Float32Array([mean / count]);
        break;

      case 'softmax':
        // Softmax для части батча
        const batchSize = params.batchSize;
        const seqLen = params.seqLen;
        const numBatches = endIdx - startIdx;
        result = new Float32Array(numBatches * seqLen);

        for (let b = 0; b < numBatches; b++) {
          const batchIdx = startIdx + b;
          const offset = batchIdx * seqLen;

          // Находим max для численной стабильности
          let maxVal = -Infinity;
          for (let i = 0; i < seqLen; i++) {
            if (a[offset + i] > maxVal) maxVal = a[offset + i];
          }

          // Вычисляем exp и сумму
          let sumExp = 0;
          for (let i = 0; i < seqLen; i++) {
            const expVal = Math.exp(a[offset + i] - maxVal);
            result[b * seqLen + i] = expVal;
            sumExp += expVal;
          }

          // Нормализуем
          for (let i = 0; i < seqLen; i++) {
            result[b * seqLen + i] /= sumExp;
          }
        }
        break;

      case 'layerNorm':
        const hiddenSize = params.hiddenSize;
        const eps = params.eps || 1e-5;
        const gamma = params.gamma;
        const beta = params.beta;
        const numSamples = endIdx - startIdx;
        result = new Float32Array(numSamples * hiddenSize);

        for (let s = 0; s < numSamples; s++) {
          const sampleIdx = startIdx + s;
          const offset = sampleIdx * hiddenSize;

          // Среднее
          let mean = 0;
          for (let i = 0; i < hiddenSize; i++) {
            mean += a[offset + i];
          }
          mean /= hiddenSize;

          // Дисперсия
          let variance = 0;
          for (let i = 0; i < hiddenSize; i++) {
            const diff = a[offset + i] - mean;
            variance += diff * diff;
          }
          variance /= hiddenSize;

          // Нормализация
          const invStd = 1 / Math.sqrt(variance + eps);
          for (let i = 0; i < hiddenSize; i++) {
            const normalized = (a[offset + i] - mean) * invStd;
            result[s * hiddenSize + i] = normalized * gamma[i] + beta[i];
          }
        }
        break;

      default:
        throw new Error('Unknown operation: ' + op);
    }

    self.postMessage({ id, result }, [result.buffer]);
  } catch (error) {
    self.postMessage({ id, result: null, error: error.message });
  }
};
`;

/**
 * Пул воркеров для CPU вычислений
 */
export class WorkerPool {
  private workers: Worker[] = [];
  private available: Worker[] = [];
  private pending: Map<number, { resolve: (data: Float32Array) => void; reject: (err: Error) => void }> = new Map();
  private messageId: number = 0;
  private numWorkers: number;

  constructor(numWorkers?: number) {
    this.numWorkers = numWorkers || navigator?.hardwareConcurrency || 4;
  }

  /**
   * Инициализирует пул воркеров
   */
  async initialize(): Promise<void> {
    if (this.workers.length > 0) return;

    // Создаём воркеров из blob URL
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);

    for (let i = 0; i < this.numWorkers; i++) {
      const worker = new Worker(url);

      worker.onmessage = (e: MessageEvent<WorkerResult>) => {
        const { id, result, error } = e.data;
        const pending = this.pending.get(id);

        if (pending) {
          this.pending.delete(id);
          this.available.push(worker);

          if (error) {
            pending.reject(new Error(error));
          } else {
            pending.resolve(result);
          }
        }
      };

      worker.onerror = (e) => {
        console.error('Worker error:', e);
      };

      this.workers.push(worker);
      this.available.push(worker);
    }
  }

  /**
   * Выполняет операцию на воркере
   */
  private async runOnWorker(message: WorkerMessage): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const getWorker = () => {
        if (this.available.length > 0) {
          return this.available.pop()!;
        }
        // Если нет свободных воркеров, используем первый (будет в очереди)
        return this.workers[0];
      };

      const worker = getWorker();
      this.pending.set(message.id, { resolve, reject });

      // Отправляем с transferable для эффективности
      const transferable: ArrayBuffer[] = [];
      if (message.data.a) {
        transferable.push(message.data.a.buffer);
      }
      if (message.data.b) {
        transferable.push(message.data.b.buffer);
      }

      worker.postMessage(message, transferable);
    });
  }

  /**
   * Поэлементное сложение с параллелизацией
   */
  async add(a: Tensor, b: Tensor): Promise<Tensor> {
    const startTime = performance.now();
    const size = a.size;
    const chunkSize = Math.ceil(size / this.numWorkers);

    const promises: Promise<Float32Array>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startIdx = i * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, size);

      if (startIdx >= size) break;

      promises.push(
        this.runOnWorker({
          id: this.messageId++,
          op: WorkerOp.ADD,
          data: {
            a: new Float32Array(a.data),
            b: new Float32Array(b.data),
            startIdx,
            endIdx,
          },
        })
      );
    }

    const results = await Promise.all(promises);
    const resultData = this.mergeResults(results, size);

    const result = new Tensor(resultData, [...a.shape], {
      dtype: a.dtype,
      requiresGrad: a.requiresGrad || b.requiresGrad,
    });

    DeviceManager.getInstance().recordOp(DeviceType.CPU, performance.now() - startTime);
    return result;
  }

  /**
   * Поэлементное умножение с параллелизацией
   */
  async mul(a: Tensor, b: Tensor): Promise<Tensor> {
    const startTime = performance.now();
    const size = a.size;
    const chunkSize = Math.ceil(size / this.numWorkers);

    const promises: Promise<Float32Array>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startIdx = i * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, size);

      if (startIdx >= size) break;

      promises.push(
        this.runOnWorker({
          id: this.messageId++,
          op: WorkerOp.MUL,
          data: {
            a: new Float32Array(a.data),
            b: new Float32Array(b.data),
            startIdx,
            endIdx,
          },
        })
      );
    }

    const results = await Promise.all(promises);
    const resultData = this.mergeResults(results, size);

    const result = new Tensor(resultData, [...a.shape], {
      dtype: a.dtype,
      requiresGrad: a.requiresGrad || b.requiresGrad,
    });

    DeviceManager.getInstance().recordOp(DeviceType.CPU, performance.now() - startTime);
    return result;
  }

  /**
   * Матричное умножение с параллелизацией по строкам
   */
  async matmul(a: Tensor, b: Tensor): Promise<Tensor> {
    const startTime = performance.now();

    if (a.ndim !== 2 || b.ndim !== 2) {
      throw new Error('Parallel matmul supports only 2D matrices');
    }

    const [M, K1] = a.shape;
    const [K2, N] = b.shape;

    if (K1 !== K2) {
      throw new Error(`Matrix dimensions don't match: [${M}, ${K1}] x [${K2}, ${N}]`);
    }

    // Распределяем строки между воркерами
    const rowsPerWorker = Math.ceil(M / this.numWorkers);
    const promises: Promise<Float32Array>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startRow = i * rowsPerWorker;
      const endRow = Math.min(startRow + rowsPerWorker, M);

      if (startRow >= M) break;

      promises.push(
        this.runOnWorker({
          id: this.messageId++,
          op: WorkerOp.MATMUL,
          data: {
            a: new Float32Array(a.data),
            b: new Float32Array(b.data),
            shapeA: [...a.shape],
            shapeB: [...b.shape],
            startIdx: startRow,
            endIdx: endRow,
          },
        })
      );
    }

    const results = await Promise.all(promises);

    // Собираем результат
    const resultData = new Float32Array(M * N);
    let offset = 0;
    for (const chunk of results) {
      resultData.set(chunk, offset);
      offset += chunk.length;
    }

    const result = new Tensor(resultData, [M, N], {
      dtype: a.dtype,
      requiresGrad: a.requiresGrad || b.requiresGrad,
    });

    DeviceManager.getInstance().recordOp(DeviceType.CPU, performance.now() - startTime);
    return result;
  }

  /**
   * ReLU с параллелизацией
   */
  async relu(input: Tensor): Promise<Tensor> {
    return this.unaryOp(input, WorkerOp.RELU);
  }

  /**
   * Sigmoid с параллелизацией
   */
  async sigmoid(input: Tensor): Promise<Tensor> {
    return this.unaryOp(input, WorkerOp.SIGMOID);
  }

  /**
   * Tanh с параллелизацией
   */
  async tanh(input: Tensor): Promise<Tensor> {
    return this.unaryOp(input, WorkerOp.TANH);
  }

  /**
   * Exp с параллелизацией
   */
  async exp(input: Tensor): Promise<Tensor> {
    return this.unaryOp(input, WorkerOp.EXP);
  }

  /**
   * GELU с параллелизацией
   */
  async gelu(input: Tensor): Promise<Tensor> {
    return this.unaryOp(input, WorkerOp.GELU);
  }

  /**
   * Softmax с параллелизацией по батчам
   */
  async softmax(input: Tensor, dim: number = -1): Promise<Tensor> {
    const startTime = performance.now();

    // Для простоты реализуем softmax только по последней размерности
    if (dim === -1) dim = input.ndim - 1;

    if (input.ndim < 2) {
      throw new Error('Softmax requires at least 2D input');
    }

    const batchSize = input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const seqLen = input.shape[input.ndim - 1];

    const batchesPerWorker = Math.ceil(batchSize / this.numWorkers);
    const promises: Promise<Float32Array>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startBatch = i * batchesPerWorker;
      const endBatch = Math.min(startBatch + batchesPerWorker, batchSize);

      if (startBatch >= batchSize) break;

      promises.push(
        this.runOnWorker({
          id: this.messageId++,
          op: WorkerOp.SOFTMAX,
          data: {
            a: new Float32Array(input.data),
            startIdx: startBatch,
            endIdx: endBatch,
            params: { batchSize, seqLen },
          },
        })
      );
    }

    const results = await Promise.all(promises);
    const resultData = this.mergeResults(results, input.size);

    const result = new Tensor(resultData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    DeviceManager.getInstance().recordOp(DeviceType.CPU, performance.now() - startTime);
    return result;
  }

  /**
   * Layer Normalization с параллелизацией
   */
  async layerNorm(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: number = 1e-5
  ): Promise<Tensor> {
    const startTime = performance.now();

    if (input.ndim < 2) {
      throw new Error('LayerNorm requires at least 2D input');
    }

    const batchSize = input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const hiddenSize = input.shape[input.ndim - 1];

    const samplesPerWorker = Math.ceil(batchSize / this.numWorkers);
    const promises: Promise<Float32Array>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startSample = i * samplesPerWorker;
      const endSample = Math.min(startSample + samplesPerWorker, batchSize);

      if (startSample >= batchSize) break;

      promises.push(
        this.runOnWorker({
          id: this.messageId++,
          op: WorkerOp.LAYER_NORM,
          data: {
            a: new Float32Array(input.data),
            startIdx: startSample,
            endIdx: endSample,
            params: {
              hiddenSize,
              eps,
              gamma: new Float32Array(gamma.data),
              beta: new Float32Array(beta.data),
            },
          },
        })
      );
    }

    const results = await Promise.all(promises);
    const resultData = this.mergeResults(results, input.size);

    const result = new Tensor(resultData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    DeviceManager.getInstance().recordOp(DeviceType.CPU, performance.now() - startTime);
    return result;
  }

  /**
   * Унарная операция с параллелизацией
   */
  private async unaryOp(input: Tensor, op: WorkerOp): Promise<Tensor> {
    const startTime = performance.now();
    const size = input.size;
    const chunkSize = Math.ceil(size / this.numWorkers);

    const promises: Promise<Float32Array>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startIdx = i * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, size);

      if (startIdx >= size) break;

      promises.push(
        this.runOnWorker({
          id: this.messageId++,
          op,
          data: {
            a: new Float32Array(input.data),
            startIdx,
            endIdx,
          },
        })
      );
    }

    const results = await Promise.all(promises);
    const resultData = this.mergeResults(results, size);

    const result = new Tensor(resultData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    DeviceManager.getInstance().recordOp(DeviceType.CPU, performance.now() - startTime);
    return result;
  }

  /**
   * Объединяет результаты от воркеров
   */
  private mergeResults(results: Float32Array[], totalSize: number): Float32Array {
    const merged = new Float32Array(totalSize);
    let offset = 0;

    for (const chunk of results) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    return merged;
  }

  /**
   * Сумма с параллельной редукцией
   */
  async sum(input: Tensor): Promise<number> {
    const size = input.size;
    const chunkSize = Math.ceil(size / this.numWorkers);

    const promises: Promise<Float32Array>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startIdx = i * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, size);

      if (startIdx >= size) break;

      promises.push(
        this.runOnWorker({
          id: this.messageId++,
          op: WorkerOp.SUM,
          data: {
            a: new Float32Array(input.data),
            startIdx,
            endIdx,
          },
        })
      );
    }

    const results = await Promise.all(promises);

    // Суммируем частичные результаты
    let total = 0;
    for (const chunk of results) {
      total += chunk[0];
    }

    return total;
  }

  /**
   * Получает количество воркеров
   */
  getNumWorkers(): number {
    return this.numWorkers;
  }

  /**
   * Проверяет инициализацию
   */
  isInitialized(): boolean {
    return this.workers.length > 0;
  }

  /**
   * Завершает работу всех воркеров
   */
  terminate(): void {
    for (const worker of this.workers) {
      worker.terminate();
    }
    this.workers = [];
    this.available = [];
    this.pending.clear();
  }
}

// Singleton пул воркеров
let workerPool: WorkerPool | null = null;

/**
 * Получает или создаёт пул воркеров
 */
export async function getWorkerPool(numWorkers?: number): Promise<WorkerPool> {
  if (!workerPool) {
    workerPool = new WorkerPool(numWorkers);
    await workerPool.initialize();
  }
  return workerPool;
}

/**
 * Завершает пул воркеров
 */
export function terminateWorkerPool(): void {
  if (workerPool) {
    workerPool.terminate();
    workerPool = null;
  }
}
