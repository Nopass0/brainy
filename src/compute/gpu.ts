/**
 * @fileoverview WebGPU backend для тензорных операций
 * @description Ускорение вычислений с помощью GPU через WebGPU API
 */

import { Tensor, zeros } from '../core/tensor';
import { DType } from '../core/dtype';
import { computeSize, computeStrides } from '../core/shape';
import { DeviceManager, DeviceType } from './device';

/**
 * Шейдеры для GPU операций (WGSL)
 */
const SHADERS = {
  // Поэлементное сложение
  add: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> result: array<f32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      if (idx < arrayLength(&result)) {
        result[idx] = a[idx] + b[idx];
      }
    }
  `,

  // Поэлементное умножение
  mul: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> result: array<f32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      if (idx < arrayLength(&result)) {
        result[idx] = a[idx] * b[idx];
      }
    }
  `,

  // Матричное умножение (тайловое для производительности)
  matmul: /* wgsl */ `
    struct Dimensions {
      M: u32,
      N: u32,
      K: u32,
      _padding: u32,
    }

    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> result: array<f32>;
    @group(0) @binding(3) var<uniform> dims: Dimensions;

    const TILE_SIZE: u32 = 16u;

    var<workgroup> tileA: array<array<f32, 16>, 16>;
    var<workgroup> tileB: array<array<f32, 16>, 16>;

    @compute @workgroup_size(16, 16)
    fn main(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) group_id: vec3<u32>
    ) {
      let row = global_id.y;
      let col = global_id.x;
      let localRow = local_id.y;
      let localCol = local_id.x;

      var sum: f32 = 0.0;
      let numTiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

      for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        // Загружаем тайл A
        let aRow = row;
        let aCol = t * TILE_SIZE + localCol;
        if (aRow < dims.M && aCol < dims.K) {
          tileA[localRow][localCol] = a[aRow * dims.K + aCol];
        } else {
          tileA[localRow][localCol] = 0.0;
        }

        // Загружаем тайл B
        let bRow = t * TILE_SIZE + localRow;
        let bCol = col;
        if (bRow < dims.K && bCol < dims.N) {
          tileB[localRow][localCol] = b[bRow * dims.N + bCol];
        } else {
          tileB[localRow][localCol] = 0.0;
        }

        workgroupBarrier();

        // Вычисляем произведение
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
          sum = sum + tileA[localRow][k] * tileB[k][localCol];
        }

        workgroupBarrier();
      }

      if (row < dims.M && col < dims.N) {
        result[row * dims.N + col] = sum;
      }
    }
  `,

  // ReLU активация
  relu: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      if (idx < arrayLength(&output)) {
        output[idx] = max(input[idx], 0.0);
      }
    }
  `,

  // Sigmoid активация
  sigmoid: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      if (idx < arrayLength(&output)) {
        output[idx] = 1.0 / (1.0 + exp(-input[idx]));
      }
    }
  `,

  // Softmax (требует два прохода)
  softmaxMax: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> maxVals: array<f32>;

    struct Dims {
      batchSize: u32,
      seqLen: u32,
    }
    @group(0) @binding(2) var<uniform> dims: Dims;

    var<workgroup> sharedMax: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
      let batch = global_id.y;
      let tid = local_id.x;

      var localMax: f32 = -3.402823e+38;

      let start = batch * dims.seqLen;
      for (var i = tid; i < dims.seqLen; i = i + 256u) {
        localMax = max(localMax, input[start + i]);
      }

      sharedMax[tid] = localMax;
      workgroupBarrier();

      // Редукция
      for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + s]);
        }
        workgroupBarrier();
      }

      if (tid == 0u) {
        maxVals[batch] = sharedMax[0];
      }
    }
  `,

  softmaxExp: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> maxVals: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    @group(0) @binding(3) var<storage, read_write> sumExp: array<f32>;

    struct Dims {
      batchSize: u32,
      seqLen: u32,
    }
    @group(0) @binding(4) var<uniform> dims: Dims;

    var<workgroup> sharedSum: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
      let batch = global_id.y;
      let tid = local_id.x;
      let maxVal = maxVals[batch];

      var localSum: f32 = 0.0;
      let start = batch * dims.seqLen;

      for (var i = tid; i < dims.seqLen; i = i + 256u) {
        let expVal = exp(input[start + i] - maxVal);
        output[start + i] = expVal;
        localSum = localSum + expVal;
      }

      sharedSum[tid] = localSum;
      workgroupBarrier();

      for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          sharedSum[tid] = sharedSum[tid] + sharedSum[tid + s];
        }
        workgroupBarrier();
      }

      if (tid == 0u) {
        sumExp[batch] = sharedSum[0];
      }
    }
  `,

  softmaxNorm: /* wgsl */ `
    @group(0) @binding(0) var<storage, read_write> output: array<f32>;
    @group(0) @binding(1) var<storage, read> sumExp: array<f32>;

    struct Dims {
      batchSize: u32,
      seqLen: u32,
    }
    @group(0) @binding(2) var<uniform> dims: Dims;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let batch = global_id.y;
      let idx = global_id.x;

      if (idx < dims.seqLen) {
        let pos = batch * dims.seqLen + idx;
        output[pos] = output[pos] / sumExp[batch];
      }
    }
  `,

  // Экспонента
  exp: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      if (idx < arrayLength(&output)) {
        output[idx] = exp(input[idx]);
      }
    }
  `,

  // Транспонирование
  transpose: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    struct Dims {
      rows: u32,
      cols: u32,
    }
    @group(0) @binding(2) var<uniform> dims: Dims;

    const TILE_SIZE: u32 = 16u;
    var<workgroup> tile: array<array<f32, 17>, 16>; // +1 для избежания bank conflicts

    @compute @workgroup_size(16, 16)
    fn main(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) group_id: vec3<u32>
    ) {
      let x = group_id.x * TILE_SIZE + local_id.x;
      let y = group_id.y * TILE_SIZE + local_id.y;

      if (x < dims.cols && y < dims.rows) {
        tile[local_id.y][local_id.x] = input[y * dims.cols + x];
      }

      workgroupBarrier();

      let newX = group_id.y * TILE_SIZE + local_id.x;
      let newY = group_id.x * TILE_SIZE + local_id.y;

      if (newX < dims.rows && newY < dims.cols) {
        output[newY * dims.rows + newX] = tile[local_id.x][local_id.y];
      }
    }
  `,

  // Редукция суммы
  reduceSum: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    struct Params {
      inputSize: u32,
      reduceSize: u32,
      outputSize: u32,
      _padding: u32,
    }
    @group(0) @binding(2) var<uniform> params: Params;

    var<workgroup> sharedData: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) group_id: vec3<u32>
    ) {
      let tid = local_id.x;
      let outputIdx = group_id.x;

      var sum: f32 = 0.0;
      let start = outputIdx * params.reduceSize;

      for (var i = tid; i < params.reduceSize; i = i + 256u) {
        if (start + i < params.inputSize) {
          sum = sum + input[start + i];
        }
      }

      sharedData[tid] = sum;
      workgroupBarrier();

      for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          sharedData[tid] = sharedData[tid] + sharedData[tid + s];
        }
        workgroupBarrier();
      }

      if (tid == 0u) {
        output[outputIdx] = sharedData[0];
      }
    }
  `,

  // GELU активация
  gelu: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    const SQRT_2_OVER_PI: f32 = 0.7978845608;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      if (idx < arrayLength(&output)) {
        let x = input[idx];
        // Приближение GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
        output[idx] = 0.5 * x * (1.0 + tanh(inner));
      }
    }
  `,

  // Layer Normalization
  layerNorm: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> gamma: array<f32>;
    @group(0) @binding(2) var<storage, read> beta: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;

    struct Params {
      batchSize: u32,
      hiddenSize: u32,
      eps: f32,
      _padding: u32,
    }
    @group(0) @binding(4) var<uniform> params: Params;

    var<workgroup> sharedMean: array<f32, 256>;
    var<workgroup> sharedVar: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
      let batch = global_id.y;
      let tid = local_id.x;
      let start = batch * params.hiddenSize;

      // Вычисляем среднее
      var localSum: f32 = 0.0;
      for (var i = tid; i < params.hiddenSize; i = i + 256u) {
        localSum = localSum + input[start + i];
      }
      sharedMean[tid] = localSum;
      workgroupBarrier();

      for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          sharedMean[tid] = sharedMean[tid] + sharedMean[tid + s];
        }
        workgroupBarrier();
      }

      let mean = sharedMean[0] / f32(params.hiddenSize);
      workgroupBarrier();

      // Вычисляем дисперсию
      var localVar: f32 = 0.0;
      for (var i = tid; i < params.hiddenSize; i = i + 256u) {
        let diff = input[start + i] - mean;
        localVar = localVar + diff * diff;
      }
      sharedVar[tid] = localVar;
      workgroupBarrier();

      for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          sharedVar[tid] = sharedVar[tid] + sharedVar[tid + s];
        }
        workgroupBarrier();
      }

      let variance = sharedVar[0] / f32(params.hiddenSize);
      let invStd = 1.0 / sqrt(variance + params.eps);
      workgroupBarrier();

      // Нормализация
      for (var i = tid; i < params.hiddenSize; i = i + 256u) {
        let normalized = (input[start + i] - mean) * invStd;
        output[start + i] = normalized * gamma[i] + beta[i];
      }
    }
  `,
};

/**
 * Кеш скомпилированных pipeline'ов
 */
const pipelineCache = new Map<string, GPUComputePipeline>();

/**
 * Создаёт compute pipeline для шейдера
 */
async function createPipeline(device: GPUDevice, shaderCode: string, label: string): Promise<GPUComputePipeline> {
  const cacheKey = label;

  if (pipelineCache.has(cacheKey)) {
    return pipelineCache.get(cacheKey)!;
  }

  const shaderModule = device.createShaderModule({
    label: `${label}_shader`,
    code: shaderCode,
  });

  const pipeline = await device.createComputePipelineAsync({
    label: `${label}_pipeline`,
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });

  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

/**
 * Safely destroy a GPU buffer (workaround for bun-webgpu bug)
 */
function safeDestroyBuffer(buffer: GPUBuffer): void {
  try {
    buffer.destroy();
  } catch {
    // Ignore destroy errors - bun-webgpu has a bug with buffer lifecycle
  }
}

/**
 * Создаёт GPU буфер из тензора
 */
function createBuffer(device: GPUDevice, tensor: Tensor, usage: GPUBufferUsageFlags): GPUBuffer {
  const buffer = device.createBuffer({
    size: tensor.data.byteLength,
    usage,
    mappedAtCreation: true,
  });

  new Float32Array(buffer.getMappedRange()).set(tensor.data as Float32Array);
  buffer.unmap();

  return buffer;
}

/**
 * Читает данные из GPU буфера
 */
async function readBuffer(device: GPUDevice, buffer: GPUBuffer, size: number): Promise<Float32Array> {
  const stagingBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
  device.queue.submit([commandEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
  stagingBuffer.unmap();
  safeDestroyBuffer(stagingBuffer);

  return data;
}

/**
 * GPU Backend класс
 */
export class GPUBackend {
  private device: GPUDevice;
  private manager: DeviceManager;

  constructor(device: GPUDevice) {
    this.device = device;
    this.manager = DeviceManager.getInstance();
  }

  /**
   * Поэлементное сложение на GPU
   */
  async add(a: Tensor, b: Tensor): Promise<Tensor> {
    const startTime = performance.now();

    // Создаём буферы
    const bufferA = createBuffer(this.device, a, GPUBufferUsage.STORAGE);
    const bufferB = createBuffer(this.device, b, GPUBufferUsage.STORAGE);
    const bufferResult = this.device.createBuffer({
      size: a.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Создаём pipeline
    const pipeline = await createPipeline(this.device, SHADERS.add, 'add');

    // Создаём bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: bufferResult } },
      ],
    });

    // Запускаем compute pass
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(a.size / 256));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Читаем результат
    const resultData = await readBuffer(this.device, bufferResult, a.data.byteLength);

    // Очищаем буферы
    safeDestroyBuffer(bufferA);
    safeDestroyBuffer(bufferB);
    safeDestroyBuffer(bufferResult);

    const result = new Tensor(resultData, [...a.shape], {
      dtype: a.dtype,
      requiresGrad: a.requiresGrad || b.requiresGrad,
    });

    this.manager.recordOp(DeviceType.GPU, performance.now() - startTime);
    return result;
  }

  /**
   * Поэлементное умножение на GPU
   */
  async mul(a: Tensor, b: Tensor): Promise<Tensor> {
    const startTime = performance.now();

    const bufferA = createBuffer(this.device, a, GPUBufferUsage.STORAGE);
    const bufferB = createBuffer(this.device, b, GPUBufferUsage.STORAGE);
    const bufferResult = this.device.createBuffer({
      size: a.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const pipeline = await createPipeline(this.device, SHADERS.mul, 'mul');

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: bufferResult } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(a.size / 256));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const resultData = await readBuffer(this.device, bufferResult, a.data.byteLength);

    safeDestroyBuffer(bufferA);
    safeDestroyBuffer(bufferB);
    safeDestroyBuffer(bufferResult);

    const result = new Tensor(resultData, [...a.shape], {
      dtype: a.dtype,
      requiresGrad: a.requiresGrad || b.requiresGrad,
    });

    this.manager.recordOp(DeviceType.GPU, performance.now() - startTime);
    return result;
  }

  /**
   * Матричное умножение на GPU (оптимизированное тайловое)
   */
  async matmul(a: Tensor, b: Tensor): Promise<Tensor> {
    const startTime = performance.now();

    if (a.ndim !== 2 || b.ndim !== 2) {
      throw new Error('GPU matmul currently supports only 2D matrices');
    }

    const [M, K1] = a.shape;
    const [K2, N] = b.shape;

    if (K1 !== K2) {
      throw new Error(`Matrix dimensions don't match: [${M}, ${K1}] x [${K2}, ${N}]`);
    }

    const bufferA = createBuffer(this.device, a, GPUBufferUsage.STORAGE);
    const bufferB = createBuffer(this.device, b, GPUBufferUsage.STORAGE);
    const bufferResult = this.device.createBuffer({
      size: M * N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Буфер для размерностей
    const dimsBuffer = this.device.createBuffer({
      size: 16, // 4 x u32
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    });
    new Uint32Array(dimsBuffer.getMappedRange()).set([M, N, K1, 0]);
    dimsBuffer.unmap();

    const pipeline = await createPipeline(this.device, SHADERS.matmul, 'matmul');

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: bufferResult } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const resultData = await readBuffer(this.device, bufferResult, M * N * 4);

    safeDestroyBuffer(bufferA);
    safeDestroyBuffer(bufferB);
    safeDestroyBuffer(bufferResult);
    safeDestroyBuffer(dimsBuffer);

    const result = new Tensor(resultData, [M, N], {
      dtype: a.dtype,
      requiresGrad: a.requiresGrad || b.requiresGrad,
    });

    this.manager.recordOp(DeviceType.GPU, performance.now() - startTime);
    return result;
  }

  /**
   * ReLU активация на GPU
   */
  async relu(input: Tensor): Promise<Tensor> {
    const startTime = performance.now();

    const bufferInput = createBuffer(this.device, input, GPUBufferUsage.STORAGE);
    const bufferOutput = this.device.createBuffer({
      size: input.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const pipeline = await createPipeline(this.device, SHADERS.relu, 'relu');

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferInput } },
        { binding: 1, resource: { buffer: bufferOutput } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(input.size / 256));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const resultData = await readBuffer(this.device, bufferOutput, input.data.byteLength);

    safeDestroyBuffer(bufferInput);
    safeDestroyBuffer(bufferOutput);

    const result = new Tensor(resultData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    this.manager.recordOp(DeviceType.GPU, performance.now() - startTime);
    return result;
  }

  /**
   * GELU активация на GPU
   */
  async gelu(input: Tensor): Promise<Tensor> {
    const startTime = performance.now();

    const bufferInput = createBuffer(this.device, input, GPUBufferUsage.STORAGE);
    const bufferOutput = this.device.createBuffer({
      size: input.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const pipeline = await createPipeline(this.device, SHADERS.gelu, 'gelu');

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferInput } },
        { binding: 1, resource: { buffer: bufferOutput } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(input.size / 256));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const resultData = await readBuffer(this.device, bufferOutput, input.data.byteLength);

    safeDestroyBuffer(bufferInput);
    safeDestroyBuffer(bufferOutput);

    const result = new Tensor(resultData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    this.manager.recordOp(DeviceType.GPU, performance.now() - startTime);
    return result;
  }

  /**
   * Sigmoid активация на GPU
   */
  async sigmoid(input: Tensor): Promise<Tensor> {
    const startTime = performance.now();

    const bufferInput = createBuffer(this.device, input, GPUBufferUsage.STORAGE);
    const bufferOutput = this.device.createBuffer({
      size: input.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const pipeline = await createPipeline(this.device, SHADERS.sigmoid, 'sigmoid');

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferInput } },
        { binding: 1, resource: { buffer: bufferOutput } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(input.size / 256));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const resultData = await readBuffer(this.device, bufferOutput, input.data.byteLength);

    safeDestroyBuffer(bufferInput);
    safeDestroyBuffer(bufferOutput);

    const result = new Tensor(resultData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    this.manager.recordOp(DeviceType.GPU, performance.now() - startTime);
    return result;
  }

  /**
   * Экспонента на GPU
   */
  async exp(input: Tensor): Promise<Tensor> {
    const startTime = performance.now();

    const bufferInput = createBuffer(this.device, input, GPUBufferUsage.STORAGE);
    const bufferOutput = this.device.createBuffer({
      size: input.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const pipeline = await createPipeline(this.device, SHADERS.exp, 'exp');

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferInput } },
        { binding: 1, resource: { buffer: bufferOutput } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(input.size / 256));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const resultData = await readBuffer(this.device, bufferOutput, input.data.byteLength);

    safeDestroyBuffer(bufferInput);
    safeDestroyBuffer(bufferOutput);

    const result = new Tensor(resultData, [...input.shape], {
      dtype: input.dtype,
      requiresGrad: input.requiresGrad,
    });

    this.manager.recordOp(DeviceType.GPU, performance.now() - startTime);
    return result;
  }

  /**
   * Очищает кеш pipeline'ов
   */
  clearCache(): void {
    pipelineCache.clear();
  }
}

/**
 * Создаёт GPU backend
 */
export function createGPUBackend(): GPUBackend | null {
  const manager = DeviceManager.getInstance();
  const device = manager.getGPUDevice();

  if (!device) {
    return null;
  }

  return new GPUBackend(device);
}

/**
 * Проверяет доступность GPU backend
 */
export function isGPUBackendAvailable(): boolean {
  const manager = DeviceManager.getInstance();
  return manager.isGPUAvailable();
}
