/**
 * @fileoverview Hugging Face Hub integration
 * @description Download and use pre-trained models from Hugging Face
 */

import { Tensor } from '../core/tensor';
import { Module } from '../nn/module';

/**
 * Hugging Face model info
 */
export interface HFModelInfo {
  modelId: string;
  revision?: string;
  filename?: string;
}

/**
 * Hugging Face Hub client
 */
export class HuggingFaceHub {
  private baseUrl = 'https://huggingface.co';
  private apiUrl = 'https://huggingface.co/api';
  private token?: string;
  private cacheDir: string;

  constructor(options?: { token?: string; cacheDir?: string }) {
    this.token = options?.token || process.env.HF_TOKEN;
    this.cacheDir = options?.cacheDir || './.hf_cache';
  }

  /**
   * Get headers for API requests
   */
  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'User-Agent': 'brainy/1.0',
    };
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    return headers;
  }

  /**
   * Get model info from Hugging Face
   */
  async getModelInfo(modelId: string): Promise<any> {
    const url = `${this.apiUrl}/models/${modelId}`;
    const response = await fetch(url, { headers: this.getHeaders() });
    if (!response.ok) {
      throw new Error(`Failed to get model info: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * List files in a model repository
   */
  async listFiles(modelId: string, revision = 'main'): Promise<string[]> {
    const info = await this.getModelInfo(modelId);
    return info.siblings?.map((s: any) => s.rfilename) || [];
  }

  /**
   * Download a file from Hugging Face
   */
  async downloadFile(
    modelId: string,
    filename: string,
    revision = 'main'
  ): Promise<ArrayBuffer> {
    const url = `${this.baseUrl}/${modelId}/resolve/${revision}/${filename}`;
    console.log(`Downloading: ${url}`);

    const response = await fetch(url, { headers: this.getHeaders() });
    if (!response.ok) {
      throw new Error(`Failed to download ${filename}: ${response.statusText}`);
    }

    return response.arrayBuffer();
  }

  /**
   * Download model weights (safetensors or pytorch format)
   */
  async downloadWeights(
    modelId: string,
    options?: { revision?: string; filename?: string }
  ): Promise<Map<string, Tensor>> {
    const revision = options?.revision || 'main';
    const files = await this.listFiles(modelId, revision);

    // Prefer safetensors format
    let weightFile = options?.filename;
    if (!weightFile) {
      weightFile = files.find(f => f.endsWith('.safetensors'));
      if (!weightFile) {
        weightFile = files.find(f => f.endsWith('.bin') || f.endsWith('.pt'));
      }
    }

    if (!weightFile) {
      throw new Error(`No weight file found in ${modelId}. Files: ${files.join(', ')}`);
    }

    console.log(`Using weight file: ${weightFile}`);
    const data = await this.downloadFile(modelId, weightFile, revision);

    if (weightFile.endsWith('.safetensors')) {
      return this.parseSafetensors(data);
    } else {
      throw new Error('Only safetensors format is currently supported. Please use a model with .safetensors weights.');
    }
  }

  /**
   * Parse safetensors format
   * Format: 8 bytes header size (u64 LE) + JSON header + raw tensor data
   */
  private parseSafetensors(data: ArrayBuffer): Map<string, Tensor> {
    const view = new DataView(data);
    const headerSize = Number(view.getBigUint64(0, true));
    const headerBytes = new Uint8Array(data, 8, headerSize);
    const headerStr = new TextDecoder().decode(headerBytes);
    const header = JSON.parse(headerStr);

    const weights = new Map<string, Tensor>();
    const dataOffset = 8 + headerSize;

    for (const [name, info] of Object.entries(header)) {
      if (name === '__metadata__') continue;

      const tensorInfo = info as {
        dtype: string;
        shape: number[];
        data_offsets: [number, number];
      };

      const [start, end] = tensorInfo.data_offsets;
      const tensorData = new Uint8Array(data, dataOffset + start, end - start);

      // Convert to Float32Array based on dtype
      let float32Data: Float32Array;

      if (tensorInfo.dtype === 'F32') {
        float32Data = new Float32Array(tensorData.buffer, tensorData.byteOffset, tensorData.byteLength / 4);
      } else if (tensorInfo.dtype === 'F16') {
        // Convert F16 to F32
        float32Data = this.float16ToFloat32(tensorData);
      } else if (tensorInfo.dtype === 'BF16') {
        // Convert BF16 to F32
        float32Data = this.bfloat16ToFloat32(tensorData);
      } else {
        console.warn(`Unsupported dtype ${tensorInfo.dtype} for ${name}, skipping`);
        continue;
      }

      weights.set(name, new Tensor(float32Data, tensorInfo.shape));
    }

    return weights;
  }

  /**
   * Convert Float16 to Float32
   */
  private float16ToFloat32(data: Uint8Array): Float32Array {
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const result = new Float32Array(data.byteLength / 2);

    for (let i = 0; i < result.length; i++) {
      const h = view.getUint16(i * 2, true);
      result[i] = this.halfToFloat(h);
    }

    return result;
  }

  /**
   * Convert BFloat16 to Float32
   */
  private bfloat16ToFloat32(data: Uint8Array): Float32Array {
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const result = new Float32Array(data.byteLength / 2);

    for (let i = 0; i < result.length; i++) {
      const bf16 = view.getUint16(i * 2, true);
      // BF16 is just the upper 16 bits of F32
      const f32bits = bf16 << 16;
      const f32view = new DataView(new ArrayBuffer(4));
      f32view.setUint32(0, f32bits, true);
      result[i] = f32view.getFloat32(0, true);
    }

    return result;
  }

  /**
   * Convert half precision to float
   */
  private halfToFloat(h: number): number {
    const sign = (h >> 15) & 1;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;

    if (exp === 0) {
      if (mant === 0) return sign ? -0 : 0;
      // Subnormal
      const f = mant / 1024;
      return (sign ? -1 : 1) * f * Math.pow(2, -14);
    } else if (exp === 31) {
      return mant ? NaN : (sign ? -Infinity : Infinity);
    }

    return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
  }

  /**
   * Download and load config.json
   */
  async downloadConfig(modelId: string, revision = 'main'): Promise<any> {
    try {
      const data = await this.downloadFile(modelId, 'config.json', revision);
      const text = new TextDecoder().decode(data);
      return JSON.parse(text);
    } catch {
      return null;
    }
  }

  /**
   * Download tokenizer files
   */
  async downloadTokenizer(modelId: string, revision = 'main'): Promise<{
    vocab?: any;
    config?: any;
    merges?: string[];
  }> {
    const result: any = {};

    try {
      const vocabData = await this.downloadFile(modelId, 'vocab.json', revision);
      result.vocab = JSON.parse(new TextDecoder().decode(vocabData));
    } catch {}

    try {
      const configData = await this.downloadFile(modelId, 'tokenizer_config.json', revision);
      result.config = JSON.parse(new TextDecoder().decode(configData));
    } catch {}

    try {
      const mergesData = await this.downloadFile(modelId, 'merges.txt', revision);
      result.merges = new TextDecoder().decode(mergesData).split('\n').filter(l => l.trim());
    } catch {}

    return result;
  }
}

/**
 * Load weights into a model
 */
export function loadWeightsIntoModel(
  model: Module,
  weights: Map<string, Tensor>,
  options?: { strict?: boolean; mapping?: Record<string, string> }
): { loaded: string[]; missing: string[]; unexpected: string[] } {
  const strict = options?.strict ?? false;
  const mapping = options?.mapping || {};

  const modelParams = model.namedParameters();
  const loaded: string[] = [];
  const missing: string[] = [];
  const unexpected: string[] = [];

  // Find all model parameter names
  const modelParamNames = new Set(modelParams.map(([name]) => name));

  // Load weights
  for (const [paramName, param] of modelParams) {
    // Check for mapped name
    let weightName = mapping[paramName] || paramName;

    // Try different naming conventions
    const namesToTry = [
      weightName,
      weightName.replace(/\./g, '/'),
      weightName.replace(/\//g, '.'),
      `model.${weightName}`,
      weightName.replace('model.', ''),
    ];

    let found = false;
    for (const name of namesToTry) {
      if (weights.has(name)) {
        const weight = weights.get(name)!;

        // Check shape compatibility
        const paramShape = param.value.shape;
        const weightShape = weight.shape;

        if (paramShape.length === weightShape.length &&
            paramShape.every((d, i) => d === weightShape[i])) {
          // Direct copy
          param.value.data.set(weight.data as Float32Array);
          loaded.push(paramName);
          found = true;
          break;
        } else {
          console.warn(`Shape mismatch for ${paramName}: model=${paramShape}, weight=${weightShape}`);
        }
      }
    }

    if (!found) {
      missing.push(paramName);
    }
  }

  // Find unexpected weights
  for (const [name] of weights) {
    if (!loaded.some(l => name.includes(l) || l.includes(name))) {
      unexpected.push(name);
    }
  }

  if (strict && missing.length > 0) {
    throw new Error(`Missing weights: ${missing.join(', ')}`);
  }

  console.log(`Loaded ${loaded.length} parameters`);
  if (missing.length > 0) {
    console.log(`Missing ${missing.length} parameters: ${missing.slice(0, 5).join(', ')}${missing.length > 5 ? '...' : ''}`);
  }

  return { loaded, missing, unexpected };
}

/**
 * Create a HuggingFace Hub instance
 */
export function createHuggingFaceHub(options?: { token?: string; cacheDir?: string }): HuggingFaceHub {
  return new HuggingFaceHub(options);
}

/**
 * Quick function to download model weights
 */
export async function downloadModel(
  modelId: string,
  options?: { token?: string; revision?: string }
): Promise<Map<string, Tensor>> {
  const hub = new HuggingFaceHub({ token: options?.token });
  return hub.downloadWeights(modelId, { revision: options?.revision });
}
