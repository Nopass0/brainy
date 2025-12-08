/**
 * @fileoverview Transformer слои
 * @description Multi-Head Attention, Feed-Forward Network и другие компоненты трансформера
 */

import { Tensor, tensor, zeros, ones, randn } from '../core/tensor';
import { DType } from '../core/dtype';
import { Module, Parameter, Sequential } from './module';
import { Linear, LayerNorm, Dropout, Embedding } from './layers';
import { GELU, ReLU, Softmax } from './activations';
import { computeSize } from '../core/shape';

/**
 * Конфигурация трансформера
 */
export interface TransformerConfig {
  /** Размер скрытого слоя */
  hiddenSize: number;
  /** Количество голов внимания */
  numHeads: number;
  /** Размер промежуточного слоя FFN */
  intermediateSize: number;
  /** Вероятность dropout */
  dropoutProb?: number;
  /** Epsilon для LayerNorm */
  layerNormEps?: number;
  /** Максимальная длина последовательности */
  maxSeqLength?: number;
  /** Размер словаря */
  vocabSize?: number;
  /** Количество слоёв */
  numLayers?: number;
  /** Использовать causal маску */
  causal?: boolean;
  /** Тип активации ('gelu' или 'relu') */
  activation?: 'gelu' | 'relu';
}

/**
 * Scaled Dot-Product Attention
 * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
 */
export class ScaledDotProductAttention extends Module {
  private scale: number;
  private dropout: Dropout;

  constructor(headDim: number, dropoutProb: number = 0.0) {
    super();
    this.scale = Math.sqrt(headDim);
    this.dropout = new Dropout(dropoutProb);
    this.registerModule('dropout', this.dropout);
  }

  /**
   * Forward pass
   * @param query - Query тензор [batch, heads, seq_len, head_dim]
   * @param key - Key тензор [batch, heads, seq_len, head_dim]
   * @param value - Value тензор [batch, heads, seq_len, head_dim]
   * @param mask - Опциональная маска [batch, 1, seq_len, seq_len]
   */
  forward(query: Tensor, key: Tensor, value: Tensor, mask?: Tensor): Tensor {
    // QK^T / sqrt(d_k)
    const scores = query.matmul(key.transpose(-2, -1)).div(this.scale);

    // Применяем маску (добавляем большое отрицательное число к маскированным позициям)
    let maskedScores = scores;
    if (mask) {
      // mask: 1 = attend, 0 = mask out
      // Конвертируем в additive mask: 0 для attend, -inf для mask
      const maskValues = mask.mul(-1).add(1).mul(-1e9);
      maskedScores = scores.add(maskValues);
    }

    // Softmax по последней размерности
    const attnWeights = this.softmax(maskedScores);

    // Dropout
    const attnWeightsDropped = this.dropout.forward(attnWeights);

    // Умножаем на V
    return attnWeightsDropped.matmul(value);
  }

  /**
   * Softmax по последней размерности
   */
  private softmax(x: Tensor): Tensor {
    // Находим max для численной стабильности
    const maxVals = x.max(-1, true).values;
    const expX = x.sub(maxVals).exp();
    const sumExp = expX.sum(-1, true);
    return expX.div(sumExp);
  }
}

/**
 * Multi-Head Attention
 */
export class MultiHeadAttention extends Module {
  private numHeads: number;
  private headDim: number;
  private hiddenSize: number;

  private queryProj: Linear;
  private keyProj: Linear;
  private valueProj: Linear;
  private outProj: Linear;
  private attention: ScaledDotProductAttention;

  constructor(hiddenSize: number, numHeads: number, dropoutProb: number = 0.0) {
    super();

    if (hiddenSize % numHeads !== 0) {
      throw new Error(`hiddenSize (${hiddenSize}) must be divisible by numHeads (${numHeads})`);
    }

    this.hiddenSize = hiddenSize;
    this.numHeads = numHeads;
    this.headDim = hiddenSize / numHeads;

    // Проекции Q, K, V
    this.queryProj = new Linear(hiddenSize, hiddenSize);
    this.keyProj = new Linear(hiddenSize, hiddenSize);
    this.valueProj = new Linear(hiddenSize, hiddenSize);
    this.outProj = new Linear(hiddenSize, hiddenSize);

    this.attention = new ScaledDotProductAttention(this.headDim, dropoutProb);

    this.registerModule('query_proj', this.queryProj);
    this.registerModule('key_proj', this.keyProj);
    this.registerModule('value_proj', this.valueProj);
    this.registerModule('out_proj', this.outProj);
    this.registerModule('attention', this.attention);
  }

  /**
   * Forward pass
   * @param query - Query тензор [batch, seq_len, hidden]
   * @param key - Key тензор [batch, seq_len, hidden]
   * @param value - Value тензор [batch, seq_len, hidden]
   * @param mask - Опциональная маска
   */
  forward(query: Tensor, key: Tensor, value: Tensor, mask?: Tensor): Tensor {
    const batchSize = query.shape[0];
    const seqLen = query.shape[1];

    // Проецируем Q, K, V
    let q = this.queryProj.forward(query);
    let k = this.keyProj.forward(key);
    let v = this.valueProj.forward(value);

    // Reshape для multi-head: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    q = this.splitHeads(q, batchSize, seqLen);
    k = this.splitHeads(k, batchSize, key.shape[1]);
    v = this.splitHeads(v, batchSize, value.shape[1]);

    // Attention
    let attnOutput = this.attention.forward(q, k, v, mask);

    // Объединяем головы: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    attnOutput = this.mergeHeads(attnOutput, batchSize, seqLen);

    // Выходная проекция
    return this.outProj.forward(attnOutput);
  }

  /**
   * Разбивает тензор на головы
   */
  private splitHeads(x: Tensor, batchSize: number, seqLen: number): Tensor {
    // [batch, seq, hidden] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    const reshaped = x.reshape(batchSize, seqLen, this.numHeads, this.headDim);
    return reshaped.transpose(1, 2);
  }

  /**
   * Объединяет головы
   */
  private mergeHeads(x: Tensor, batchSize: number, seqLen: number): Tensor {
    // [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim] -> [batch, seq, hidden]
    const transposed = x.transpose(1, 2);
    return transposed.reshape(batchSize, seqLen, this.hiddenSize);
  }
}

/**
 * Position-wise Feed-Forward Network
 */
export class FeedForward extends Module {
  private fc1: Linear;
  private fc2: Linear;
  private activation: Module;
  private dropout: Dropout;

  constructor(
    hiddenSize: number,
    intermediateSize: number,
    activation: 'gelu' | 'relu' = 'gelu',
    dropoutProb: number = 0.0
  ) {
    super();

    this.fc1 = new Linear(hiddenSize, intermediateSize);
    this.fc2 = new Linear(intermediateSize, hiddenSize);
    this.activation = activation === 'gelu' ? new GELU() : new ReLU();
    this.dropout = new Dropout(dropoutProb);

    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
    this.registerModule('activation', this.activation);
    this.registerModule('dropout', this.dropout);
  }

  forward(x: Tensor): Tensor {
    let hidden = this.fc1.forward(x);
    hidden = this.activation.forward(hidden);
    hidden = this.dropout.forward(hidden);
    hidden = this.fc2.forward(hidden);
    return hidden;
  }
}

/**
 * Transformer Encoder Block
 */
export class TransformerEncoderBlock extends Module {
  private attention: MultiHeadAttention;
  private feedForward: FeedForward;
  private norm1: LayerNorm;
  private norm2: LayerNorm;
  private dropout: Dropout;

  constructor(config: TransformerConfig) {
    super();

    const dropoutProb = config.dropoutProb ?? 0.1;
    const eps = config.layerNormEps ?? 1e-5;

    this.attention = new MultiHeadAttention(
      config.hiddenSize,
      config.numHeads,
      dropoutProb
    );
    this.feedForward = new FeedForward(
      config.hiddenSize,
      config.intermediateSize,
      config.activation ?? 'gelu',
      dropoutProb
    );
    this.norm1 = new LayerNorm(config.hiddenSize, eps);
    this.norm2 = new LayerNorm(config.hiddenSize, eps);
    this.dropout = new Dropout(dropoutProb);

    this.registerModule('attention', this.attention);
    this.registerModule('feed_forward', this.feedForward);
    this.registerModule('norm1', this.norm1);
    this.registerModule('norm2', this.norm2);
    this.registerModule('dropout', this.dropout);
  }

  forward(x: Tensor, mask?: Tensor): Tensor {
    // Self-attention с residual connection
    const attnOutput = this.attention.forward(x, x, x, mask);
    let hidden = x.add(this.dropout.forward(attnOutput));
    hidden = this.norm1.forward(hidden);

    // Feed-forward с residual connection
    const ffOutput = this.feedForward.forward(hidden);
    hidden = hidden.add(this.dropout.forward(ffOutput));
    hidden = this.norm2.forward(hidden);

    return hidden;
  }
}

/**
 * Transformer Decoder Block (с causal masking)
 */
export class TransformerDecoderBlock extends Module {
  private selfAttention: MultiHeadAttention;
  private crossAttention: MultiHeadAttention | null;
  private feedForward: FeedForward;
  private norm1: LayerNorm;
  private norm2: LayerNorm;
  private norm3: LayerNorm | null;
  private dropout: Dropout;

  constructor(config: TransformerConfig, hasCrossAttention: boolean = false) {
    super();

    const dropoutProb = config.dropoutProb ?? 0.1;
    const eps = config.layerNormEps ?? 1e-5;

    this.selfAttention = new MultiHeadAttention(
      config.hiddenSize,
      config.numHeads,
      dropoutProb
    );
    this.feedForward = new FeedForward(
      config.hiddenSize,
      config.intermediateSize,
      config.activation ?? 'gelu',
      dropoutProb
    );
    this.norm1 = new LayerNorm(config.hiddenSize, eps);
    this.norm2 = new LayerNorm(config.hiddenSize, eps);
    this.dropout = new Dropout(dropoutProb);

    this.registerModule('self_attention', this.selfAttention);
    this.registerModule('feed_forward', this.feedForward);
    this.registerModule('norm1', this.norm1);
    this.registerModule('norm2', this.norm2);
    this.registerModule('dropout', this.dropout);

    if (hasCrossAttention) {
      this.crossAttention = new MultiHeadAttention(
        config.hiddenSize,
        config.numHeads,
        dropoutProb
      );
      this.norm3 = new LayerNorm(config.hiddenSize, eps);
      this.registerModule('cross_attention', this.crossAttention);
      this.registerModule('norm3', this.norm3);
    } else {
      this.crossAttention = null;
      this.norm3 = null;
    }
  }

  forward(x: Tensor, causalMask: Tensor, encoderOutput?: Tensor, encoderMask?: Tensor): Tensor {
    // Causal self-attention
    const selfAttnOutput = this.selfAttention.forward(x, x, x, causalMask);
    let hidden = x.add(this.dropout.forward(selfAttnOutput));
    hidden = this.norm1.forward(hidden);

    // Cross-attention (если есть encoder output)
    if (this.crossAttention && this.norm3 && encoderOutput) {
      const crossAttnOutput = this.crossAttention.forward(
        hidden,
        encoderOutput,
        encoderOutput,
        encoderMask
      );
      hidden = hidden.add(this.dropout.forward(crossAttnOutput));
      hidden = this.norm3.forward(hidden);
    }

    // Feed-forward
    const ffOutput = this.feedForward.forward(hidden);
    hidden = hidden.add(this.dropout.forward(ffOutput));
    hidden = this.norm2.forward(hidden);

    return hidden;
  }
}

/**
 * Positional Encoding (sinusoidal)
 */
export class PositionalEncoding extends Module {
  private encoding: Tensor;
  private dropout: Dropout;
  private maxSeqLength: number;

  constructor(hiddenSize: number, maxSeqLength: number = 512, dropoutProb: number = 0.1) {
    super();

    this.maxSeqLength = maxSeqLength;
    this.dropout = new Dropout(dropoutProb);
    this.registerModule('dropout', this.dropout);

    // Создаём sinusoidal positional encoding
    const pe = new Float32Array(maxSeqLength * hiddenSize);

    for (let pos = 0; pos < maxSeqLength; pos++) {
      for (let i = 0; i < hiddenSize; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / hiddenSize);
        pe[pos * hiddenSize + i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      }
    }

    this.encoding = new Tensor(pe, [maxSeqLength, hiddenSize]);
  }

  forward(x: Tensor): Tensor {
    const seqLen = x.shape[1];

    // Получаем позиционные эмбеддинги для текущей длины
    const pe = this.getPositionalEncoding(seqLen);

    // Добавляем к входу
    const result = x.add(pe);
    return this.dropout.forward(result);
  }

  /**
   * Получает позиционные эмбеддинги для заданной длины
   */
  private getPositionalEncoding(seqLen: number): Tensor {
    if (seqLen > this.maxSeqLength) {
      throw new Error(`Sequence length ${seqLen} exceeds maximum ${this.maxSeqLength}`);
    }

    // Slice encoding to seqLen
    const hiddenSize = this.encoding.shape[1];
    const data = new Float32Array(seqLen * hiddenSize);

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < hiddenSize; j++) {
        data[i * hiddenSize + j] = this.encoding.data[i * hiddenSize + j];
      }
    }

    return new Tensor(data, [1, seqLen, hiddenSize]);
  }
}

/**
 * Learnable Positional Embedding
 */
export class LearnedPositionalEmbedding extends Module {
  private embedding: Embedding;

  constructor(maxSeqLength: number, hiddenSize: number) {
    super();
    this.embedding = new Embedding(maxSeqLength, hiddenSize);
    this.registerModule('embedding', this.embedding);
  }

  forward(x: Tensor): Tensor {
    const seqLen = x.shape[1];

    // Создаём позиционные индексы [0, 1, 2, ..., seqLen-1]
    const positions = new Float32Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
      positions[i] = i;
    }
    const positionIds = new Tensor(positions, [1, seqLen], { dtype: DType.Int32 });

    // Получаем позиционные эмбеддинги
    const posEmb = this.embedding.forward(positionIds);

    // Добавляем к входу
    return x.add(posEmb);
  }
}

/**
 * Transformer Encoder (стек encoder блоков)
 */
export class TransformerEncoder extends Module {
  private layers: TransformerEncoderBlock[] = [];
  private norm: LayerNorm;

  constructor(config: TransformerConfig) {
    super();

    const numLayers = config.numLayers ?? 6;

    for (let i = 0; i < numLayers; i++) {
      const layer = new TransformerEncoderBlock(config);
      this.layers.push(layer);
      this.registerModule(`layer_${i}`, layer);
    }

    this.norm = new LayerNorm(config.hiddenSize, config.layerNormEps ?? 1e-5);
    this.registerModule('norm', this.norm);
  }

  forward(x: Tensor, mask?: Tensor): Tensor {
    let hidden = x;

    for (const layer of this.layers) {
      hidden = layer.forward(hidden, mask);
    }

    return this.norm.forward(hidden);
  }
}

/**
 * Transformer Decoder (стек decoder блоков)
 */
export class TransformerDecoder extends Module {
  private layers: TransformerDecoderBlock[] = [];
  private norm: LayerNorm;
  private config: TransformerConfig;

  constructor(config: TransformerConfig, hasCrossAttention: boolean = false) {
    super();

    this.config = config;
    const numLayers = config.numLayers ?? 6;

    for (let i = 0; i < numLayers; i++) {
      const layer = new TransformerDecoderBlock(config, hasCrossAttention);
      this.layers.push(layer);
      this.registerModule(`layer_${i}`, layer);
    }

    this.norm = new LayerNorm(config.hiddenSize, config.layerNormEps ?? 1e-5);
    this.registerModule('norm', this.norm);
  }

  forward(x: Tensor, encoderOutput?: Tensor, encoderMask?: Tensor): Tensor {
    let hidden = x;
    const seqLen = x.shape[1];

    // Создаём causal маску
    const causalMask = this.createCausalMask(seqLen);

    for (const layer of this.layers) {
      hidden = layer.forward(hidden, causalMask, encoderOutput, encoderMask);
    }

    return this.norm.forward(hidden);
  }

  /**
   * Создаёт causal маску для decoder
   */
  private createCausalMask(seqLen: number): Tensor {
    // Нижнетреугольная матрица: 1 = attend, 0 = mask
    const mask = new Float32Array(seqLen * seqLen);

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j <= i; j++) {
        mask[i * seqLen + j] = 1;
      }
    }

    return new Tensor(mask, [1, 1, seqLen, seqLen]);
  }
}

/**
 * Rotary Position Embedding (RoPE)
 * Более современный подход к позиционному кодированию
 */
export class RotaryEmbedding extends Module {
  private dim: number;
  private maxSeqLen: number;
  private base: number;
  private cosCache: Tensor | null = null;
  private sinCache: Tensor | null = null;

  constructor(dim: number, maxSeqLen: number = 2048, base: number = 10000) {
    super();
    this.dim = dim;
    this.maxSeqLen = maxSeqLen;
    this.base = base;
    this.buildCache();
  }

  /**
   * Строит кеш sin/cos значений
   */
  private buildCache(): void {
    const halfDim = this.dim / 2;

    // inv_freq = 1 / (base^(2i/dim))
    const invFreq = new Float32Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
      invFreq[i] = 1.0 / Math.pow(this.base, (2 * i) / this.dim);
    }

    // positions = [0, 1, 2, ..., maxSeqLen-1]
    // freqs = positions * inv_freq (outer product)
    const cosData = new Float32Array(this.maxSeqLen * halfDim);
    const sinData = new Float32Array(this.maxSeqLen * halfDim);

    for (let pos = 0; pos < this.maxSeqLen; pos++) {
      for (let i = 0; i < halfDim; i++) {
        const freq = pos * invFreq[i];
        cosData[pos * halfDim + i] = Math.cos(freq);
        sinData[pos * halfDim + i] = Math.sin(freq);
      }
    }

    this.cosCache = new Tensor(cosData, [this.maxSeqLen, halfDim]);
    this.sinCache = new Tensor(sinData, [this.maxSeqLen, halfDim]);
  }

  /**
   * Применяет RoPE к query и key
   */
  forward(q: Tensor, k: Tensor, startPos: number = 0): [Tensor, Tensor] {
    const seqLen = q.shape[2];
    const halfDim = this.dim / 2;

    // Получаем cos/sin для текущих позиций
    const cosSlice = this.sliceCache(this.cosCache!, startPos, seqLen, halfDim);
    const sinSlice = this.sliceCache(this.sinCache!, startPos, seqLen, halfDim);

    // Применяем rotation
    const qRotated = this.applyRotation(q, cosSlice, sinSlice);
    const kRotated = this.applyRotation(k, cosSlice, sinSlice);

    return [qRotated, kRotated];
  }

  /**
   * Получает slice из кеша
   */
  private sliceCache(cache: Tensor, startPos: number, seqLen: number, halfDim: number): Tensor {
    const data = new Float32Array(seqLen * halfDim);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < halfDim; j++) {
        data[i * halfDim + j] = cache.data[(startPos + i) * halfDim + j];
      }
    }
    return new Tensor(data, [1, 1, seqLen, halfDim]);
  }

  /**
   * Применяет rotation к тензору
   */
  private applyRotation(x: Tensor, cos: Tensor, sin: Tensor): Tensor {
    const shape = x.shape;
    const halfDim = shape[shape.length - 1] / 2;

    // Разделяем на две половины
    const batchSize = shape[0];
    const numHeads = shape[1];
    const seqLen = shape[2];

    const resultData = new Float32Array(x.size);

    for (let b = 0; b < batchSize; b++) {
      for (let h = 0; h < numHeads; h++) {
        for (let s = 0; s < seqLen; s++) {
          const baseIdx = ((b * numHeads + h) * seqLen + s) * (halfDim * 2);
          const cosIdx = s * halfDim;

          for (let d = 0; d < halfDim; d++) {
            const x1 = x.data[baseIdx + d];
            const x2 = x.data[baseIdx + halfDim + d];
            const c = cos.data[cosIdx + d];
            const s_val = sin.data[cosIdx + d];

            // Rotation: [cos*x1 - sin*x2, sin*x1 + cos*x2]
            resultData[baseIdx + d] = c * x1 - s_val * x2;
            resultData[baseIdx + halfDim + d] = s_val * x1 + c * x2;
          }
        }
      }
    }

    return new Tensor(resultData, [...shape]);
  }
}

/**
 * Создаёт causal маску для attention
 */
export function createCausalMask(seqLen: number): Tensor {
  const mask = new Float32Array(seqLen * seqLen);

  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j <= i; j++) {
      mask[i * seqLen + j] = 1;
    }
  }

  return new Tensor(mask, [1, 1, seqLen, seqLen]);
}

/**
 * Создаёт padding маску из attention mask
 */
export function createPaddingMask(attentionMask: Tensor): Tensor {
  // attentionMask: [batch, seq_len] -> [batch, 1, 1, seq_len]
  const batchSize = attentionMask.shape[0];
  const seqLen = attentionMask.shape[1];

  const mask = new Float32Array(batchSize * seqLen);
  for (let i = 0; i < batchSize * seqLen; i++) {
    mask[i] = attentionMask.data[i];
  }

  return new Tensor(mask, [batchSize, 1, 1, seqLen]);
}
