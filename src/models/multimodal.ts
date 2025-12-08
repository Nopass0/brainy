/**
 * @fileoverview Multimodal Few-Shot Model
 * @description Мультимодальная модель для быстрого обучения на нескольких примерах
 *
 * Поддерживает:
 * - Текст (embeddings)
 * - Изображения (CNN encoder)
 * - Числовые последовательности
 * - Логические задачи
 */

import { Tensor, tensor, zeros, randn, cat, stack } from '../core/tensor';
import { Module, Sequential, Parameter } from '../nn/module';
import { Linear, Conv2d, Flatten, Dropout, LayerNorm, Embedding } from '../nn/layers';
import { ReLU, GELU, Softmax, Tanh, Sigmoid } from '../nn/activations';
import { MultiHeadAttention } from '../nn/transformer';

/**
 * Конфигурация мультимодальной модели
 */
export interface MultimodalConfig {
  /** Размер общего embedding пространства */
  embeddingDim: number;
  /** Размер скрытых слоёв */
  hiddenDim: number;
  /** Максимальная длина текста */
  maxTextLen?: number;
  /** Размер словаря для текста */
  vocabSize?: number;
  /** Размер изображения (квадратное) */
  imageSize?: number;
  /** Количество каналов изображения */
  imageChannels?: number;
  /** Dropout */
  dropout?: number;
  /** Количество голов внимания */
  numHeads?: number;
}

/**
 * Энкодер текста
 */
class TextEncoder extends Module {
  private embedding: Embedding;
  private encoder: Sequential;
  private outputProj: Linear;

  constructor(
    vocabSize: number,
    embeddingDim: number,
    hiddenDim: number,
    maxLen: number
  ) {
    super();

    this.embedding = new Embedding(vocabSize, hiddenDim);
    this.registerModule('embedding', this.embedding);

    this.encoder = new Sequential(
      new Linear(hiddenDim * maxLen, hiddenDim * 2),
      new GELU(),
      new Linear(hiddenDim * 2, hiddenDim),
      new GELU()
    );
    this.registerModule('encoder', this.encoder);

    this.outputProj = new Linear(hiddenDim, embeddingDim);
    this.registerModule('outputProj', this.outputProj);
  }

  forward(tokenIds: Tensor): Tensor {
    // tokenIds: [batch, seqLen]
    const embedded = this.embedding.forward(tokenIds);
    // Flatten sequence
    const batchSize = tokenIds.shape[0];
    const flatData = new Float32Array(embedded.data);
    const flatLen = embedded.data.length / batchSize;
    const flat = new Tensor(flatData, [batchSize, flatLen], { requiresGrad: embedded.requiresGrad });

    const encoded = this.encoder.forward(flat);
    return this.outputProj.forward(encoded);
  }
}

/**
 * Энкодер изображений (простой CNN)
 */
class ImageEncoder extends Module {
  private encoder: Sequential;
  private outputProj: Linear;

  constructor(
    imageSize: number,
    channels: number,
    embeddingDim: number,
    hiddenDim: number
  ) {
    super();

    // Простой MLP для изображений (flatten -> linear)
    const inputSize = imageSize * imageSize * channels;

    this.encoder = new Sequential(
      new Linear(inputSize, hiddenDim * 2),
      new GELU(),
      new Linear(hiddenDim * 2, hiddenDim),
      new GELU()
    );
    this.registerModule('encoder', this.encoder);

    this.outputProj = new Linear(hiddenDim, embeddingDim);
    this.registerModule('outputProj', this.outputProj);
  }

  forward(image: Tensor): Tensor {
    // image: [batch, channels, height, width] или [batch, flat]
    const batchSize = image.shape[0];
    const flatSize = image.data.length / batchSize;

    // Flatten if needed
    let flat: Tensor;
    if (image.shape.length > 2) {
      flat = new Tensor(new Float32Array(image.data), [batchSize, flatSize], { requiresGrad: image.requiresGrad });
    } else {
      flat = image;
    }

    const encoded = this.encoder.forward(flat);
    return this.outputProj.forward(encoded);
  }
}

/**
 * Энкодер числовых последовательностей
 */
class SequenceEncoder extends Module {
  private encoder: Sequential;
  private outputProj: Linear;

  constructor(maxSeqLen: number, embeddingDim: number, hiddenDim: number) {
    super();

    this.encoder = new Sequential(
      new Linear(maxSeqLen, hiddenDim),
      new GELU(),
      new Linear(hiddenDim, hiddenDim),
      new GELU()
    );
    this.registerModule('encoder', this.encoder);

    this.outputProj = new Linear(hiddenDim, embeddingDim);
    this.registerModule('outputProj', this.outputProj);
  }

  forward(sequence: Tensor): Tensor {
    const encoded = this.encoder.forward(sequence);
    return this.outputProj.forward(encoded);
  }
}

/**
 * Декодер для разных модальностей
 */
class MultimodalDecoder extends Module {
  private textDecoder: Sequential;
  private classDecoder: Sequential;
  private valueDecoder: Sequential;

  constructor(embeddingDim: number, hiddenDim: number, vocabSize: number, numClasses: number) {
    super();

    // Текстовый декодер (упрощённый - выдаёт один токен)
    this.textDecoder = new Sequential(
      new Linear(embeddingDim, hiddenDim),
      new GELU(),
      new Linear(hiddenDim, vocabSize)
    );
    this.registerModule('textDecoder', this.textDecoder);

    // Классификационный декодер
    this.classDecoder = new Sequential(
      new Linear(embeddingDim, hiddenDim),
      new GELU(),
      new Linear(hiddenDim, numClasses)
    );
    this.registerModule('classDecoder', this.classDecoder);

    // Регрессионный декодер
    this.valueDecoder = new Sequential(
      new Linear(embeddingDim, hiddenDim),
      new GELU(),
      new Linear(hiddenDim, 1)
    );
    this.registerModule('valueDecoder', this.valueDecoder);
  }

  decodeText(embedding: Tensor): Tensor {
    return this.textDecoder.forward(embedding);
  }

  decodeClass(embedding: Tensor): Tensor {
    return this.classDecoder.forward(embedding);
  }

  decodeValue(embedding: Tensor): Tensor {
    return this.valueDecoder.forward(embedding);
  }
}

/**
 * Cross-modal Attention для объединения модальностей
 */
class CrossModalFusion extends Module {
  private queryProj: Linear;
  private keyProj: Linear;
  private valueProj: Linear;
  private outputProj: Linear;
  private numHeads: number;
  private headDim: number;

  constructor(embeddingDim: number, numHeads: number = 4) {
    super();

    this.numHeads = numHeads;
    this.headDim = embeddingDim / numHeads;

    this.queryProj = new Linear(embeddingDim, embeddingDim);
    this.keyProj = new Linear(embeddingDim, embeddingDim);
    this.valueProj = new Linear(embeddingDim, embeddingDim);
    this.outputProj = new Linear(embeddingDim, embeddingDim);

    this.registerModule('queryProj', this.queryProj);
    this.registerModule('keyProj', this.keyProj);
    this.registerModule('valueProj', this.valueProj);
    this.registerModule('outputProj', this.outputProj);
  }

  forward(query: Tensor, context: Tensor): Tensor {
    // Simplified attention
    const q = this.queryProj.forward(query);
    const k = this.keyProj.forward(context);
    const v = this.valueProj.forward(context);

    // Simple dot-product attention
    const scores = q.matmul(k.transpose(0, 1));
    const scale = Math.sqrt(q.shape[q.shape.length - 1]);

    // Softmax
    const scoreData = new Float32Array(scores.data.length);
    const batchSize = scores.shape[0];
    const seqLen = scores.shape[1];

    for (let b = 0; b < batchSize; b++) {
      let max = -Infinity;
      for (let i = 0; i < seqLen; i++) {
        const idx = b * seqLen + i;
        scoreData[idx] = scores.data[idx] / scale;
        if (scoreData[idx] > max) max = scoreData[idx];
      }

      let sum = 0;
      for (let i = 0; i < seqLen; i++) {
        const idx = b * seqLen + i;
        scoreData[idx] = Math.exp(scoreData[idx] - max);
        sum += scoreData[idx];
      }

      for (let i = 0; i < seqLen; i++) {
        const idx = b * seqLen + i;
        scoreData[idx] /= sum;
      }
    }

    const attnWeights = new Tensor(scoreData, scores.shape, { requiresGrad: scores.requiresGrad });
    const attended = attnWeights.matmul(v);

    return this.outputProj.forward(attended);
  }
}

/**
 * Мультимодальная Few-Shot модель
 *
 * Объединяет текст, изображения и числовые данные в общее пространство
 * и использует prototypical networks для few-shot обучения.
 */
export class MultimodalFewShot extends Module {
  private config: Required<MultimodalConfig>;
  private textEncoder: TextEncoder;
  private imageEncoder: ImageEncoder;
  private sequenceEncoder: SequenceEncoder;
  private fusion: CrossModalFusion;
  private decoder: MultimodalDecoder;
  private taskEncoder: Sequential;

  constructor(config: MultimodalConfig) {
    super();

    this.config = {
      maxTextLen: 32,
      vocabSize: 1000,
      imageSize: 8,
      imageChannels: 1,
      dropout: 0.1,
      numHeads: 4,
      ...config,
    };

    const { embeddingDim, hiddenDim, vocabSize, maxTextLen, imageSize, imageChannels, numHeads } = this.config;

    // Энкодеры
    this.textEncoder = new TextEncoder(vocabSize!, embeddingDim, hiddenDim, maxTextLen!);
    this.imageEncoder = new ImageEncoder(imageSize!, imageChannels!, embeddingDim, hiddenDim);
    this.sequenceEncoder = new SequenceEncoder(maxTextLen!, embeddingDim, hiddenDim);

    this.registerModule('textEncoder', this.textEncoder);
    this.registerModule('imageEncoder', this.imageEncoder);
    this.registerModule('sequenceEncoder', this.sequenceEncoder);

    // Fusion
    this.fusion = new CrossModalFusion(embeddingDim, numHeads!);
    this.registerModule('fusion', this.fusion);

    // Task encoder (для task conditioning)
    this.taskEncoder = new Sequential(
      new Linear(embeddingDim, hiddenDim),
      new GELU(),
      new Linear(hiddenDim, embeddingDim)
    );
    this.registerModule('taskEncoder', this.taskEncoder);

    // Decoder
    this.decoder = new MultimodalDecoder(embeddingDim, hiddenDim, vocabSize!, 10);
    this.registerModule('decoder', this.decoder);
  }

  /**
   * Кодирует входные данные в общее пространство
   */
  encode(
    data: {
      text?: Tensor;
      image?: Tensor;
      sequence?: Tensor;
    }
  ): Tensor {
    const embeddings: Tensor[] = [];

    if (data.text) {
      embeddings.push(this.textEncoder.forward(data.text));
    }
    if (data.image) {
      embeddings.push(this.imageEncoder.forward(data.image));
    }
    if (data.sequence) {
      embeddings.push(this.sequenceEncoder.forward(data.sequence));
    }

    if (embeddings.length === 0) {
      throw new Error('At least one modality must be provided');
    }

    if (embeddings.length === 1) {
      return embeddings[0];
    }

    // Combine embeddings (average)
    let combined = embeddings[0];
    for (let i = 1; i < embeddings.length; i++) {
      combined = combined.add(embeddings[i]);
    }

    // Normalize
    const scale = 1 / embeddings.length;
    const scaledData = new Float32Array(combined.data.length);
    for (let i = 0; i < combined.data.length; i++) {
      scaledData[i] = combined.data[i] * scale;
    }

    return new Tensor(scaledData, combined.shape, { requiresGrad: combined.requiresGrad });
  }

  /**
   * Few-shot классификация через prototypical networks
   */
  fewShotClassify(
    supportSet: { data: { text?: Tensor; image?: Tensor; sequence?: Tensor }; label: number }[],
    querySet: { text?: Tensor; image?: Tensor; sequence?: Tensor }[]
  ): number[] {
    // Compute class prototypes
    const prototypes: Map<number, Float32Array> = new Map();
    const counts: Map<number, number> = new Map();

    for (const { data, label } of supportSet) {
      const embedding = this.encode(data);

      if (!prototypes.has(label)) {
        prototypes.set(label, new Float32Array(embedding.data.length).fill(0));
        counts.set(label, 0);
      }

      const proto = prototypes.get(label)!;
      for (let i = 0; i < embedding.data.length; i++) {
        proto[i] += embedding.data[i];
      }
      counts.set(label, counts.get(label)! + 1);
    }

    // Average prototypes
    for (const [label, proto] of prototypes.entries()) {
      const count = counts.get(label)!;
      for (let i = 0; i < proto.length; i++) {
        proto[i] /= count;
      }
    }

    // Classify queries
    const predictions: number[] = [];

    for (const query of querySet) {
      const queryEmb = this.encode(query);

      let bestLabel = -1;
      let bestDist = Infinity;

      for (const [label, proto] of prototypes.entries()) {
        // Euclidean distance
        let dist = 0;
        for (let i = 0; i < queryEmb.data.length; i++) {
          const diff = queryEmb.data[i] - proto[i];
          dist += diff * diff;
        }
        dist = Math.sqrt(dist);

        if (dist < bestDist) {
          bestDist = dist;
          bestLabel = label;
        }
      }

      predictions.push(bestLabel);
    }

    return predictions;
  }

  /**
   * Few-shot регрессия
   */
  fewShotRegress(
    supportSet: { data: { text?: Tensor; image?: Tensor; sequence?: Tensor }; value: number }[],
    querySet: { text?: Tensor; image?: Tensor; sequence?: Tensor }[]
  ): number[] {
    // Encode support set
    const supportEmbs: Float32Array[] = [];
    const supportVals: number[] = [];

    for (const { data, value } of supportSet) {
      const emb = this.encode(data);
      supportEmbs.push(new Float32Array(emb.data));
      supportVals.push(value);
    }

    // Predict for queries using weighted average
    const predictions: number[] = [];

    for (const query of querySet) {
      const queryEmb = this.encode(query);

      // Compute weights based on similarity
      const weights: number[] = [];
      let totalWeight = 0;

      for (const supportEmb of supportEmbs) {
        // Cosine similarity
        let dot = 0, normQ = 0, normS = 0;
        for (let i = 0; i < queryEmb.data.length; i++) {
          dot += queryEmb.data[i] * supportEmb[i];
          normQ += queryEmb.data[i] * queryEmb.data[i];
          normS += supportEmb[i] * supportEmb[i];
        }

        const sim = dot / (Math.sqrt(normQ) * Math.sqrt(normS) + 1e-8);
        const weight = Math.exp(sim * 5);  // Temperature scaling
        weights.push(weight);
        totalWeight += weight;
      }

      // Weighted average prediction
      let pred = 0;
      for (let i = 0; i < weights.length; i++) {
        pred += (weights[i] / totalWeight) * supportVals[i];
      }

      predictions.push(pred);
    }

    return predictions;
  }

  /**
   * Генерация текста (один токен)
   */
  generateNextToken(context: { text?: Tensor; image?: Tensor; sequence?: Tensor }): Tensor {
    const embedding = this.encode(context);
    return this.decoder.decodeText(embedding);
  }

  /**
   * Классификация
   */
  classify(input: { text?: Tensor; image?: Tensor; sequence?: Tensor }): Tensor {
    const embedding = this.encode(input);
    return this.decoder.decodeClass(embedding);
  }

  /**
   * Регрессия
   */
  regress(input: { text?: Tensor; image?: Tensor; sequence?: Tensor }): Tensor {
    const embedding = this.encode(input);
    return this.decoder.decodeValue(embedding);
  }
}

/**
 * Factory function для создания small multimodal model
 */
export function createSmallMultimodal(): MultimodalFewShot {
  return new MultimodalFewShot({
    embeddingDim: 64,
    hiddenDim: 128,
    vocabSize: 256,
    maxTextLen: 16,
    imageSize: 8,
    imageChannels: 1,
    numHeads: 4,
  });
}

/**
 * Factory function для создания medium multimodal model
 */
export function createMediumMultimodal(): MultimodalFewShot {
  return new MultimodalFewShot({
    embeddingDim: 128,
    hiddenDim: 256,
    vocabSize: 1000,
    maxTextLen: 32,
    imageSize: 16,
    imageChannels: 3,
    numHeads: 8,
  });
}
