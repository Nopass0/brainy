/**
 * @fileoverview GPT-подобная модель для генерации текста
 * @description Decoder-only трансформер с causal attention
 */

import { Tensor, tensor, zeros, randn, noGrad } from '../core/tensor';
import { DType } from '../core/dtype';
import { Module, Parameter, Sequential } from '../nn/module';
import { Linear, LayerNorm, Dropout, Embedding } from '../nn/layers';
import {
  TransformerConfig,
  TransformerDecoder,
  LearnedPositionalEmbedding,
  createCausalMask,
} from '../nn/transformer';
import { Tokenizer } from '../text/tokenizer';
import { GradNode, GradContext, isNoGradEnabled } from '../core/autograd';

/**
 * Конфигурация GPT модели
 */
export interface GPTConfig extends TransformerConfig {
  /** Размер словаря */
  vocabSize: number;
  /** Максимальная длина последовательности */
  maxSeqLength: number;
  /** Привязывать ли веса embedding и lm_head */
  tieWeights?: boolean;
}

/**
 * Параметры генерации текста
 */
export interface GenerationConfig {
  /** Максимальное количество новых токенов */
  maxNewTokens?: number;
  /** Температура (0 = greedy, > 1 = более случайный) */
  temperature?: number;
  /** Top-k sampling (0 = отключено) */
  topK?: number;
  /** Top-p (nucleus) sampling (1.0 = отключено) */
  topP?: number;
  /** Repetition penalty */
  repetitionPenalty?: number;
  /** ID токена конца последовательности */
  eosTokenId?: number;
  /** ID токена padding */
  padTokenId?: number;
  /** Делать ли sampling или greedy */
  doSample?: boolean;
}

/**
 * GPT (Generative Pre-trained Transformer) модель
 */
export class GPT extends Module {
  private config: GPTConfig;
  private tokenEmbedding: Embedding;
  private positionEmbedding: LearnedPositionalEmbedding;
  private dropout: Dropout;
  private decoder: TransformerDecoder;
  private lmHead: Linear;
  private norm: LayerNorm;

  constructor(config: GPTConfig) {
    super();

    this.config = {
      tieWeights: true,
      dropoutProb: 0.1,
      layerNormEps: 1e-5,
      numLayers: 6,
      activation: 'gelu',
      ...config,
    };

    // Token embeddings
    this.tokenEmbedding = new Embedding(config.vocabSize, config.hiddenSize);
    this.registerModule('token_embedding', this.tokenEmbedding);

    // Position embeddings
    this.positionEmbedding = new LearnedPositionalEmbedding(
      config.maxSeqLength,
      config.hiddenSize
    );
    this.registerModule('position_embedding', this.positionEmbedding);

    // Dropout
    this.dropout = new Dropout(this.config.dropoutProb!);
    this.registerModule('dropout', this.dropout);

    // Transformer decoder
    this.decoder = new TransformerDecoder(this.config, false);
    this.registerModule('decoder', this.decoder);

    // Final layer norm
    this.norm = new LayerNorm(config.hiddenSize, this.config.layerNormEps);
    this.registerModule('norm', this.norm);

    // Language modeling head
    this.lmHead = new Linear(config.hiddenSize, config.vocabSize, false);
    this.registerModule('lm_head', this.lmHead);

    // Tie weights
    if (this.config.tieWeights) {
      this.lmHead.weight = this.tokenEmbedding.weight;
    }
  }

  /**
   * Forward pass
   * @param inputIds - ID токенов [batch, seq_len]
   * @param attentionMask - Маска внимания (опционально)
   * @returns Logits [batch, seq_len, vocab_size]
   */
  forward(inputIds: Tensor, attentionMask?: Tensor): Tensor {
    const batchSize = inputIds.shape[0];
    const seqLen = inputIds.shape[1];

    // Token embeddings
    let hidden = this.tokenEmbedding.forward(inputIds);

    // Position embeddings
    hidden = this.positionEmbedding.forward(hidden);

    // Dropout
    hidden = this.dropout.forward(hidden);

    // Decoder
    hidden = this.decoder.forward(hidden);

    // Final norm
    hidden = this.norm.forward(hidden);

    // LM head
    const logits = this.lmHead.forward(hidden);

    return logits;
  }

  /**
   * Вычисляет loss для языкового моделирования
   * @param inputIds - ID токенов [batch, seq_len]
   * @param labels - Метки (те же что inputIds - будут сдвинуты внутри) [batch, seq_len]
   */
  computeLoss(inputIds: Tensor, labels: Tensor): Tensor {
    const logits = this.forward(inputIds);

    const batchSize = logits.shape[0];
    const seqLen = logits.shape[1];
    const vocabSize = logits.shape[2];

    // Cross entropy loss с proper gradient support
    // Для позиции t предсказываем labels[t+1]

    // Collect targets - labels is already shifted (labels[t] is the target for input[t])
    // For position t, logits[t] predicts the next token which is labels[t]
    const targets = new Float32Array(batchSize * seqLen);
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        targets[b * seqLen + t] = labels.data[b * seqLen + t];
      }
    }

    // Compute loss
    let totalLoss = 0;
    let count = 0;

    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        const rowIdx = b * seqLen + t;
        const targetToken = Math.floor(targets[b * seqLen + t]);

        if (targetToken >= 0 && targetToken < vocabSize) {
          // Numerically stable log softmax
          let maxVal = -Infinity;
          for (let v = 0; v < vocabSize; v++) {
            const val = logits.data[(rowIdx * vocabSize) + v];
            if (val > maxVal) maxVal = val;
          }

          let sumExp = 0;
          for (let v = 0; v < vocabSize; v++) {
            sumExp += Math.exp(logits.data[(rowIdx * vocabSize) + v] - maxVal);
          }

          const logProb = logits.data[(rowIdx * vocabSize) + targetToken] - maxVal - Math.log(sumExp + 1e-10);
          totalLoss -= logProb;
          count++;
        }
      }
    }

    const avgLoss = count > 0 ? totalLoss / count : 0;

    // Create result tensor with proper gradient
    const result = new Tensor(new Float32Array([avgLoss]), [1], {
      requiresGrad: logits.requiresGrad,
    });

    // Setup autograd
    if (logits.requiresGrad && !isNoGradEnabled()) {
      const capturedLogits = logits;
      const capturedTargets = targets;
      const capturedCount = count;
      const capturedSeqLen = seqLen;
      const capturedVocabSize = vocabSize;
      const capturedBatchSize = batchSize;

      result.gradNode = new GradNode(
        (gradOutput: Tensor) => {
          const gradData = new Float32Array(capturedLogits.size);

          for (let b = 0; b < capturedBatchSize; b++) {
            for (let t = 0; t < capturedSeqLen; t++) {
              const rowIdx = b * capturedSeqLen + t;
              const targetToken = Math.floor(capturedTargets[b * capturedSeqLen + t]);

              if (targetToken >= 0 && targetToken < capturedVocabSize) {
                // Compute softmax for this position
                let maxVal = -Infinity;
                for (let v = 0; v < capturedVocabSize; v++) {
                  const val = capturedLogits.data[(rowIdx * capturedVocabSize) + v];
                  if (val > maxVal) maxVal = val;
                }

                let sumExp = 0;
                const expVals = new Float32Array(capturedVocabSize);
                for (let v = 0; v < capturedVocabSize; v++) {
                  expVals[v] = Math.exp(capturedLogits.data[(rowIdx * capturedVocabSize) + v] - maxVal);
                  sumExp += expVals[v];
                }

                const gradScale = gradOutput.data[0] / capturedCount;

                for (let v = 0; v < capturedVocabSize; v++) {
                  const softmax = expVals[v] / sumExp;
                  const targetVal = v === targetToken ? 1 : 0;
                  gradData[(rowIdx * capturedVocabSize) + v] = gradScale * (softmax - targetVal);
                }
              }
            }
          }

          return [new Tensor(gradData, [...capturedLogits.shape], { dtype: capturedLogits.dtype })];
        },
        [logits],
        new GradContext()
      );
    }

    return result;
  }

  /**
   * Генерирует текст
   * @param inputIds - Начальные токены [batch, seq_len]
   * @param config - Параметры генерации
   */
  generate(inputIds: Tensor, config: GenerationConfig = {}): Tensor {
    const {
      maxNewTokens = 50,
      temperature = 1.0,
      topK = 0,
      topP = 1.0,
      repetitionPenalty = 1.0,
      eosTokenId,
      doSample = true,
    } = config;

    this.eval();
    let currentIds = inputIds;

    return noGrad(() => {
      for (let i = 0; i < maxNewTokens; i++) {
        // Обрезаем до maxSeqLength если нужно
        if (currentIds.shape[1] > this.config.maxSeqLength) {
          const startIdx = currentIds.shape[1] - this.config.maxSeqLength;
          const newData = new Float32Array(currentIds.shape[0] * this.config.maxSeqLength);
          for (let b = 0; b < currentIds.shape[0]; b++) {
            for (let s = 0; s < this.config.maxSeqLength; s++) {
              newData[b * this.config.maxSeqLength + s] =
                currentIds.data[b * currentIds.shape[1] + startIdx + s];
            }
          }
          currentIds = new Tensor(newData, [currentIds.shape[0], this.config.maxSeqLength], {
            dtype: DType.Int32,
          });
        }

        // Forward pass
        const logits = this.forward(currentIds);

        // Берём logits последнего токена
        const lastLogits = this.getLastLogits(logits);

        // Применяем temperature
        let scaledLogits = lastLogits;
        if (temperature !== 1.0) {
          scaledLogits = lastLogits.div(temperature);
        }

        // Применяем repetition penalty
        if (repetitionPenalty !== 1.0) {
          scaledLogits = this.applyRepetitionPenalty(
            scaledLogits,
            currentIds,
            repetitionPenalty
          );
        }

        // Sampling
        let nextToken: number;
        if (doSample) {
          nextToken = this.sample(scaledLogits, topK, topP);
        } else {
          nextToken = this.argmax(scaledLogits);
        }

        // Проверяем EOS
        if (eosTokenId !== undefined && nextToken === eosTokenId) {
          break;
        }

        // Добавляем токен
        currentIds = this.appendToken(currentIds, nextToken);
      }

      return currentIds;
    });
  }

  /**
   * Получает logits последнего токена
   */
  private getLastLogits(logits: Tensor): Tensor {
    const batchSize = logits.shape[0];
    const seqLen = logits.shape[1];
    const vocabSize = logits.shape[2];

    const lastLogitsData = new Float32Array(batchSize * vocabSize);
    for (let b = 0; b < batchSize; b++) {
      for (let v = 0; v < vocabSize; v++) {
        lastLogitsData[b * vocabSize + v] = logits.data[(b * seqLen + seqLen - 1) * vocabSize + v];
      }
    }

    return new Tensor(lastLogitsData, [batchSize, vocabSize]);
  }

  /**
   * Применяет repetition penalty
   */
  private applyRepetitionPenalty(
    logits: Tensor,
    inputIds: Tensor,
    penalty: number
  ): Tensor {
    const batchSize = logits.shape[0];
    const vocabSize = logits.shape[1];
    const seqLen = inputIds.shape[1];

    const penalizedData = new Float32Array(logits.data);

    for (let b = 0; b < batchSize; b++) {
      const seenTokens = new Set<number>();
      for (let s = 0; s < seqLen; s++) {
        seenTokens.add(Math.floor(inputIds.data[b * seqLen + s]));
      }

      for (const token of seenTokens) {
        const idx = b * vocabSize + token;
        if (penalizedData[idx] > 0) {
          penalizedData[idx] /= penalty;
        } else {
          penalizedData[idx] *= penalty;
        }
      }
    }

    return new Tensor(penalizedData, [...logits.shape]);
  }

  /**
   * Sampling с top-k и top-p
   */
  private sample(logits: Tensor, topK: number, topP: number): number {
    const vocabSize = logits.shape[1];

    // Получаем вероятности для первого элемента батча
    const logitsArray = Array.from(logits.data.slice(0, vocabSize));

    // Top-k filtering
    let indices = logitsArray.map((_, i) => i);
    if (topK > 0 && topK < vocabSize) {
      indices.sort((a, b) => logitsArray[b] - logitsArray[a]);
      indices = indices.slice(0, topK);
    }

    // Softmax
    let maxLogit = -Infinity;
    for (const i of indices) {
      if (logitsArray[i] > maxLogit) maxLogit = logitsArray[i];
    }

    const probs: number[] = [];
    let sumProb = 0;
    for (const i of indices) {
      const prob = Math.exp(logitsArray[i] - maxLogit);
      probs.push(prob);
      sumProb += prob;
    }

    // Нормализация
    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumProb;
    }

    // Top-p (nucleus) filtering
    if (topP < 1.0) {
      const sortedIndices = indices
        .map((idx, i) => ({ idx, prob: probs[i] }))
        .sort((a, b) => b.prob - a.prob);

      let cumProb = 0;
      const filteredIndices: number[] = [];
      const filteredProbs: number[] = [];

      for (const { idx, prob } of sortedIndices) {
        if (cumProb >= topP && filteredIndices.length > 0) break;
        filteredIndices.push(idx);
        filteredProbs.push(prob);
        cumProb += prob;
      }

      // Перенормализация
      const totalProb = filteredProbs.reduce((a, b) => a + b, 0);
      for (let i = 0; i < filteredProbs.length; i++) {
        filteredProbs[i] /= totalProb;
      }

      indices = filteredIndices;
      probs.length = 0;
      probs.push(...filteredProbs);
    }

    // Sampling
    const r = Math.random();
    let cumProb = 0;
    for (let i = 0; i < indices.length; i++) {
      cumProb += probs[i];
      if (r < cumProb) {
        return indices[i];
      }
    }

    return indices[indices.length - 1];
  }

  /**
   * Argmax (greedy decoding)
   */
  private argmax(logits: Tensor): number {
    const vocabSize = logits.shape[1];
    let maxIdx = 0;
    let maxVal = logits.data[0];

    for (let i = 1; i < vocabSize; i++) {
      if (logits.data[i] > maxVal) {
        maxVal = logits.data[i];
        maxIdx = i;
      }
    }

    return maxIdx;
  }

  /**
   * Добавляет токен к последовательности
   */
  private appendToken(inputIds: Tensor, token: number): Tensor {
    const batchSize = inputIds.shape[0];
    const seqLen = inputIds.shape[1];

    const newData = new Float32Array(batchSize * (seqLen + 1));
    for (let b = 0; b < batchSize; b++) {
      for (let s = 0; s < seqLen; s++) {
        newData[b * (seqLen + 1) + s] = inputIds.data[b * seqLen + s];
      }
      newData[b * (seqLen + 1) + seqLen] = token;
    }

    return new Tensor(newData, [batchSize, seqLen + 1], { dtype: DType.Int32 });
  }

  /**
   * Генерирует текст из строки
   */
  async generateText(
    tokenizer: Tokenizer,
    prompt: string,
    config: GenerationConfig = {}
  ): Promise<string> {
    const encoded = tokenizer.encode(prompt, true);
    const inputIds = tensor([encoded.inputIds], { dtype: DType.Int32 });

    const outputIds = this.generate(inputIds, {
      eosTokenId: tokenizer.eosTokenId,
      ...config,
    });

    return tokenizer.decode(Array.from(outputIds.data).map((x) => Math.floor(x)), true);
  }

  /**
   * Получает конфигурацию модели
   */
  getConfig(): GPTConfig {
    return { ...this.config };
  }
}

/**
 * Создаёт маленькую GPT модель для тестирования
 */
export function createSmallGPT(vocabSize: number = 1000): GPT {
  return new GPT({
    vocabSize,
    hiddenSize: 128,
    numHeads: 4,
    numLayers: 2,
    intermediateSize: 512,
    maxSeqLength: 256,
    dropoutProb: 0.1,
  });
}

/**
 * Создаёт среднюю GPT модель
 */
export function createMediumGPT(vocabSize: number = 10000): GPT {
  return new GPT({
    vocabSize,
    hiddenSize: 512,
    numHeads: 8,
    numLayers: 6,
    intermediateSize: 2048,
    maxSeqLength: 512,
    dropoutProb: 0.1,
  });
}

/**
 * Создаёт большую GPT модель
 */
export function createLargeGPT(vocabSize: number = 50000): GPT {
  return new GPT({
    vocabSize,
    hiddenSize: 1024,
    numHeads: 16,
    numLayers: 12,
    intermediateSize: 4096,
    maxSeqLength: 1024,
    dropoutProb: 0.1,
  });
}
