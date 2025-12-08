/**
 * @fileoverview Утилиты для дообучения (Fine-tuning)
 * @description Инструменты для эффективного дообучения предобученных моделей
 */

import { Tensor, tensor, zeros, randn } from '../core/tensor';
import { DType } from '../core/dtype';
import { Module, Parameter } from '../nn/module';
import { Linear } from '../nn/layers';
import { Optimizer, Adam, AdamW } from '../optim/optimizer';
import { Dataset, DataLoader } from '../data/dataloader';

/**
 * Стратегия дообучения
 */
export enum FineTuneStrategy {
  /** Обучаем все параметры */
  FULL = 'full',
  /** Замораживаем backbone, обучаем только голову */
  HEAD_ONLY = 'head_only',
  /** Постепенно размораживаем слои */
  GRADUAL_UNFREEZE = 'gradual_unfreeze',
  /** LoRA (Low-Rank Adaptation) */
  LORA = 'lora',
  /** Prefix tuning */
  PREFIX_TUNING = 'prefix_tuning',
  /** Adapter tuning */
  ADAPTER = 'adapter',
}

/**
 * Конфигурация дообучения
 */
export interface FineTuneConfig {
  /** Стратегия дообучения */
  strategy: FineTuneStrategy;
  /** Learning rate */
  lr: number;
  /** Weight decay */
  weightDecay?: number;
  /** Количество эпох */
  epochs: number;
  /** Размер батча */
  batchSize: number;
  /** Warmup steps */
  warmupSteps?: number;
  /** Максимальное количество шагов */
  maxSteps?: number;
  /** Градиентный клиппинг */
  maxGradNorm?: number;
  /** Аккумуляция градиентов */
  gradientAccumulationSteps?: number;
  /** LoRA rank (для LORA стратегии) */
  loraRank?: number;
  /** LoRA alpha */
  loraAlpha?: number;
  /** Слои для LoRA */
  loraTargetModules?: string[];
  /** Prefix length (для PREFIX_TUNING) */
  prefixLength?: number;
  /** Размер скрытого слоя адаптера */
  adapterSize?: number;
  /** Callback после каждого шага */
  onStep?: (step: number, loss: number) => void;
  /** Callback после каждой эпохи */
  onEpoch?: (epoch: number, metrics: Record<string, number>) => void;
}

/**
 * LoRA слой (Low-Rank Adaptation)
 * Добавляет низкоранговую матрицу к весам: W' = W + BA
 */
export class LoRALayer extends Module {
  private rank: number;
  private alpha: number;
  private scaling: number;

  readonly loraA: Parameter;
  readonly loraB: Parameter;

  private originalWeight: Tensor;

  constructor(
    originalModule: Linear,
    rank: number = 8,
    alpha: number = 16
  ) {
    super();

    this.rank = rank;
    this.alpha = alpha;
    this.scaling = alpha / rank;

    const { inFeatures, outFeatures } = originalModule;

    // LoRA матрицы: A [r, in], B [out, r]
    // Инициализация: A ~ N(0, 1), B = 0
    const loraAData = randn([rank, inFeatures]).div(Math.sqrt(rank));
    const loraBData = zeros([outFeatures, rank]);

    this.loraA = new Parameter(loraAData);
    this.loraB = new Parameter(loraBData);

    this.registerParameter('lora_A', this.loraA);
    this.registerParameter('lora_B', this.loraB);

    // Сохраняем оригинальные веса (замороженные)
    this.originalWeight = originalModule.weight.data.clone();
  }

  forward(x: Tensor): Tensor {
    // Оригинальный проход: x @ W^T
    const originalOut = x.matmul(this.originalWeight.T);

    // LoRA добавка: x @ A^T @ B^T * scaling
    const loraOut = x.matmul(this.loraA.data.T).matmul(this.loraB.data.T);

    return originalOut.add(loraOut.mul(this.scaling));
  }

  /**
   * Объединяет LoRA веса с оригинальными
   */
  mergeWeights(): Tensor {
    // W' = W + scaling * B @ A
    const delta = this.loraB.data.matmul(this.loraA.data).mul(this.scaling);
    return this.originalWeight.add(delta);
  }
}

/**
 * Adapter слой
 * Добавляет bottleneck слой между трансформер блоками
 */
export class AdapterLayer extends Module {
  private downProj: Linear;
  private upProj: Linear;

  constructor(hiddenSize: number, adapterSize: number = 64) {
    super();

    this.downProj = new Linear(hiddenSize, adapterSize);
    this.upProj = new Linear(adapterSize, hiddenSize);

    this.registerModule('down_proj', this.downProj);
    this.registerModule('up_proj', this.upProj);

    // Инициализация близкой к identity
    // up_proj инициализируется нулями для начального residual
    const upWeightData = this.upProj.weight.data;
    for (let i = 0; i < upWeightData.size; i++) {
      upWeightData.data[i] = 0;
    }
  }

  forward(x: Tensor): Tensor {
    // Down projection -> ReLU -> Up projection
    let hidden = this.downProj.forward(x);

    // ReLU
    const reluData = new Float32Array(hidden.size);
    for (let i = 0; i < hidden.size; i++) {
      reluData[i] = Math.max(0, hidden.data[i]);
    }
    hidden = new Tensor(reluData, [...hidden.shape], { requiresGrad: true });

    // Up projection
    const output = this.upProj.forward(hidden);

    // Residual connection
    return x.add(output);
  }
}

/**
 * Prefix Tuning слой
 * Добавляет обучаемые prefix токены к ключам и значениям attention
 */
export class PrefixTuningLayer extends Module {
  private prefixLength: number;
  private hiddenSize: number;

  readonly prefixKeys: Parameter;
  readonly prefixValues: Parameter;

  constructor(prefixLength: number, hiddenSize: number, numHeads: number) {
    super();

    this.prefixLength = prefixLength;
    this.hiddenSize = hiddenSize;

    // Инициализируем prefix токены
    const prefixKeysData = randn([prefixLength, hiddenSize]).div(Math.sqrt(hiddenSize));
    const prefixValuesData = randn([prefixLength, hiddenSize]).div(Math.sqrt(hiddenSize));

    this.prefixKeys = new Parameter(prefixKeysData);
    this.prefixValues = new Parameter(prefixValuesData);

    this.registerParameter('prefix_keys', this.prefixKeys);
    this.registerParameter('prefix_values', this.prefixValues);
  }

  /**
   * Получает prefix для ключей
   */
  getKeys(batchSize: number): Tensor {
    // Расширяем до [batch, prefix_len, hidden]
    const data = new Float32Array(batchSize * this.prefixLength * this.hiddenSize);
    for (let b = 0; b < batchSize; b++) {
      for (let i = 0; i < this.prefixLength * this.hiddenSize; i++) {
        data[b * this.prefixLength * this.hiddenSize + i] = this.prefixKeys.data.data[i];
      }
    }
    return new Tensor(data, [batchSize, this.prefixLength, this.hiddenSize]);
  }

  /**
   * Получает prefix для значений
   */
  getValues(batchSize: number): Tensor {
    const data = new Float32Array(batchSize * this.prefixLength * this.hiddenSize);
    for (let b = 0; b < batchSize; b++) {
      for (let i = 0; i < this.prefixLength * this.hiddenSize; i++) {
        data[b * this.prefixLength * this.hiddenSize + i] = this.prefixValues.data.data[i];
      }
    }
    return new Tensor(data, [batchSize, this.prefixLength, this.hiddenSize]);
  }

  forward(x: Tensor): Tensor {
    // Этот метод не используется напрямую
    // Prefix добавляются в attention механизм
    return x;
  }
}

/**
 * Trainer для дообучения
 */
export class FineTuneTrainer {
  private model: Module;
  private config: FineTuneConfig;
  private optimizer: Optimizer | null = null;
  private globalStep: number = 0;
  private loraLayers: Map<string, LoRALayer> = new Map();
  private adapterLayers: Map<string, AdapterLayer> = new Map();

  constructor(model: Module, config: FineTuneConfig) {
    this.model = model;
    this.config = {
      weightDecay: 0.01,
      warmupSteps: 0,
      maxGradNorm: 1.0,
      gradientAccumulationSteps: 1,
      loraRank: 8,
      loraAlpha: 16,
      prefixLength: 10,
      adapterSize: 64,
      ...config,
    };
  }

  /**
   * Подготавливает модель к дообучению
   */
  prepare(): void {
    switch (this.config.strategy) {
      case FineTuneStrategy.FULL:
        // Все параметры обучаемы
        break;

      case FineTuneStrategy.HEAD_ONLY:
        this.freezeBackbone();
        break;

      case FineTuneStrategy.LORA:
        this.applyLoRA();
        break;

      case FineTuneStrategy.ADAPTER:
        this.applyAdapters();
        break;

      case FineTuneStrategy.PREFIX_TUNING:
        this.applyPrefixTuning();
        break;
    }

    // Создаём оптимизатор только для обучаемых параметров
    const trainableParams = this.getTrainableParameters();
    this.optimizer = new AdamW(trainableParams, this.config.lr, {
      weightDecay: this.config.weightDecay,
    });
  }

  /**
   * Замораживает backbone модели
   */
  private freezeBackbone(): void {
    // Замораживаем все параметры кроме последних слоёв
    const params = Array.from(this.model.parameters());
    const totalLayers = params.length;
    const unfreezeCount = Math.max(1, Math.floor(totalLayers * 0.1)); // Оставляем 10% слоёв

    for (let i = 0; i < totalLayers - unfreezeCount; i++) {
      (params[i] as any).requiresGrad = false;
    }
  }

  /**
   * Применяет LoRA к модели
   */
  private applyLoRA(): void {
    const targetModules = this.config.loraTargetModules || ['query_proj', 'value_proj'];

    for (const [name, module] of this.model.namedModules()) {
      // Проверяем, соответствует ли имя целевым модулям
      const isTarget = targetModules.some((target) => name.includes(target));

      if (isTarget && module instanceof Linear) {
        const loraLayer = new LoRALayer(
          module,
          this.config.loraRank,
          this.config.loraAlpha
        );
        this.loraLayers.set(name, loraLayer);

        // Замораживаем оригинальные веса
        module.weight.requiresGrad = false;
        if (module.biasParam) {
          module.biasParam.requiresGrad = false;
        }
      }
    }
  }

  /**
   * Применяет Adapter к модели
   */
  private applyAdapters(): void {
    // Добавляем адаптеры после каждого трансформер блока
    // Это упрощённая версия - в реальности нужно модифицировать forward
  }

  /**
   * Применяет Prefix Tuning
   */
  private applyPrefixTuning(): void {
    // Добавляем prefix слои для attention
    // Это упрощённая версия
  }

  /**
   * Получает обучаемые параметры
   */
  private getTrainableParameters(): Parameter[] {
    const params: Parameter[] = [];

    switch (this.config.strategy) {
      case FineTuneStrategy.LORA:
        // Только LoRA параметры
        for (const loraLayer of this.loraLayers.values()) {
          params.push(loraLayer.loraA, loraLayer.loraB);
        }
        break;

      default:
        // Все параметры с requiresGrad=true
        for (const param of this.model.parameters()) {
          if ((param as any).requiresGrad !== false) {
            params.push(param);
          }
        }
    }

    return params;
  }

  /**
   * Выполняет один шаг обучения
   */
  step(input: Tensor, target: Tensor, lossFn: (pred: Tensor, target: Tensor) => Tensor): number {
    if (!this.optimizer) {
      throw new Error('Trainer not prepared. Call prepare() first.');
    }

    this.model.train();

    // Forward pass
    const output = this.model.forward(input);
    const loss = lossFn(output, target);

    // Backward pass
    this.optimizer.zeroGrad();
    loss.backward();

    // Gradient clipping
    if (this.config.maxGradNorm) {
      this.clipGradients(this.config.maxGradNorm);
    }

    // Optimizer step
    if ((this.globalStep + 1) % this.config.gradientAccumulationSteps! === 0) {
      this.optimizer.step();
    }

    this.globalStep++;

    const lossValue = loss.item();

    // Callback
    if (this.config.onStep) {
      this.config.onStep(this.globalStep, lossValue);
    }

    return lossValue;
  }

  /**
   * Обучает на датасете
   */
  async train(
    trainDataset: Dataset<{ input: Tensor; target: Tensor }>,
    lossFn: (pred: Tensor, target: Tensor) => Tensor,
    valDataset?: Dataset<{ input: Tensor; target: Tensor }>
  ): Promise<void> {
    if (!this.optimizer) {
      throw new Error('Trainer not prepared. Call prepare() first.');
    }

    const dataLoader = new DataLoader(trainDataset, {
      batchSize: this.config.batchSize,
      shuffle: true,
    });

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      let totalLoss = 0;
      let numBatches = 0;

      for (const batch of dataLoader) {
        const { input, target } = batch as unknown as { input: Tensor; target: Tensor };
        const loss = this.step(input, target, lossFn);
        totalLoss += loss;
        numBatches++;

        // Проверяем maxSteps
        if (this.config.maxSteps && this.globalStep >= this.config.maxSteps) {
          break;
        }
      }

      const avgLoss = totalLoss / numBatches;
      const metrics: Record<string, number> = { loss: avgLoss };

      // Валидация
      if (valDataset) {
        const valLoss = this.evaluate(valDataset, lossFn);
        metrics.valLoss = valLoss;
      }

      // Callback
      if (this.config.onEpoch) {
        this.config.onEpoch(epoch, metrics);
      }

      // Проверяем maxSteps
      if (this.config.maxSteps && this.globalStep >= this.config.maxSteps) {
        break;
      }
    }
  }

  /**
   * Оценивает модель на валидационном датасете
   */
  evaluate(
    dataset: Dataset<{ input: Tensor; target: Tensor }>,
    lossFn: (pred: Tensor, target: Tensor) => Tensor
  ): number {
    this.model.eval();

    const dataLoader = new DataLoader(dataset, {
      batchSize: this.config.batchSize,
      shuffle: false,
    });

    let totalLoss = 0;
    let numBatches = 0;

    for (const batch of dataLoader) {
      const { input, target } = batch as unknown as { input: Tensor; target: Tensor };
      const output = this.model.forward(input);
      const loss = lossFn(output, target);
      totalLoss += loss.item();
      numBatches++;
    }

    return totalLoss / numBatches;
  }

  /**
   * Клиппинг градиентов
   */
  private clipGradients(maxNorm: number): void {
    let totalNorm = 0;

    for (const param of this.getTrainableParameters()) {
      if (param.data.grad) {
        for (let i = 0; i < param.data.grad.size; i++) {
          totalNorm += param.data.grad.data[i] * param.data.grad.data[i];
        }
      }
    }

    totalNorm = Math.sqrt(totalNorm);

    if (totalNorm > maxNorm) {
      const scale = maxNorm / (totalNorm + 1e-6);
      for (const param of this.getTrainableParameters()) {
        if (param.data.grad) {
          for (let i = 0; i < param.data.grad.size; i++) {
            param.data.grad.data[i] *= scale;
          }
        }
      }
    }
  }

  /**
   * Объединяет LoRA веса с оригинальными (для inference)
   */
  mergeLoRA(): void {
    for (const [name, loraLayer] of this.loraLayers) {
      const mergedWeight = loraLayer.mergeWeights();
      // Обновляем веса в оригинальном модуле
      // Это упрощённая версия
    }
    this.loraLayers.clear();
  }

  /**
   * Получает текущий learning rate
   */
  getLearningRate(): number {
    return this.optimizer?.lr || this.config.lr;
  }

  /**
   * Устанавливает learning rate
   */
  setLearningRate(lr: number): void {
    if (this.optimizer) {
      this.optimizer.lr = lr;
    }
  }

  /**
   * Получает количество обучаемых параметров
   */
  getTrainableParamsCount(): number {
    let count = 0;
    for (const param of this.getTrainableParameters()) {
      count += param.data.size;
    }
    return count;
  }

  /**
   * Получает текущий шаг
   */
  getGlobalStep(): number {
    return this.globalStep;
  }
}

/**
 * Создаёт LoRA версию модели
 */
export function createLoRAModel(
  model: Module,
  rank: number = 8,
  alpha: number = 16,
  targetModules: string[] = ['query_proj', 'value_proj']
): Module {
  const trainer = new FineTuneTrainer(model, {
    strategy: FineTuneStrategy.LORA,
    lr: 1e-4,
    epochs: 1,
    batchSize: 1,
    loraRank: rank,
    loraAlpha: alpha,
    loraTargetModules: targetModules,
  });

  trainer.prepare();
  return model;
}

/**
 * Быстрое дообучение модели
 */
export async function fineTune(
  model: Module,
  trainData: { input: Tensor; target: Tensor }[],
  config: Partial<FineTuneConfig> = {}
): Promise<Module> {
  const fullConfig: FineTuneConfig = {
    strategy: FineTuneStrategy.FULL,
    lr: 1e-4,
    epochs: 3,
    batchSize: 8,
    ...config,
  };

  const trainer = new FineTuneTrainer(model, fullConfig);
  trainer.prepare();

  // Создаём простой датасет
  const dataset = {
    length: trainData.length,
    getItem: (i: number) => trainData[i],
    *[Symbol.iterator]() {
      for (let i = 0; i < trainData.length; i++) {
        yield trainData[i];
      }
    },
  } as Dataset<{ input: Tensor; target: Tensor }>;

  // MSE loss по умолчанию
  const lossFn = (pred: Tensor, target: Tensor) => {
    let sum = 0;
    for (let i = 0; i < pred.size; i++) {
      const diff = pred.data[i] - target.data[i];
      sum += diff * diff;
    }
    return tensor([[sum / pred.size]], { requiresGrad: true });
  };

  await trainer.train(dataset, lossFn);

  return model;
}
