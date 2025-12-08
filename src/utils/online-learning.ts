/**
 * @fileoverview Online/Real-time Learning
 * @description Обучение модели во время работы (inference-time learning)
 *
 * Позволяет моделям адаптироваться в реальном времени к новым данным
 * без полной переобучения.
 */

import { Tensor, tensor, zeros } from '../core/tensor';
import { Module } from '../nn/module';
import { Optimizer, SGD, Adam } from '../optim/optimizer';

/**
 * Конфигурация онлайн-обучения
 */
export interface OnlineLearningConfig {
  /** Learning rate для онлайн-обновлений */
  lr?: number;
  /** Размер буфера для хранения недавних примеров */
  bufferSize?: number;
  /** Частота обновления (каждые N примеров) */
  updateFrequency?: number;
  /** Использовать адаптивный learning rate */
  adaptiveLR?: boolean;
  /** Минимальный confidence для обучения */
  confidenceThreshold?: number;
  /** Экспоненциальное затухание для старых примеров */
  decayRate?: number;
}

/**
 * Буфер опыта для онлайн-обучения
 */
class ExperienceBuffer<T> {
  private buffer: T[] = [];
  private maxSize: number;
  private weights: number[] = [];

  constructor(maxSize: number) {
    this.maxSize = maxSize;
  }

  push(item: T, weight: number = 1.0): void {
    if (this.buffer.length >= this.maxSize) {
      // Удаляем самый старый
      this.buffer.shift();
      this.weights.shift();
    }
    this.buffer.push(item);
    this.weights.push(weight);
  }

  sample(n: number): { items: T[]; weights: number[] } {
    if (n >= this.buffer.length) {
      return { items: [...this.buffer], weights: [...this.weights] };
    }

    const indices: number[] = [];
    const items: T[] = [];
    const weights: number[] = [];

    // Weighted sampling (prefer recent)
    const totalWeight = this.weights.reduce((a, b) => a + b, 0);
    const probs = this.weights.map(w => w / totalWeight);

    while (indices.length < n) {
      let r = Math.random();
      for (let i = 0; i < probs.length; i++) {
        r -= probs[i];
        if (r <= 0 && !indices.includes(i)) {
          indices.push(i);
          break;
        }
      }
    }

    for (const i of indices) {
      items.push(this.buffer[i]);
      weights.push(this.weights[i]);
    }

    return { items, weights };
  }

  get length(): number {
    return this.buffer.length;
  }

  clear(): void {
    this.buffer = [];
    this.weights = [];
  }

  /**
   * Применяет decay ко всем весам
   */
  applyDecay(rate: number): void {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] *= rate;
    }
  }
}

/**
 * Обёртка для онлайн-обучения модели
 */
export class OnlineLearner<T extends Module> {
  private model: T;
  private optimizer: Optimizer;
  private config: Required<OnlineLearningConfig>;
  private buffer: ExperienceBuffer<{ x: Tensor; y: Tensor }>;
  private stepCount: number = 0;
  private totalLoss: number = 0;
  private lossCount: number = 0;

  constructor(model: T, config: OnlineLearningConfig = {}) {
    this.model = model;
    this.config = {
      lr: 0.001,
      bufferSize: 100,
      updateFrequency: 1,
      adaptiveLR: true,
      confidenceThreshold: 0.5,
      decayRate: 0.99,
      ...config,
    };

    this.optimizer = new Adam(model.parameters(), this.config.lr);
    this.buffer = new ExperienceBuffer(this.config.bufferSize);
  }

  /**
   * Предсказание с опциональным обучением
   */
  predict(x: Tensor): Tensor {
    return this.model.forward(x);
  }

  /**
   * Добавляет пример для обучения
   */
  addExample(x: Tensor, y: Tensor, importance: number = 1.0): void {
    this.buffer.push({ x: x.clone(), y: y.clone() }, importance);
    this.stepCount++;

    // Применяем decay
    if (this.stepCount % 10 === 0) {
      this.buffer.applyDecay(this.config.decayRate);
    }

    // Обновляем модель если нужно
    if (this.stepCount % this.config.updateFrequency === 0) {
      this.update();
    }
  }

  /**
   * Предсказание + обратная связь (для supervised онлайн)
   */
  predictAndLearn(x: Tensor, y: Tensor): { prediction: Tensor; loss: number } {
    const prediction = this.predict(x);

    // Вычисляем loss для мониторинга
    const loss = prediction.sub(y).pow(2).mean().item();

    // Добавляем пример
    this.addExample(x, y);

    return { prediction, loss };
  }

  /**
   * Выполняет шаг обучения
   */
  private update(): void {
    if (this.buffer.length === 0) return;

    const batchSize = Math.min(8, this.buffer.length);
    const { items } = this.buffer.sample(batchSize);

    let totalLoss = 0;

    for (const { x, y } of items) {
      const pred = this.model.forward(x);
      const loss = pred.sub(y).pow(2).mean();

      this.optimizer.zeroGrad();
      loss.backward();
      this.optimizer.step();

      totalLoss += loss.item();
    }

    this.totalLoss += totalLoss / items.length;
    this.lossCount++;

    // Адаптивный LR
    if (this.config.adaptiveLR && this.lossCount > 10) {
      const avgLoss = this.totalLoss / this.lossCount;
      if (avgLoss < 0.01) {
        // Уменьшаем LR при хорошем результате
        this.optimizer.setLearningRate(this.config.lr * 0.5);
      }
    }
  }

  /**
   * Получает статистику обучения
   */
  getStats(): {
    stepCount: number;
    bufferSize: number;
    avgLoss: number;
  } {
    return {
      stepCount: this.stepCount,
      bufferSize: this.buffer.length,
      avgLoss: this.lossCount > 0 ? this.totalLoss / this.lossCount : 0,
    };
  }

  /**
   * Сбрасывает буфер
   */
  reset(): void {
    this.buffer.clear();
    this.stepCount = 0;
    this.totalLoss = 0;
    this.lossCount = 0;
  }

  /**
   * Получает модель
   */
  getModel(): T {
    return this.model;
  }
}

/**
 * Continual Learning - обучение на потоке задач без забывания
 */
export class ContinualLearner<T extends Module> {
  private model: T;
  private optimizer: Optimizer;
  private taskMemory: Map<string, { x: Tensor; y: Tensor }[]> = new Map();
  private memoryPerTask: number;
  private replayRatio: number;

  constructor(
    model: T,
    lr: number = 0.001,
    memoryPerTask: number = 50,
    replayRatio: number = 0.3
  ) {
    this.model = model;
    this.optimizer = new Adam(model.parameters(), lr);
    this.memoryPerTask = memoryPerTask;
    this.replayRatio = replayRatio;
  }

  /**
   * Обучает на новой задаче с replay старых
   */
  trainOnTask(
    taskId: string,
    data: { x: Tensor; y: Tensor }[],
    epochs: number = 10
  ): number[] {
    const losses: number[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let count = 0;

      // Текущие данные
      for (const { x, y } of data) {
        const pred = this.model.forward(x);
        const loss = pred.sub(y).pow(2).mean();

        this.optimizer.zeroGrad();
        loss.backward();
        this.optimizer.step();

        totalLoss += loss.item();
        count++;
      }

      // Replay старых задач
      const replayCount = Math.floor(data.length * this.replayRatio);
      for (const [id, memory] of this.taskMemory.entries()) {
        if (id === taskId) continue;

        const samples = this.sampleFromMemory(memory, Math.min(replayCount, memory.length));
        for (const { x, y } of samples) {
          const pred = this.model.forward(x);
          const loss = pred.sub(y).pow(2).mean();

          this.optimizer.zeroGrad();
          loss.backward();
          this.optimizer.step();

          totalLoss += loss.item();
          count++;
        }
      }

      losses.push(totalLoss / count);
    }

    // Сохраняем примеры в память
    this.updateMemory(taskId, data);

    return losses;
  }

  private sampleFromMemory(
    memory: { x: Tensor; y: Tensor }[],
    n: number
  ): { x: Tensor; y: Tensor }[] {
    const shuffled = [...memory].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, n);
  }

  private updateMemory(taskId: string, data: { x: Tensor; y: Tensor }[]): void {
    // Случайная выборка для памяти
    const samples = this.sampleFromMemory(data, this.memoryPerTask);
    this.taskMemory.set(taskId, samples.map(s => ({
      x: s.x.clone(),
      y: s.y.clone(),
    })));
  }

  /**
   * Оценивает модель на всех задачах
   */
  evaluate(tasks: Map<string, { x: Tensor; y: Tensor }[]>): Map<string, number> {
    const results = new Map<string, number>();

    for (const [taskId, data] of tasks.entries()) {
      let totalLoss = 0;
      for (const { x, y } of data) {
        const pred = this.model.forward(x);
        totalLoss += pred.sub(y).pow(2).mean().item();
      }
      results.set(taskId, totalLoss / data.length);
    }

    return results;
  }

  getModel(): T {
    return this.model;
  }
}

/**
 * Meta-Learning для быстрой адаптации (упрощённый MAML)
 */
export class MetaLearner<T extends Module> {
  private model: T;
  private metaOptimizer: Optimizer;
  private innerLR: number;
  private innerSteps: number;

  constructor(
    model: T,
    metaLR: number = 0.001,
    innerLR: number = 0.01,
    innerSteps: number = 5
  ) {
    this.model = model;
    this.metaOptimizer = new Adam(model.parameters(), metaLR);
    this.innerLR = innerLR;
    this.innerSteps = innerSteps;
  }

  /**
   * Быстрая адаптация на новую задачу (few-shot)
   */
  adapt(
    supportX: Tensor,
    supportY: Tensor,
    steps?: number
  ): void {
    const numSteps = steps ?? this.innerSteps;

    // Создаём временный оптимизатор для адаптации
    const adaptOptimizer = new SGD(this.model.parameters(), this.innerLR);

    for (let i = 0; i < numSteps; i++) {
      const pred = this.model.forward(supportX);
      const loss = pred.sub(supportY).pow(2).mean();

      adaptOptimizer.zeroGrad();
      loss.backward();
      adaptOptimizer.step();
    }
  }

  /**
   * Предсказание после адаптации
   */
  predictAfterAdapt(
    supportX: Tensor,
    supportY: Tensor,
    queryX: Tensor
  ): Tensor {
    // Сохраняем текущие параметры
    const savedState = this.model.stateDict();

    // Адаптируемся
    this.adapt(supportX, supportY);

    // Предсказываем
    const pred = this.model.forward(queryX);

    // Восстанавливаем параметры
    this.model.loadStateDict(savedState);

    return pred;
  }

  getModel(): T {
    return this.model;
  }
}

/**
 * Self-Training - модель обучается на своих уверенных предсказаниях
 */
export class SelfTrainer<T extends Module> {
  private model: T;
  private optimizer: Optimizer;
  private confidenceThreshold: number;
  private pseudoLabeledData: { x: Tensor; y: Tensor }[] = [];

  constructor(
    model: T,
    lr: number = 0.001,
    confidenceThreshold: number = 0.9
  ) {
    this.model = model;
    this.optimizer = new Adam(model.parameters(), lr);
    this.confidenceThreshold = confidenceThreshold;
  }

  /**
   * Добавляет unlabeled данные для pseudo-labeling
   */
  addUnlabeled(x: Tensor): { accepted: boolean; pseudoLabel?: Tensor } {
    const pred = this.model.forward(x);

    // Для классификации: проверяем уверенность (max prob)
    const maxProb = Math.max(...pred.data);

    if (maxProb >= this.confidenceThreshold) {
      this.pseudoLabeledData.push({
        x: x.clone(),
        y: pred.clone(),
      });
      return { accepted: true, pseudoLabel: pred };
    }

    return { accepted: false };
  }

  /**
   * Обучение на pseudo-labeled данных
   */
  trainOnPseudoLabels(epochs: number = 5): number {
    if (this.pseudoLabeledData.length === 0) return 0;

    let totalLoss = 0;
    let count = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const { x, y } of this.pseudoLabeledData) {
        const pred = this.model.forward(x);
        const loss = pred.sub(y).pow(2).mean();

        this.optimizer.zeroGrad();
        loss.backward();
        this.optimizer.step();

        totalLoss += loss.item();
        count++;
      }
    }

    // Очищаем после обучения
    this.pseudoLabeledData = [];

    return totalLoss / count;
  }

  getModel(): T {
    return this.model;
  }
}
