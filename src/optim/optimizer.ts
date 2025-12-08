/**
 * @fileoverview Оптимизаторы для обучения нейронных сетей
 * @description Реализация SGD, Adam, RMSprop и других оптимизаторов
 */

import { Tensor, zeros, scalar } from '../core/tensor';
import { Parameter } from '../nn/module';

/**
 * Базовый класс для всех оптимизаторов
 * Аналог torch.optim.Optimizer в PyTorch
 */
export abstract class Optimizer {
  /** Параметры для оптимизации */
  protected params: Parameter[];
  /** Скорость обучения */
  lr: number;
  /** Текущий шаг оптимизации */
  protected stepCount: number = 0;

  /**
   * Создаёт оптимизатор
   * @param params - Параметры для оптимизации
   * @param lr - Скорость обучения
   */
  constructor(params: Parameter[] | IterableIterator<Parameter>, lr: number) {
    this.params = Array.isArray(params) ? params : Array.from(params);
    this.lr = lr;
  }

  /**
   * Выполняет шаг оптимизации
   */
  abstract step(): void;

  /**
   * Обнуляет градиенты всех параметров
   */
  zeroGrad(): void {
    for (const param of this.params) {
      param.data.zeroGrad();
    }
  }

  /**
   * Возвращает состояние оптимизатора (для сериализации)
   */
  stateDict(): Map<string, unknown> {
    return new Map([
      ['lr', this.lr],
      ['step', this.stepCount],
    ]);
  }

  /**
   * Загружает состояние оптимизатора
   */
  loadStateDict(state: Map<string, unknown>): void {
    if (state.has('lr')) this.lr = state.get('lr') as number;
    if (state.has('step')) this.stepCount = state.get('step') as number;
  }
}

/**
 * Stochastic Gradient Descent оптимизатор
 * Поддерживает momentum и weight decay
 * 
 * @example
 * const optimizer = new SGD(model.parameters(), 0.01, { momentum: 0.9 });
 * optimizer.zeroGrad();
 * loss.backward();
 * optimizer.step();
 */
export class SGD extends Optimizer {
  /** Momentum коэффициент */
  readonly momentum: number;
  /** Weight decay (L2 регуляризация) */
  readonly weightDecay: number;
  /** Dampening для momentum */
  readonly dampening: number;
  /** Использовать Nesterov momentum */
  readonly nesterov: boolean;
  /** Буферы для momentum */
  private velocities: Map<Parameter, Float32Array> = new Map();

  /**
   * Создаёт SGD оптимизатор
   * @param params - Параметры для оптимизации
   * @param lr - Скорость обучения
   * @param options - Дополнительные параметры
   */
  constructor(
    params: Parameter[] | IterableIterator<Parameter>,
    lr: number,
    options: {
      momentum?: number;
      weightDecay?: number;
      dampening?: number;
      nesterov?: boolean;
    } = {}
  ) {
    super(params, lr);
    this.momentum = options.momentum ?? 0;
    this.weightDecay = options.weightDecay ?? 0;
    this.dampening = options.dampening ?? 0;
    this.nesterov = options.nesterov ?? false;

    if (this.nesterov && (this.momentum <= 0 || this.dampening !== 0)) {
      throw new Error('Nesterov momentum requires a momentum and zero dampening');
    }
  }

  /**
   * Выполняет шаг SGD
   */
  step(): void {
    for (const param of this.params) {
      if (!param.data.grad) continue;

      const grad = param.data.grad;
      let d_p = grad.clone();

      // Weight decay
      if (this.weightDecay !== 0) {
        d_p = d_p.add(param.data.mul(this.weightDecay));
      }

      // Momentum
      if (this.momentum !== 0) {
        let velocity = this.velocities.get(param);

        if (!velocity) {
          velocity = new Float32Array(param.data.size);
          for (let i = 0; i < param.data.size; i++) {
            velocity[i] = d_p.data[i];
          }
          this.velocities.set(param, velocity);
        } else {
          for (let i = 0; i < param.data.size; i++) {
            velocity[i] = this.momentum * velocity[i] + (1 - this.dampening) * d_p.data[i];
          }
        }

        if (this.nesterov) {
          for (let i = 0; i < d_p.size; i++) {
            (d_p.data as Float32Array)[i] = d_p.data[i] + this.momentum * velocity[i];
          }
        } else {
          for (let i = 0; i < d_p.size; i++) {
            (d_p.data as Float32Array)[i] = velocity[i];
          }
        }
      }

      // Обновляем параметры
      for (let i = 0; i < param.data.size; i++) {
        (param.data.data as Float32Array)[i] -= this.lr * d_p.data[i];
      }
    }

    this.stepCount++;
  }

  toString(): string {
    return `SGD(lr=${this.lr}, momentum=${this.momentum}, weight_decay=${this.weightDecay})`;
  }
}

/**
 * Adam оптимизатор
 * Adaptive Moment Estimation
 * 
 * @example
 * const optimizer = new Adam(model.parameters(), 0.001);
 */
export class Adam extends Optimizer {
  /** Beta1 для первого момента */
  readonly beta1: number;
  /** Beta2 для второго момента */
  readonly beta2: number;
  /** Epsilon для численной стабильности */
  readonly eps: number;
  /** Weight decay */
  readonly weightDecay: number;
  /** Использовать AMSGrad */
  readonly amsgrad: boolean;

  /** Буферы первого момента */
  private m: Map<Parameter, Float32Array> = new Map();
  /** Буферы второго момента */
  private v: Map<Parameter, Float32Array> = new Map();
  /** Буферы максимума v для AMSGrad */
  private vMax: Map<Parameter, Float32Array> = new Map();

  /**
   * Создаёт Adam оптимизатор
   * @param params - Параметры для оптимизации
   * @param lr - Скорость обучения (по умолчанию 0.001)
   * @param options - Дополнительные параметры
   */
  constructor(
    params: Parameter[] | IterableIterator<Parameter>,
    lr: number = 0.001,
    options: {
      beta1?: number;
      beta2?: number;
      eps?: number;
      weightDecay?: number;
      amsgrad?: boolean;
    } = {}
  ) {
    super(params, lr);
    this.beta1 = options.beta1 ?? 0.9;
    this.beta2 = options.beta2 ?? 0.999;
    this.eps = options.eps ?? 1e-8;
    this.weightDecay = options.weightDecay ?? 0;
    this.amsgrad = options.amsgrad ?? false;
  }

  /**
   * Выполняет шаг Adam
   */
  step(): void {
    this.stepCount++;

    for (const param of this.params) {
      if (!param.data.grad) continue;

      let grad = param.data.grad;

      // Weight decay (L2)
      if (this.weightDecay !== 0) {
        grad = grad.add(param.data.mul(this.weightDecay));
      }

      // Инициализация буферов
      if (!this.m.has(param)) {
        this.m.set(param, new Float32Array(param.data.size));
        this.v.set(param, new Float32Array(param.data.size));
        if (this.amsgrad) {
          this.vMax.set(param, new Float32Array(param.data.size));
        }
      }

      const m = this.m.get(param)!;
      const v = this.v.get(param)!;

      // Обновляем моменты
      for (let i = 0; i < param.data.size; i++) {
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * grad.data[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * grad.data[i] * grad.data[i];
      }

      // Bias correction
      const biasCorrection1 = 1 - Math.pow(this.beta1, this.stepCount);
      const biasCorrection2 = 1 - Math.pow(this.beta2, this.stepCount);

      // Обновляем параметры
      if (this.amsgrad) {
        const vMax = this.vMax.get(param)!;
        for (let i = 0; i < param.data.size; i++) {
          vMax[i] = Math.max(vMax[i], v[i]);
          const mHat = m[i] / biasCorrection1;
          const vMaxHat = vMax[i] / biasCorrection2;
          (param.data.data as Float32Array)[i] -= this.lr * mHat / (Math.sqrt(vMaxHat) + this.eps);
        }
      } else {
        for (let i = 0; i < param.data.size; i++) {
          const mHat = m[i] / biasCorrection1;
          const vHat = v[i] / biasCorrection2;
          (param.data.data as Float32Array)[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
        }
      }
    }
  }

  toString(): string {
    return `Adam(lr=${this.lr}, betas=(${this.beta1}, ${this.beta2}), eps=${this.eps})`;
  }
}

/**
 * AdamW оптимизатор
 * Adam с decoupled weight decay
 */
export class AdamW extends Adam {
  /**
   * Выполняет шаг AdamW
   */
  step(): void {
    this.stepCount++;

    for (const param of this.params) {
      if (!param.data.grad) continue;

      const grad = param.data.grad;

      // Инициализация буферов
      if (!this['m'].has(param)) {
        this['m'].set(param, new Float32Array(param.data.size));
        this['v'].set(param, new Float32Array(param.data.size));
      }

      const m = this['m'].get(param)!;
      const v = this['v'].get(param)!;

      // Decoupled weight decay
      if (this.weightDecay !== 0) {
        for (let i = 0; i < param.data.size; i++) {
          (param.data.data as Float32Array)[i] *= (1 - this.lr * this.weightDecay);
        }
      }

      // Обновляем моменты
      for (let i = 0; i < param.data.size; i++) {
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * grad.data[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * grad.data[i] * grad.data[i];
      }

      // Bias correction
      const biasCorrection1 = 1 - Math.pow(this.beta1, this.stepCount);
      const biasCorrection2 = 1 - Math.pow(this.beta2, this.stepCount);

      // Обновляем параметры
      for (let i = 0; i < param.data.size; i++) {
        const mHat = m[i] / biasCorrection1;
        const vHat = v[i] / biasCorrection2;
        (param.data.data as Float32Array)[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
      }
    }
  }

  toString(): string {
    return `AdamW(lr=${this.lr}, betas=(${this.beta1}, ${this.beta2}), weight_decay=${this.weightDecay})`;
  }
}

/**
 * RMSprop оптимизатор
 * Root Mean Square Propagation
 */
export class RMSprop extends Optimizer {
  /** Alpha для скользящего среднего */
  readonly alpha: number;
  /** Epsilon для стабильности */
  readonly eps: number;
  /** Weight decay */
  readonly weightDecay: number;
  /** Momentum */
  readonly momentum: number;
  /** Centered RMSprop */
  readonly centered: boolean;

  /** Буферы скользящего среднего квадратов */
  private squareAvg: Map<Parameter, Float32Array> = new Map();
  /** Буферы градиентного среднего (для centered) */
  private gradAvg: Map<Parameter, Float32Array> = new Map();
  /** Буферы momentum */
  private momentumBuf: Map<Parameter, Float32Array> = new Map();

  constructor(
    params: Parameter[] | IterableIterator<Parameter>,
    lr: number = 0.01,
    options: {
      alpha?: number;
      eps?: number;
      weightDecay?: number;
      momentum?: number;
      centered?: boolean;
    } = {}
  ) {
    super(params, lr);
    this.alpha = options.alpha ?? 0.99;
    this.eps = options.eps ?? 1e-8;
    this.weightDecay = options.weightDecay ?? 0;
    this.momentum = options.momentum ?? 0;
    this.centered = options.centered ?? false;
  }

  step(): void {
    for (const param of this.params) {
      if (!param.data.grad) continue;

      let grad = param.data.grad;

      // Weight decay
      if (this.weightDecay !== 0) {
        grad = grad.add(param.data.mul(this.weightDecay));
      }

      // Инициализация буферов
      if (!this.squareAvg.has(param)) {
        this.squareAvg.set(param, new Float32Array(param.data.size));
        if (this.centered) {
          this.gradAvg.set(param, new Float32Array(param.data.size));
        }
        if (this.momentum > 0) {
          this.momentumBuf.set(param, new Float32Array(param.data.size));
        }
      }

      const squareAvg = this.squareAvg.get(param)!;

      // Обновляем скользящее среднее квадратов
      for (let i = 0; i < param.data.size; i++) {
        squareAvg[i] = this.alpha * squareAvg[i] + (1 - this.alpha) * grad.data[i] * grad.data[i];
      }

      let avg: Float32Array;
      if (this.centered) {
        const gradAvg = this.gradAvg.get(param)!;
        for (let i = 0; i < param.data.size; i++) {
          gradAvg[i] = this.alpha * gradAvg[i] + (1 - this.alpha) * grad.data[i];
        }
        avg = new Float32Array(param.data.size);
        for (let i = 0; i < param.data.size; i++) {
          avg[i] = squareAvg[i] - gradAvg[i] * gradAvg[i];
        }
      } else {
        avg = squareAvg;
      }

      if (this.momentum > 0) {
        const buf = this.momentumBuf.get(param)!;
        for (let i = 0; i < param.data.size; i++) {
          buf[i] = this.momentum * buf[i] + grad.data[i] / (Math.sqrt(avg[i]) + this.eps);
          (param.data.data as Float32Array)[i] -= this.lr * buf[i];
        }
      } else {
        for (let i = 0; i < param.data.size; i++) {
          (param.data.data as Float32Array)[i] -= this.lr * grad.data[i] / (Math.sqrt(avg[i]) + this.eps);
        }
      }
    }

    this.stepCount++;
  }

  toString(): string {
    return `RMSprop(lr=${this.lr}, alpha=${this.alpha}, eps=${this.eps})`;
  }
}

/**
 * Adagrad оптимизатор
 * Адаптивный learning rate для каждого параметра
 */
export class Adagrad extends Optimizer {
  readonly eps: number;
  readonly weightDecay: number;
  readonly lrDecay: number;
  readonly initialAccumulatorValue: number;

  private sumSquares: Map<Parameter, Float32Array> = new Map();

  constructor(
    params: Parameter[] | IterableIterator<Parameter>,
    lr: number = 0.01,
    options: {
      eps?: number;
      weightDecay?: number;
      lrDecay?: number;
      initialAccumulatorValue?: number;
    } = {}
  ) {
    super(params, lr);
    this.eps = options.eps ?? 1e-10;
    this.weightDecay = options.weightDecay ?? 0;
    this.lrDecay = options.lrDecay ?? 0;
    this.initialAccumulatorValue = options.initialAccumulatorValue ?? 0;
  }

  step(): void {
    this.stepCount++;
    const clr = this.lr / (1 + (this.stepCount - 1) * this.lrDecay);

    for (const param of this.params) {
      if (!param.data.grad) continue;

      let grad = param.data.grad;

      if (this.weightDecay !== 0) {
        grad = grad.add(param.data.mul(this.weightDecay));
      }

      if (!this.sumSquares.has(param)) {
        const ss = new Float32Array(param.data.size);
        ss.fill(this.initialAccumulatorValue);
        this.sumSquares.set(param, ss);
      }

      const sumSquares = this.sumSquares.get(param)!;

      for (let i = 0; i < param.data.size; i++) {
        sumSquares[i] += grad.data[i] * grad.data[i];
        (param.data.data as Float32Array)[i] -= clr * grad.data[i] / (Math.sqrt(sumSquares[i]) + this.eps);
      }
    }
  }

  toString(): string {
    return `Adagrad(lr=${this.lr}, lr_decay=${this.lrDecay}, eps=${this.eps})`;
  }
}

// ============================================
// LEARNING RATE SCHEDULERS
// ============================================

/**
 * Базовый класс для LR schedulers
 */
export abstract class LRScheduler {
  protected optimizer: Optimizer;
  protected lastEpoch: number = -1;
  protected baseLRs: number[];

  constructor(optimizer: Optimizer) {
    this.optimizer = optimizer;
    this.baseLRs = [optimizer.lr];
  }

  abstract getLR(): number[];

  step(epoch?: number): void {
    if (epoch === undefined) {
      this.lastEpoch++;
    } else {
      this.lastEpoch = epoch;
    }

    const newLRs = this.getLR();
    this.optimizer.lr = newLRs[0];
  }
}

/**
 * Step LR scheduler
 * Уменьшает LR каждые step_size эпох
 */
export class StepLR extends LRScheduler {
  readonly stepSize: number;
  readonly gamma: number;

  constructor(optimizer: Optimizer, stepSize: number, gamma: number = 0.1) {
    super(optimizer);
    this.stepSize = stepSize;
    this.gamma = gamma;
  }

  getLR(): number[] {
    const factor = Math.pow(this.gamma, Math.floor(this.lastEpoch / this.stepSize));
    return this.baseLRs.map(lr => lr * factor);
  }
}

/**
 * Exponential LR scheduler
 * LR *= gamma каждую эпоху
 */
export class ExponentialLR extends LRScheduler {
  readonly gamma: number;

  constructor(optimizer: Optimizer, gamma: number) {
    super(optimizer);
    this.gamma = gamma;
  }

  getLR(): number[] {
    return this.baseLRs.map(lr => lr * Math.pow(this.gamma, this.lastEpoch));
  }
}

/**
 * Cosine Annealing LR scheduler
 * Косинусное затухание LR
 */
export class CosineAnnealingLR extends LRScheduler {
  readonly tMax: number;
  readonly etaMin: number;

  constructor(optimizer: Optimizer, tMax: number, etaMin: number = 0) {
    super(optimizer);
    this.tMax = tMax;
    this.etaMin = etaMin;
  }

  getLR(): number[] {
    return this.baseLRs.map(baseLR => {
      return this.etaMin + (baseLR - this.etaMin) * (1 + Math.cos(Math.PI * this.lastEpoch / this.tMax)) / 2;
    });
  }
}

/**
 * Reduce LR On Plateau
 * Уменьшает LR когда метрика перестаёт улучшаться
 */
export class ReduceLROnPlateau {
  private optimizer: Optimizer;
  readonly factor: number;
  readonly patience: number;
  readonly threshold: number;
  readonly mode: 'min' | 'max';
  readonly cooldown: number;
  readonly minLR: number;

  private best: number;
  private numBadEpochs: number = 0;
  private cooldownCounter: number = 0;

  constructor(
    optimizer: Optimizer,
    options: {
      factor?: number;
      patience?: number;
      threshold?: number;
      mode?: 'min' | 'max';
      cooldown?: number;
      minLR?: number;
    } = {}
  ) {
    this.optimizer = optimizer;
    this.factor = options.factor ?? 0.1;
    this.patience = options.patience ?? 10;
    this.threshold = options.threshold ?? 1e-4;
    this.mode = options.mode ?? 'min';
    this.cooldown = options.cooldown ?? 0;
    this.minLR = options.minLR ?? 0;
    this.best = this.mode === 'min' ? Infinity : -Infinity;
  }

  step(metric: number): void {
    if (this.cooldownCounter > 0) {
      this.cooldownCounter--;
      this.numBadEpochs = 0;
      return;
    }

    const isImproved = this.mode === 'min'
      ? metric < this.best * (1 - this.threshold)
      : metric > this.best * (1 + this.threshold);

    if (isImproved) {
      this.best = metric;
      this.numBadEpochs = 0;
    } else {
      this.numBadEpochs++;
    }

    if (this.numBadEpochs > this.patience) {
      const newLR = Math.max(this.optimizer.lr * this.factor, this.minLR);
      this.optimizer.lr = newLR;
      this.cooldownCounter = this.cooldown;
      this.numBadEpochs = 0;
    }
  }
}
