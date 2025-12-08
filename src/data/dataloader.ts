/**
 * @fileoverview Классы Dataset и DataLoader для работы с данными
 * @description Утилиты для батчинга, shuffle и итерации по данным
 */

import { Tensor, tensor, stack } from '../core/tensor';

/**
 * Абстрактный базовый класс для датасетов
 * Аналог torch.utils.data.Dataset в PyTorch
 * 
 * @example
 * class MyDataset extends Dataset<{input: Tensor, target: Tensor}> {
 *   constructor(data, labels) {
 *     super();
 *     this.data = data;
 *     this.labels = labels;
 *   }
 *   
 *   get length() { return this.data.length; }
 *   getItem(idx) { return { input: this.data[idx], target: this.labels[idx] }; }
 * }
 */
export abstract class Dataset<T = { input: Tensor; target: Tensor }> {
  /**
   * Возвращает количество элементов в датасете
   */
  abstract get length(): number;

  /**
   * Возвращает элемент по индексу
   * @param index - Индекс элемента
   * @returns Элемент датасета
   */
  abstract getItem(index: number): T;

  /**
   * Итератор по датасету
   */
  *[Symbol.iterator](): Generator<T> {
    for (let i = 0; i < this.length; i++) {
      yield this.getItem(i);
    }
  }
}

/**
 * Датасет из тензоров
 * Простой случай когда данные уже в виде тензоров
 */
export class TensorDataset extends Dataset<{ input: Tensor; target: Tensor }> {
  private inputs: Tensor;
  private targets: Tensor;

  /**
   * Создаёт TensorDataset
   * @param inputs - Входные данные [N, ...]
   * @param targets - Целевые значения [N, ...]
   */
  constructor(inputs: Tensor, targets: Tensor) {
    super();
    if (inputs.shape[0] !== targets.shape[0]) {
      throw new Error('Inputs and targets must have the same first dimension');
    }
    this.inputs = inputs;
    this.targets = targets;
  }

  get length(): number {
    return this.inputs.shape[0];
  }

  getItem(index: number): { input: Tensor; target: Tensor } {
    return {
      input: this.inputs.getRow(index),
      target: this.targets.getRow(index),
    };
  }
}

/**
 * Датасет из массивов
 */
export class ArrayDataset extends Dataset<{ input: Tensor; target: Tensor }> {
  private inputs: number[][];
  private targets: number[][];

  constructor(inputs: number[][], targets: number[][]) {
    super();
    if (inputs.length !== targets.length) {
      throw new Error('Inputs and targets must have the same length');
    }
    this.inputs = inputs;
    this.targets = targets;
  }

  get length(): number {
    return this.inputs.length;
  }

  getItem(index: number): { input: Tensor; target: Tensor } {
    return {
      input: tensor(this.inputs[index]),
      target: tensor(this.targets[index]),
    };
  }
}

/**
 * Интерфейс для batch
 */
export interface Batch {
  input: Tensor;
  target: Tensor;
}

/**
 * DataLoader для итерации по батчам данных
 * Аналог torch.utils.data.DataLoader в PyTorch
 * 
 * @example
 * const loader = new DataLoader(dataset, { batchSize: 32, shuffle: true });
 * for (const batch of loader) {
 *   const output = model.forward(batch.input);
 *   const loss = criterion.forward(output, batch.target);
 * }
 */
export class DataLoader {
  private dataset: Dataset;
  /** Размер батча */
  readonly batchSize: number;
  /** Перемешивать данные */
  readonly shuffle: boolean;
  /** Отбрасывать неполный последний батч */
  readonly dropLast: boolean;

  /**
   * Создаёт DataLoader
   * @param dataset - Датасет
   * @param options - Опции загрузчика
   */
  constructor(
    dataset: Dataset,
    options: {
      batchSize?: number;
      shuffle?: boolean;
      dropLast?: boolean;
    } = {}
  ) {
    this.dataset = dataset;
    this.batchSize = options.batchSize ?? 1;
    this.shuffle = options.shuffle ?? false;
    this.dropLast = options.dropLast ?? false;
  }

  /**
   * Количество батчей
   */
  get length(): number {
    if (this.dropLast) {
      return Math.floor(this.dataset.length / this.batchSize);
    }
    return Math.ceil(this.dataset.length / this.batchSize);
  }

  /**
   * Итератор по батчам
   */
  *[Symbol.iterator](): Generator<Batch> {
    // Создаём индексы
    const indices: number[] = [];
    for (let i = 0; i < this.dataset.length; i++) {
      indices.push(i);
    }

    // Перемешиваем если нужно
    if (this.shuffle) {
      shuffleArray(indices);
    }

    // Итерируем по батчам
    for (let i = 0; i < indices.length; i += this.batchSize) {
      const batchIndices = indices.slice(i, i + this.batchSize);

      // Пропускаем неполный последний батч если dropLast
      if (this.dropLast && batchIndices.length < this.batchSize) {
        continue;
      }

      // Собираем батч
      const inputs: Tensor[] = [];
      const targets: Tensor[] = [];

      for (const idx of batchIndices) {
        const item = this.dataset.getItem(idx);
        inputs.push((item as { input: Tensor; target: Tensor }).input);
        targets.push((item as { input: Tensor; target: Tensor }).target);
      }

      yield {
        input: stack(inputs, 0),
        target: stack(targets, 0),
      };
    }
  }

  /**
   * Преобразует DataLoader в массив батчей
   */
  toArray(): Batch[] {
    return Array.from(this);
  }
}

/**
 * Перемешивает массив in-place (Fisher-Yates shuffle)
 * @param array - Массив для перемешивания
 */
function shuffleArray<T>(array: T[]): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

/**
 * Разделяет датасет на train/test
 * @param dataset - Датасет
 * @param trainRatio - Доля train данных (0-1)
 * @param shuffle - Перемешать перед разделением
 * @returns [trainDataset, testDataset]
 */
export function trainTestSplit<T>(
  dataset: Dataset<T>,
  trainRatio: number = 0.8,
  shuffle: boolean = true
): [Dataset<T>, Dataset<T>] {
  const indices: number[] = [];
  for (let i = 0; i < dataset.length; i++) {
    indices.push(i);
  }

  if (shuffle) {
    shuffleArray(indices);
  }

  const splitIdx = Math.floor(dataset.length * trainRatio);
  const trainIndices = indices.slice(0, splitIdx);
  const testIndices = indices.slice(splitIdx);

  return [
    new SubsetDataset(dataset, trainIndices),
    new SubsetDataset(dataset, testIndices),
  ];
}

/**
 * Subset датасета по индексам
 */
class SubsetDataset<T> extends Dataset<T> {
  private dataset: Dataset<T>;
  private indices: number[];

  constructor(dataset: Dataset<T>, indices: number[]) {
    super();
    this.dataset = dataset;
    this.indices = indices;
  }

  get length(): number {
    return this.indices.length;
  }

  getItem(index: number): T {
    return this.dataset.getItem(this.indices[index]);
  }
}

/**
 * Конкатенация датасетов
 */
export class ConcatDataset<T> extends Dataset<T> {
  private datasets: Dataset<T>[];
  private cumulativeSizes: number[];

  constructor(...datasets: Dataset<T>[]) {
    super();
    this.datasets = datasets;
    this.cumulativeSizes = [];

    let total = 0;
    for (const ds of datasets) {
      total += ds.length;
      this.cumulativeSizes.push(total);
    }
  }

  get length(): number {
    return this.cumulativeSizes[this.cumulativeSizes.length - 1] || 0;
  }

  getItem(index: number): T {
    let datasetIdx = 0;
    for (let i = 0; i < this.cumulativeSizes.length; i++) {
      if (index < this.cumulativeSizes[i]) {
        datasetIdx = i;
        break;
      }
    }

    const prevSize = datasetIdx > 0 ? this.cumulativeSizes[datasetIdx - 1] : 0;
    return this.datasets[datasetIdx].getItem(index - prevSize);
  }
}
