/**
 * @fileoverview Классы Dataset и DataLoader для работы с данными
 * @description Утилиты для батчинга, shuffle и итерации по данным
 */

import { Tensor, tensor, stack } from '../core/tensor';

// ============================================
// БАЗОВЫЕ КЛАССЫ
// ============================================

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

// ============================================
// ПОТОКОВЫЙ DATASET (STREAMING)
// ============================================

/**
 * Конфигурация потоковой загрузки
 */
export interface StreamConfig {
  /** URL источника данных */
  url: string;
  /** Заголовки запроса */
  headers?: Record<string, string>;
  /** Размер буфера */
  bufferSize?: number;
  /** Трансформация данных */
  transform?: (data: any) => { input: any; target: any };
}

/**
 * Потоковый Dataset для загрузки данных по мере необходимости
 */
export class StreamingDataset extends Dataset<{ input: Tensor; target: Tensor }> {
  private config: StreamConfig;
  private buffer: Array<{ input: any; target: any }> = [];
  private isLoading: boolean = false;
  private totalLoaded: number = 0;

  constructor(config: StreamConfig) {
    super();
    this.config = {
      bufferSize: 1000,
      ...config,
    };
  }

  get length(): number {
    return Math.max(this.buffer.length, this.totalLoaded, 1000);
  }

  /**
   * Начинает потоковую загрузку
   */
  async startStreaming(): Promise<void> {
    if (this.isLoading) return;
    this.isLoading = true;

    try {
      const response = await fetch(this.config.url, {
        headers: this.config.headers,
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              let item = JSON.parse(line);
              if (this.config.transform) {
                item = this.config.transform(item);
              }
              this.buffer.push(item);
              this.totalLoaded++;

              if (this.buffer.length > this.config.bufferSize!) {
                this.buffer.shift();
              }
            } catch {
              // Skip invalid JSON
            }
          }
        }
      }
    } finally {
      this.isLoading = false;
    }
  }

  getItem(index: number): { input: Tensor; target: Tensor } {
    const idx = index % Math.max(this.buffer.length, 1);
    const item = this.buffer[idx];

    if (!item) {
      return {
        input: tensor([0]),
        target: tensor([0]),
      };
    }

    const input = Array.isArray(item.input) ? tensor(item.input) : tensor([item.input]);
    const target = Array.isArray(item.target) ? tensor(item.target) : tensor([item.target]);

    return { input, target };
  }

  /**
   * Асинхронный итератор для потоковых батчей
   */
  async *streamBatches(batchSize: number): AsyncGenerator<{ input: Tensor; target: Tensor }> {
    while (this.isLoading || this.buffer.length >= batchSize) {
      if (this.buffer.length < batchSize) {
        await new Promise(resolve => setTimeout(resolve, 100));
        continue;
      }

      const batch = this.buffer.splice(0, batchSize);
      const inputs: Tensor[] = [];
      const targets: Tensor[] = [];

      for (const item of batch) {
        inputs.push(Array.isArray(item.input) ? tensor(item.input) : tensor([item.input]));
        targets.push(Array.isArray(item.target) ? tensor(item.target) : tensor([item.target]));
      }

      yield {
        input: stack(inputs, 0),
        target: stack(targets, 0),
      };
    }
  }
}

// ============================================
// HUGGINGFACE DATASET
// ============================================

/**
 * Конфигурация HuggingFace
 */
export interface HuggingFaceConfig {
  /** Название датасета */
  dataset: string;
  /** Сплит (train, test, validation) */
  split?: string;
  /** Колонки для загрузки */
  columns?: string[];
  /** Максимальное количество примеров */
  maxSamples?: number;
  /** API токен */
  token?: string;
}

/**
 * Dataset для загрузки данных из HuggingFace Hub
 */
export class HuggingFaceDataset extends Dataset<{ input: Tensor; target: Tensor }> {
  private config: HuggingFaceConfig;
  private data: any[] = [];
  private loaded: boolean = false;

  constructor(config: HuggingFaceConfig) {
    super();
    this.config = {
      split: 'train',
      maxSamples: 1000,
      ...config,
    };
  }

  get length(): number {
    return this.data.length || 1;
  }

  /**
   * Загружает данные из HuggingFace
   */
  async load(): Promise<void> {
    if (this.loaded) return;

    const baseUrl = 'https://datasets-server.huggingface.co';
    const url = `${baseUrl}/rows?dataset=${this.config.dataset}&split=${this.config.split}&offset=0&length=${this.config.maxSamples}`;

    const headers: Record<string, string> = {};
    if (this.config.token) {
      headers['Authorization'] = `Bearer ${this.config.token}`;
    }

    try {
      const response = await fetch(url, { headers });
      if (!response.ok) {
        throw new Error(`Failed to load: ${response.status}`);
      }

      const result = await response.json();
      this.data = (result.rows || []).map((row: any) => row.row);
      this.loaded = true;
      console.log(`Loaded ${this.data.length} samples from ${this.config.dataset}`);
    } catch (error) {
      console.error('Error loading HuggingFace dataset:', error);
      throw error;
    }
  }

  getItem(index: number): { input: Tensor; target: Tensor } {
    const item = this.data[index];
    if (!item) {
      return { input: tensor([0]), target: tensor([0]) };
    }

    // Автоматически определяем поля
    const keys = Object.keys(item);
    let input: any = item[keys[0]];
    let target: any = item.label ?? item[keys[1]] ?? 0;

    const inputTensor = Array.isArray(input)
      ? tensor(input)
      : typeof input === 'number'
        ? tensor([input])
        : tensor([0]);

    const targetTensor = typeof target === 'number'
      ? tensor([target])
      : Array.isArray(target)
        ? tensor(target)
        : tensor([0]);

    return { input: inputTensor, target: targetTensor };
  }
}

// ============================================
// ГЕНЕРАТОР ДАННЫХ
// ============================================

/**
 * Генератор синтетических данных
 */
export class DataGenerator {
  // Имена (русские и английские)
  private static readonly FIRST_NAMES_RU = [
    'Александр', 'Михаил', 'Дмитрий', 'Иван', 'Сергей', 'Андрей', 'Алексей',
    'Мария', 'Анна', 'Елена', 'Ольга', 'Наталья', 'Екатерина', 'Татьяна',
  ];
  private static readonly LAST_NAMES_RU = [
    'Иванов', 'Петров', 'Сидоров', 'Козлов', 'Новиков', 'Морозов', 'Волков',
    'Иванова', 'Петрова', 'Сидорова', 'Козлова', 'Новикова', 'Морозова',
  ];
  private static readonly FIRST_NAMES_EN = [
    'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Joseph',
    'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan',
  ];
  private static readonly LAST_NAMES_EN = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Davis', 'Miller',
    'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White',
  ];

  private static readonly STREETS_RU = [
    'Ленина', 'Пушкина', 'Гагарина', 'Мира', 'Советская', 'Центральная',
  ];
  private static readonly CITIES_RU = [
    'Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань',
  ];
  private static readonly STREETS_EN = [
    'Main', 'Oak', 'Pine', 'Maple', 'Cedar', 'First', 'Second',
  ];
  private static readonly CITIES_EN = [
    'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
  ];

  /**
   * Генерирует случайное имя
   */
  static generateName(lang: 'ru' | 'en' = 'en'): { firstName: string; lastName: string; fullName: string } {
    const firstNames = lang === 'ru' ? this.FIRST_NAMES_RU : this.FIRST_NAMES_EN;
    const lastNames = lang === 'ru' ? this.LAST_NAMES_RU : this.LAST_NAMES_EN;

    const firstName = firstNames[Math.floor(Math.random() * firstNames.length)];
    const lastName = lastNames[Math.floor(Math.random() * lastNames.length)];

    return {
      firstName,
      lastName,
      fullName: `${firstName} ${lastName}`,
    };
  }

  /**
   * Генерирует случайный адрес
   */
  static generateAddress(lang: 'ru' | 'en' = 'en'): {
    street: string;
    city: string;
    zipCode: string;
    fullAddress: string;
  } {
    const streets = lang === 'ru' ? this.STREETS_RU : this.STREETS_EN;
    const cities = lang === 'ru' ? this.CITIES_RU : this.CITIES_EN;

    const street = streets[Math.floor(Math.random() * streets.length)];
    const city = cities[Math.floor(Math.random() * cities.length)];
    const number = Math.floor(Math.random() * 200) + 1;
    const zipCode = String(Math.floor(Math.random() * 900000) + 100000);

    const fullAddress = lang === 'ru'
      ? `ул. ${street}, д. ${number}, ${city}, ${zipCode}`
      : `${number} ${street} St, ${city}, ${zipCode}`;

    return { street, city, zipCode, fullAddress };
  }

  /**
   * Генерирует изображение с текстом
   */
  static generateTextImage(
    text: string,
    width: number = 28,
    height: number = 28
  ): Float32Array {
    const image = new Float32Array(width * height);

    // Простая растеризация текста
    const charWidth = Math.floor(width / Math.max(text.length, 1));
    const charHeight = height;

    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      const startX = i * charWidth;

      // Простой паттерн на основе charCode
      for (let y = 0; y < charHeight; y++) {
        for (let x = 0; x < charWidth && (startX + x) < width; x++) {
          const idx = y * width + (startX + x);
          // Используем биты charCode для создания паттерна
          const bit = (charCode >> (x % 8)) & 1;
          const yPattern = ((charCode >> (y % 8)) & 1);
          image[idx] = (bit ^ yPattern) * 0.8 + 0.1;
        }
      }
    }

    return image;
  }

  /**
   * Генерирует изображение с эмодзи
   */
  static generateEmojiImage(
    emojiIndex: number,
    width: number = 28,
    height: number = 28
  ): Float32Array {
    const image = new Float32Array(width * height);
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 3;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const dx = x - centerX;
        const dy = y - centerY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const idx = y * width + x;

        if (dist < radius) {
          // Круглое лицо
          image[idx] = 0.9;

          // Глаза
          const eyeY = centerY - radius * 0.3;
          const leftEyeX = centerX - radius * 0.35;
          const rightEyeX = centerX + radius * 0.35;

          if (Math.abs(y - eyeY) < 2 && (Math.abs(x - leftEyeX) < 2 || Math.abs(x - rightEyeX) < 2)) {
            image[idx] = 0.1;
          }

          // Рот (разный для разных эмодзи)
          const mouthY = centerY + radius * 0.3;
          if (Math.abs(y - mouthY) < 2 && Math.abs(x - centerX) < radius * 0.4) {
            if (emojiIndex % 3 === 0) {
              // Улыбка
              if (y > mouthY - (Math.abs(x - centerX) * 0.2)) {
                image[idx] = 0.1;
              }
            } else if (emojiIndex % 3 === 1) {
              // Грусть
              if (y < mouthY + (Math.abs(x - centerX) * 0.2)) {
                image[idx] = 0.1;
              }
            } else {
              // Нейтральный
              image[idx] = 0.1;
            }
          }
        }
      }
    }

    return image;
  }

  /**
   * Генерирует числовую последовательность
   */
  static generateSequence(
    type: 'arithmetic' | 'geometric' | 'fibonacci' | 'square' | 'random',
    length: number = 10
  ): { sequence: number[]; nextValue: number; step?: number } {
    let sequence: number[] = [];
    let nextValue: number;
    let step: number | undefined;

    switch (type) {
      case 'arithmetic':
        step = Math.floor(Math.random() * 10) + 1;
        const start = Math.floor(Math.random() * 20);
        for (let i = 0; i < length; i++) {
          sequence.push(start + i * step);
        }
        nextValue = start + length * step;
        break;

      case 'geometric':
        step = Math.floor(Math.random() * 3) + 2;
        let gStart = Math.floor(Math.random() * 5) + 1;
        for (let i = 0; i < length; i++) {
          sequence.push(gStart * Math.pow(step, i));
        }
        nextValue = gStart * Math.pow(step, length);
        break;

      case 'fibonacci':
        sequence = [1, 1];
        for (let i = 2; i < length; i++) {
          sequence.push(sequence[i - 1] + sequence[i - 2]);
        }
        nextValue = sequence[length - 1] + sequence[length - 2];
        break;

      case 'square':
        for (let i = 1; i <= length; i++) {
          sequence.push(i * i);
        }
        nextValue = (length + 1) * (length + 1);
        break;

      case 'random':
      default:
        step = Math.floor(Math.random() * 10) + 1;
        for (let i = 0; i < length; i++) {
          sequence.push(Math.floor(Math.random() * 100));
        }
        nextValue = Math.floor(Math.random() * 100);
        break;
    }

    return { sequence, nextValue, step };
  }
}

/**
 * Dataset для числовых последовательностей
 */
export class SequenceDataset extends Dataset<{ input: Tensor; target: Tensor }> {
  private sequences: Array<{ input: number[]; target: number }> = [];

  constructor(
    numSamples: number = 1000,
    seqLength: number = 5,
    types: Array<'arithmetic' | 'geometric' | 'fibonacci' | 'square'> = ['arithmetic']
  ) {
    super();

    for (let i = 0; i < numSamples; i++) {
      const type = types[Math.floor(Math.random() * types.length)];
      const { sequence, nextValue } = DataGenerator.generateSequence(type, seqLength + 1);

      this.sequences.push({
        input: sequence.slice(0, seqLength),
        target: nextValue,
      });
    }
  }

  get length(): number {
    return this.sequences.length;
  }

  getItem(index: number): { input: Tensor; target: Tensor } {
    const item = this.sequences[index];
    return {
      input: tensor(item.input),
      target: tensor([item.target]),
    };
  }
}

/**
 * Dataset с генерируемыми именами
 */
export class NameDataset extends Dataset<{ input: Tensor; target: Tensor }> {
  private names: Array<{ chars: number[]; label: number }> = [];
  private maxLength: number;

  constructor(
    numSamples: number = 1000,
    maxLength: number = 20,
    lang: 'ru' | 'en' | 'mixed' = 'mixed'
  ) {
    super();
    this.maxLength = maxLength;

    for (let i = 0; i < numSamples; i++) {
      const useLang = lang === 'mixed'
        ? (Math.random() > 0.5 ? 'ru' : 'en')
        : lang;

      const { fullName } = DataGenerator.generateName(useLang);
      const chars = this.encodeString(fullName);

      this.names.push({
        chars,
        label: useLang === 'ru' ? 1 : 0,
      });
    }
  }

  private encodeString(str: string): number[] {
    const encoded: number[] = [];
    for (let i = 0; i < this.maxLength; i++) {
      if (i < str.length) {
        encoded.push(str.charCodeAt(i) / 256);
      } else {
        encoded.push(0);
      }
    }
    return encoded;
  }

  get length(): number {
    return this.names.length;
  }

  getItem(index: number): { input: Tensor; target: Tensor } {
    const item = this.names[index];
    return {
      input: tensor(item.chars),
      target: tensor([item.label]),
    };
  }
}

// ============================================
// УТИЛИТЫ ДЛЯ ЗАГРУЗКИ
// ============================================

/**
 * Загружает JSON данные из URL
 */
export async function loadJson(url: string): Promise<any> {
  const response = await fetch(url);
  return response.json();
}

/**
 * Загружает JSONL (JSON Lines) из URL
 */
export async function loadJsonl(url: string): Promise<any[]> {
  const response = await fetch(url);
  const text = await response.text();
  return text
    .split('\n')
    .filter(line => line.trim())
    .map(line => JSON.parse(line));
}

/**
 * Загружает CSV из URL
 */
export async function loadCsv(url: string): Promise<Record<string, string | number>[]> {
  const response = await fetch(url);
  const text = await response.text();
  const lines = text.split('\n');
  const headers = lines[0].split(',').map(h => h.trim());

  return lines.slice(1).filter(l => l.trim()).map(line => {
    const values = line.split(',');
    const obj: Record<string, string | number> = {};
    headers.forEach((h, i) => {
      const val = values[i]?.trim();
      obj[h] = isNaN(Number(val)) ? val : Number(val);
    });
    return obj;
  });
}

/**
 * Создаёт DataLoader из HuggingFace датасета
 */
export async function createHuggingFaceLoader(
  datasetName: string,
  batchSize: number = 32,
  config: Partial<HuggingFaceConfig> = {}
): Promise<DataLoader> {
  const dataset = new HuggingFaceDataset({
    dataset: datasetName,
    ...config,
  });

  await dataset.load();

  return new DataLoader(dataset, { batchSize, shuffle: true });
}
