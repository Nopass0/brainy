/**
 * @fileoverview Основной класс Tensor - ядро фреймворка Brainy
 * @description Многомерный массив с поддержкой автоматического дифференцирования,
 * broadcasting и оптимизированных математических операций
 */

import { DType, Shape, TypedArray, getTypedArrayConstructor, NestedArray } from './dtype';
import {
  computeSize,
  computeStrides,
  indicesToOffset,
  offsetToIndices,
  broadcastShapes,
  shapesEqual,
  inferShape,
  flattenArray,
  validateShape,
  squeezeShape,
  unsqueezeShape,
} from './shape';
import { GradNode, GradContext, isNoGradEnabled, backward as autoBackward } from './autograd';

/**
 * Опции для создания тензора
 */
export interface TensorOptions {
  /** Тип данных (по умолчанию Float32) */
  dtype?: DType;
  /** Требуется ли отслеживание градиентов */
  requiresGrad?: boolean;
}

/**
 * Основной класс Tensor
 * Представляет многомерный массив с поддержкой автоматического дифференцирования
 * 
 * @example
 * const t = new Tensor([[1, 2], [3, 4]]);
 * const sum = t.sum();
 * sum.backward();
 * console.log(t.grad); // градиенты
 */
export class Tensor {
  /** Данные тензора в плоском виде */
  readonly data: TypedArray;
  /** Форма тензора */
  readonly shape: readonly number[];
  /** Тип данных */
  readonly dtype: DType;
  /** Strides для индексации */
  readonly strides: readonly number[];
  /** Общее количество элементов */
  readonly size: number;
  /** Количество измерений (ранг тензора) */
  readonly ndim: number;
  /** Требуется ли отслеживание градиентов */
  readonly requiresGrad: boolean;
  /** Узел вычислительного графа для autograd */
  gradNode: GradNode | null = null;
  /** Накопленный градиент */
  grad: Tensor | null = null;

  /**
   * Создаёт новый тензор
   * @param data - Данные (вложенный массив, TypedArray или плоский массив + shape)
   * @param shape - Форма (если data - плоский массив или TypedArray)
   * @param options - Дополнительные опции
   */
  constructor(
    data: NestedArray | TypedArray | number[],
    shape?: Shape,
    options: TensorOptions = {}
  ) {
    const { dtype = DType.Float32, requiresGrad = false } = options;

    this.dtype = dtype;
    this.requiresGrad = requiresGrad;

    // Определяем форму и данные
    if (ArrayBuffer.isView(data)) {
      // TypedArray
      this.shape = shape ? [...shape] : [data.length];
      this.data = data as TypedArray;
    } else if (shape) {
      // Плоский массив + форма
      const TypedArrayClass = getTypedArrayConstructor(dtype);
      this.shape = [...shape];
      this.data = new TypedArrayClass(data as number[]);
    } else {
      // Вложенный массив - определяем форму автоматически
      this.shape = inferShape(data);
      const flat = flattenArray(data);
      const TypedArrayClass = getTypedArrayConstructor(dtype);
      this.data = new TypedArrayClass(flat);
    }

    validateShape(this.shape);
    this.strides = computeStrides(this.shape);
    this.size = computeSize(this.shape);
    this.ndim = this.shape.length;

    // Проверка размера данных
    if (this.data.length !== this.size) {
      throw new Error(
        `Data size ${this.data.length} doesn't match shape ${this.shape} (expected ${this.size})`
      );
    }
  }

  // ============================================
  // ИНДЕКСАЦИЯ И ДОСТУП К ЭЛЕМЕНТАМ
  // ============================================

  /**
   * Получает элемент по индексам
   * @param indices - Индексы по каждой оси
   * @returns Значение элемента
   */
  get(...indices: number[]): number {
    const offset = indicesToOffset(indices, this.strides as number[]);
    return this.data[offset];
  }

  /**
   * Устанавливает значение элемента по индексам
   * @param indices - Индексы, последний элемент - значение
   */
  set(...indicesAndValue: number[]): void {
    const value = indicesAndValue.pop()!;
    const offset = indicesToOffset(indicesAndValue, this.strides as number[]);
    (this.data as Float32Array)[offset] = value;
  }

  /**
   * Получает строку (row) из матрицы
   * @param index - Индекс строки
   * @returns Новый тензор
   */
  getRow(index: number): Tensor {
    if (this.ndim < 1) throw new Error('Cannot get row from scalar');
    const start = index * this.strides[0];
    const size = this.size / this.shape[0];
    const newShape = this.shape.slice(1);
    const newData = this.data.slice(start, start + size);
    return new Tensor(newData, newShape.length ? newShape : [1], { dtype: this.dtype });
  }

  /**
   * Преобразует тензор в обычный JavaScript массив
   * @returns Вложенный массив
   */
  toArray(): NestedArray {
    if (this.ndim === 0) {
      return this.data[0];
    }

    const buildArray = (offset: number, dim: number): NestedArray => {
      if (dim === this.ndim - 1) {
        const result: number[] = [];
        for (let i = 0; i < this.shape[dim]; i++) {
          result.push(this.data[offset + i]);
        }
        return result;
      }

      const result: NestedArray[] = [];
      for (let i = 0; i < this.shape[dim]; i++) {
        result.push(buildArray(offset + i * this.strides[dim], dim + 1));
      }
      return result;
    };

    return buildArray(0, 0);
  }

  /**
   * Возвращает скалярное значение (для тензоров размера 1)
   * @returns Число
   */
  item(): number {
    if (this.size !== 1) {
      throw new Error('item() can only be called on single-element tensors');
    }
    return this.data[0];
  }

  // ============================================
  // ОПЕРАЦИИ ИЗМЕНЕНИЯ ФОРМЫ
  // ============================================

  /**
   * Изменяет форму тензора без изменения данных
   * @param newShape - Новая форма (-1 для автоматического вычисления одной размерности)
   * @returns Новый тензор с изменённой формой
   */
  reshape(...newShape: number[]): Tensor {
    // Обрабатываем -1
    let unknownIdx = -1;
    let knownSize = 1;
    for (let i = 0; i < newShape.length; i++) {
      if (newShape[i] === -1) {
        if (unknownIdx !== -1) throw new Error('Can only have one -1 in reshape');
        unknownIdx = i;
      } else {
        knownSize *= newShape[i];
      }
    }

    if (unknownIdx !== -1) {
      newShape[unknownIdx] = this.size / knownSize;
    }

    if (computeSize(newShape) !== this.size) {
      throw new Error(`Cannot reshape tensor of size ${this.size} to shape [${newShape}]`);
    }

    const result = new Tensor(this.data, newShape, {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
    });

    // Autograd
    if (this.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveValue('originalShape', this.shape);
      result.gradNode = new GradNode(
        (gradOutput) => [gradOutput.reshape(...(this.shape as number[]))],
        [this],
        ctx
      );
    }

    return result;
  }

  /**
   * Выравнивает тензор в одномерный
   * @param startDim - Начальная размерность для выравнивания
   * @param endDim - Конечная размерность
   * @returns Выровненный тензор
   */
  flatten(startDim: number = 0, endDim: number = -1): Tensor {
    if (endDim === -1) endDim = this.ndim - 1;

    const newShape: number[] = [];
    let flattenedSize = 1;

    for (let i = 0; i < this.ndim; i++) {
      if (i < startDim) {
        newShape.push(this.shape[i]);
      } else if (i <= endDim) {
        flattenedSize *= this.shape[i];
      } else {
        if (flattenedSize > 0) {
          newShape.push(flattenedSize);
          flattenedSize = 0;
        }
        newShape.push(this.shape[i]);
      }
    }

    if (flattenedSize > 0) {
      newShape.push(flattenedSize);
    }

    return this.reshape(...newShape);
  }

  /**
   * Транспонирует тензор (меняет местами оси)
   * @param dim0 - Первая ось
   * @param dim1 - Вторая ось
   * @returns Транспонированный тензор
   */
  transpose(dim0: number = 0, dim1: number = 1): Tensor {
    if (this.ndim < 2) {
      return this.clone();
    }

    // Нормализуем отрицательные индексы
    if (dim0 < 0) dim0 += this.ndim;
    if (dim1 < 0) dim1 += this.ndim;

    const newShape = [...this.shape];
    [newShape[dim0], newShape[dim1]] = [newShape[dim1], newShape[dim0]];

    const newData = new (getTypedArrayConstructor(this.dtype))(this.size);
    const newStrides = computeStrides(newShape);

    // Перестановка данных
    for (let i = 0; i < this.size; i++) {
      const oldIndices = offsetToIndices(i, this.shape);
      [oldIndices[dim0], oldIndices[dim1]] = [oldIndices[dim1], oldIndices[dim0]];
      const newOffset = indicesToOffset(oldIndices, newStrides);
      newData[newOffset] = this.data[i];
    }

    const result = new Tensor(newData, newShape, {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
    });

    // Autograd
    if (this.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveValue('dim0', dim0);
      ctx.saveValue('dim1', dim1);
      result.gradNode = new GradNode(
        (gradOutput) => [gradOutput.transpose(dim0, dim1)],
        [this],
        ctx
      );
    }

    return result;
  }

  /**
   * Сокращённое транспонирование для матриц
   * @returns Транспонированная матрица
   */
  get T(): Tensor {
    return this.transpose(0, 1);
  }

  /**
   * Удаляет размерности равные 1
   * @param dim - Конкретная размерность для удаления (опционально)
   * @returns Сжатый тензор
   */
  squeeze(dim?: number): Tensor {
    const newShape = squeezeShape(this.shape, dim);
    return this.reshape(...newShape);
  }

  /**
   * Добавляет размерность 1 в указанную позицию
   * @param dim - Позиция для вставки
   * @returns Расширенный тензор
   */
  unsqueeze(dim: number): Tensor {
    const newShape = unsqueezeShape(this.shape, dim);
    return this.reshape(...newShape);
  }

  // ============================================
  // МАТЕМАТИЧЕСКИЕ ОПЕРАЦИИ
  // ============================================

  /**
   * Создаёт копию тензора
   * @returns Новый тензор с копией данных
   */
  clone(): Tensor {
    const TypedArrayClass = getTypedArrayConstructor(this.dtype);
    const newData = new TypedArrayClass(this.data);
    return new Tensor(newData, this.shape, {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
    });
  }

  /**
   * Выполняет поэлементную операцию с broadcasting
   * @param other - Другой тензор или число
   * @param op - Функция операции
   * @param gradFn - Функция для вычисления градиентов
   * @returns Результат операции
   */
  private binaryOp(
    other: Tensor | number,
    op: (a: number, b: number) => number,
    gradFn?: (gradOutput: Tensor, a: Tensor, b: Tensor) => [Tensor, Tensor]
  ): Tensor {
    const otherTensor = typeof other === 'number' ? scalar(other) : other;

    // Broadcasting
    const resultShape = broadcastShapes(this.shape, otherTensor.shape);
    if (!resultShape) {
      throw new Error(
        `Cannot broadcast shapes [${this.shape}] and [${otherTensor.shape}]`
      );
    }

    const resultSize = computeSize(resultShape);
    const TypedArrayClass = getTypedArrayConstructor(this.dtype);
    const resultData = new TypedArrayClass(resultSize);

    // Выполняем операцию с broadcast
    // Вычисляем strides для обоих тензоров с учётом broadcast
    const aStrides = computeStrides(this.shape);
    const bStrides = computeStrides(otherTensor.shape);

    for (let i = 0; i < resultSize; i++) {
      const indices = offsetToIndices(i, resultShape);

      // Broadcast индексы для каждого тензора
      // Для каждого тензора нужно пропустить ведущие измерения результата
      // если тензор имеет меньше измерений

      let aOffset = 0;
      const aDimOffset = resultShape.length - this.shape.length;
      for (let d = 0; d < this.shape.length; d++) {
        const resultIdx = indices[aDimOffset + d];
        const idx = this.shape[d] === 1 ? 0 : resultIdx;
        aOffset += idx * aStrides[d];
      }

      let bOffset = 0;
      const bDimOffset = resultShape.length - otherTensor.shape.length;
      for (let d = 0; d < otherTensor.shape.length; d++) {
        const resultIdx = indices[bDimOffset + d];
        const idx = otherTensor.shape[d] === 1 ? 0 : resultIdx;
        bOffset += idx * bStrides[d];
      }

      resultData[i] = op(this.data[aOffset], otherTensor.data[bOffset]);
    }

    const result = new Tensor(resultData, resultShape, {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad || otherTensor.requiresGrad,
    });

    // Autograd
    if ((this.requiresGrad || otherTensor.requiresGrad) && !isNoGradEnabled() && gradFn) {
      const ctx = new GradContext();
      ctx.saveTensors(this, otherTensor);
      result.gradNode = new GradNode(
        (gradOutput) => gradFn(gradOutput, this, otherTensor),
        [this, otherTensor],
        ctx
      );
    }

    return result;
  }

  /**
   * Сложение тензоров
   * @param other - Другой тензор или число
   * @returns Результат сложения
   */
  add(other: Tensor | number): Tensor {
    return this.binaryOp(other, (a, b) => a + b, (gradOutput, a, b) => {
      // Градиенты для сложения: просто пропускаем gradient, но нужно учесть broadcast
      let gradA = gradOutput;
      let gradB = gradOutput;

      // Суммируем по broadcast размерностям
      if (!shapesEqual(gradA.shape, a.shape)) {
        gradA = sumToShape(gradOutput, a.shape);
      }
      if (!shapesEqual(gradB.shape, b.shape)) {
        gradB = sumToShape(gradOutput, b.shape);
      }

      return [gradA, gradB];
    });
  }

  /**
   * Вычитание тензоров
   * @param other - Другой тензор или число  
   * @returns Результат вычитания
   */
  sub(other: Tensor | number): Tensor {
    return this.binaryOp(other, (a, b) => a - b, (gradOutput, a, b) => {
      let gradA = gradOutput;
      let gradB = gradOutput.neg();

      if (!shapesEqual(gradA.shape, a.shape)) {
        gradA = sumToShape(gradOutput, a.shape);
      }
      if (!shapesEqual(gradB.shape, b.shape)) {
        gradB = sumToShape(gradB, b.shape);
      }

      return [gradA, gradB];
    });
  }

  /**
   * Умножение тензоров (поэлементное)
   * @param other - Другой тензор или число
   * @returns Результат умножения
   */
  mul(other: Tensor | number): Tensor {
    return this.binaryOp(other, (a, b) => a * b, (gradOutput, a, b) => {
      let gradA = gradOutput.mul(b);
      let gradB = gradOutput.mul(a);

      if (!shapesEqual(gradA.shape, a.shape)) {
        gradA = sumToShape(gradA, a.shape);
      }
      if (!shapesEqual(gradB.shape, b.shape)) {
        gradB = sumToShape(gradB, b.shape);
      }

      return [gradA, gradB];
    });
  }

  /**
   * Деление тензоров (поэлементное)
   * @param other - Другой тензор или число
   * @returns Результат деления
   */
  div(other: Tensor | number): Tensor {
    return this.binaryOp(other, (a, b) => a / b, (gradOutput, a, b) => {
      // d/da (a/b) = 1/b
      // d/db (a/b) = -a/b^2
      let gradA = gradOutput.div(b);
      let gradB = gradOutput.mul(a).neg().div(b.mul(b));

      if (!shapesEqual(gradA.shape, a.shape)) {
        gradA = sumToShape(gradA, a.shape);
      }
      if (!shapesEqual(gradB.shape, b.shape)) {
        gradB = sumToShape(gradB, b.shape);
      }

      return [gradA, gradB];
    });
  }

  /**
   * Возведение в степень
   * @param exponent - Показатель степени
   * @returns Результат возведения в степень
   */
  pow(exponent: number): Tensor {
    const resultData = this.data.map(v => Math.pow(v, exponent));
    const result = new Tensor(resultData as unknown as number[], [...this.shape], {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
    });

    if (this.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveTensors(this);
      ctx.saveValue('exponent', exponent);
      result.gradNode = new GradNode(
        (gradOutput) => {
          // d/dx (x^n) = n * x^(n-1)
          const grad = gradOutput.mul(this.pow(exponent - 1).mul(exponent));
          return [grad];
        },
        [this],
        ctx
      );
    }

    return result;
  }

  /**
   * Квадратный корень
   * @returns Тензор с квадратными корнями
   */
  sqrt(): Tensor {
    return this.pow(0.5);
  }

  /**
   * Экспонента
   * @returns e^x для каждого элемента
   */
  exp(): Tensor {
    const resultData = this.data.map(v => Math.exp(v));
    const result = new Tensor(resultData as unknown as number[], [...this.shape], {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
    });

    if (this.requiresGrad && !isNoGradEnabled()) {
      result.gradNode = new GradNode(
        (gradOutput) => [gradOutput.mul(result)],
        [this],
        new GradContext()
      );
    }

    return result;
  }

  /**
   * Натуральный логарифм
   * @returns ln(x) для каждого элемента
   */
  log(): Tensor {
    const resultData = this.data.map(v => Math.log(v));
    const result = new Tensor(resultData as unknown as number[], [...this.shape], {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
    });

    if (this.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveTensors(this);
      result.gradNode = new GradNode(
        (gradOutput) => [gradOutput.div(this)],
        [this],
        ctx
      );
    }

    return result;
  }

  /**
   * Отрицание (умножение на -1)
   * @returns Отрицательный тензор
   */
  neg(): Tensor {
    return this.mul(-1);
  }

  /**
   * Абсолютное значение
   * @returns Тензор с абсолютными значениями
   */
  abs(): Tensor {
    const resultData = this.data.map(v => Math.abs(v));
    const result = new Tensor(resultData as unknown as number[], [...this.shape], {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
    });

    if (this.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveTensors(this);
      result.gradNode = new GradNode(
        (gradOutput) => {
          const sign = this.data.map(v => v >= 0 ? 1 : -1);
          const signTensor = new Tensor(sign as unknown as number[], [...this.shape], { dtype: this.dtype });
          return [gradOutput.mul(signTensor)];
        },
        [this],
        ctx
      );
    }

    return result;
  }

  // ============================================
  // МАТРИЧНЫЕ ОПЕРАЦИИ
  // ============================================

  /**
   * Матричное умножение
   * @param other - Другая матрица
   * @returns Результат матричного умножения
   */
  matmul(other: Tensor): Tensor {
    // Поддержка batch матричного умножения
    if (this.ndim < 1 || other.ndim < 1) {
      throw new Error('matmul requires at least 1D tensors');
    }

    // Для 1D тензоров
    if (this.ndim === 1 && other.ndim === 1) {
      // Скалярное произведение векторов
      if (this.shape[0] !== other.shape[0]) {
        throw new Error(`Vector dimensions must match: ${this.shape[0]} vs ${other.shape[0]}`);
      }
      let sum = 0;
      for (let i = 0; i < this.shape[0]; i++) {
        sum += this.data[i] * other.data[i];
      }
      return scalar(sum, { requiresGrad: this.requiresGrad || other.requiresGrad });
    }

    // Для 2D матриц
    if (this.ndim === 2 && other.ndim === 2) {
      const [m, k1] = this.shape;
      const [k2, n] = other.shape;

      if (k1 !== k2) {
        throw new Error(`Matrix dimensions must match: [${m}, ${k1}] x [${k2}, ${n}]`);
      }

      const resultData = new Float32Array(m * n);

      // Наивное матричное умножение (можно оптимизировать)
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let sum = 0;
          for (let k = 0; k < k1; k++) {
            sum += this.data[i * k1 + k] * other.data[k * n + j];
          }
          resultData[i * n + j] = sum;
        }
      }

      const result = new Tensor(resultData, [m, n], {
        dtype: this.dtype,
        requiresGrad: this.requiresGrad || other.requiresGrad,
      });

      // Autograd
      if ((this.requiresGrad || other.requiresGrad) && !isNoGradEnabled()) {
        const ctx = new GradContext();
        ctx.saveTensors(this, other);
        result.gradNode = new GradNode(
          (gradOutput) => {
            // dL/dA = dL/dC @ B^T
            // dL/dB = A^T @ dL/dC
            const gradA = gradOutput.matmul(other.T);
            const gradB = this.T.matmul(gradOutput);
            return [gradA, gradB];
          },
          [this, other],
          ctx
        );
      }

      return result;
    }

    // Для 1D @ 2D или 2D @ 1D
    if (this.ndim === 1) {
      return this.unsqueeze(0).matmul(other).squeeze(0);
    }
    if (other.ndim === 1) {
      return this.matmul(other.unsqueeze(1)).squeeze(1);
    }

    // Batch matrix multiplication (для тензоров > 2D)
    // Алгоритм:
    // 1. Broadcast формы (кроме последних 2 измерений)
    // 2. reshape в [batch_size, m, k] и [batch_size, k, n]
    // 3. цикл по batch_size
    // 4. reshape обратно

    const shapeA = this.shape;
    const shapeB = other.shape;
    
    // check matrix dims
    if (shapeA[shapeA.length - 1] !== shapeB[shapeB.length - 2]) {
      throw new Error(`MatMul: Incompatible matrix dimensions: ...x${shapeA[shapeA.length - 1]} and ...x${shapeB[shapeB.length - 2]}x...`);
    }

    const batchShapeA = shapeA.slice(0, -2);
    const batchShapeB = shapeB.slice(0, -2);
    const batchShape = broadcastShapes(batchShapeA, batchShapeB);
    
    if (!batchShape) {
       throw new Error(`MatMul: Cannot broadcast batch shapes [${batchShapeA}] and [${batchShapeB}]`);
    }

    const m = shapeA[shapeA.length - 2];
    const k = shapeA[shapeA.length - 1]; // == shapeB[shapeB.length - 2]
    const n = shapeB[shapeB.length - 1];

    // Expand inputs to full batch shape
    const expandedA = this.expand(...batchShape, m, k);
    const expandedB = other.expand(...batchShape, k, n);

    const batchSize = computeSize(batchShape);
    const resultData = new Float32Array(batchSize * m * n);
    
    // Strides
    const param = this.dtype; // unused but just ensuring access
    
    // Эффективнее было бы реализовать через доступ к underlying buffer, но пока через data
    // Так как мы сделали expand, данные могут быть не contiguous. 
    // Для простоты реализации (но не эффективности) - просто проитерируемся по батчам как по view

    // TODO: Здесь стоит оптимизировать, если данные копируются Expand'ом - это ок, если это View - будет медленно
    // Сейчас Expand создает новый тензор с копией данных, если были broadcasted dims. 
    
    // У нас нет view-механики полноценной, expand реально копирует данные в текущей реализации (см expand)
    // Значит expandedA и expandedB имеют "плоские" данные правильного размера.
    
    for (let b = 0; b < batchSize; b++) {
       const offsetA = b * m * k;
       const offsetB = b * k * n;
       const offsetRes = b * m * n;
       
       for (let i = 0; i < m; i++) {
         for (let j = 0; j < n; j++) {
           let sum = 0;
           for (let x = 0; x < k; x++) {
             sum += expandedA.data[offsetA + i * k + x] * expandedB.data[offsetB + x * n + j];
           }
           resultData[offsetRes + i * n + j] = sum;
         }
       }
    }

    const resultShape = [...batchShape, m, n];
    const result = new Tensor(resultData, resultShape, {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad || other.requiresGrad,
    });

    if ((this.requiresGrad || other.requiresGrad) && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveTensors(this, other);
      result.gradNode = new GradNode(
        (gradOutput) => {
          // dL/dA = grad * B^T
          // dL/dB = A^T * grad
          // Transpose last two dims for B and A
          const BT = other.transpose(-2, -1);
          const AT = this.transpose(-2, -1);
          
          let gradA = gradOutput.matmul(BT);
          let gradB = AT.matmul(gradOutput);
          
          // Sum reduction if broadcast happened
          if (!shapesEqual(gradA.shape, this.shape)) {
             gradA = sumToShape(gradA, this.shape);
          }
          if (!shapesEqual(gradB.shape, other.shape)) {
             gradB = sumToShape(gradB, other.shape);
          }
          
          return [gradA, gradB];
        },
        [this, other],
        ctx
      );
    }
    
    return result;
  }

  // ============================================
  // ОПЕРАЦИИ РЕДУКЦИИ
  // ============================================

  /**
   * Сумма элементов
   * @param dim - Размерность для суммирования (опционально)
   * @param keepdim - Сохранять размерность
   * @returns Сумма
   */
  sum(dim?: number, keepdim: boolean = false): Tensor {
    if (dim === undefined) {
      // Сумма всех элементов
      let sum = 0;
      for (let i = 0; i < this.size; i++) {
        sum += this.data[i];
      }
      const result = scalar(sum, { requiresGrad: this.requiresGrad });

      if (this.requiresGrad && !isNoGradEnabled()) {
        result.gradNode = new GradNode(
          (gradOutput) => [ones(this.shape).mul(gradOutput)],
          [this],
          new GradContext()
        );
      }

      return result;
    }

    // Сумма по конкретной размерности
    if (dim < 0) dim += this.ndim;

    const newShape = this.shape.filter((_, i) => i !== dim);
    const resultShape = keepdim
      ? this.shape.map((s, i) => (i === dim ? 1 : s))
      : newShape;

    const resultSize = computeSize(resultShape);
    const resultData = new Float32Array(resultSize);

    // Вычисляем сумму
    for (let i = 0; i < this.size; i++) {
      const indices = offsetToIndices(i, this.shape);
      const newIndices = keepdim
        ? indices.map((idx, d) => (d === dim ? 0 : idx))
        : indices.filter((_, d) => d !== dim);
      const resultIdx = indicesToOffset(newIndices, computeStrides(resultShape));
      resultData[resultIdx] += this.data[i];
    }

    const result = new Tensor(resultData, resultShape, {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
    });

    if (this.requiresGrad && !isNoGradEnabled()) {
      const ctx = new GradContext();
      ctx.saveValue('dim', dim);
      ctx.saveValue('keepdim', keepdim);
      ctx.saveValue('inputShape', this.shape);
      result.gradNode = new GradNode(
        (gradOutput) => {
          // Расширяем градиент обратно до исходной формы
          let grad = gradOutput;
          if (!keepdim) {
            grad = grad.unsqueeze(dim);
          }
          // Broadcast градиент
          return [grad.expand(...(this.shape as number[]))];
        },
        [this],
        ctx
      );
    }

    return result;
  }

  /**
   * Среднее значение элементов
   * @param dim - Размерность для усреднения (опционально)
   * @param keepdim - Сохранять размерность
   * @returns Среднее
   */
  mean(dim?: number, keepdim: boolean = false): Tensor {
    if (dim === undefined) {
      return this.sum().div(this.size);
    }

    if (dim < 0) dim += this.ndim;
    const sumResult = this.sum(dim, keepdim);
    return sumResult.div(this.shape[dim]);
  }

  /**
   * Максимальное значение
   * @param dim - Размерность (опционально)
   * @param keepdim - Сохранять размерность
   * @returns Максимум и индексы
   */
  max(dim?: number, keepdim: boolean = false): { values: Tensor; indices?: Tensor } {
    if (dim === undefined) {
      let maxVal = -Infinity;
      for (let i = 0; i < this.size; i++) {
        if (this.data[i] > maxVal) maxVal = this.data[i];
      }
      return { values: scalar(maxVal) };
    }

    if (dim < 0) dim += this.ndim;

    const newShape = keepdim
      ? this.shape.map((s, i) => (i === dim ? 1 : s))
      : this.shape.filter((_, i) => i !== dim);

    const resultSize = computeSize(newShape);
    const resultData = new Float32Array(resultSize).fill(-Infinity);
    const indicesData = new Float32Array(resultSize);

    for (let i = 0; i < this.size; i++) {
      const indices = offsetToIndices(i, this.shape);
      const dimIndex = indices[dim];
      const newIndices = keepdim
        ? indices.map((idx, d) => (d === dim ? 0 : idx))
        : indices.filter((_, d) => d !== dim);
      const resultIdx = indicesToOffset(newIndices, computeStrides(newShape));

      if (this.data[i] > resultData[resultIdx]) {
        resultData[resultIdx] = this.data[i];
        indicesData[resultIdx] = dimIndex;
      }
    }

    // Autograd for max
    if (this.requiresGrad && !isNoGradEnabled()) {
       const ctx = new GradContext();
       ctx.saveValue('dim', dim);
       ctx.saveValue('keepdim', keepdim);
       ctx.saveValue('inputShape', this.shape);
       // Save indices for backward
       ctx.saveValue('indices', new Tensor(indicesData, newShape, { dtype: DType.Int32 })); 
       
       const maxVals = new Tensor(resultData, newShape, { dtype: this.dtype, requiresGrad: true });
       maxVals.gradNode = new GradNode(
         (gradOutput) => {
           // Create a zero tensor of input shape
           const gradInputData = new Float32Array(this.size);
           
           // Scatter gradients
           const indicesTensor = ctx.getValue('indices') as Tensor;
           const gOutData = gradOutput.data; 
           const indData = indicesTensor.data;
           
           // Need to map output linear indices back to input linear indices
           // Since we don't have a standardized 'scatter' ops yet, we do raw loop with strides
           
           // It's easier if we iterate over the OUTPUT size, because we know exactly where each grad comes from
           for(let i=0; i<gradOutput.size; i++) {
              const gradVal = gOutData[i];
              if (gradVal === 0) continue;
              
              const outIndices = offsetToIndices(i, gradOutput.shape); // indices in result
              const selectedIndexInDim = indData[i]; // which index was selected along 'dim'
              
              // Construct input indices
              const inIndices = [...outIndices];
              if (!keepdim) {
                // If dim was removed, we need to insert the index back
                // outIndices has length N-1. 
                // We need to insert selectedIndexInDim at 'dim'
                 inIndices.splice(dim!, 0, selectedIndexInDim);
              } else {
                 // if keepdim, outIndices has length N, but the value at 'dim' is 0
                 // we replace it with the actual selected index
                 inIndices[dim!] = selectedIndexInDim;
              }
              
              const inputOffset = indicesToOffset(inIndices, this.strides as number[]);
              gradInputData[inputOffset] += gradVal; // accumulate just in case (though max is unique usually)
           }
           
           const gradInput = new Tensor(gradInputData, this.shape, { dtype: this.dtype });
           return [gradInput];
         },
         [this],
         ctx
       );
       return {
          values: maxVals,
          indices: new Tensor(indicesData, newShape, { dtype: DType.Int32 })
       };
    }

    return {
      values: new Tensor(resultData, newShape, { dtype: this.dtype }),
      indices: new Tensor(indicesData, newShape, { dtype: DType.Int32 }),
    };
  }

  /**
   * Минимальное значение
   * @param dim - Размерность (опционально)
   * @param keepdim - Сохранять размерность
   * @returns Минимум и индексы
   */
  min(dim?: number, keepdim: boolean = false): { values: Tensor; indices?: Tensor } {
    if (dim === undefined) {
      let minVal = Infinity;
      for (let i = 0; i < this.size; i++) {
        if (this.data[i] < minVal) minVal = this.data[i];
      }
      return { values: scalar(minVal) };
    }

    if (dim < 0) dim += this.ndim;

    const newShape = keepdim
      ? this.shape.map((s, i) => (i === dim ? 1 : s))
      : this.shape.filter((_, i) => i !== dim);

    const resultSize = computeSize(newShape);
    const resultData = new Float32Array(resultSize).fill(Infinity);
    const indicesData = new Float32Array(resultSize);

    for (let i = 0; i < this.size; i++) {
      const indices = offsetToIndices(i, this.shape);
      const dimIndex = indices[dim];
      const newIndices = keepdim
        ? indices.map((idx, d) => (d === dim ? 0 : idx))
        : indices.filter((_, d) => d !== dim);
      const resultIdx = indicesToOffset(newIndices, computeStrides(newShape));

      if (this.data[i] < resultData[resultIdx]) {
        resultData[resultIdx] = this.data[i];
        indicesData[resultIdx] = dimIndex;
      }
    }

    // Autograd for min
    if (this.requiresGrad && !isNoGradEnabled()) {
       const ctx = new GradContext();
       ctx.saveValue('dim', dim);
       ctx.saveValue('keepdim', keepdim);
       ctx.saveValue('indices', new Tensor(indicesData, newShape, { dtype: DType.Int32 }));
       
       const minVals = new Tensor(resultData, newShape, { dtype: this.dtype, requiresGrad: true });
       minVals.gradNode = new GradNode(
         (gradOutput) => {
           const gradInputData = new Float32Array(this.size);
           const indicesTensor = ctx.getValue('indices') as Tensor;
           const gOutData = gradOutput.data;
           const indData = indicesTensor.data;
           
           for(let i=0; i<gradOutput.size; i++) {
              const gradVal = gOutData[i];
              if(gradVal === 0) continue;
              
              const outIndices = offsetToIndices(i, gradOutput.shape);
              const selectedIndexInDim = indData[i];
              
              const inIndices = [...outIndices];
              if (!keepdim) {
                 inIndices.splice(dim!, 0, selectedIndexInDim);
              } else {
                 inIndices[dim!] = selectedIndexInDim;
              }
              
              const inputOffset = indicesToOffset(inIndices, this.strides as number[]);
              gradInputData[inputOffset] += gradVal;
           }
           
           return [new Tensor(gradInputData, this.shape, { dtype: this.dtype })];
         },
         [this],
         ctx
       );
       
       return {
          values: minVals,
          indices: new Tensor(indicesData, newShape, { dtype: DType.Int32 })
       };
    }

    return {
      values: new Tensor(resultData, newShape, { dtype: this.dtype }),
      indices: new Tensor(indicesData, newShape, { dtype: DType.Int32 }),
    };
  }

  /**
   * Индекс максимального элемента
   * @param dim - Размерность (по умолчанию выравнивает тензор)
   * @returns Тензор индексов
   */
  argmax(dim?: number): Tensor {
    if (dim === undefined) {
      let maxIdx = 0;
      let maxVal = this.data[0];
      for (let i = 1; i < this.size; i++) {
        if (this.data[i] > maxVal) {
          maxVal = this.data[i];
          maxIdx = i;
        }
      }
      return scalar(maxIdx, { dtype: DType.Int32 });
    }
    return this.max(dim).indices!;
  }

  /**
   * Индекс минимального элемента
   * @param dim - Размерность (по умолчанию выравнивает тензор)
   * @returns Тензор индексов
   */
  argmin(dim?: number): Tensor {
    if (dim === undefined) {
      let minIdx = 0;
      let minVal = this.data[0];
      for (let i = 1; i < this.size; i++) {
        if (this.data[i] < minVal) {
          minVal = this.data[i];
          minIdx = i;
        }
      }
      return scalar(minIdx, { dtype: DType.Int32 });
    }
    return this.min(dim).indices!;
  }

  // ============================================
  // BROADCASTING И EXPANSION
  // ============================================

  /**
   * Расширяет тензор до указанной формы (через broadcasting)
   * @param shape - Целевая форма
   * @returns Расширенный тензор
   */
  expand(...shape: number[]): Tensor {
    // Проверяем совместимость
    if (shape.length < this.ndim) {
      throw new Error('expand shape must have at least as many dims as tensor');
    }

    const newShape: number[] = [];
    const offset = shape.length - this.ndim;

    for (let i = 0; i < shape.length; i++) {
      if (i < offset) {
        newShape.push(shape[i]);
      } else {
        const tensorDim = this.shape[i - offset];
        if (shape[i] === -1) {
          newShape.push(tensorDim);
        } else if (tensorDim === 1 || tensorDim === shape[i]) {
          newShape.push(shape[i]);
        } else {
          throw new Error(`Cannot expand dim ${i} from ${tensorDim} to ${shape[i]}`);
        }
      }
    }

    const resultSize = computeSize(newShape);
    const resultData = new Float32Array(resultSize);

    for (let i = 0; i < resultSize; i++) {
      const indices = offsetToIndices(i, newShape);
      const sourceIndices = indices.slice(offset).map((idx, dim) => {
        return this.shape[dim] === 1 ? 0 : idx;
      });
      const sourceOffset = indicesToOffset(sourceIndices, this.strides as number[]);
      resultData[i] = this.data[sourceOffset];
    }

    if (this.requiresGrad && !isNoGradEnabled()) {
        const ctx = new GradContext();
        ctx.saveValue('inputShape', this.shape);
        const res = new Tensor(resultData, newShape, {
            dtype: this.dtype,
            requiresGrad: this.requiresGrad,
        });
        res.gradNode = new GradNode(
            (gradOutput) => {
                // backward of expand is sumToShape
                return [sumToShape(gradOutput, this.shape)];
            },
            [this],
            ctx
        );
        return res;
    }

    return new Tensor(resultData, newShape, {
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
    });
  }

  // ============================================
  // AUTOGRAD
  // ============================================

  /**
   * Выполняет обратное распространение градиентов
   * @param gradOutput - Начальный градиент (по умолчанию 1 для скаляров)
   */
  backward(gradOutput?: Tensor): void {
    if (!this.requiresGrad) {
      throw new Error('Cannot call backward on tensor with requiresGrad=false');
    }
    autoBackward(this, gradOutput);
  }

  /**
   * Обнуляет накопленный градиент
   */
  zeroGrad(): void {
    this.grad = null;
  }

  /**
   * Создаёт копию тензора без отслеживания градиентов
   * @returns Тензор с requiresGrad=false
   */
  detach(): Tensor {
    const result = new Tensor(this.data, this.shape, {
      dtype: this.dtype,
      requiresGrad: false,
    });
    return result;
  }

  // ============================================
  // УТИЛИТЫ
  // ============================================

  /**
   * Строковое представление тензора
   * @returns Строка с информацией о тензоре
   */
  toString(): string {
    const dataStr = this.size <= 20
      ? Array.from(this.data).join(', ')
      : `${Array.from(this.data.slice(0, 10)).join(', ')} ... ${Array.from(this.data.slice(-5)).join(', ')}`;
    return `Tensor(shape=[${this.shape}], dtype=${this.dtype}, data=[${dataStr}])`;
  }

  /**
   * Выводит тензор в консоль в красивом формате
   */
  print(): void {
    console.log(this.toString());
    if (this.ndim <= 2) {
      console.log(this.toArray());
    }
  }
}

// ============================================
// ФАБРИЧНЫЕ ФУНКЦИИ
// ============================================

/**
 * Создаёт скалярный тензор
 * @param value - Скалярное значение
 * @param options - Опции тензора
 * @returns Скалярный тензор формы [1]
 */
export function scalar(value: number, options: TensorOptions = {}): Tensor {
  return new Tensor([value], [1], options);
}

/**
 * Создаёт тензор из вложенного массива
 * @param data - Вложенный массив
 * @param options - Опции тензора
 * @returns Новый тензор
 */
export function tensor(data: NestedArray, options: TensorOptions = {}): Tensor {
  return new Tensor(data, undefined, options);
}

/**
 * Создаёт тензор заполненный нулями
 * @param shape - Форма тензора
 * @param options - Опции тензора
 * @returns Тензор с нулями
 */
export function zeros(shape: Shape, options: TensorOptions = {}): Tensor {
  const size = computeSize(shape);
  const TypedArrayClass = getTypedArrayConstructor(options.dtype || DType.Float32);
  const data = new TypedArrayClass(size);
  return new Tensor(data, shape, options);
}

/**
 * Создаёт тензор заполненный единицами
 * @param shape - Форма тензора
 * @param options - Опции тензора
 * @returns Тензор с единицами
 */
export function ones(shape: Shape, options: TensorOptions = {}): Tensor {
  const size = computeSize(shape);
  const TypedArrayClass = getTypedArrayConstructor(options.dtype || DType.Float32);
  const data = new TypedArrayClass(size).fill(1);
  return new Tensor(data, shape, options);
}

/**
 * Создаёт тензор с равномерно распределёнными случайными числами [0, 1)
 * @param shape - Форма тензора
 * @param options - Опции тензора
 * @returns Случайный тензор
 */
export function rand(shape: Shape, options: TensorOptions = {}): Tensor {
  const size = computeSize(shape);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = Math.random();
  }
  return new Tensor(data, shape, options);
}

/**
 * Создаёт тензор с нормально распределёнными случайными числами
 * @param shape - Форма тензора
 * @param mean - Среднее (по умолчанию 0)
 * @param std - Стандартное отклонение (по умолчанию 1)
 * @param options - Опции тензора
 * @returns Случайный тензор
 */
export function randn(shape: Shape, mean: number = 0, std: number = 1, options: TensorOptions = {}): Tensor {
  const size = computeSize(shape);
  const data = new Float32Array(size);

  // Box-Muller transform
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.random();
    const u2 = Math.random();
    const mag = std * Math.sqrt(-2 * Math.log(u1));
    data[i] = mag * Math.cos(2 * Math.PI * u2) + mean;
    if (i + 1 < size) {
      data[i + 1] = mag * Math.sin(2 * Math.PI * u2) + mean;
    }
  }

  return new Tensor(data, shape, options);
}

/**
 * Создаёт единичную матрицу
 * @param n - Размер матрицы
 * @param options - Опции тензора
 * @returns Единичная матрица n x n
 */
export function eye(n: number, options: TensorOptions = {}): Tensor {
  const data = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    data[i * n + i] = 1;
  }
  return new Tensor(data, [n, n], options);
}

/**
 * Создаёт тензор с линейно распределёнными значениями
 * @param start - Начальное значение
 * @param end - Конечное значение
 * @param steps - Количество шагов
 * @param options - Опции тензора
 * @returns Тензор с линейными значениями
 */
export function linspace(start: number, end: number, steps: number, options: TensorOptions = {}): Tensor {
  const data = new Float32Array(steps);
  const step = (end - start) / (steps - 1);
  for (let i = 0; i < steps; i++) {
    data[i] = start + i * step;
  }
  return new Tensor(data, [steps], options);
}

/**
 * Создаёт тензор с диапазоном целых чисел
 * @param start - Начало (или конец если step не указан)
 * @param end - Конец (опционально)
 * @param step - Шаг (опционально)
 * @param options - Опции тензора
 * @returns Тензор с диапазоном
 */
export function arange(start: number, end?: number, step: number = 1, options: TensorOptions = {}): Tensor {
  if (end === undefined) {
    end = start;
    start = 0;
  }

  const size = Math.ceil((end - start) / step);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = start + i * step;
  }
  return new Tensor(data, [size], options);
}

/**
 * Создаёт тензор заполненный указанным значением
 * @param shape - Форма тензора
 * @param value - Значение для заполнения
 * @param options - Опции тензора
 * @returns Заполненный тензор
 */
export function full(shape: Shape, value: number, options: TensorOptions = {}): Tensor {
  const size = computeSize(shape);
  const TypedArrayClass = getTypedArrayConstructor(options.dtype || DType.Float32);
  const data = new TypedArrayClass(size).fill(value);
  return new Tensor(data, shape, options);
}

/**
 * Конкатенирует тензоры вдоль указанной оси
 * @param tensors - Массив тензоров
 * @param dim - Ось для конкатенации
 * @returns Конкатенированный тензор
 */
export function cat(tensors: Tensor[], dim: number = 0): Tensor {
  if (tensors.length === 0) {
    throw new Error('Cannot concatenate empty list of tensors');
  }

  if (dim < 0) dim += tensors[0].ndim;

  // Проверяем совместимость форм
  const baseShape = [...tensors[0].shape];
  for (let i = 1; i < tensors.length; i++) {
    if (tensors[i].ndim !== tensors[0].ndim) {
      throw new Error('All tensors must have the same number of dimensions');
    }
    for (let d = 0; d < tensors[0].ndim; d++) {
      if (d !== dim && tensors[i].shape[d] !== tensors[0].shape[d]) {
        throw new Error(`Tensor dimensions must match except for concat dimension`);
      }
    }
  }

  // Вычисляем новую форму
  const newShape = [...baseShape];
  newShape[dim] = tensors.reduce((sum, t) => sum + t.shape[dim], 0);

  // Копируем данные
  const resultSize = computeSize(newShape);
  const resultData = new Float32Array(resultSize);
  const resultStrides = computeStrides(newShape);

  let dimOffset = 0;
  for (const t of tensors) {
    for (let i = 0; i < t.size; i++) {
      const indices = offsetToIndices(i, t.shape);
      indices[dim] += dimOffset;
      const resultIdx = indicesToOffset(indices, resultStrides);
      resultData[resultIdx] = t.data[i];
    }
    dimOffset += t.shape[dim];
  }

  return new Tensor(resultData, newShape, {
    dtype: tensors[0].dtype,
    requiresGrad: tensors.some(t => t.requiresGrad),
  });
}

/**
 * Стекает тензоры вдоль новой оси
 * @param tensors - Массив тензоров
 * @param dim - Позиция новой оси
 * @returns Стек тензоров
 */
export function stack(tensors: Tensor[], dim: number = 0): Tensor {
  const expanded = tensors.map(t => t.unsqueeze(dim));
  return cat(expanded, dim);
}

// ============================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================

/**
 * Суммирует тензор до указанной формы (для градиентов при broadcast)
 * @param tensor - Исходный тензор
 * @param targetShape - Целевая форма
 * @returns Суммированный тензор
 */
export function sumToShape(tensor: Tensor, targetShape: Shape): Tensor {
  let result = tensor;

  // Суммируем по лишним размерностям
  while (result.ndim > targetShape.length) {
    result = result.sum(0);
  }

  // Суммируем по broadcast размерностям
  for (let i = 0; i < result.ndim; i++) {
    if (i < targetShape.length && targetShape[i] === 1 && result.shape[i] !== 1) {
      result = result.sum(i, true);
    }
  }

  return result;
}

export { noGrad, noGradContext, isNoGradEnabled } from './autograd';
