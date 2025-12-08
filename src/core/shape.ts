/**
 * @fileoverview Утилиты для работы с формами тензоров
 * @description Функции для вычисления размеров, строк и broadcasting
 */

import { Shape } from './dtype';

/**
 * Вычисляет общее количество элементов в тензоре по его форме
 * @param shape - Форма тензора
 * @returns Общее количество элементов
 * @example
 * computeSize([2, 3, 4]) // 24
 */
export function computeSize(shape: Shape): number {
  if (shape.length === 0) return 1; // scalar
  return shape.reduce((acc, dim) => acc * dim, 1);
}

/**
 * Вычисляет strides для тензора (шаги для перехода между элементами по каждой оси)
 * Используется row-major (C-style) порядок хранения
 * @param shape - Форма тензора
 * @returns Массив strides
 * @example
 * computeStrides([2, 3, 4]) // [12, 4, 1]
 */
export function computeStrides(shape: Shape): number[] {
  const strides: number[] = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Преобразует многомерный индекс в линейный индекс
 * @param indices - Многомерный индекс
 * @param strides - Strides тензора
 * @returns Линейный индекс
 * @example
 * indicesToOffset([1, 2, 3], [12, 4, 1]) // 1*12 + 2*4 + 3*1 = 23
 */
export function indicesToOffset(indices: number[], strides: number[]): number {
  let offset = 0;
  for (let i = 0; i < indices.length; i++) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

/**
 * Преобразует линейный индекс в многомерный индекс
 * @param offset - Линейный индекс
 * @param shape - Форма тензора
 * @returns Многомерный индекс
 */
export function offsetToIndices(offset: number, shape: Shape): number[] {
  const indices: number[] = new Array(shape.length);
  let remaining = offset;
  for (let i = shape.length - 1; i >= 0; i--) {
    indices[i] = remaining % shape[i];
    remaining = Math.floor(remaining / shape[i]);
  }
  return indices;
}

/**
 * Вычисляет результирующую форму после broadcasting двух тензоров
 * Реализует правила NumPy broadcasting
 * @param shape1 - Форма первого тензора
 * @param shape2 - Форма второго тензора
 * @returns Результирующая форма или null если broadcasting невозможен
 * @example
 * broadcastShapes([3, 1, 4], [1, 5, 4]) // [3, 5, 4]
 */
export function broadcastShapes(shape1: Shape, shape2: Shape): number[] | null {
  const maxLength = Math.max(shape1.length, shape2.length);
  const result: number[] = new Array(maxLength);

  for (let i = 0; i < maxLength; i++) {
    const dim1 = i < shape1.length ? shape1[shape1.length - 1 - i] : 1;
    const dim2 = i < shape2.length ? shape2[shape2.length - 1 - i] : 1;

    if (dim1 === dim2) {
      result[maxLength - 1 - i] = dim1;
    } else if (dim1 === 1) {
      result[maxLength - 1 - i] = dim2;
    } else if (dim2 === 1) {
      result[maxLength - 1 - i] = dim1;
    } else {
      return null; // Incompatible shapes
    }
  }

  return result;
}

/**
 * Проверяет, совпадают ли две формы
 * @param shape1 - Первая форма
 * @param shape2 - Вторая форма
 * @returns true если формы идентичны
 */
export function shapesEqual(shape1: Shape, shape2: Shape): boolean {
  if (shape1.length !== shape2.length) return false;
  for (let i = 0; i < shape1.length; i++) {
    if (shape1[i] !== shape2[i]) return false;
  }
  return true;
}

/**
 * Определяет форму вложенного массива
 * @param data - Вложенный массив
 * @returns Форма массива
 * @example
 * inferShape([[1, 2], [3, 4]]) // [2, 2]
 */
export function inferShape(data: unknown): number[] {
  const shape: number[] = [];
  let current: unknown = data;

  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }

  return shape;
}

/**
 * Расплющивает вложенный массив в одномерный
 * @param data - Вложенный массив
 * @returns Одномерный массив чисел
 */
export function flattenArray(data: unknown): number[] {
  const result: number[] = [];

  function flatten(arr: unknown): void {
    if (Array.isArray(arr)) {
      for (const item of arr) {
        flatten(item);
      }
    } else if (typeof arr === 'number') {
      result.push(arr);
    } else if (typeof arr === 'boolean') {
      result.push(arr ? 1 : 0);
    }
  }

  flatten(data);
  return result;
}

/**
 * Валидирует форму (все размерности должны быть положительными целыми)
 * @param shape - Форма для проверки
 * @throws Error если форма невалидна
 */
export function validateShape(shape: Shape): void {
  for (let i = 0; i < shape.length; i++) {
    if (!Number.isInteger(shape[i]) || shape[i] < 0) {
      throw new Error(`Invalid shape dimension at index ${i}: ${shape[i]}`);
    }
  }
}

/**
 * Вычисляет новую форму после squeeze (удаление размерностей равных 1)
 * @param shape - Исходная форма
 * @param dim - Конкретная размерность для squeeze (опционально)
 * @returns Новая форма
 */
export function squeezeShape(shape: Shape, dim?: number): number[] {
  if (dim !== undefined) {
    if (shape[dim] !== 1) {
      return [...shape];
    }
    return [...shape.slice(0, dim), ...shape.slice(dim + 1)];
  }
  return shape.filter(d => d !== 1);
}

/**
 * Вычисляет новую форму после unsqueeze (добавление размерности 1)
 * @param shape - Исходная форма
 * @param dim - Позиция для вставки новой размерности
 * @returns Новая форма
 */
export function unsqueezeShape(shape: Shape, dim: number): number[] {
  const result = [...shape];
  result.splice(dim, 0, 1);
  return result;
}
