/**
 * @fileoverview Определение типов данных для тензоров
 * @description Поддерживаемые типы данных (dtype) для хранения элементов тензора
 */

/**
 * Перечисление поддерживаемых типов данных
 * Аналог torch.dtype в PyTorch
 */
export enum DType {
  /** 32-битное число с плавающей точкой (по умолчанию) */
  Float32 = 'float32',
  /** 64-битное число с плавающей точкой */
  Float64 = 'float64',
  /** 32-битное целое число */
  Int32 = 'int32',
  /** 8-битное целое число без знака */
  Uint8 = 'uint8',
  /** Логический тип (true/false) */
  Bool = 'bool',
}

/**
 * Тип, представляющий форму тензора
 * Массив чисел, где каждое число - размер по соответствующей оси
 * Например: [2, 3, 4] - тензор размером 2x3x4
 */
export type Shape = readonly number[];

/**
 * Получает соответствующий TypedArray конструктор для типа данных
 * @param dtype - Тип данных
 * @returns Конструктор TypedArray
 */
export function getTypedArrayConstructor(dtype: DType): Float32ArrayConstructor | Float64ArrayConstructor | Int32ArrayConstructor | Uint8ArrayConstructor {
  switch (dtype) {
    case DType.Float32:
      return Float32Array;
    case DType.Float64:
      return Float64Array;
    case DType.Int32:
      return Int32Array;
    case DType.Uint8:
    case DType.Bool:
      return Uint8Array;
    default:
      return Float32Array;
  }
}

/**
 * Возвращает размер в байтах для элемента данного типа
 * @param dtype - Тип данных
 * @returns Размер в байтах
 */
export function getDTypeSize(dtype: DType): number {
  switch (dtype) {
    case DType.Float32:
    case DType.Int32:
      return 4;
    case DType.Float64:
      return 8;
    case DType.Uint8:
    case DType.Bool:
      return 1;
    default:
      return 4;
  }
}

/**
 * Тип данных для TypedArray
 */
export type TypedArray = Float32Array | Float64Array | Int32Array | Uint8Array;

/**
 * Вложенный массив чисел произвольной глубины
 * Используется для инициализации тензоров
 */
export type NestedArray = number | NestedArray[];
