/**
 * @fileoverview Инициализация весов нейронных сетей
 * @description Различные стратегии инициализации для оптимального обучения
 */

import { Tensor, zeros, ones, randn } from '../core/tensor';
import { Parameter } from './module';

/**
 * Xavier/Glorot uniform инициализация
 * Хорошо работает с sigmoid и tanh активациями
 * @param tensor - Тензор для инициализации
 * @param gain - Коэффициент усиления (по умолчанию 1.0)
 */
export function xavierUniform(tensor: Tensor, gain: number = 1.0): void {
  const [fanIn, fanOut] = calculateFanInFanOut(tensor);
  const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
  const bound = Math.sqrt(3.0) * std;

  for (let i = 0; i < tensor.size; i++) {
    (tensor.data as Float32Array)[i] = (Math.random() * 2 - 1) * bound;
  }
}

/**
 * Xavier/Glorot normal инициализация
 * @param tensor - Тензор для инициализации
 * @param gain - Коэффициент усиления
 */
export function xavierNormal(tensor: Tensor, gain: number = 1.0): void {
  const [fanIn, fanOut] = calculateFanInFanOut(tensor);
  const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));

  for (let i = 0; i < tensor.size; i += 2) {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    const mag = std * Math.sqrt(-2 * Math.log(u1));
    (tensor.data as Float32Array)[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < tensor.size) {
      (tensor.data as Float32Array)[i + 1] = mag * Math.sin(2 * Math.PI * u2);
    }
  }
}

/**
 * Kaiming/He uniform инициализация
 * Оптимальна для ReLU и его вариантов
 * @param tensor - Тензор для инициализации
 * @param a - Параметр для LeakyReLU (по умолчанию 0 для ReLU)
 * @param mode - 'fan_in' или 'fan_out'
 * @param nonlinearity - Тип нелинейности ('relu', 'leaky_relu')
 */
export function kaimingUniform(
  tensor: Tensor,
  a: number = 0,
  mode: 'fan_in' | 'fan_out' = 'fan_in',
  nonlinearity: 'relu' | 'leaky_relu' = 'relu'
): void {
  const [fanIn, fanOut] = calculateFanInFanOut(tensor);
  const fan = mode === 'fan_in' ? fanIn : fanOut;
  const gain = calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  const bound = Math.sqrt(3.0) * std;

  for (let i = 0; i < tensor.size; i++) {
    (tensor.data as Float32Array)[i] = (Math.random() * 2 - 1) * bound;
  }
}

/**
 * Kaiming/He normal инициализация
 * @param tensor - Тензор для инициализации
 * @param a - Параметр для LeakyReLU
 * @param mode - 'fan_in' или 'fan_out'
 * @param nonlinearity - Тип нелинейности
 */
export function kaimingNormal(
  tensor: Tensor,
  a: number = 0,
  mode: 'fan_in' | 'fan_out' = 'fan_in',
  nonlinearity: 'relu' | 'leaky_relu' = 'relu'
): void {
  const [fanIn, fanOut] = calculateFanInFanOut(tensor);
  const fan = mode === 'fan_in' ? fanIn : fanOut;
  const gain = calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);

  for (let i = 0; i < tensor.size; i += 2) {
    const u1 = Math.random();
    const u2 = Math.random();
    const mag = std * Math.sqrt(-2 * Math.log(u1));
    (tensor.data as Float32Array)[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < tensor.size) {
      (tensor.data as Float32Array)[i + 1] = mag * Math.sin(2 * Math.PI * u2);
    }
  }
}

/**
 * Uniform инициализация в заданном диапазоне
 * @param tensor - Тензор для инициализации
 * @param a - Нижняя граница
 * @param b - Верхняя граница
 */
export function uniform(tensor: Tensor, a: number = 0, b: number = 1): void {
  for (let i = 0; i < tensor.size; i++) {
    (tensor.data as Float32Array)[i] = a + Math.random() * (b - a);
  }
}

/**
 * Normal инициализация
 * @param tensor - Тензор для инициализации
 * @param mean - Среднее
 * @param std - Стандартное отклонение
 */
export function normal(tensor: Tensor, mean: number = 0, std: number = 1): void {
  for (let i = 0; i < tensor.size; i += 2) {
    const u1 = Math.random();
    const u2 = Math.random();
    const mag = std * Math.sqrt(-2 * Math.log(u1));
    (tensor.data as Float32Array)[i] = mag * Math.cos(2 * Math.PI * u2) + mean;
    if (i + 1 < tensor.size) {
      (tensor.data as Float32Array)[i + 1] = mag * Math.sin(2 * Math.PI * u2) + mean;
    }
  }
}

/**
 * Constant инициализация
 * @param tensor - Тензор для инициализации
 * @param value - Значение
 */
export function constant(tensor: Tensor, value: number): void {
  for (let i = 0; i < tensor.size; i++) {
    (tensor.data as Float32Array)[i] = value;
  }
}

/**
 * Инициализация нулями
 * @param tensor - Тензор для инициализации
 */
export function zeros_(tensor: Tensor): void {
  for (let i = 0; i < tensor.size; i++) {
    (tensor.data as Float32Array)[i] = 0;
  }
}

/**
 * Инициализация единицами
 * @param tensor - Тензор для инициализации
 */
export function ones_(tensor: Tensor): void {
  for (let i = 0; i < tensor.size; i++) {
    (tensor.data as Float32Array)[i] = 1;
  }
}

/**
 * Инициализация единичной матрицей
 * @param tensor - Тензор для инициализации (должен быть 2D)
 */
export function eye_(tensor: Tensor): void {
  if (tensor.ndim !== 2) {
    throw new Error('eye_ requires 2D tensor');
  }
  const [rows, cols] = tensor.shape;
  for (let i = 0; i < tensor.size; i++) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    (tensor.data as Float32Array)[i] = row === col ? 1 : 0;
  }
}

/**
 * Orthogonal инициализация
 * Генерирует ортогональную матрицу с помощью QR-разложения
 * @param tensor - Тензор для инициализации
 * @param gain - Коэффициент усиления
 */
export function orthogonal(tensor: Tensor, gain: number = 1.0): void {
  if (tensor.ndim < 2) {
    throw new Error('orthogonal requires at least 2D tensor');
  }

  const rows = tensor.shape[0];
  const cols = tensor.shape.slice(1).reduce((a, b) => a * b, 1);
  const flat = new Float32Array(rows * cols);

  // Генерируем случайную матрицу
  for (let i = 0; i < flat.length; i += 2) {
    const u1 = Math.random();
    const u2 = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u1));
    flat[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < flat.length) {
      flat[i + 1] = mag * Math.sin(2 * Math.PI * u2);
    }
  }

  // Упрощённая QR факторизация через Gram-Schmidt
  const Q = gramSchmidt(flat, rows, cols);

  for (let i = 0; i < tensor.size; i++) {
    (tensor.data as Float32Array)[i] = Q[i] * gain;
  }
}

/**
 * Gram-Schmidt ортогонализация
 */
function gramSchmidt(A: Float32Array, rows: number, cols: number): Float32Array {
  const Q = new Float32Array(A.length);
  const numVecs = Math.min(rows, cols);

  for (let i = 0; i < numVecs; i++) {
    // Копируем i-й вектор
    for (let j = 0; j < (rows > cols ? rows : cols); j++) {
      if (rows >= cols) {
        Q[j * cols + i] = A[j * cols + i];
      } else {
        Q[i * cols + j] = A[i * cols + j];
      }
    }

    // Ортогонализуем относительно предыдущих
    for (let k = 0; k < i; k++) {
      let dot = 0;
      const len = rows >= cols ? rows : cols;
      for (let j = 0; j < len; j++) {
        if (rows >= cols) {
          dot += Q[j * cols + k] * A[j * cols + i];
        } else {
          dot += Q[k * cols + j] * A[i * cols + j];
        }
      }
      for (let j = 0; j < len; j++) {
        if (rows >= cols) {
          Q[j * cols + i] -= dot * Q[j * cols + k];
        } else {
          Q[i * cols + j] -= dot * Q[k * cols + j];
        }
      }
    }

    // Нормализуем
    let norm = 0;
    const len = rows >= cols ? rows : cols;
    for (let j = 0; j < len; j++) {
      if (rows >= cols) {
        norm += Q[j * cols + i] * Q[j * cols + i];
      } else {
        norm += Q[i * cols + j] * Q[i * cols + j];
      }
    }
    norm = Math.sqrt(norm);
    if (norm > 1e-10) {
      for (let j = 0; j < len; j++) {
        if (rows >= cols) {
          Q[j * cols + i] /= norm;
        } else {
          Q[i * cols + j] /= norm;
        }
      }
    }
  }

  return Q;
}

/**
 * Вычисляет fan_in и fan_out для тензора весов
 * @param tensor - Тензор весов
 * @returns [fan_in, fan_out]
 */
export function calculateFanInFanOut(tensor: Tensor): [number, number] {
  const dims = tensor.ndim;
  if (dims < 2) {
    throw new Error('Fan-in/fan-out calculation requires at least 2D tensor');
  }

  const numInputFmaps = tensor.shape[1];
  const numOutputFmaps = tensor.shape[0];
  const receptiveFieldSize = dims > 2
    ? tensor.shape.slice(2).reduce((a, b) => a * b, 1)
    : 1;

  const fanIn = numInputFmaps * receptiveFieldSize;
  const fanOut = numOutputFmaps * receptiveFieldSize;

  return [fanIn, fanOut];
}

/**
 * Вычисляет gain для заданной нелинейности
 * @param nonlinearity - Тип активации
 * @param param - Параметр (для leaky_relu)
 * @returns Коэффициент gain
 */
export function calculateGain(nonlinearity: string, param?: number): number {
  switch (nonlinearity) {
    case 'linear':
    case 'conv1d':
    case 'conv2d':
    case 'conv3d':
      return 1;
    case 'sigmoid':
      return 1;
    case 'tanh':
      return 5 / 3;
    case 'relu':
      return Math.sqrt(2);
    case 'leaky_relu':
      const negativeSlope = param ?? 0.01;
      return Math.sqrt(2 / (1 + negativeSlope * negativeSlope));
    case 'selu':
      return 3 / 4;
    default:
      return 1;
  }
}
