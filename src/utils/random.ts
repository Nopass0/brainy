/**
 * @fileoverview Утилиты для генерации случайных чисел
 * @description Контроль seed и генерация случайных чисел
 */

/**
 * Простой генератор случайных чисел с seed (Mulberry32)
 */
class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed;
  }

  /**
   * Генерирует следующее случайное число [0, 1)
   */
  random(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /**
   * Генерирует случайное целое число в диапазоне [min, max]
   */
  randint(min: number, max: number): number {
    return Math.floor(this.random() * (max - min + 1)) + min;
  }

  /**
   * Генерирует случайное число из нормального распределения
   */
  randn(mean: number = 0, std: number = 1): number {
    const u1 = this.random();
    const u2 = this.random();
    const mag = Math.sqrt(-2 * Math.log(u1));
    return mag * Math.cos(2 * Math.PI * u2) * std + mean;
  }
}

let globalRng: SeededRandom | null = null;

/**
 * Устанавливает глобальный seed для воспроизводимости
 * @param seed - Seed значение
 */
export function manualSeed(seed: number): void {
  globalRng = new SeededRandom(seed);
}

/**
 * Получает текущий RNG (или создаёт новый на основе Math.random)
 */
export function getRng(): { random: () => number; randn: (mean?: number, std?: number) => number } {
  if (globalRng) {
    return globalRng;
  }
  return {
    random: Math.random,
    randn: (mean: number = 0, std: number = 1) => {
      const u1 = Math.random();
      const u2 = Math.random();
      const mag = Math.sqrt(-2 * Math.log(u1));
      return mag * Math.cos(2 * Math.PI * u2) * std + mean;
    },
  };
}

/**
 * Сбрасывает глобальный RNG
 */
export function resetRng(): void {
  globalRng = null;
}

/**
 * Генерирует случайное число [0, 1)
 */
export function random(): number {
  return getRng().random();
}

/**
 * Генерирует случайное целое число в диапазоне [min, max]
 */
export function randint(min: number, max: number): number {
  return Math.floor(getRng().random() * (max - min + 1)) + min;
}

/**
 * Генерирует случайное число из нормального распределения
 */
export function randn(mean: number = 0, std: number = 1): number {
  return getRng().randn(mean, std);
}

/**
 * Выбирает случайный элемент из массива
 */
export function choice<T>(array: T[]): T {
  const idx = Math.floor(getRng().random() * array.length);
  return array[idx];
}

/**
 * Перемешивает массив (Fisher-Yates)
 */
export function shuffle<T>(array: T[]): T[] {
  const result = [...array];
  const rng = getRng();
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
