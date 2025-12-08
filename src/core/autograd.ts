/**
 * @fileoverview Движок автоматического дифференцирования (autograd)
 * @description Реализация обратного распространения градиентов через вычислительный граф
 */

import type { Tensor } from './tensor';

/**
 * Тип функции для вычисления градиентов
 * Получает градиент выхода и возвращает градиенты для каждого входа
 */
export type GradientFunction = (gradOutput: Tensor) => Tensor[];

/**
 * Контекст для сохранения промежуточных значений при forward pass
 * Используется backward функцией для вычисления градиентов
 */
export class GradContext {
  /** Сохранённые тензоры для backward pass */
  private savedTensors: Tensor[] = [];
  /** Произвольные сохранённые значения */
  private savedValues: Map<string, unknown> = new Map();

  /**
   * Сохраняет тензоры для использования в backward pass
   * @param tensors - Тензоры для сохранения
   */
  saveTensors(...tensors: Tensor[]): void {
    this.savedTensors.push(...tensors);
  }

  /**
   * Получает сохранённые тензоры
   * @returns Массив сохранённых тензоров
   */
  getSavedTensors(): Tensor[] {
    return this.savedTensors;
  }

  /**
   * Сохраняет произвольное значение по ключу
   * @param key - Ключ
   * @param value - Значение
   */
  saveValue(key: string, value: unknown): void {
    this.savedValues.set(key, value);
  }

  /**
   * Получает сохранённое значение по ключу
   * @param key - Ключ
   * @returns Сохранённое значение
   */
  getValue<T>(key: string): T | undefined {
    return this.savedValues.get(key) as T | undefined;
  }
}

/**
 * Узел вычислительного графа для autograd
 * Хранит информацию о операции и её входах для обратного прохода
 */
export class GradNode {
  /** Функция для вычисления градиентов */
  readonly gradFn: GradientFunction;
  /** Входные тензоры операции */
  readonly inputs: Tensor[];
  /** Контекст с сохранёнными значениями */
  readonly context: GradContext;

  /**
   * Создаёт новый узел вычислительного графа
   * @param gradFn - Функция для вычисления градиентов
   * @param inputs - Входные тензоры
   * @param context - Контекст с сохранёнными значениями
   */
  constructor(gradFn: GradientFunction, inputs: Tensor[], context: GradContext) {
    this.gradFn = gradFn;
    this.inputs = inputs;
    this.context = context;
  }
}

/**
 * Выполняет топологическую сортировку графа для backward pass
 * @param root - Корневой тензор (обычно loss)
 * @returns Отсортированный список тензоров
 */
export function topologicalSort(root: Tensor): Tensor[] {
  const visited = new Set<Tensor>();
  const sorted: Tensor[] = [];

  function visit(tensor: Tensor): void {
    if (visited.has(tensor)) return;
    visited.add(tensor);

    if (tensor.gradNode) {
      for (const input of tensor.gradNode.inputs) {
        visit(input);
      }
    }

    sorted.push(tensor);
  }

  visit(root);
  return sorted.reverse();
}

/**
 * Выполняет backward pass от корневого тензора
 * Накапливает градиенты во всех тензорах с requires_grad=true
 * @param root - Корневой тензор (должен быть скаляром)
 * @param gradOutput - Начальный градиент (по умолчанию 1)
 */
export function backward(root: Tensor, gradOutput?: Tensor): void {
  // Импортируем здесь чтобы избежать циклических зависимостей
  const { ones } = require('./tensor');
  
  // Инициализируем градиент корня
  if (!gradOutput) {
    if (root.size !== 1) {
      throw new Error('backward() requires gradOutput for non-scalar tensors');
    }
    gradOutput = ones(root.shape);
  }

  root.grad = gradOutput;

  // Топологическая сортировка
  const sorted = topologicalSort(root);

  // Обратный проход
  for (const tensor of sorted) {
    if (!tensor.gradNode || !tensor.grad) continue;

    const grads = tensor.gradNode.gradFn(tensor.grad);

    for (let i = 0; i < tensor.gradNode.inputs.length; i++) {
      const input = tensor.gradNode.inputs[i];
      const inputGrad = grads[i];

      if (!input.requiresGrad) continue;

      if (input.grad) {
        // Накапливаем градиенты
        input.grad = input.grad.add(inputGrad);
      } else {
        input.grad = inputGrad;
      }
    }
  }
}

/**
 * Режим без градиентов - отключает отслеживание операций
 * Полезно для inference или валидации
 */
let noGradMode = false;

/**
 * Проверяет, включён ли режим без градиентов
 * @returns true если градиенты отключены
 */
export function isNoGradEnabled(): boolean {
  return noGradMode;
}

/**
 * Выполняет функцию без отслеживания градиентов
 * @param fn - Функция для выполнения
 * @returns Результат функции
 */
export function noGrad<T>(fn: () => T): T {
  const prev = noGradMode;
  noGradMode = true;
  try {
    return fn();
  } finally {
    noGradMode = prev;
  }
}

/**
 * Контекстный менеджер для отключения градиентов
 * Использование: const ctx = noGradContext(); ... ctx.exit();
 */
export function noGradContext(): { exit: () => void } {
  const prev = noGradMode;
  noGradMode = true;
  return {
    exit: () => {
      noGradMode = prev;
    }
  };
}
