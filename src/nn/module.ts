/**
 * @fileoverview Базовый класс Module для нейронных сетей
 * @description Абстракция для создания нейросетевых моделей с управлением параметрами,
 * режимами train/eval и сериализацией
 */

import { Tensor, zeros, randn } from '../core/tensor';
import { noGrad } from '../core/autograd';

/**
 * Параметр модели - тензор с автоматическим отслеживанием градиентов
 * Аналог nn.Parameter в PyTorch
 */
export class Parameter {
  /** Тензор данных */
  data: Tensor;
  /** Имя параметра */
  name: string;

  /**
   * Создаёт новый параметр
   * @param data - Тензор или форма для инициализации
   * @param name - Имя параметра
   */
  constructor(data: Tensor | number[], name: string = '') {
    if (data instanceof Tensor) {
      // Создаём копию с requiresGrad=true
      this.data = new Tensor(data.data, data.shape, {
        dtype: data.dtype,
        requiresGrad: true,
      });
    } else {
      // Инициализация случайными значениями
      this.data = randn(data, 0, 0.01, { requiresGrad: true });
    }
    this.name = name;
  }

  /**
   * Обнуляет градиент параметра
   */
  zeroGrad(): void {
    this.data.zeroGrad();
  }

  /**
   * Возвращает форму параметра
   */
  get shape(): readonly number[] {
    return this.data.shape;
  }

  /**
   * Возвращает градиент
   */
  get grad(): Tensor | null {
    return this.data.grad;
  }
}

/**
 * Абстрактный базовый класс для всех нейросетевых модулей
 * Аналог nn.Module в PyTorch
 * 
 * @example
 * class MyModel extends Module {
 *   constructor() {
 *     super();
 *     this.linear = new Linear(10, 5);
 *     this.registerModule('linear', this.linear);
 *   }
 *   
 *   forward(x: Tensor): Tensor {
 *     return this.linear.forward(x);
 *   }
 * }
 */
export abstract class Module {
  /** Режим обучения (true) или inference (false) */
  private _training: boolean = true;
  /** Зарегистрированные параметры */
  private _parameters: Map<string, Parameter> = new Map();
  /** Зарегистрированные подмодули */
  private _modules: Map<string, Module> = new Map();

  /**
   * Прямой проход нейронной сети
   * @param input - Входной тензор
   * @returns Выходной тензор
   */
  abstract forward(input: Tensor): Tensor;

  /**
   * Вызов модуля как функции
   * @param input - Входной тензор
   * @returns Результат forward()
   */
  call(input: Tensor): Tensor {
    return this.forward(input);
  }

  /**
   * Регистрирует параметр в модуле
   * @param name - Имя параметра
   * @param param - Параметр
   */
  protected registerParameter(name: string, param: Parameter): void {
    param.name = name;
    this._parameters.set(name, param);
  }

  /**
   * Регистрирует подмодуль
   * @param name - Имя подмодуля
   * @param module - Подмодуль
   */
  protected registerModule(name: string, module: Module): void {
    this._modules.set(name, module);
  }

  /**
   * Возвращает итератор по всем параметрам модуля (включая подмодули)
   * @param recurse - Включать параметры подмодулей
   * @yields Параметры модуля
   */
  *parameters(recurse: boolean = true): Generator<Parameter> {
    // Собственные параметры
    for (const param of this._parameters.values()) {
      yield param;
    }

    // Параметры подмодулей
    if (recurse) {
      for (const module of this._modules.values()) {
        yield* module.parameters(true);
      }
    }
  }

  /**
   * Возвращает массив всех параметров
   * @returns Массив параметров
   */
  getParameters(): Parameter[] {
    return Array.from(this.parameters());
  }

  /**
   * Возвращает итератор по именованным параметрам
   * @param prefix - Префикс для имён
   * @param recurse - Включать подмодули
   * @yields Пары [имя, параметр]
   */
  *namedParameters(prefix: string = '', recurse: boolean = true): Generator<[string, Parameter]> {
    for (const [name, param] of this._parameters) {
      yield [prefix ? `${prefix}.${name}` : name, param];
    }

    if (recurse) {
      for (const [name, module] of this._modules) {
        const modulePrefix = prefix ? `${prefix}.${name}` : name;
        yield* module.namedParameters(modulePrefix, true);
      }
    }
  }

  /**
   * Возвращает итератор по подмодулям
   * @yields Подмодули
   */
  *modules(): Generator<Module> {
    yield this;
    for (const module of this._modules.values()) {
      yield* module.modules();
    }
  }

  /**
   * Возвращает итератор по именованным подмодулям
   * @param prefix - Префикс для имён
   * @yields Пары [имя, модуль]
   */
  *namedModules(prefix: string = ''): Generator<[string, Module]> {
    yield [prefix, this];
    for (const [name, module] of this._modules) {
      const modulePrefix = prefix ? `${prefix}.${name}` : name;
      yield* module.namedModules(modulePrefix);
    }
  }

  /**
   * Переключает модуль в режим обучения
   * @returns this для цепочки вызовов
   */
  train(): this {
    this._training = true;
    for (const module of this._modules.values()) {
      module.train();
    }
    return this;
  }

  /**
   * Переключает модуль в режим inference
   * @returns this для цепочки вызовов
   */
  eval(): this {
    this._training = false;
    for (const module of this._modules.values()) {
      module.eval();
    }
    return this;
  }

  /**
   * Проверяет, находится ли модуль в режиме обучения
   */
  get training(): boolean {
    return this._training;
  }

  /**
   * Обнуляет градиенты всех параметров
   */
  zeroGrad(): void {
    for (const param of this.parameters()) {
      param.zeroGrad();
    }
  }

  /**
   * Подсчитывает общее количество параметров
   * @returns Количество параметров
   */
  numParameters(): number {
    let count = 0;
    for (const param of this.parameters()) {
      count += param.data.size;
    }
    return count;
  }

  /**
   * Возвращает состояние модуля (для сериализации)
   * @returns Словарь с параметрами
   */
  stateDict(): Map<string, Tensor> {
    const state = new Map<string, Tensor>();
    for (const [name, param] of this.namedParameters()) {
      state.set(name, param.data.clone());
    }
    return state;
  }

  /**
   * Загружает состояние модуля
   * @param state - Словарь с параметрами
   */
  loadStateDict(state: Map<string, Tensor>): void {
    for (const [name, param] of this.namedParameters()) {
      const loadedTensor = state.get(name);
      if (loadedTensor) {
        // Копируем данные
        for (let i = 0; i < param.data.size; i++) {
          (param.data.data as Float32Array)[i] = loadedTensor.data[i];
        }
      }
    }
  }

  /**
   * Применяет функцию ко всем модулям
   * @param fn - Функция для применения
   * @returns this для цепочки вызовов
   */
  apply(fn: (module: Module) => void): this {
    for (const module of this.modules()) {
      fn(module);
    }
    return this;
  }

  /**
   * Строковое представление модуля
   */
  toString(): string {
    const lines: string[] = [this.constructor.name + '('];
    for (const [name, module] of this._modules) {
      const moduleStr = module.toString().split('\n').map(l => '  ' + l).join('\n');
      lines.push(`  (${name}): ${moduleStr.trim()}`);
    }
    lines.push(')');
    return lines.join('\n');
  }
}

/**
 * Контейнер для последовательного применения модулей
 * Аналог nn.Sequential в PyTorch
 * 
 * @example
 * const model = new Sequential(
 *   new Linear(784, 256),
 *   new ReLU(),
 *   new Linear(256, 10)
 * );
 */
export class Sequential extends Module {
  private layers: Module[] = [];

  /**
   * Создаёт последовательность модулей
   * @param modules - Модули для последовательного применения
   */
  constructor(...modules: Module[]) {
    super();
    this.layers = modules;
    modules.forEach((m, i) => this.registerModule(i.toString(), m));
  }

  /**
   * Добавляет модуль в конец последовательности
   * @param module - Модуль для добавления
   */
  add(module: Module): void {
    const idx = this.layers.length;
    this.layers.push(module);
    this.registerModule(idx.toString(), module);
  }

  /**
   * Последовательно применяет все модули к входу
   * @param input - Входной тензор
   * @returns Выходной тензор
   */
  forward(input: Tensor): Tensor {
    let x = input;
    for (const layer of this.layers) {
      x = layer.forward(x);
    }
    return x;
  }

  /**
   * Возвращает количество слоёв
   */
  get length(): number {
    return this.layers.length;
  }
}

export { Parameter as Param };
