/**
 * @fileoverview Утилиты для сериализации моделей
 * @description Сохранение и загрузка моделей и их состояния
 */

import { Tensor } from '../core/tensor';
import { Module } from '../nn/module';

/**
 * Сериализует state_dict модели в JSON-совместимый формат
 * @param stateDict - Словарь состояния модели
 * @returns Сериализованные данные
 */
export function serializeStateDict(stateDict: Map<string, Tensor>): object {
  const serialized: Record<string, {
    data: number[];
    shape: number[];
    dtype: string;
  }> = {};

  for (const [name, tensor] of stateDict) {
    serialized[name] = {
      data: Array.from(tensor.data),
      shape: [...tensor.shape],
      dtype: tensor.dtype,
    };
  }

  return serialized;
}

/**
 * Десериализует state_dict из JSON-формата
 * @param serialized - Сериализованные данные
 * @returns Map с тензорами
 */
export function deserializeStateDict(serialized: object): Map<string, Tensor> {
  const stateDict = new Map<string, Tensor>();

  for (const [name, data] of Object.entries(serialized as Record<string, {
    data: number[];
    shape: number[];
    dtype: string;
  }>)) {
    const tensor = new Tensor(data.data, data.shape);
    stateDict.set(name, tensor);
  }

  return stateDict;
}

/**
 * Сохраняет модель в файл (JSON формат)
 * @param model - Модель для сохранения
 * @param filepath - Путь к файлу
 */
export async function saveModel(model: Module, filepath: string): Promise<void> {
  const stateDict = model.stateDict();
  const serialized = serializeStateDict(stateDict);
  const json = JSON.stringify(serialized, null, 2);
  
  await Bun.write(filepath, json);
}

/**
 * Загружает состояние модели из файла
 * @param model - Модель для загрузки состояния
 * @param filepath - Путь к файлу
 */
export async function loadModel(model: Module, filepath: string): Promise<void> {
  const file = Bun.file(filepath);
  const json = await file.text();
  const serialized = JSON.parse(json);
  const stateDict = deserializeStateDict(serialized);
  model.loadStateDict(stateDict);
}

/**
 * Сохраняет только state_dict в файл
 * @param stateDict - Словарь состояния
 * @param filepath - Путь к файлу
 */
export async function saveStateDict(stateDict: Map<string, Tensor>, filepath: string): Promise<void> {
  const serialized = serializeStateDict(stateDict);
  const json = JSON.stringify(serialized, null, 2);
  await Bun.write(filepath, json);
}

/**
 * Загружает state_dict из файла
 * @param filepath - Путь к файлу
 * @returns Map с тензорами
 */
export async function loadStateDict(filepath: string): Promise<Map<string, Tensor>> {
  const file = Bun.file(filepath);
  const json = await file.text();
  const serialized = JSON.parse(json);
  return deserializeStateDict(serialized);
}

/**
 * Экспортирует модель в формат для inference (без градиентов)
 * @param model - Модель
 * @returns Объект для сериализации
 */
export function exportForInference(model: Module): object {
  model.eval();
  const stateDict = model.stateDict();
  
  return {
    version: '1.0.0',
    framework: 'brainy',
    state_dict: serializeStateDict(stateDict),
    metadata: {
      num_parameters: model.numParameters(),
      exported_at: new Date().toISOString(),
    },
  };
}
