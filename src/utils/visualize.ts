/**
 * @fileoverview Визуализация архитектуры моделей
 * @description Генерация ASCII-диаграмм из структуры модели
 */

import { Module } from '../nn/module';

/**
 * Опции визуализации
 */
export interface VisualizeOptions {
  /** Показывать размеры параметров */
  showParams?: boolean;
  /** Показывать формы тензоров */
  showShapes?: boolean;
  /** Максимальная ширина */
  maxWidth?: number;
  /** Стиль рамки */
  boxStyle?: 'simple' | 'double' | 'rounded';
  /** Показывать skip connections */
  showSkipConnections?: boolean;
}

/**
 * Информация о слое для визуализации
 */
interface LayerInfo {
  name: string;
  type: string;
  params: number;
  inputShape?: string;
  outputShape?: string;
  children: LayerInfo[];
  isSequential?: boolean;
  hasResidual?: boolean;
}

/**
 * Символы для рисования
 */
const BoxChars = {
  simple: {
    tl: '┌', tr: '┐', bl: '└', br: '┘',
    h: '─', v: '│', cross: '┼',
    lt: '├', rt: '┤', tt: '┬', bt: '┴',
    arrow: '↓', arrowRight: '→', arrowLeft: '←',
  },
  double: {
    tl: '╔', tr: '╗', bl: '╚', br: '╝',
    h: '═', v: '║', cross: '╬',
    lt: '╠', rt: '╣', tt: '╦', bt: '╩',
    arrow: '↓', arrowRight: '→', arrowLeft: '←',
  },
  rounded: {
    tl: '╭', tr: '╮', bl: '╰', br: '╯',
    h: '─', v: '│', cross: '┼',
    lt: '├', rt: '┤', tt: '┬', bt: '┴',
    arrow: '↓', arrowRight: '→', arrowLeft: '←',
  },
};

/**
 * Извлекает информацию о структуре модели
 */
function extractModelInfo(module: Module, name: string = 'Model'): LayerInfo {
  const children: LayerInfo[] = [];
  const modules = (module as any)._modules as Map<string, Module> | undefined;

  if (modules) {
    for (const [childName, childModule] of modules) {
      children.push(extractModelInfo(childModule, childName));
    }
  }

  // Определяем тип модуля
  const type = module.constructor.name;

  // Подсчитываем параметры
  let params = 0;
  const parameters = (module as any)._parameters as Map<string, any> | undefined;
  if (parameters) {
    for (const [, param] of parameters) {
      if (param?.data) {
        params += param.data.size;
      }
    }
  }

  // Определяем специальные свойства
  const isSequential = type === 'Sequential';
  const hasResidual = type.toLowerCase().includes('residual') ||
                      type.toLowerCase().includes('skip') ||
                      name.toLowerCase().includes('residual');

  // Извлекаем информацию о форме из свойств модуля
  let inputShape: string | undefined;
  let outputShape: string | undefined;

  if ((module as any).inFeatures !== undefined) {
    inputShape = `[*, ${(module as any).inFeatures}]`;
  }
  if ((module as any).outFeatures !== undefined) {
    outputShape = `[*, ${(module as any).outFeatures}]`;
  }
  if ((module as any).inChannels !== undefined) {
    inputShape = `[N, ${(module as any).inChannels}, H, W]`;
  }
  if ((module as any).outChannels !== undefined) {
    outputShape = `[N, ${(module as any).outChannels}, H', W']`;
  }

  return {
    name,
    type,
    params,
    inputShape,
    outputShape,
    children,
    isSequential,
    hasResidual,
  };
}

/**
 * Форматирует число параметров
 */
function formatParams(params: number): string {
  if (params >= 1e9) return `${(params / 1e9).toFixed(2)}B`;
  if (params >= 1e6) return `${(params / 1e6).toFixed(2)}M`;
  if (params >= 1e3) return `${(params / 1e3).toFixed(2)}K`;
  return params.toString();
}

/**
 * Создаёт строку с заданной шириной
 */
function padCenter(text: string, width: number): string {
  const padding = width - text.length;
  const left = Math.floor(padding / 2);
  const right = padding - left;
  return ' '.repeat(left) + text + ' '.repeat(right);
}

/**
 * Создаёт блок с рамкой
 */
function createBox(content: string[], width: number, chars: typeof BoxChars.simple): string[] {
  const innerWidth = width - 2;
  const lines: string[] = [];

  lines.push(chars.tl + chars.h.repeat(innerWidth) + chars.tr);

  for (const line of content) {
    const paddedLine = padCenter(line, innerWidth);
    lines.push(chars.v + paddedLine + chars.v);
  }

  lines.push(chars.bl + chars.h.repeat(innerWidth) + chars.br);

  return lines;
}

/**
 * Генерирует простую вертикальную диаграмму
 */
function generateVerticalDiagram(
  info: LayerInfo,
  options: VisualizeOptions,
  depth: number = 0
): string[] {
  const chars = BoxChars[options.boxStyle || 'simple'];
  const width = options.maxWidth || 40;
  const lines: string[] = [];
  const indent = '  '.repeat(depth);

  // Создаём контент блока
  const content: string[] = [];
  content.push(info.name);
  content.push(`(${info.type})`);

  if (options.showParams && info.params > 0) {
    content.push(`params: ${formatParams(info.params)}`);
  }

  if (options.showShapes) {
    if (info.inputShape) content.push(`in: ${info.inputShape}`);
    if (info.outputShape) content.push(`out: ${info.outputShape}`);
  }

  // Создаём блок
  const boxLines = createBox(content, width - depth * 2, chars);

  for (const line of boxLines) {
    lines.push(indent + line);
  }

  // Добавляем дочерние элементы
  if (info.children.length > 0) {
    // Стрелка вниз
    lines.push(indent + padCenter(chars.arrow, width - depth * 2));

    for (let i = 0; i < info.children.length; i++) {
      const child = info.children[i];
      const childLines = generateVerticalDiagram(child, options, depth + 1);
      lines.push(...childLines);

      if (i < info.children.length - 1) {
        lines.push(indent + '  ' + padCenter(chars.arrow, width - depth * 2 - 2));
      }
    }
  }

  return lines;
}

/**
 * Генерирует компактную диаграмму
 */
function generateCompactDiagram(info: LayerInfo, options: VisualizeOptions): string[] {
  const lines: string[] = [];
  const chars = BoxChars[options.boxStyle || 'simple'];

  function traverse(node: LayerInfo, prefix: string = '', isLast: boolean = true): void {
    const connector = isLast ? '└── ' : '├── ';
    const extension = isLast ? '    ' : '│   ';

    let text = `${node.type}`;
    if (node.name !== node.type) {
      text = `${node.name}: ${node.type}`;
    }

    if (options.showParams && node.params > 0) {
      text += ` (${formatParams(node.params)})`;
    }

    if (options.showShapes && node.inputShape) {
      text += ` ${node.inputShape} → ${node.outputShape || '?'}`;
    }

    lines.push(prefix + connector + text);

    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i];
      const isChildLast = i === node.children.length - 1;
      traverse(child, prefix + extension, isChildLast);
    }
  }

  lines.push(info.name + ` (${info.type})`);
  if (options.showParams) {
    lines.push(`Total params: ${formatParams(info.params)}`);
  }
  lines.push('');

  for (let i = 0; i < info.children.length; i++) {
    const isLast = i === info.children.length - 1;
    traverse(info.children[i], '', isLast);
  }

  return lines;
}

/**
 * Генерирует диаграмму с residual connections
 */
function generateResidualDiagram(info: LayerInfo, options: VisualizeOptions): string[] {
  const chars = BoxChars[options.boxStyle || 'rounded'];
  const width = options.maxWidth || 50;
  const lines: string[] = [];

  // Заголовок
  lines.push(chars.tl + chars.h.repeat(width - 2) + chars.tr);
  lines.push(chars.v + padCenter(`${info.name} (${info.type})`, width - 2) + chars.v);
  if (options.showParams) {
    lines.push(chars.v + padCenter(`Total: ${formatParams(info.params)} params`, width - 2) + chars.v);
  }
  lines.push(chars.bl + chars.h.repeat(width - 2) + chars.br);
  lines.push('');

  // Проходим по детям
  for (const child of info.children) {
    if (child.hasResidual || child.type.includes('Residual')) {
      // Рисуем residual block
      lines.push(padCenter('Input (x)', width));
      lines.push(padCenter(chars.arrow, width));
      lines.push(padCenter('┌──────┴──────┐', width));
      lines.push(padCenter(chars.arrow + '             ' + chars.arrow, width));

      // Основная ветка
      for (const subChild of child.children) {
        const boxContent = [subChild.type];
        if (options.showParams && subChild.params > 0) {
          boxContent.push(`(${formatParams(subChild.params)})`);
        }
        const subBox = createBox(boxContent, 16, chars);
        for (const boxLine of subBox) {
          lines.push(padCenter(boxLine + '      Identity', width));
        }
        lines.push(padCenter(chars.arrow + '             │', width));
      }

      lines.push(padCenter('└──────+──────┘', width));
      lines.push(padCenter(chars.arrow, width));
      lines.push(padCenter('Output', width));
      lines.push('');
    } else {
      // Обычный блок
      const boxContent = [child.name, `(${child.type})`];
      if (options.showParams && child.params > 0) {
        boxContent.push(`${formatParams(child.params)} params`);
      }
      const box = createBox(boxContent, Math.min(30, width - 10), chars);
      for (const boxLine of box) {
        lines.push(padCenter(boxLine, width));
      }
      lines.push(padCenter(chars.arrow, width));
    }
  }

  return lines;
}

/**
 * Генерирует flow-диаграмму
 */
function generateFlowDiagram(info: LayerInfo, options: VisualizeOptions): string[] {
  const lines: string[] = [];
  const chars = BoxChars[options.boxStyle || 'simple'];

  // Собираем все слои в линейную последовательность
  const layers: LayerInfo[] = [];

  function collectLayers(node: LayerInfo): void {
    if (node.children.length === 0 || node.params > 0) {
      layers.push(node);
    }
    for (const child of node.children) {
      collectLayers(child);
    }
  }

  collectLayers(info);

  // Заголовок
  lines.push(`${info.name} Architecture`);
  lines.push('=' .repeat(40));
  lines.push('');

  // Рисуем flow
  let flowLine = '';
  const flowParts: string[] = [];

  for (let i = 0; i < layers.length; i++) {
    const layer = layers[i];
    let part = layer.type;

    // Сокращаем длинные имена
    if (part.length > 8) {
      part = part.substring(0, 6) + '..';
    }

    flowParts.push(part);
  }

  // Создаём многострочную flow диаграмму
  const maxPerLine = 5;
  for (let i = 0; i < flowParts.length; i += maxPerLine) {
    const chunk = flowParts.slice(i, i + maxPerLine);
    const line = chunk.map(p => `[${p}]`).join(' → ');
    lines.push(line);
    if (i + maxPerLine < flowParts.length) {
      lines.push('    ↓');
    }
  }

  lines.push('');

  // Таблица параметров
  if (options.showParams) {
    lines.push('Layer Details:');
    lines.push('-'.repeat(50));
    lines.push('Layer'.padEnd(20) + 'Type'.padEnd(15) + 'Params');
    lines.push('-'.repeat(50));

    for (const layer of layers) {
      if (layer.params > 0) {
        lines.push(
          layer.name.padEnd(20) +
          layer.type.padEnd(15) +
          formatParams(layer.params)
        );
      }
    }
    lines.push('-'.repeat(50));
    lines.push('Total'.padEnd(35) + formatParams(info.params));
  }

  return lines;
}

/**
 * Визуализирует модель в виде ASCII-диаграммы
 */
export function visualize(
  model: Module,
  options: VisualizeOptions = {}
): string {
  const info = extractModelInfo(model, model.constructor.name);

  const defaultOptions: VisualizeOptions = {
    showParams: true,
    showShapes: false,
    maxWidth: 50,
    boxStyle: 'simple',
    showSkipConnections: true,
    ...options,
  };

  const lines = generateCompactDiagram(info, defaultOptions);
  return lines.join('\n');
}

/**
 * Визуализирует модель в вертикальном стиле
 */
export function visualizeVertical(
  model: Module,
  options: VisualizeOptions = {}
): string {
  const info = extractModelInfo(model, model.constructor.name);

  const defaultOptions: VisualizeOptions = {
    showParams: true,
    showShapes: true,
    maxWidth: 40,
    boxStyle: 'rounded',
    ...options,
  };

  const lines = generateVerticalDiagram(info, defaultOptions, 0);
  return lines.join('\n');
}

/**
 * Визуализирует модель в flow-стиле
 */
export function visualizeFlow(
  model: Module,
  options: VisualizeOptions = {}
): string {
  const info = extractModelInfo(model, model.constructor.name);

  const defaultOptions: VisualizeOptions = {
    showParams: true,
    showShapes: false,
    maxWidth: 60,
    boxStyle: 'simple',
    ...options,
  };

  const lines = generateFlowDiagram(info, defaultOptions);
  return lines.join('\n');
}

/**
 * Печатает визуализацию модели в консоль
 */
export function printModel(model: Module, style: 'compact' | 'vertical' | 'flow' = 'compact'): void {
  let output: string;

  switch (style) {
    case 'vertical':
      output = visualizeVertical(model);
      break;
    case 'flow':
      output = visualizeFlow(model);
      break;
    default:
      output = visualize(model);
  }

  console.log(output);
}

/**
 * Генерирует summary модели (как в PyTorch/Keras)
 */
export function summary(model: Module, inputShape?: number[]): string {
  const info = extractModelInfo(model, model.constructor.name);
  const lines: string[] = [];

  lines.push('═'.repeat(65));
  lines.push('Model Summary');
  lines.push('═'.repeat(65));

  lines.push('Layer (type)'.padEnd(30) + 'Output Shape'.padEnd(20) + 'Param #');
  lines.push('─'.repeat(65));

  let totalParams = 0;
  let trainableParams = 0;

  function printLayer(node: LayerInfo, depth: number = 0): void {
    const indent = '  '.repeat(depth);
    const name = indent + node.name + ` (${node.type})`;
    const shape = node.outputShape || 'N/A';
    const params = node.params;

    if (params > 0 || node.children.length === 0) {
      lines.push(name.padEnd(30) + shape.padEnd(20) + formatParams(params));
      totalParams += params;
      trainableParams += params; // Assuming all are trainable
    }

    for (const child of node.children) {
      printLayer(child, depth + 1);
    }
  }

  for (const child of info.children) {
    printLayer(child);
  }

  lines.push('═'.repeat(65));
  lines.push(`Total params: ${formatParams(totalParams)}`);
  lines.push(`Trainable params: ${formatParams(trainableParams)}`);
  lines.push(`Non-trainable params: 0`);
  lines.push('─'.repeat(65));

  if (inputShape) {
    lines.push(`Input shape: [${inputShape.join(', ')}]`);
  }

  // Оценка размера модели
  const sizeBytes = totalParams * 4; // Float32
  const sizeMB = sizeBytes / (1024 * 1024);
  lines.push(`Model size: ${sizeMB.toFixed(2)} MB (FP32)`);
  lines.push('═'.repeat(65));

  return lines.join('\n');
}

/**
 * Добавляет метод visualize к Module
 */
export function extendModuleWithVisualize(): void {
  (Module.prototype as any).visualize = function(style?: 'compact' | 'vertical' | 'flow'): string {
    switch (style) {
      case 'vertical':
        return visualizeVertical(this);
      case 'flow':
        return visualizeFlow(this);
      default:
        return visualize(this);
    }
  };

  (Module.prototype as any).print = function(style?: 'compact' | 'vertical' | 'flow'): void {
    printModel(this, style);
  };

  (Module.prototype as any).summary = function(inputShape?: number[]): string {
    return summary(this, inputShape);
  };
}

// Автоматически расширяем Module
extendModuleWithVisualize();
