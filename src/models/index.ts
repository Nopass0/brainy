/**
 * @fileoverview Модуль моделей
 * @description Экспорт готовых моделей для различных задач
 */

export {
  GPT,
  createSmallGPT,
  createMediumGPT,
  createLargeGPT,
} from './gpt';
export type { GPTConfig, GenerationConfig } from './gpt';

export {
  VAE,
  ConvVAE,
  createMNISTVAE,
  createCIFARVAE,
} from './vae';
export type { VAEConfig, VAEOutput } from './vae';
