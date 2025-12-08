/**
 * @fileoverview Variational Autoencoder (VAE) для генерации изображений
 * @description Модель для обучения и генерации изображений
 */

import { Tensor, tensor, zeros, randn } from '../core/tensor';
import { DType } from '../core/dtype';
import { Module, Parameter, Sequential } from '../nn/module';
import { Linear, Conv2d, BatchNorm2d, Dropout, Flatten } from '../nn/layers';
import { ReLU, LeakyReLU, Sigmoid, Tanh } from '../nn/activations';
import { computeSize } from '../core/shape';

/**
 * Конфигурация VAE
 */
export interface VAEConfig {
  /** Размер входного изображения (квадратное) */
  imageSize: number;
  /** Количество каналов изображения (1 для grayscale, 3 для RGB) */
  imageChannels: number;
  /** Размер латентного пространства */
  latentDim: number;
  /** Размеры скрытых слоёв encoder */
  encoderHiddenDims?: number[];
  /** Размеры скрытых слоёв decoder */
  decoderHiddenDims?: number[];
  /** Использовать batch normalization */
  useBatchNorm?: boolean;
  /** Вес KL divergence в loss */
  klWeight?: number;
}

/**
 * Результат forward pass VAE
 */
export interface VAEOutput {
  /** Реконструкция */
  reconstruction: Tensor;
  /** Среднее латентного распределения */
  mu: Tensor;
  /** Log variance латентного распределения */
  logVar: Tensor;
  /** Сэмпл из латентного пространства */
  z: Tensor;
}

/**
 * Encoder часть VAE
 */
class VAEEncoder extends Module {
  private layers: Module[] = [];
  private fcMu: Linear;
  private fcLogVar: Linear;
  private flattenSize: number;

  constructor(config: VAEConfig) {
    super();

    const hiddenDims = config.encoderHiddenDims || [32, 64, 128, 256];
    let inChannels = config.imageChannels;
    let currentSize = config.imageSize;

    // Convolutional layers
    for (let i = 0; i < hiddenDims.length; i++) {
      const outChannels = hiddenDims[i];

      const conv = new Conv2d(inChannels, outChannels, 3, 2, 1);
      this.layers.push(conv);
      this.registerModule(`conv_${i}`, conv);

      if (config.useBatchNorm) {
        const bn = new BatchNorm2d(outChannels);
        this.layers.push(bn);
        this.registerModule(`bn_${i}`, bn);
      }

      const relu = new LeakyReLU(0.2);
      this.layers.push(relu);
      this.registerModule(`relu_${i}`, relu);

      inChannels = outChannels;
      currentSize = Math.floor((currentSize + 2 * 1 - 3) / 2 + 1); // После conv с stride=2
    }

    // Flatten и FC слои для mu и logvar
    this.flattenSize = hiddenDims[hiddenDims.length - 1] * currentSize * currentSize;

    this.fcMu = new Linear(this.flattenSize, config.latentDim);
    this.fcLogVar = new Linear(this.flattenSize, config.latentDim);

    this.registerModule('fc_mu', this.fcMu);
    this.registerModule('fc_logvar', this.fcLogVar);
  }

  forward(x: Tensor): { mu: Tensor; logVar: Tensor } {
    let hidden = x;

    for (const layer of this.layers) {
      hidden = layer.forward(hidden);
    }

    // Flatten
    const batchSize = hidden.shape[0];
    hidden = hidden.reshape(batchSize, -1);

    // Получаем mu и logvar
    const mu = this.fcMu.forward(hidden);
    const logVar = this.fcLogVar.forward(hidden);

    return { mu, logVar };
  }
}

/**
 * Decoder часть VAE
 */
class VAEDecoder extends Module {
  private fcInput: Linear;
  private layers: Module[] = [];
  private initialSize: number;
  private initialChannels: number;

  constructor(config: VAEConfig) {
    super();

    const hiddenDims = config.decoderHiddenDims ||
      (config.encoderHiddenDims || [32, 64, 128, 256]).slice().reverse();

    // Вычисляем начальный размер после всех свёрток encoder
    let currentSize = config.imageSize;
    const numConvs = (config.encoderHiddenDims || [32, 64, 128, 256]).length;
    for (let i = 0; i < numConvs; i++) {
      currentSize = Math.floor((currentSize + 2 * 1 - 3) / 2 + 1);
    }

    this.initialSize = currentSize;
    this.initialChannels = hiddenDims[0];

    // FC слой из латентного пространства
    this.fcInput = new Linear(
      config.latentDim,
      this.initialChannels * this.initialSize * this.initialSize
    );
    this.registerModule('fc_input', this.fcInput);

    // Deconvolutional (transposed conv) layers
    let inChannels = hiddenDims[0];

    for (let i = 0; i < hiddenDims.length - 1; i++) {
      const outChannels = hiddenDims[i + 1];

      // Используем обычную свёртку + upsample вместо transposed conv для простоты
      const conv = new Conv2d(inChannels, outChannels, 3, 1, 1);
      this.layers.push(conv);
      this.registerModule(`conv_${i}`, conv);

      if (config.useBatchNorm) {
        const bn = new BatchNorm2d(outChannels);
        this.layers.push(bn);
        this.registerModule(`bn_${i}`, bn);
      }

      const relu = new ReLU();
      this.layers.push(relu);
      this.registerModule(`relu_${i}`, relu);

      inChannels = outChannels;
    }

    // Финальный слой для восстановления изображения
    const finalConv = new Conv2d(inChannels, config.imageChannels, 3, 1, 1);
    this.layers.push(finalConv);
    this.registerModule('final_conv', finalConv);

    const sigmoid = new Sigmoid();
    this.layers.push(sigmoid);
    this.registerModule('sigmoid', sigmoid);
  }

  forward(z: Tensor): Tensor {
    const batchSize = z.shape[0];

    // FC + reshape
    let hidden = this.fcInput.forward(z);
    hidden = new ReLU().forward(hidden);
    hidden = hidden.reshape(batchSize, this.initialChannels, this.initialSize, this.initialSize);

    // Декодирование с upsampling
    for (let i = 0; i < this.layers.length; i++) {
      hidden = this.layers[i].forward(hidden);

      // Upsample после каждого conv блока (кроме последнего)
      if (i < this.layers.length - 2 && this.layers[i] instanceof Conv2d) {
        hidden = this.upsample(hidden, 2);
      }
    }

    return hidden;
  }

  /**
   * Простой upsample (nearest neighbor)
   */
  private upsample(x: Tensor, scale: number): Tensor {
    const [batch, channels, height, width] = x.shape;
    const newHeight = height * scale;
    const newWidth = width * scale;

    const resultData = new Float32Array(batch * channels * newHeight * newWidth);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let h = 0; h < newHeight; h++) {
          for (let w = 0; w < newWidth; w++) {
            const srcH = Math.floor(h / scale);
            const srcW = Math.floor(w / scale);
            const srcIdx = ((b * channels + c) * height + srcH) * width + srcW;
            const dstIdx = ((b * channels + c) * newHeight + h) * newWidth + w;
            resultData[dstIdx] = x.data[srcIdx];
          }
        }
      }
    }

    return new Tensor(resultData, [batch, channels, newHeight, newWidth], {
      requiresGrad: x.requiresGrad,
    });
  }
}

/**
 * Variational Autoencoder
 */
export class VAE extends Module {
  private config: VAEConfig;
  private encoder: VAEEncoder;
  private decoder: VAEDecoder;

  constructor(config: VAEConfig) {
    super();

    this.config = {
      useBatchNorm: true,
      klWeight: 0.00025,
      ...config,
    };

    this.encoder = new VAEEncoder(this.config);
    this.decoder = new VAEDecoder(this.config);

    this.registerModule('encoder', this.encoder);
    this.registerModule('decoder', this.decoder);
  }

  /**
   * Reparameterization trick
   * z = mu + std * epsilon, где epsilon ~ N(0, 1)
   */
  private reparameterize(mu: Tensor, logVar: Tensor): Tensor {
    // std = exp(0.5 * logVar)
    const std = logVar.mul(0.5).exp();

    // epsilon ~ N(0, 1)
    const eps = randn(mu.shape);

    // z = mu + std * eps
    return mu.add(std.mul(eps));
  }

  /**
   * Forward pass
   */
  forward(x: Tensor): VAEOutput {
    // Encode
    const { mu, logVar } = this.encoder.forward(x);

    // Reparameterize
    const z = this.reparameterize(mu, logVar);

    // Decode
    const reconstruction = this.decoder.forward(z);

    return { reconstruction, mu, logVar, z };
  }

  /**
   * Вычисляет ELBO loss
   * loss = reconstruction_loss + kl_weight * kl_divergence
   */
  computeLoss(x: Tensor, output: VAEOutput): {
    loss: Tensor;
    reconLoss: number;
    klLoss: number;
  } {
    const { reconstruction, mu, logVar } = output;

    // Reconstruction loss (MSE или BCE)
    const reconLoss = this.reconstructionLoss(x, reconstruction);

    // KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    const klLoss = this.klDivergence(mu, logVar);

    // Total loss
    const totalLoss = reconLoss + this.config.klWeight! * klLoss;

    return {
      loss: tensor([[totalLoss]], { requiresGrad: true }),
      reconLoss,
      klLoss,
    };
  }

  /**
   * Reconstruction loss (MSE)
   */
  private reconstructionLoss(x: Tensor, reconstruction: Tensor): number {
    let loss = 0;
    for (let i = 0; i < x.size; i++) {
      const diff = x.data[i] - reconstruction.data[i];
      loss += diff * diff;
    }
    return loss / x.size;
  }

  /**
   * KL Divergence для нормального распределения
   */
  private klDivergence(mu: Tensor, logVar: Tensor): number {
    // KL = -0.5 * sum(1 + logVar - mu^2 - exp(logVar))
    let kl = 0;
    for (let i = 0; i < mu.size; i++) {
      const m = mu.data[i];
      const lv = logVar.data[i];
      kl += -0.5 * (1 + lv - m * m - Math.exp(lv));
    }
    return kl / mu.shape[0]; // Среднее по batch
  }

  /**
   * Генерирует изображения из случайного шума
   */
  generate(numSamples: number = 1): Tensor {
    this.eval();

    // Сэмплируем из стандартного нормального распределения
    const z = randn([numSamples, this.config.latentDim]);

    // Декодируем
    return this.decoder.forward(z);
  }

  /**
   * Интерполирует между двумя точками в латентном пространстве
   */
  interpolate(x1: Tensor, x2: Tensor, numSteps: number = 10): Tensor[] {
    this.eval();

    // Кодируем
    const { mu: mu1 } = this.encoder.forward(x1);
    const { mu: mu2 } = this.encoder.forward(x2);

    const results: Tensor[] = [];

    for (let i = 0; i <= numSteps; i++) {
      const alpha = i / numSteps;

      // Линейная интерполяция в латентном пространстве
      const z = mu1.mul(1 - alpha).add(mu2.mul(alpha));

      // Декодируем
      const img = this.decoder.forward(z);
      results.push(img);
    }

    return results;
  }

  /**
   * Реконструирует изображение
   */
  reconstruct(x: Tensor): Tensor {
    this.eval();
    const { reconstruction } = this.forward(x);
    return reconstruction;
  }

  /**
   * Получает латентное представление
   */
  encode(x: Tensor): Tensor {
    const { mu } = this.encoder.forward(x);
    return mu;
  }

  /**
   * Декодирует из латентного пространства
   */
  decode(z: Tensor): Tensor {
    return this.decoder.forward(z);
  }

  /**
   * Получает конфигурацию
   */
  getConfig(): VAEConfig {
    return { ...this.config };
  }
}

/**
 * Convolutional VAE с более мощной архитектурой
 */
export class ConvVAE extends Module {
  private config: VAEConfig;
  private encoderConvs: Module[] = [];
  private decoderConvs: Module[] = [];
  private fcMu: Linear;
  private fcLogVar: Linear;
  private fcDecode: Linear;
  private flattenSize: number;
  private decoderInitSize: number;
  private decoderInitChannels: number;

  constructor(config: VAEConfig) {
    super();

    this.config = {
      useBatchNorm: true,
      klWeight: 0.00025,
      encoderHiddenDims: [32, 64, 128, 256],
      ...config,
    };

    const hiddenDims = this.config.encoderHiddenDims!;

    // === Encoder ===
    let inChannels = config.imageChannels;
    let currentSize = config.imageSize;

    for (let i = 0; i < hiddenDims.length; i++) {
      const outChannels = hiddenDims[i];

      // Conv block: Conv -> BN -> LeakyReLU
      const conv = new Conv2d(inChannels, outChannels, 4, 2, 1);
      this.encoderConvs.push(conv);
      this.registerModule(`enc_conv_${i}`, conv);

      if (this.config.useBatchNorm && i > 0) {
        const bn = new BatchNorm2d(outChannels);
        this.encoderConvs.push(bn);
        this.registerModule(`enc_bn_${i}`, bn);
      }

      const act = new LeakyReLU(0.2);
      this.encoderConvs.push(act);
      this.registerModule(`enc_act_${i}`, act);

      inChannels = outChannels;
      currentSize = Math.floor(currentSize / 2);
    }

    this.flattenSize = hiddenDims[hiddenDims.length - 1] * currentSize * currentSize;
    this.decoderInitSize = currentSize;
    this.decoderInitChannels = hiddenDims[hiddenDims.length - 1];

    // FC layers for mu and logvar
    this.fcMu = new Linear(this.flattenSize, config.latentDim);
    this.fcLogVar = new Linear(this.flattenSize, config.latentDim);
    this.registerModule('fc_mu', this.fcMu);
    this.registerModule('fc_logvar', this.fcLogVar);

    // === Decoder ===
    this.fcDecode = new Linear(config.latentDim, this.flattenSize);
    this.registerModule('fc_decode', this.fcDecode);

    const decoderDims = [...hiddenDims].reverse();

    for (let i = 0; i < decoderDims.length - 1; i++) {
      const inCh = decoderDims[i];
      const outCh = decoderDims[i + 1];

      // Transposed conv simulation: upsample + conv
      const conv = new Conv2d(inCh, outCh, 3, 1, 1);
      this.decoderConvs.push(conv);
      this.registerModule(`dec_conv_${i}`, conv);

      if (this.config.useBatchNorm) {
        const bn = new BatchNorm2d(outCh);
        this.decoderConvs.push(bn);
        this.registerModule(`dec_bn_${i}`, bn);
      }

      const act = new ReLU();
      this.decoderConvs.push(act);
      this.registerModule(`dec_act_${i}`, act);
    }

    // Final layer
    const finalConv = new Conv2d(decoderDims[decoderDims.length - 1], config.imageChannels, 3, 1, 1);
    this.decoderConvs.push(finalConv);
    this.registerModule('dec_final', finalConv);

    const sigmoid = new Sigmoid();
    this.decoderConvs.push(sigmoid);
    this.registerModule('dec_sigmoid', sigmoid);
  }

  /**
   * Encoder forward
   */
  private encodeForward(x: Tensor): { mu: Tensor; logVar: Tensor } {
    let hidden = x;

    for (const layer of this.encoderConvs) {
      hidden = layer.forward(hidden);
    }

    // Flatten
    const batchSize = hidden.shape[0];
    hidden = hidden.reshape(batchSize, -1);

    const mu = this.fcMu.forward(hidden);
    const logVar = this.fcLogVar.forward(hidden);

    return { mu, logVar };
  }

  /**
   * Decoder forward
   */
  private decodeForward(z: Tensor): Tensor {
    const batchSize = z.shape[0];

    // FC + reshape
    let hidden = this.fcDecode.forward(z);
    hidden = new ReLU().forward(hidden);
    hidden = hidden.reshape(
      batchSize,
      this.decoderInitChannels,
      this.decoderInitSize,
      this.decoderInitSize
    );

    // Decode with upsampling
    let layerIdx = 0;
    for (const layer of this.decoderConvs) {
      // Upsample before conv (except first and last)
      if (layer instanceof Conv2d && layerIdx > 0 && layerIdx < this.decoderConvs.length - 2) {
        hidden = this.upsample(hidden, 2);
      }

      hidden = layer.forward(hidden);
      layerIdx++;
    }

    return hidden;
  }

  /**
   * Upsample (nearest neighbor)
   */
  private upsample(x: Tensor, scale: number): Tensor {
    const [batch, channels, height, width] = x.shape;
    const newHeight = height * scale;
    const newWidth = width * scale;

    const resultData = new Float32Array(batch * channels * newHeight * newWidth);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let h = 0; h < newHeight; h++) {
          for (let w = 0; w < newWidth; w++) {
            const srcH = Math.floor(h / scale);
            const srcW = Math.floor(w / scale);
            const srcIdx = ((b * channels + c) * height + srcH) * width + srcW;
            const dstIdx = ((b * channels + c) * newHeight + h) * newWidth + w;
            resultData[dstIdx] = x.data[srcIdx];
          }
        }
      }
    }

    return new Tensor(resultData, [batch, channels, newHeight, newWidth], {
      requiresGrad: x.requiresGrad,
    });
  }

  /**
   * Reparameterization trick
   */
  private reparameterize(mu: Tensor, logVar: Tensor): Tensor {
    const std = logVar.mul(0.5).exp();
    const eps = randn(mu.shape);
    return mu.add(std.mul(eps));
  }

  forward(x: Tensor): VAEOutput {
    const { mu, logVar } = this.encodeForward(x);
    const z = this.reparameterize(mu, logVar);
    const reconstruction = this.decodeForward(z);
    return { reconstruction, mu, logVar, z };
  }

  computeLoss(x: Tensor, output: VAEOutput): {
    loss: Tensor;
    reconLoss: number;
    klLoss: number;
  } {
    const { reconstruction, mu, logVar } = output;

    // MSE reconstruction loss
    let reconLoss = 0;
    for (let i = 0; i < x.size; i++) {
      const diff = x.data[i] - reconstruction.data[i];
      reconLoss += diff * diff;
    }
    reconLoss /= x.size;

    // KL divergence
    let klLoss = 0;
    for (let i = 0; i < mu.size; i++) {
      const m = mu.data[i];
      const lv = logVar.data[i];
      klLoss += -0.5 * (1 + lv - m * m - Math.exp(lv));
    }
    klLoss /= mu.shape[0];

    const totalLoss = reconLoss + this.config.klWeight! * klLoss;

    return {
      loss: tensor([[totalLoss]], { requiresGrad: true }),
      reconLoss,
      klLoss,
    };
  }

  generate(numSamples: number = 1): Tensor {
    this.eval();
    const z = randn([numSamples, this.config.latentDim]);
    return this.decodeForward(z);
  }

  getConfig(): VAEConfig {
    return { ...this.config };
  }
}

/**
 * Создаёт простой VAE для MNIST (28x28 grayscale)
 */
export function createMNISTVAE(latentDim: number = 20): VAE {
  return new VAE({
    imageSize: 28,
    imageChannels: 1,
    latentDim,
    encoderHiddenDims: [32, 64],
    useBatchNorm: false,
    klWeight: 0.0001,
  });
}

/**
 * Создаёт VAE для цветных изображений (например, CIFAR-10: 32x32 RGB)
 */
export function createCIFARVAE(latentDim: number = 128): ConvVAE {
  return new ConvVAE({
    imageSize: 32,
    imageChannels: 3,
    latentDim,
    encoderHiddenDims: [32, 64, 128, 256],
    useBatchNorm: true,
    klWeight: 0.00025,
  });
}
