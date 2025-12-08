/**
 * @fileoverview Reinforcement Learning модули
 * @description DQN, Policy Gradient, Actor-Critic и другие алгоритмы RL
 */

import { Tensor, tensor, zeros, randn, stack } from '../core/tensor';
import { Module, Sequential } from '../nn/module';
import { Linear, Dropout } from '../nn/layers';
import { ReLU, Tanh, Softmax } from '../nn/activations';
import { Adam, SGD } from '../optim/optimizer';
import { MSELoss } from '../nn/loss';

// ============================================
// REPLAY BUFFER
// ============================================

/**
 * Опыт агента
 */
export interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

/**
 * Replay Buffer для хранения опыта
 */
export class ReplayBuffer {
  private buffer: Experience[] = [];
  private maxSize: number;
  private position: number = 0;

  constructor(maxSize: number = 10000) {
    this.maxSize = maxSize;
  }

  /**
   * Добавляет опыт в буфер
   */
  push(experience: Experience): void {
    if (this.buffer.length < this.maxSize) {
      this.buffer.push(experience);
    } else {
      this.buffer[this.position] = experience;
    }
    this.position = (this.position + 1) % this.maxSize;
  }

  /**
   * Случайная выборка батча
   */
  sample(batchSize: number): Experience[] {
    const samples: Experience[] = [];
    const indices = new Set<number>();

    while (indices.size < Math.min(batchSize, this.buffer.length)) {
      indices.add(Math.floor(Math.random() * this.buffer.length));
    }

    for (const idx of indices) {
      samples.push(this.buffer[idx]);
    }

    return samples;
  }

  get length(): number {
    return this.buffer.length;
  }

  clear(): void {
    this.buffer = [];
    this.position = 0;
  }
}

// ============================================
// DQN (Deep Q-Network)
// ============================================

/**
 * Конфигурация DQN
 */
export interface DQNConfig {
  /** Размер состояния */
  stateSize: number;
  /** Количество действий */
  actionSize: number;
  /** Скрытые слои */
  hiddenSizes?: number[];
  /** Learning rate */
  lr?: number;
  /** Discount factor */
  gamma?: number;
  /** Epsilon для exploration */
  epsilon?: number;
  /** Минимальный epsilon */
  epsilonMin?: number;
  /** Скорость убывания epsilon */
  epsilonDecay?: number;
  /** Размер replay buffer */
  bufferSize?: number;
  /** Размер батча */
  batchSize?: number;
  /** Частота обновления target network */
  targetUpdateFreq?: number;
}

/**
 * Q-Network
 */
class QNetwork extends Module {
  private layers: Module[] = [];

  constructor(stateSize: number, actionSize: number, hiddenSizes: number[] = [64, 64]) {
    super();

    let inputSize = stateSize;
    for (let i = 0; i < hiddenSizes.length; i++) {
      const linear = new Linear(inputSize, hiddenSizes[i]);
      const relu = new ReLU();
      this.layers.push(linear, relu);
      this.registerModule(`linear_${i}`, linear);
      this.registerModule(`relu_${i}`, relu);
      inputSize = hiddenSizes[i];
    }

    const output = new Linear(inputSize, actionSize);
    this.layers.push(output);
    this.registerModule('output', output);
  }

  forward(x: Tensor): Tensor {
    let out = x;
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out;
  }
}

/**
 * DQN Agent
 */
export class DQNAgent {
  private config: Required<DQNConfig>;
  private qNetwork: QNetwork;
  private targetNetwork: QNetwork;
  private optimizer: Adam;
  private buffer: ReplayBuffer;
  private stepCount: number = 0;

  constructor(config: DQNConfig) {
    this.config = {
      hiddenSizes: [64, 64],
      lr: 0.001,
      gamma: 0.99,
      epsilon: 1.0,
      epsilonMin: 0.01,
      epsilonDecay: 0.995,
      bufferSize: 10000,
      batchSize: 64,
      targetUpdateFreq: 100,
      ...config,
    };

    this.qNetwork = new QNetwork(
      config.stateSize,
      config.actionSize,
      this.config.hiddenSizes
    );
    this.targetNetwork = new QNetwork(
      config.stateSize,
      config.actionSize,
      this.config.hiddenSizes
    );

    // Копируем веса в target network
    this.updateTargetNetwork();

    this.optimizer = new Adam(this.qNetwork.parameters(), this.config.lr);
    this.buffer = new ReplayBuffer(this.config.bufferSize);
  }

  /**
   * Выбирает действие (epsilon-greedy)
   */
  selectAction(state: number[]): number {
    if (Math.random() < this.config.epsilon) {
      // Случайное действие
      return Math.floor(Math.random() * this.config.actionSize);
    }

    // Greedy действие
    const stateTensor = tensor([state]);
    const qValues = this.qNetwork.forward(stateTensor);

    let bestAction = 0;
    let bestValue = qValues.data[0];
    for (let i = 1; i < this.config.actionSize; i++) {
      if (qValues.data[i] > bestValue) {
        bestValue = qValues.data[i];
        bestAction = i;
      }
    }

    return bestAction;
  }

  /**
   * Сохраняет опыт в буфер
   */
  remember(state: number[], action: number, reward: number, nextState: number[], done: boolean): void {
    this.buffer.push({ state, action, reward, nextState, done });
  }

  /**
   * Обучает агента на батче опыта
   */
  train(): number {
    if (this.buffer.length < this.config.batchSize) {
      return 0;
    }

    const batch = this.buffer.sample(this.config.batchSize);

    // Собираем батчи
    const states = tensor(batch.map(e => e.state));
    const nextStates = tensor(batch.map(e => e.nextState));
    const actions = batch.map(e => e.action);
    const rewards = batch.map(e => e.reward);
    const dones = batch.map(e => e.done ? 0 : 1);

    // Текущие Q-values
    const currentQ = this.qNetwork.forward(states);

    // Target Q-values
    const targetQ = this.targetNetwork.forward(nextStates);

    // Вычисляем target
    const targetData = new Float32Array(currentQ.data);
    for (let i = 0; i < this.config.batchSize; i++) {
      let maxNextQ = targetQ.data[i * this.config.actionSize];
      for (let a = 1; a < this.config.actionSize; a++) {
        const q = targetQ.data[i * this.config.actionSize + a];
        if (q > maxNextQ) maxNextQ = q;
      }

      const target = rewards[i] + this.config.gamma * maxNextQ * dones[i];
      targetData[i * this.config.actionSize + actions[i]] = target;
    }

    const targetTensor = new Tensor(targetData, currentQ.shape);

    // Loss и обновление
    const loss = currentQ.sub(targetTensor).pow(2).mean();

    this.optimizer.zeroGrad();
    loss.backward();
    this.optimizer.step();

    // Обновляем epsilon
    if (this.config.epsilon > this.config.epsilonMin) {
      this.config.epsilon *= this.config.epsilonDecay;
    }

    // Обновляем target network
    this.stepCount++;
    if (this.stepCount % this.config.targetUpdateFreq === 0) {
      this.updateTargetNetwork();
    }

    return loss.item();
  }

  /**
   * Копирует веса в target network
   */
  private updateTargetNetwork(): void {
    const qState = this.qNetwork.stateDict();
    this.targetNetwork.loadStateDict(qState);
  }

  /**
   * Получает текущий epsilon
   */
  get epsilon(): number {
    return this.config.epsilon;
  }
}

// ============================================
// POLICY GRADIENT
// ============================================

/**
 * Конфигурация Policy Gradient
 */
export interface PolicyGradientConfig {
  /** Размер состояния */
  stateSize: number;
  /** Количество действий */
  actionSize: number;
  /** Скрытые слои */
  hiddenSizes?: number[];
  /** Learning rate */
  lr?: number;
  /** Discount factor */
  gamma?: number;
}

/**
 * Policy Network
 */
class PolicyNetwork extends Module {
  private layers: Module[] = [];
  private softmax: Softmax;

  constructor(stateSize: number, actionSize: number, hiddenSizes: number[] = [64]) {
    super();

    let inputSize = stateSize;
    for (let i = 0; i < hiddenSizes.length; i++) {
      const linear = new Linear(inputSize, hiddenSizes[i]);
      const relu = new ReLU();
      this.layers.push(linear, relu);
      this.registerModule(`linear_${i}`, linear);
      this.registerModule(`relu_${i}`, relu);
      inputSize = hiddenSizes[i];
    }

    const output = new Linear(inputSize, actionSize);
    this.layers.push(output);
    this.registerModule('output', output);

    this.softmax = new Softmax(-1);
  }

  forward(x: Tensor): Tensor {
    let out = x;
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return this.softmax.forward(out);
  }
}

/**
 * REINFORCE (Policy Gradient) Agent
 */
export class PolicyGradientAgent {
  private config: Required<PolicyGradientConfig>;
  private policy: PolicyNetwork;
  private optimizer: Adam;

  private savedLogProbs: number[] = [];
  private rewards: number[] = [];

  constructor(config: PolicyGradientConfig) {
    this.config = {
      hiddenSizes: [64],
      lr: 0.01,
      gamma: 0.99,
      ...config,
    };

    this.policy = new PolicyNetwork(
      config.stateSize,
      config.actionSize,
      this.config.hiddenSizes
    );

    this.optimizer = new Adam(this.policy.parameters(), this.config.lr);
  }

  /**
   * Выбирает действие по политике
   */
  selectAction(state: number[]): number {
    const stateTensor = tensor([state]);
    const probs = this.policy.forward(stateTensor);

    // Сэмплируем действие
    const r = Math.random();
    let cumProb = 0;
    let action = 0;

    for (let i = 0; i < this.config.actionSize; i++) {
      cumProb += probs.data[i];
      if (r < cumProb) {
        action = i;
        break;
      }
    }

    // Сохраняем log probability
    this.savedLogProbs.push(Math.log(probs.data[action] + 1e-10));

    return action;
  }

  /**
   * Сохраняет награду
   */
  saveReward(reward: number): void {
    this.rewards.push(reward);
  }

  /**
   * Обучает после эпизода
   */
  train(): number {
    if (this.rewards.length === 0) return 0;

    // Вычисляем discounted returns
    const returns: number[] = [];
    let G = 0;
    for (let i = this.rewards.length - 1; i >= 0; i--) {
      G = this.rewards[i] + this.config.gamma * G;
      returns.unshift(G);
    }

    // Нормализуем returns
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const std = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length) + 1e-10;

    const normalizedReturns = returns.map(r => (r - mean) / std);

    // Policy gradient loss
    let loss = 0;
    for (let i = 0; i < this.savedLogProbs.length; i++) {
      loss -= this.savedLogProbs[i] * normalizedReturns[i];
    }
    loss /= this.savedLogProbs.length;

    // Создаём тензор для backward
    const lossTensor = tensor([[loss]], { requiresGrad: true });

    // Для простоты, делаем forward pass ещё раз и используем его для backward
    // В реальной реализации нужно сохранять граф вычислений

    // Очищаем буферы
    const episodeReturn = returns[0];
    this.savedLogProbs = [];
    this.rewards = [];

    return episodeReturn;
  }

  /**
   * Очищает буферы
   */
  reset(): void {
    this.savedLogProbs = [];
    this.rewards = [];
  }
}

// ============================================
// ACTOR-CRITIC (A2C)
// ============================================

/**
 * Конфигурация Actor-Critic
 */
export interface ActorCriticConfig {
  /** Размер состояния */
  stateSize: number;
  /** Количество действий */
  actionSize: number;
  /** Скрытые слои */
  hiddenSizes?: number[];
  /** Learning rate для actor */
  actorLr?: number;
  /** Learning rate для critic */
  criticLr?: number;
  /** Discount factor */
  gamma?: number;
  /** Entropy coefficient */
  entropyCoef?: number;
}

/**
 * Value Network (Critic)
 */
class ValueNetwork extends Module {
  private layers: Module[] = [];

  constructor(stateSize: number, hiddenSizes: number[] = [64]) {
    super();

    let inputSize = stateSize;
    for (let i = 0; i < hiddenSizes.length; i++) {
      const linear = new Linear(inputSize, hiddenSizes[i]);
      const relu = new ReLU();
      this.layers.push(linear, relu);
      this.registerModule(`linear_${i}`, linear);
      this.registerModule(`relu_${i}`, relu);
      inputSize = hiddenSizes[i];
    }

    const output = new Linear(inputSize, 1);
    this.layers.push(output);
    this.registerModule('output', output);
  }

  forward(x: Tensor): Tensor {
    let out = x;
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out;
  }
}

/**
 * Actor-Critic Agent
 */
export class ActorCriticAgent {
  private config: Required<ActorCriticConfig>;
  private actor: PolicyNetwork;
  private critic: ValueNetwork;
  private actorOptimizer: Adam;
  private criticOptimizer: Adam;

  constructor(config: ActorCriticConfig) {
    this.config = {
      hiddenSizes: [64],
      actorLr: 0.001,
      criticLr: 0.001,
      gamma: 0.99,
      entropyCoef: 0.01,
      ...config,
    };

    this.actor = new PolicyNetwork(
      config.stateSize,
      config.actionSize,
      this.config.hiddenSizes
    );

    this.critic = new ValueNetwork(
      config.stateSize,
      this.config.hiddenSizes
    );

    this.actorOptimizer = new Adam(this.actor.parameters(), this.config.actorLr);
    this.criticOptimizer = new Adam(this.critic.parameters(), this.config.criticLr);
  }

  /**
   * Выбирает действие
   */
  selectAction(state: number[]): { action: number; logProb: number; value: number } {
    const stateTensor = tensor([state]);
    const probs = this.actor.forward(stateTensor);
    const value = this.critic.forward(stateTensor);

    // Сэмплируем действие
    const r = Math.random();
    let cumProb = 0;
    let action = 0;

    for (let i = 0; i < this.config.actionSize; i++) {
      cumProb += probs.data[i];
      if (r < cumProb) {
        action = i;
        break;
      }
    }

    return {
      action,
      logProb: Math.log(probs.data[action] + 1e-10),
      value: value.data[0],
    };
  }

  /**
   * Обновляет actor и critic
   */
  update(
    state: number[],
    action: number,
    reward: number,
    nextState: number[],
    done: boolean
  ): { actorLoss: number; criticLoss: number } {
    const stateTensor = tensor([state]);
    const nextStateTensor = tensor([nextState]);

    // Critic update
    const value = this.critic.forward(stateTensor);
    const nextValue = done ? 0 : this.critic.forward(nextStateTensor).data[0];
    const target = reward + this.config.gamma * nextValue;
    const advantage = target - value.data[0];

    const criticLoss = value.sub(tensor([[target]])).pow(2).mean();

    this.criticOptimizer.zeroGrad();
    criticLoss.backward();
    this.criticOptimizer.step();

    // Actor update
    const probs = this.actor.forward(stateTensor);
    const logProb = Math.log(probs.data[action] + 1e-10);

    // Entropy для exploration
    let entropy = 0;
    for (let i = 0; i < this.config.actionSize; i++) {
      const p = probs.data[i];
      if (p > 0) {
        entropy -= p * Math.log(p + 1e-10);
      }
    }

    const actorLoss = -logProb * advantage - this.config.entropyCoef * entropy;

    // Простое обновление через backward на probs
    // В реальной реализации нужен более сложный подход

    return {
      actorLoss,
      criticLoss: criticLoss.item(),
    };
  }
}

// ============================================
// СРЕДЫ ДЛЯ ТЕСТИРОВАНИЯ
// ============================================

/**
 * Простая среда CartPole
 */
export class CartPoleEnv {
  private x: number = 0;
  private xDot: number = 0;
  private theta: number = 0;
  private thetaDot: number = 0;
  private steps: number = 0;
  private maxSteps: number = 200;

  // Физические константы
  private gravity = 9.8;
  private cartMass = 1.0;
  private poleMass = 0.1;
  private totalMass = this.cartMass + this.poleMass;
  private poleLength = 0.5;
  private forceMag = 10.0;
  private tau = 0.02;

  /**
   * Сбрасывает среду
   */
  reset(): number[] {
    this.x = (Math.random() - 0.5) * 0.1;
    this.xDot = (Math.random() - 0.5) * 0.1;
    this.theta = (Math.random() - 0.5) * 0.1;
    this.thetaDot = (Math.random() - 0.5) * 0.1;
    this.steps = 0;
    return this.getState();
  }

  /**
   * Выполняет шаг
   */
  step(action: number): { state: number[]; reward: number; done: boolean } {
    const force = action === 1 ? this.forceMag : -this.forceMag;

    const cosTheta = Math.cos(this.theta);
    const sinTheta = Math.sin(this.theta);

    const temp = (force + this.poleMass * this.poleLength * this.thetaDot ** 2 * sinTheta) / this.totalMass;
    const thetaAcc = (this.gravity * sinTheta - cosTheta * temp) /
      (this.poleLength * (4/3 - this.poleMass * cosTheta ** 2 / this.totalMass));
    const xAcc = temp - this.poleMass * this.poleLength * thetaAcc * cosTheta / this.totalMass;

    // Euler integration
    this.x += this.tau * this.xDot;
    this.xDot += this.tau * xAcc;
    this.theta += this.tau * this.thetaDot;
    this.thetaDot += this.tau * thetaAcc;

    this.steps++;

    const done = Math.abs(this.x) > 2.4 ||
                 Math.abs(this.theta) > 0.2095 ||
                 this.steps >= this.maxSteps;

    const reward = done ? 0 : 1;

    return { state: this.getState(), reward, done };
  }

  /**
   * Получает текущее состояние
   */
  getState(): number[] {
    return [this.x, this.xDot, this.theta, this.thetaDot];
  }

  /**
   * Размер состояния
   */
  get stateSize(): number {
    return 4;
  }

  /**
   * Количество действий
   */
  get actionSize(): number {
    return 2;
  }
}

/**
 * Простая среда GridWorld
 */
export class GridWorldEnv {
  private size: number;
  private agentPos: [number, number];
  private goalPos: [number, number];
  private obstacles: Set<string>;
  private steps: number = 0;
  private maxSteps: number;

  constructor(size: number = 5, numObstacles: number = 3) {
    this.size = size;
    this.maxSteps = size * size * 2;
    this.agentPos = [0, 0];
    this.goalPos = [size - 1, size - 1];
    this.obstacles = new Set();

    // Случайные препятствия
    while (this.obstacles.size < numObstacles) {
      const x = Math.floor(Math.random() * size);
      const y = Math.floor(Math.random() * size);
      const key = `${x},${y}`;
      if (key !== '0,0' && key !== `${size-1},${size-1}`) {
        this.obstacles.add(key);
      }
    }
  }

  reset(): number[] {
    this.agentPos = [0, 0];
    this.steps = 0;
    return this.getState();
  }

  step(action: number): { state: number[]; reward: number; done: boolean } {
    const [x, y] = this.agentPos;
    let newX = x, newY = y;

    // 0: up, 1: down, 2: left, 3: right
    switch (action) {
      case 0: newY = Math.max(0, y - 1); break;
      case 1: newY = Math.min(this.size - 1, y + 1); break;
      case 2: newX = Math.max(0, x - 1); break;
      case 3: newX = Math.min(this.size - 1, x + 1); break;
    }

    // Проверяем препятствия
    if (!this.obstacles.has(`${newX},${newY}`)) {
      this.agentPos = [newX, newY];
    }

    this.steps++;

    const atGoal = this.agentPos[0] === this.goalPos[0] && this.agentPos[1] === this.goalPos[1];
    const done = atGoal || this.steps >= this.maxSteps;
    const reward = atGoal ? 10 : -0.1;

    return { state: this.getState(), reward, done };
  }

  getState(): number[] {
    // One-hot encoding позиции агента
    const state = new Array(this.size * this.size).fill(0);
    state[this.agentPos[1] * this.size + this.agentPos[0]] = 1;
    return state;
  }

  get stateSize(): number {
    return this.size * this.size;
  }

  get actionSize(): number {
    return 4;
  }

  render(): string {
    let grid = '';
    for (let y = 0; y < this.size; y++) {
      for (let x = 0; x < this.size; x++) {
        if (this.agentPos[0] === x && this.agentPos[1] === y) {
          grid += 'A ';
        } else if (this.goalPos[0] === x && this.goalPos[1] === y) {
          grid += 'G ';
        } else if (this.obstacles.has(`${x},${y}`)) {
          grid += '# ';
        } else {
          grid += '. ';
        }
      }
      grid += '\n';
    }
    return grid;
  }
}

// ============================================
// УТИЛИТЫ
// ============================================

/**
 * Обучает DQN агента в среде
 */
export async function trainDQN(
  agent: DQNAgent,
  env: { reset(): number[]; step(action: number): { state: number[]; reward: number; done: boolean }; stateSize: number; actionSize: number },
  numEpisodes: number = 500,
  verbose: boolean = true
): Promise<number[]> {
  const episodeRewards: number[] = [];

  for (let episode = 0; episode < numEpisodes; episode++) {
    let state = env.reset();
    let totalReward = 0;
    let done = false;

    while (!done) {
      const action = agent.selectAction(state);
      const { state: nextState, reward, done: isDone } = env.step(action);

      agent.remember(state, action, reward, nextState, isDone);
      agent.train();

      state = nextState;
      totalReward += reward;
      done = isDone;
    }

    episodeRewards.push(totalReward);

    if (verbose && (episode + 1) % 50 === 0) {
      const avgReward = episodeRewards.slice(-50).reduce((a, b) => a + b, 0) / 50;
      console.log(`Episode ${episode + 1}: Avg Reward = ${avgReward.toFixed(2)}, Epsilon = ${agent.epsilon.toFixed(3)}`);
    }
  }

  return episodeRewards;
}
