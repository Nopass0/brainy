/**
 * @fileoverview Reinforcement Learning - DQN и Actor-Critic
 * @description Обучение агентов в средах CartPole и GridWorld
 */

import {
  DQNAgent,
  CartPoleEnv,
  GridWorldEnv,
  ActorCriticAgent,
  trainDQN,
  // Model visualization
  summary,
  Sequential,
  Linear,
  ReLU,
} from '../src';

console.log('='.repeat(60));
console.log('Reinforcement Learning - DQN и Actor-Critic');
console.log('='.repeat(60));

// ============================================
// 1. DQN на CartPole
// ============================================
console.log('\n[1] DQN на CartPole');
console.log('-'.repeat(40));

const cartPole = new CartPoleEnv();
console.log(`Состояние: ${cartPole.stateSize} измерений (x, x_dot, theta, theta_dot)`);
console.log(`Действия: ${cartPole.actionSize} (влево, вправо)`);

const dqnAgent = new DQNAgent({
  stateSize: cartPole.stateSize,
  actionSize: cartPole.actionSize,
  hiddenSizes: [32, 32],
  lr: 0.001,
  gamma: 0.99,
  epsilon: 1.0,
  epsilonMin: 0.01,
  epsilonDecay: 0.995,
  batchSize: 32,
  bufferSize: 5000,
  targetUpdateFreq: 50,
});

console.log('\nОбучение DQN агента...');
const numEpisodes = 100;
const episodeRewards: number[] = [];

for (let episode = 0; episode < numEpisodes; episode++) {
  let state = cartPole.reset();
  let totalReward = 0;
  let done = false;

  while (!done) {
    const action = dqnAgent.selectAction(state);
    const result = cartPole.step(action);

    dqnAgent.remember(state, action, result.reward, result.state, result.done);
    dqnAgent.train();

    state = result.state;
    totalReward += result.reward;
    done = result.done;
  }

  episodeRewards.push(totalReward);

  if ((episode + 1) % 20 === 0) {
    const avgReward = episodeRewards.slice(-20).reduce((a, b) => a + b, 0) / 20;
    console.log(`  Эпизод ${episode + 1}: Средняя награда = ${avgReward.toFixed(1)}, Epsilon = ${dqnAgent.epsilon.toFixed(3)}`);
  }
}

const finalAvg = episodeRewards.slice(-20).reduce((a, b) => a + b, 0) / 20;
console.log(`\nИтоговая средняя награда: ${finalAvg.toFixed(1)}`);

// ============================================
// 2. GridWorld
// ============================================
console.log('\n[2] DQN на GridWorld');
console.log('-'.repeat(40));

const gridWorld = new GridWorldEnv(5, 3);
console.log('Начальное состояние:');
console.log(gridWorld.render());

const gridAgent = new DQNAgent({
  stateSize: gridWorld.stateSize,
  actionSize: gridWorld.actionSize,
  hiddenSizes: [64, 32],
  lr: 0.001,
  gamma: 0.95,
  epsilon: 1.0,
  epsilonMin: 0.05,
  epsilonDecay: 0.99,
  batchSize: 32,
  bufferSize: 5000,
  targetUpdateFreq: 50,
});

console.log('Обучение агента в GridWorld...');
const gridRewards: number[] = [];

for (let episode = 0; episode < 200; episode++) {
  let state = gridWorld.reset();
  let totalReward = 0;
  let done = false;

  while (!done) {
    const action = gridAgent.selectAction(state);
    const result = gridWorld.step(action);

    gridAgent.remember(state, action, result.reward, result.state, result.done);
    gridAgent.train();

    state = result.state;
    totalReward += result.reward;
    done = result.done;
  }

  gridRewards.push(totalReward);

  if ((episode + 1) % 50 === 0) {
    const avgReward = gridRewards.slice(-50).reduce((a, b) => a + b, 0) / 50;
    console.log(`  Эпизод ${episode + 1}: Средняя награда = ${avgReward.toFixed(2)}, Epsilon = ${gridAgent.epsilon.toFixed(3)}`);
  }
}

// Тестируем обученного агента
console.log('\nТестирование обученного агента:');
gridWorld.reset();
console.log('Начало:');
console.log(gridWorld.render());

let testState = gridWorld.reset();
let testSteps = 0;
let testDone = false;

while (!testDone && testSteps < 20) {
  const action = gridAgent.selectAction(testState);
  const result = gridWorld.step(action);
  testState = result.state;
  testDone = result.done;
  testSteps++;
}

console.log(`После ${testSteps} шагов:`);
console.log(gridWorld.render());
console.log(testDone ? 'Агент достиг цели!' : 'Агент не достиг цели');

// ============================================
// 3. Actor-Critic на CartPole
// ============================================
console.log('\n[3] Actor-Critic на CartPole');
console.log('-'.repeat(40));

const cartPole2 = new CartPoleEnv();
const a2cAgent = new ActorCriticAgent({
  stateSize: cartPole2.stateSize,
  actionSize: cartPole2.actionSize,
  hiddenSizes: [32],
  actorLr: 0.001,
  criticLr: 0.005,
  gamma: 0.99,
  entropyCoef: 0.01,
});

console.log('Обучение A2C агента...');
const a2cRewards: number[] = [];

for (let episode = 0; episode < 100; episode++) {
  let state = cartPole2.reset();
  let totalReward = 0;
  let done = false;

  while (!done) {
    const { action } = a2cAgent.selectAction(state);
    const result = cartPole2.step(action);

    a2cAgent.update(state, action, result.reward, result.state, result.done);

    state = result.state;
    totalReward += result.reward;
    done = result.done;
  }

  a2cRewards.push(totalReward);

  if ((episode + 1) % 20 === 0) {
    const avgReward = a2cRewards.slice(-20).reduce((a, b) => a + b, 0) / 20;
    console.log(`  Эпизод ${episode + 1}: Средняя награда = ${avgReward.toFixed(1)}`);
  }
}

// ============================================
// 4. Сравнение методов
// ============================================
console.log('\n[4] Сравнение DQN и A2C');
console.log('-'.repeat(40));

const dqnFinal = episodeRewards.slice(-20).reduce((a, b) => a + b, 0) / 20;
const a2cFinal = a2cRewards.slice(-20).reduce((a, b) => a + b, 0) / 20;

console.log(`DQN средняя награда (последние 20 эпизодов): ${dqnFinal.toFixed(1)}`);
console.log(`A2C средняя награда (последние 20 эпизодов): ${a2cFinal.toFixed(1)}`);

// ============================================
// 5. Визуализация Q-Network
// ============================================
console.log('\n[5] Архитектура Q-Network');
console.log('-'.repeat(40));

// Создаём модель Q-network для визуализации
const qNetwork = new Sequential(
  new Linear(4, 32),
  new ReLU(),
  new Linear(32, 32),
  new ReLU(),
  new Linear(32, 2)
);

console.log(summary(qNetwork, [1, 4]));

console.log('\n' + '='.repeat(60));
console.log('Reinforcement Learning демо завершено!');
console.log('='.repeat(60));
