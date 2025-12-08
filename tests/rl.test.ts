/**
 * @fileoverview Tests for Reinforcement Learning modules
 */

import { describe, test, expect } from 'bun:test';
import {
  ReplayBuffer,
  DQNAgent,
  PolicyGradientAgent,
  ActorCriticAgent,
  CartPoleEnv,
  GridWorldEnv,
} from '../src';

describe('ReplayBuffer', () => {
  test('push adds experience', () => {
    const buffer = new ReplayBuffer(100);
    buffer.push({
      state: [1, 2, 3],
      action: 0,
      reward: 1,
      nextState: [4, 5, 6],
      done: false,
    });
    expect(buffer.length).toBe(1);
  });

  test('buffer respects max size', () => {
    const buffer = new ReplayBuffer(5);
    for (let i = 0; i < 10; i++) {
      buffer.push({
        state: [i],
        action: 0,
        reward: 0,
        nextState: [i + 1],
        done: false,
      });
    }
    expect(buffer.length).toBe(5);
  });

  test('sample returns requested size', () => {
    const buffer = new ReplayBuffer(100);
    for (let i = 0; i < 50; i++) {
      buffer.push({
        state: [i],
        action: 0,
        reward: 0,
        nextState: [i + 1],
        done: false,
      });
    }
    const samples = buffer.sample(10);
    expect(samples.length).toBe(10);
  });

  test('clear empties buffer', () => {
    const buffer = new ReplayBuffer(100);
    buffer.push({
      state: [1],
      action: 0,
      reward: 0,
      nextState: [2],
      done: false,
    });
    buffer.clear();
    expect(buffer.length).toBe(0);
  });
});

describe('DQNAgent', () => {
  test('selectAction returns valid action', () => {
    const agent = new DQNAgent({
      stateSize: 4,
      actionSize: 2,
    });
    const action = agent.selectAction([0, 0, 0, 0]);
    expect(action).toBeGreaterThanOrEqual(0);
    expect(action).toBeLessThan(2);
  });

  test('remember stores experience', () => {
    const agent = new DQNAgent({
      stateSize: 4,
      actionSize: 2,
      bufferSize: 100,
    });
    agent.remember([0, 0, 0, 0], 1, 1, [1, 1, 1, 1], false);
    // No direct access to buffer, but train should work after
  });

  test('epsilon starts at 1.0', () => {
    const agent = new DQNAgent({
      stateSize: 4,
      actionSize: 2,
      epsilon: 1.0,
    });
    expect(agent.epsilon).toBe(1.0);
  });

  test('train decreases epsilon', () => {
    const agent = new DQNAgent({
      stateSize: 4,
      actionSize: 2,
      epsilon: 1.0,
      epsilonDecay: 0.99,
      batchSize: 2,
    });

    // Fill buffer
    for (let i = 0; i < 10; i++) {
      agent.remember([0, 0, 0, 0], 0, 1, [1, 1, 1, 1], false);
    }

    const initialEpsilon = agent.epsilon;
    agent.train();
    expect(agent.epsilon).toBeLessThan(initialEpsilon);
  });
});

describe('PolicyGradientAgent', () => {
  test('selectAction returns valid action', () => {
    const agent = new PolicyGradientAgent({
      stateSize: 4,
      actionSize: 2,
    });
    const action = agent.selectAction([0, 0, 0, 0]);
    expect(action).toBeGreaterThanOrEqual(0);
    expect(action).toBeLessThan(2);
  });

  test('saveReward stores rewards', () => {
    const agent = new PolicyGradientAgent({
      stateSize: 4,
      actionSize: 2,
    });
    agent.selectAction([0, 0, 0, 0]);
    agent.saveReward(1);
    // No direct access, but train should work
  });

  test('train returns episode return', () => {
    const agent = new PolicyGradientAgent({
      stateSize: 4,
      actionSize: 2,
    });

    agent.selectAction([0, 0, 0, 0]);
    agent.saveReward(1);
    agent.selectAction([1, 1, 1, 1]);
    agent.saveReward(2);

    const ret = agent.train();
    expect(ret).toBeGreaterThan(0);
  });

  test('reset clears buffers', () => {
    const agent = new PolicyGradientAgent({
      stateSize: 4,
      actionSize: 2,
    });

    agent.selectAction([0, 0, 0, 0]);
    agent.saveReward(1);
    agent.reset();
    const ret = agent.train();
    expect(ret).toBe(0);  // Empty buffers
  });
});

describe('ActorCriticAgent', () => {
  test('selectAction returns action, logProb, value', () => {
    const agent = new ActorCriticAgent({
      stateSize: 4,
      actionSize: 2,
    });
    const result = agent.selectAction([0, 0, 0, 0]);
    expect(result.action).toBeGreaterThanOrEqual(0);
    expect(result.action).toBeLessThan(2);
    expect(typeof result.logProb).toBe('number');
    expect(typeof result.value).toBe('number');
  });

  test('update returns losses', () => {
    const agent = new ActorCriticAgent({
      stateSize: 4,
      actionSize: 2,
    });
    const losses = agent.update(
      [0, 0, 0, 0],
      0,
      1,
      [1, 1, 1, 1],
      false
    );
    expect(typeof losses.actorLoss).toBe('number');
    expect(typeof losses.criticLoss).toBe('number');
  });
});

describe('CartPoleEnv', () => {
  test('reset returns initial state', () => {
    const env = new CartPoleEnv();
    const state = env.reset();
    expect(state.length).toBe(4);
  });

  test('step returns state, reward, done', () => {
    const env = new CartPoleEnv();
    env.reset();
    const result = env.step(0);
    expect(result.state.length).toBe(4);
    expect(typeof result.reward).toBe('number');
    expect(typeof result.done).toBe('boolean');
  });

  test('stateSize is 4', () => {
    const env = new CartPoleEnv();
    expect(env.stateSize).toBe(4);
  });

  test('actionSize is 2', () => {
    const env = new CartPoleEnv();
    expect(env.actionSize).toBe(2);
  });

  test('episode terminates', () => {
    const env = new CartPoleEnv();
    env.reset();

    let done = false;
    let steps = 0;
    while (!done && steps < 500) {
      const result = env.step(Math.random() > 0.5 ? 1 : 0);
      done = result.done;
      steps++;
    }

    expect(done).toBe(true);
  });
});

describe('GridWorldEnv', () => {
  test('reset returns initial state', () => {
    const env = new GridWorldEnv(5, 2);
    const state = env.reset();
    expect(state.length).toBe(25);  // 5x5 grid
  });

  test('step moves agent', () => {
    const env = new GridWorldEnv(5, 0);  // No obstacles
    env.reset();

    // Move right
    const result = env.step(3);
    expect(result.state.length).toBe(25);
  });

  test('stateSize matches grid', () => {
    const env = new GridWorldEnv(4, 0);
    expect(env.stateSize).toBe(16);  // 4x4
  });

  test('actionSize is 4', () => {
    const env = new GridWorldEnv(5, 0);
    expect(env.actionSize).toBe(4);  // up, down, left, right
  });

  test('render returns grid string', () => {
    const env = new GridWorldEnv(3, 0);
    env.reset();
    const grid = env.render();
    expect(grid).toContain('A');  // Agent
    expect(grid).toContain('G');  // Goal
  });

  test('reaching goal gives reward', () => {
    const env = new GridWorldEnv(2, 0);  // 2x2 grid, goal at (1,1)
    env.reset();  // Agent at (0,0)

    // Move right to (1,0)
    env.step(3);
    // Move down to (1,1) - goal
    const result = env.step(1);

    expect(result.reward).toBe(10);
    expect(result.done).toBe(true);
  });
});
