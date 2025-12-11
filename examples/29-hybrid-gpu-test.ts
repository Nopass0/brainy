/**
 * Test hybrid GPU engine
 */

import { Tensor } from '../src';
import { getHybridEngine, disposeHybridEngine } from '../src/compute/hybrid';
import { DeviceManager, DeviceType } from '../src/compute/device';

async function main() {
  console.log('=== Hybrid GPU Engine Test ===\n');

  // Reset and initialize device manager with HYBRID mode
  DeviceManager.reset();
  const deviceManager = DeviceManager.getInstance({ type: DeviceType.HYBRID });
  await deviceManager.initialize();

  console.log('Device info:', {
    gpu: deviceManager.isGPUAvailable(),
    type: deviceManager.getConfig().type,
  });

  // Get hybrid engine
  console.log('\nInitializing hybrid engine...');
  const engine = await getHybridEngine({
    gpuThreshold: 64, // Low threshold to test GPU
    gpuPriority: 1.0, // Always prefer GPU
    autoBalance: false,
    profiling: true,
  });

  console.log('Engine initialized:', engine.getDeviceInfo());

  // Test basic operations
  const a = new Tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
  const b = new Tensor([1, 1, 1, 1, 1, 1, 1, 1], [2, 4]);

  console.log('\n1. Testing add...');
  const addResult = await engine.add(a, b);
  console.log('   a:', Array.from(a.data));
  console.log('   b:', Array.from(b.data));
  console.log('   a + b:', Array.from(addResult.data));

  console.log('\n2. Testing mul...');
  const mulResult = await engine.mul(a, b);
  console.log('   a * b:', Array.from(mulResult.data));

  console.log('\n3. Testing relu...');
  const negTensor = new Tensor([-1, -2, 3, 4, -5, 6, -7, 8], [2, 4]);
  const reluResult = await engine.relu(negTensor);
  console.log('   input:', Array.from(negTensor.data));
  console.log('   relu:', Array.from(reluResult.data));

  console.log('\n4. Testing sigmoid...');
  const sigInput = new Tensor([-2, -1, 0, 1, 2, 3, 4, 5], [2, 4]);
  const sigResult = await engine.sigmoid(sigInput);
  console.log('   input:', Array.from(sigInput.data));
  const sigValues = Array.from(sigResult.data).map(v => v.toFixed(4));
  console.log('   sigmoid:', sigValues);

  console.log('\n5. Testing matmul...');
  const m1 = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const m2 = new Tensor([1, 2, 3, 4, 5, 6], [3, 2]);
  const matResult = await engine.matmul(m1, m2);
  console.log('   m1 [2x3]:', Array.from(m1.data));
  console.log('   m2 [3x2]:', Array.from(m2.data));
  console.log('   m1 @ m2 [2x2]:', Array.from(matResult.data));

  // Show operation stats
  console.log('\n=== Operation Stats ===');
  const stats = engine.getOpStats();
  for (const [op, stat] of stats) {
    const gpuTime = stat.gpuTime.toFixed(2);
    const cpuTime = stat.cpuTime.toFixed(2);
    console.log(op + ': GPU=' + stat.gpuCount + ' (' + gpuTime + 'ms), CPU=' + stat.cpuCount + ' (' + cpuTime + 'ms)');
  }

  disposeHybridEngine();
  console.log('\n=== SUCCESS ===');
}

main().catch(console.error);
