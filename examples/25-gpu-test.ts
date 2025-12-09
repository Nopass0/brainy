/**
 * GPU Test Example
 * Tests GPU initialization and basic operations
 */

import {
  createDevice,
  DeviceType,
  isWebGPUSupportedAsync,
  getDevice,
  createGPUBackend,
  isGPUBackendAvailable,
  getWebGPUProviderInfoAsync,
  tensor,
  Tensor,
} from '../src';

async function main() {
  console.log('='.repeat(60));
  console.log('GPU TEST');
  console.log('='.repeat(60));

  // Check runtime and WebGPU support (use async version for Bun!)
  console.log('\n1. Checking environment...');
  const providerInfo = await getWebGPUProviderInfoAsync();
  console.log(`   Runtime: ${providerInfo.runtime}`);
  console.log(`   WebGPU provider: ${providerInfo.type}`);
  console.log(`   WebGPU available: ${providerInfo.available}`);

  if (!providerInfo.available) {
    console.log('\n   WebGPU is not available.');
    if (providerInfo.runtime === 'bun') {
      console.log('   For Bun: bun add bun-webgpu');
    } else if (providerInfo.runtime === 'node') {
      console.log('   For Node.js: npm install webgpu');
    } else {
      console.log('   For Browser: Use Chrome/Edge with WebGPU enabled');
    }
    return;
  }

  // Initialize GPU device
  console.log('\n2. Initializing GPU device...');
  try {
    const device = await createDevice({ type: DeviceType.GPU });
    console.log('   GPU device initialized successfully!');

    // Get GPU info
    const gpuInfo = device.getGPUInfo();
    console.log('\n3. GPU Information:');
    console.log(`   Available: ${gpuInfo.available}`);
    console.log(`   Name: ${gpuInfo.name || 'Unknown'}`);
    console.log(`   Vendor: ${gpuInfo.vendor || 'Unknown'}`);
    console.log(`   Max Buffer Size: ${gpuInfo.maxBufferSize ? (gpuInfo.maxBufferSize / 1024 / 1024).toFixed(2) + ' MB' : 'Unknown'}`);

    // Create GPU backend
    console.log('\n4. Creating GPU backend...');
    const gpuBackend = createGPUBackend();

    if (!gpuBackend) {
      console.log('   Failed to create GPU backend');
      return;
    }
    console.log('   GPU backend created successfully!');

    // Test basic operations
    console.log('\n5. Testing GPU operations...');

    // Test 1: Addition
    console.log('\n   Test 1: Element-wise Addition');
    const a = tensor([[1, 2, 3], [4, 5, 6]]);
    const b = tensor([[7, 8, 9], [10, 11, 12]]);

    const startAdd = performance.now();
    const addResult = await gpuBackend.add(a, b);
    const timeAdd = performance.now() - startAdd;

    console.log(`   Input A: ${JSON.stringify(Array.from(a.data))}`);
    console.log(`   Input B: ${JSON.stringify(Array.from(b.data))}`);
    console.log(`   Result:  ${JSON.stringify(Array.from(addResult.data))}`);
    console.log(`   Time: ${timeAdd.toFixed(2)}ms`);

    // Test 2: Multiplication
    console.log('\n   Test 2: Element-wise Multiplication');
    const startMul = performance.now();
    const mulResult = await gpuBackend.mul(a, b);
    const timeMul = performance.now() - startMul;

    console.log(`   Result: ${JSON.stringify(Array.from(mulResult.data))}`);
    console.log(`   Time: ${timeMul.toFixed(2)}ms`);

    // Test 3: Matrix multiplication
    console.log('\n   Test 3: Matrix Multiplication');
    const matA = tensor([[1, 2], [3, 4], [5, 6]]); // 3x2
    const matB = tensor([[7, 8, 9], [10, 11, 12]]); // 2x3

    const startMatmul = performance.now();
    const matmulResult = await gpuBackend.matmul(matA, matB);
    const timeMatmul = performance.now() - startMatmul;

    console.log(`   Matrix A (3x2): ${JSON.stringify(Array.from(matA.data))}`);
    console.log(`   Matrix B (2x3): ${JSON.stringify(Array.from(matB.data))}`);
    console.log(`   Result (3x3): ${JSON.stringify(Array.from(matmulResult.data))}`);
    console.log(`   Time: ${timeMatmul.toFixed(2)}ms`);

    // Test 4: ReLU
    console.log('\n   Test 4: ReLU Activation');
    const negInput = tensor([[-1, 2, -3], [4, -5, 6]]);

    const startRelu = performance.now();
    const reluResult = await gpuBackend.relu(negInput);
    const timeRelu = performance.now() - startRelu;

    console.log(`   Input:  ${JSON.stringify(Array.from(negInput.data))}`);
    console.log(`   Result: ${JSON.stringify(Array.from(reluResult.data))}`);
    console.log(`   Time: ${timeRelu.toFixed(2)}ms`);

    // Test 5: Large matrix multiplication benchmark
    console.log('\n6. Performance Benchmark (Large Matrix Multiplication)');
    const sizes = [64, 128, 256, 512];

    for (const size of sizes) {
      const largeA = new Tensor(
        new Float32Array(size * size).fill(0).map(() => Math.random()),
        [size, size]
      );
      const largeB = new Tensor(
        new Float32Array(size * size).fill(0).map(() => Math.random()),
        [size, size]
      );

      const start = performance.now();
      await gpuBackend.matmul(largeA, largeB);
      const time = performance.now() - start;

      const gflops = (2 * size * size * size) / (time * 1e6);
      console.log(`   ${size}x${size} matmul: ${time.toFixed(2)}ms (${gflops.toFixed(2)} GFLOPS)`);
    }

    // Get stats
    console.log('\n7. Performance Statistics:');
    const stats = device.getStats();
    console.log(`   Total operations: ${stats.totalOps}`);
    console.log(`   GPU operations: ${stats.gpuOps}`);
    console.log(`   Total GPU time: ${stats.gpuTimeMs.toFixed(2)}ms`);

    console.log('\n' + '='.repeat(60));
    console.log('GPU TEST COMPLETED SUCCESSFULLY!');
    console.log('='.repeat(60));

    // Cleanup
    device.dispose();
  } catch (error) {
    console.error('\nError:', error);
    console.log('\nTroubleshooting:');
    console.log('1. Make sure you have a compatible GPU');
    console.log('2. Update your GPU drivers');
    console.log('3. For NVIDIA: Install Vulkan SDK');
    console.log('4. For AMD/Intel: Ensure Vulkan support is enabled');
  }
}

main().catch(console.error);
