/**
 * Minimal GPU test - direct WebGPU usage without our backend
 */

async function main() {
  console.log('=== Minimal GPU Test ===\n');

  // Load bun-webgpu
  const bunWebGPU = await import('bun-webgpu');
  bunWebGPU.setupGlobals();

  const gpu = (navigator as any).gpu;
  if (!gpu) {
    console.log('WebGPU not available');
    return;
  }

  console.log('1. Requesting adapter...');
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    console.log('No adapter found');
    return;
  }
  console.log('   Adapter found');

  console.log('\n2. Requesting device...');
  const device = await adapter.requestDevice();
  console.log('   Device created');

  // Simple shader - just copies input to output
  const shaderCode = `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      let idx = id.x;
      if (idx < arrayLength(&output)) {
        output[idx] = input[idx] * 2.0;
      }
    }
  `;

  console.log('\n3. Creating shader module...');
  const shaderModule = device.createShaderModule({ code: shaderCode });
  console.log('   Shader created');

  console.log('\n4. Creating pipeline...');
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
  });
  console.log('   Pipeline created');

  // Create buffers
  const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const size = data.byteLength;

  console.log('\n5. Creating buffers...');
  const inputBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const stagingBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  console.log('   Buffers created');

  console.log('\n6. Writing data to input buffer...');
  device.queue.writeBuffer(inputBuffer, 0, data);
  console.log('   Data written');

  console.log('\n7. Creating bind group...');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  });
  console.log('   Bind group created');

  console.log('\n8. Running compute pass...');
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(1);
  passEncoder.end();

  // Copy output to staging buffer
  commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, size);

  device.queue.submit([commandEncoder.finish()]);
  console.log('   Commands submitted');

  console.log('\n9. Waiting for GPU work to complete...');
  await device.queue.onSubmittedWorkDone();
  console.log('   GPU work done');

  console.log('\n10. Reading results...');
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(stagingBuffer.getMappedRange().slice(0));
  stagingBuffer.unmap();
  console.log('    Input:', Array.from(data));
  console.log('    Output:', Array.from(result));

  console.log('\n11. Cleanup...');
  // Don't destroy buffers - let them be garbage collected
  // inputBuffer.destroy();
  // outputBuffer.destroy();
  // stagingBuffer.destroy();
  console.log('    Skipping buffer.destroy() due to bun-webgpu bug');

  console.log('\n=== SUCCESS ===');
}

main().catch(console.error);
