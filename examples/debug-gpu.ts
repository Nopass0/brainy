/**
 * Debug GPU detection
 */

console.log('=== GPU Debug ===\n');

// Check runtime
console.log('1. Runtime check:');
console.log('   IS_BUN:', typeof globalThis.Bun !== 'undefined');
console.log('   Bun version:', typeof Bun !== 'undefined' ? Bun.version : 'N/A');

// Try to load bun-webgpu directly
console.log('\n2. Loading bun-webgpu:');
try {
  const bunWebGPU = require('bun-webgpu');
  console.log('   Module loaded:', !!bunWebGPU);
  console.log('   Module keys:', Object.keys(bunWebGPU));
  console.log('   setupGlobals:', typeof bunWebGPU.setupGlobals);

  if (bunWebGPU.setupGlobals) {
    console.log('\n3. Calling setupGlobals():');
    try {
      bunWebGPU.setupGlobals();
      console.log('   setupGlobals() called successfully');
    } catch (e: any) {
      console.log('   setupGlobals() ERROR:', e.message);
    }
  }

  console.log('\n4. Checking navigator.gpu:');
  console.log('   navigator exists:', typeof navigator !== 'undefined');
  if (typeof navigator !== 'undefined') {
    console.log('   navigator.gpu:', (navigator as any).gpu);
  }

  // Try to get adapter directly
  console.log('\n5. Trying to get GPU adapter:');
  if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
    const gpu = (navigator as any).gpu;
    gpu.requestAdapter().then((adapter: any) => {
      if (adapter) {
        console.log('   Adapter found!');
        adapter.requestAdapterInfo().then((info: any) => {
          console.log('   GPU:', info.description || info.device || 'Unknown');
          console.log('   Vendor:', info.vendor);
        });
      } else {
        console.log('   No adapter available');
      }
    }).catch((e: any) => {
      console.log('   Adapter error:', e.message);
    });
  } else {
    console.log('   navigator.gpu not available');
  }

} catch (e: any) {
  console.log('   Failed to load bun-webgpu:', e.message);
  console.log('   Stack:', e.stack);
}
