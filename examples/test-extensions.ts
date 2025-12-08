
import { tensor, ones, randn, noGrad, Tensor } from '../src';

console.log('🧪 Testing Brainy Extensions (Batch Matmul & Autograd)\n');

// 1. Batch Matmul Test
console.log('1. Batch Matrix Multiplication');
const B = 2, M = 3, K = 4, N = 5;
const t1 = randn([B, M, K], 0, 1, { requiresGrad: true });
const t2 = randn([B, K, N], 0, 1, { requiresGrad: true });

console.log(`t1 shape: [${t1.shape}]`);
console.log(`t2 shape: [${t2.shape}]`);

const t3 = t1.matmul(t2);
console.log(`Result shape: [${t3.shape}]`);

if (t3.shape[0] === B && t3.shape[1] === M && t3.shape[2] === N) {
    console.log('✅ Shape check passed');
} else {
    console.error('❌ Shape check failed');
}

// Simple logic verify with ones
const o1 = ones([2, 2, 2]);
const o2 = ones([2, 2, 2]);
const o3 = o1.matmul(o2);
// [ [ [1,1],[1,1] ], ... ] @ [ [ [1,1],[1,1] ], ... ] -> [ [ [2,2],[2,2] ], ... ]
const val = o3.flatten().toArray()[0];
if (val === 2) {
    console.log('✅ Value check passed');
} else {
    console.error(`❌ Value check failed. Expected 2, got ${val}`);
}

// Backward check
const loss = t3.sum();
loss.backward();
console.log('✅ Backward pass ran without error');
if (t1.grad && t2.grad) {
    console.log('✅ Gradients computed');
} else {
    console.error('❌ Gradients missing');
}


// 2. Max/Min Autograd Test
console.log('\n2. Max/Min Autograd');
const t4 = tensor([1, 5, 2, 8, 3], { requiresGrad: true });
const maxRes = t4.max();
console.log(`Max value: ${maxRes.values.item()}`);

maxRes.values.backward();
console.log(`Grad: ${t4.grad!.toArray()}`);
// Expected grad: [0, 0, 0, 1, 0]
const gradArr = t4.grad!.toArray() as number[];
if (gradArr[3] === 1 && gradArr[0] === 0) {
    console.log('✅ Max grad check passed');
} else {
    console.error(`❌ Max grad check failed: ${gradArr}`);
}

t4.zeroGrad();
const minRes = t4.min();
minRes.values.backward();
console.log(`Min value: ${minRes.values.item()}`);
console.log(`Min Grad: ${t4.grad!.toArray()}`);
// Expected: [1, 0, 0, 0, 0]
const gradArrMin = t4.grad!.toArray() as number[];
if (gradArrMin[0] === 1 && gradArrMin[1] === 0) {
    console.log('✅ Min grad check passed');
} else {
    console.error(`❌ Min grad check failed: ${gradArrMin}`);
}


// 3. Expand Autograd Test
console.log('\n3. Expand Autograd');
const t5 = tensor([1, 2, 3], { requiresGrad: true }); // shape [3]
const expanded = t5.expand(2, 3); // shape [2, 3] -> [[1,2,3], [1,2,3]]
const sumExpanded = expanded.sum(); // sum = (1+2+3)*2 = 12
console.log(`Sum: ${sumExpanded.item()}`);

sumExpanded.backward();
console.log(`Grad: ${t5.grad!.toArray()}`);
// Grad should be [2, 2, 2] because each element was duplicated 2 times in the sum
const gradExp = t5.grad!.toArray() as number[];
if (gradExp[0] === 2 && gradExp[1] === 2 && gradExp[2] === 2) {
    console.log('✅ Expand grad check passed');
} else {
    console.error(`❌ Expand grad check failed: ${gradExp}`);
}

console.log('\n✨ All tests finished');
