/**
 * Пример 01: Базовые операции с тензорами
 * 
 * Этот пример демонстрирует:
 * - Создание тензоров различными способами
 * - Базовые математические операции
 * - Изменение формы и индексацию
 * - Broadcasting
 * 
 * Запуск: bun run examples/01-basic-tensors.ts
 */

import {
  tensor,
  zeros,
  ones,
  rand,
  randn,
  eye,
  linspace,
  arange,
  DType,
} from '../src';

console.log('🧠 Brainy - Пример 01: Базовые операции с тензорами\n');
console.log('='.repeat(60));

// ============================================
// 1. Создание тензоров
// ============================================
console.log('\n📦 1. Создание тензоров\n');

// Из вложенного массива
const t1 = tensor([[1, 2, 3], [4, 5, 6]]);
console.log('Тензор из массива [[1,2,3], [4,5,6]]:');
console.log(`  shape: [${t1.shape}], dtype: ${t1.dtype}`);
console.log(`  data: ${t1.toArray()}`);

// Нули и единицы
const z = zeros([2, 3]);
const o = ones([2, 3]);
console.log(`\nНули [2,3]: ${z.toArray()}`);
console.log(`Единицы [2,3]: ${o.toArray()}`);

// Случайные тензоры
const r = rand([2, 2]);
const rn = randn([2, 2]);
console.log(`\nСлучайный [0,1): ${JSON.stringify(r.toArray())}`);
console.log(`Нормальный N(0,1): ${JSON.stringify(rn.toArray())}`);

// Специальные тензоры
const I = eye(3);
console.log(`\nЕдиничная матрица 3x3:`);
console.log(I.toArray());

const lin = linspace(0, 10, 5);
console.log(`\nlinspace(0, 10, 5): ${lin.toArray()}`);

const ar = arange(0, 10, 2);
console.log(`arange(0, 10, 2): ${ar.toArray()}`);

// ============================================
// 2. Математические операции
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔢 2. Математические операции\n');

const a = tensor([[1, 2], [3, 4]]);
const b = tensor([[5, 6], [7, 8]]);

console.log(`a = ${JSON.stringify(a.toArray())}`);
console.log(`b = ${JSON.stringify(b.toArray())}`);

// Поэлементные операции
console.log(`\na + b = ${JSON.stringify(a.add(b).toArray())}`);
console.log(`a - b = ${JSON.stringify(a.sub(b).toArray())}`);
console.log(`a * b = ${JSON.stringify(a.mul(b).toArray())}`);
console.log(`a / b = ${JSON.stringify(a.div(b).toArray())}`);

// Скалярные операции
console.log(`\na + 10 = ${JSON.stringify(a.add(10).toArray())}`);
console.log(`a * 2 = ${JSON.stringify(a.mul(2).toArray())}`);
console.log(`a ^ 2 = ${JSON.stringify(a.pow(2).toArray())}`);

// Другие операции
console.log(`\nsqrt(a) = ${JSON.stringify(a.sqrt().toArray())}`);
console.log(`exp(a) = ${JSON.stringify(a.exp().toArray())}`);
console.log(`log(a) = ${JSON.stringify(a.log().toArray())}`);
console.log(`abs(-a) = ${JSON.stringify(a.neg().abs().toArray())}`);

// ============================================
// 3. Матричные операции
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n📐 3. Матричные операции\n');

const m1 = tensor([[1, 2], [3, 4]]);
const m2 = tensor([[5, 6], [7, 8]]);

console.log('Matrix multiplication (m1 @ m2):');
const mm = m1.matmul(m2);
console.log(mm.toArray());

console.log('\nТранспонирование m1.T:');
console.log(m1.T.toArray());

// ============================================
// 4. Операции редукции
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n📊 4. Операции редукции\n');

const t = tensor([[1, 2, 3], [4, 5, 6]]);
console.log(`t = ${JSON.stringify(t.toArray())}`);

console.log(`\nsum(): ${t.sum().item()}`);
console.log(`mean(): ${t.mean().item()}`);
console.log(`max(): ${t.max().values.item()}`);
console.log(`min(): ${t.min().values.item()}`);

console.log(`\nsum(dim=0): ${t.sum(0).toArray()}`);
console.log(`sum(dim=1): ${t.sum(1).toArray()}`);
console.log(`mean(dim=1): ${t.mean(1).toArray()}`);

console.log(`\nargmax(): ${t.argmax().item()}`);
console.log(`argmax(dim=1): ${t.argmax(1).toArray()}`);

// ============================================
// 5. Изменение формы
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔄 5. Изменение формы\n');

const orig = arange(12);
console.log(`Оригинал: shape=[${orig.shape}]`);

const reshaped = orig.reshape(3, 4);
console.log(`reshape(3, 4):`);
console.log(reshaped.toArray());

const reshaped2 = orig.reshape(2, -1);
console.log(`\nreshape(2, -1) (автоматическое вычисление):`);
console.log(reshaped2.toArray());

const flat = reshaped.flatten();
console.log(`\nflatten(): [${flat.toArray()}]`);

// Squeeze и unsqueeze
const s = tensor([[1, 2, 3]]);
console.log(`\nОригинал shape: [${s.shape}]`);
console.log(`squeeze(): shape=[${s.squeeze().shape}]`);
console.log(`unsqueeze(0): shape=[${s.unsqueeze(0).shape}]`);

// ============================================
// 6. Broadcasting
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n📡 6. Broadcasting\n');

const x = tensor([[1], [2], [3]]); // [3, 1]
const y = tensor([10, 20, 30]);    // [3]

console.log(`x shape: [${x.shape}]`);
console.log(`y shape: [${y.shape}]`);

const result = x.add(y);
console.log(`\nx + y (broadcast [3,1] + [3] -> [3,3]):`);
console.log(result.toArray());

// ============================================
// 7. Индексация и доступ
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔍 7. Индексация и доступ\n');

const arr = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
console.log('Матрица:');
console.log(arr.toArray());

console.log(`\narr.get(0, 0) = ${arr.get(0, 0)}`);
console.log(`arr.get(1, 2) = ${arr.get(1, 2)}`);
console.log(`arr.getRow(1) = ${arr.getRow(1).toArray()}`);
console.log(`arr.item() (для скаляра): ${arr.sum().item()}`);

// ============================================
// Итоги
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n✅ Все базовые операции работают корректно!');
console.log('\nСледующий пример: bun run examples/02-autograd.ts');
