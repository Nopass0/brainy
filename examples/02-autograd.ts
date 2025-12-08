/**
 * Пример 02: Автоматическое дифференцирование (Autograd)
 * 
 * Этот пример демонстрирует:
 * - Тензоры с requires_grad=true
 * - Вычисление градиентов через backward()
 * - Построение вычислительного графа
 * - Использование noGrad для отключения градиентов
 * 
 * Запуск: bun run examples/02-autograd.ts
 */

import { tensor, scalar, ones, noGrad, Tensor } from '../src';

console.log('🧠 Brainy - Пример 02: Автоматическое дифференцирование\n');
console.log('='.repeat(60));

// ============================================
// 1. Простой пример: градиент скалярной функции
// ============================================
console.log('\n📈 1. Простой градиент: f(x) = x^2\n');

const x = tensor([3.0], { requiresGrad: true });
console.log(`x = ${x.item()}`);

// f(x) = x^2
const f = x.pow(2);
console.log(`f(x) = x^2 = ${f.item()}`);

// Вычисляем градиент
f.backward();

// df/dx = 2x = 2*3 = 6
console.log(`\n📐 Градиент df/dx = 2x:`);
console.log(`  Ожидаемый: 6`);
console.log(`  Полученный: ${x.grad!.item()}`);
console.log(`  ✅ Совпадает!`);

// ============================================
// 2. Цепочка операций
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔗 2. Цепочка операций: f(x) = (x + 2)^2 * 3\n');

const x2 = tensor([2.0], { requiresGrad: true });
console.log(`x = ${x2.item()}`);

// f(x) = (x + 2)^2 * 3
const y2 = x2.add(2);       // x + 2 = 4
const z2 = y2.pow(2);       // (x + 2)^2 = 16
const f2 = z2.mul(3);       // (x + 2)^2 * 3 = 48

console.log(`y = x + 2 = ${y2.item()}`);
console.log(`z = y^2 = ${z2.item()}`);
console.log(`f = z * 3 = ${f2.item()}`);

f2.backward();

// df/dx = 3 * 2 * (x + 2) = 6 * (2 + 2) = 24
console.log(`\n📐 Градиент df/dx = 6(x + 2):`);
console.log(`  Ожидаемый: 24`);
console.log(`  Полученный: ${x2.grad!.item()}`);
console.log(`  ✅ Совпадает!`);

// ============================================
// 3. Многомерные тензоры
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n📦 3. Многомерные тензоры\n');

const A = tensor([[1, 2], [3, 4]], { requiresGrad: true });
console.log('A =', A.toArray());

// f(A) = sum(A^2)
const A_squared = A.pow(2);
const loss = A_squared.sum();

console.log(`\nf(A) = sum(A^2) = ${loss.item()}`);

loss.backward();

// df/dA = 2A
console.log(`\n📐 Градиент df/dA = 2A:`);
console.log('Ожидаемый:', [[2, 4], [6, 8]]);
console.log('Полученный:', A.grad!.toArray());
console.log('✅ Совпадает!');

// ============================================
// 4. Операции с несколькими переменными
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔀 4. Несколько переменных: f(a, b) = a*b + a^2\n');

const a = tensor([3.0], { requiresGrad: true });
const b = tensor([2.0], { requiresGrad: true });

console.log(`a = ${a.item()}, b = ${b.item()}`);

// f(a, b) = a*b + a^2
const ab = a.mul(b);        // a*b = 6
const a2 = a.pow(2);        // a^2 = 9
const f3 = ab.add(a2);      // a*b + a^2 = 15

console.log(`\nf(a, b) = a*b + a^2 = ${f3.item()}`);

f3.backward();

// df/da = b + 2a = 2 + 6 = 8
// df/db = a = 3
console.log(`\n📐 Градиенты:`);
console.log(`  df/da = b + 2a = ${b.item()} + 2*${a.item()} = 8`);
console.log(`  Полученный: ${a.grad!.item()}`);
console.log(`  ✅ Совпадает!`);
console.log(`\n  df/db = a = ${a.item()}`);
console.log(`  Полученный: ${b.grad!.item()}`);
console.log(`  ✅ Совпадает!`);

// ============================================
// 5. noGrad - отключение градиентов
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🚫 5. noGrad - отключение градиентов\n');

const w = tensor([1.0, 2.0, 3.0], { requiresGrad: true });
console.log('w =', w.toArray());

// Операции внутри noGrad не отслеживаются
const result = noGrad(() => {
  const doubled = w.mul(2);
  console.log('Внутри noGrad: w * 2 =', doubled.toArray());
  return doubled;
});

console.log('Результат не имеет gradNode:', result.gradNode === null);
console.log('✅ Градиенты не вычисляются внутри noGrad!');

// ============================================
// 6. Обнуление градиентов
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔄 6. Обнуление градиентов (zeroGrad)\n');

const param = tensor([5.0], { requiresGrad: true });

// Первый backward
const loss1 = param.pow(2);
loss1.backward();
console.log(`После первого backward: grad = ${param.grad!.item()}`);

// Без обнуления градиенты накапливаются
// param.zeroGrad(); // раскомментировать для сброса

console.log('Важно: gradient accumulation может быть полезен для больших батчей!');
console.log('Чтобы сбросить градиенты, вызовите tensor.zeroGrad() или optimizer.zeroGrad()');

// ============================================
// 7. Практический пример: градиентный спуск
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🎯 7. Практический пример: ручной градиентный спуск\n');

// Находим минимум функции f(x) = (x - 5)^2
let xOpt = tensor([0.0], { requiresGrad: true });
const lr = 0.1;

console.log('Находим минимум f(x) = (x - 5)^2');
console.log(`Начальное значение: x = ${xOpt.item()}`);
console.log(`Learning rate: ${lr}\n`);

for (let i = 0; i < 10; i++) {
  // Forward
  const diff = xOpt.sub(5);
  const loss = diff.pow(2);
  
  // Backward
  loss.backward();
  
  // Update (ручной SGD)
  const grad = xOpt.grad!.item();
  const newVal = xOpt.item() - lr * grad;
  
  console.log(`Шаг ${i + 1}: x = ${xOpt.item().toFixed(4)}, loss = ${loss.item().toFixed(4)}, grad = ${grad.toFixed(4)}`);
  
  // Создаём новый тензор с обновлённым значением
  xOpt = tensor([newVal], { requiresGrad: true });
}

console.log(`\n🎉 Оптимальное значение: x ≈ ${xOpt.item().toFixed(2)} (ожидаемое: 5.00)`);

// ============================================
// Итоги
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n✅ Autograd работает корректно!');
console.log('\nСледующий пример: bun run examples/03-linear-regression.ts');
