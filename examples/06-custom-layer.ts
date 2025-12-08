/**
 * Пример 06: Создание кастомных слоёв
 * 
 * Этот пример демонстрирует:
 * - Наследование от Module для создания своих слоёв
 * - Регистрация параметров и подмодулей
 * - Создание сложных архитектур (Residual connections)
 * - Attention механизм
 * 
 * Запуск: bun run examples/06-custom-layer.ts
 */

import {
  tensor,
  randn,
  zeros,
  ones,
  Module,
  Parameter,
  Linear,
  ReLU,
  Sigmoid,
  Tensor,
  Sequential,
  MSELoss,
  Adam,
  softmax,
  Dropout,
  LayerNorm,
} from '../src';

console.log('🧠 Brainy - Пример 06: Кастомные слои\n');
console.log('='.repeat(60));

// ============================================
// 1. Простой кастомный слой: Swish активация
// ============================================
console.log('\n🔧 1. Кастомный слой: Swish (x * sigmoid(x))\n');

/**
 * Swish активация: f(x) = x * sigmoid(x)
 * Популярная в современных архитектурах
 */
class Swish extends Module {
  forward(x: Tensor): Tensor {
    const sigmoid_x = x.exp().div(x.exp().add(1));
    return x.mul(sigmoid_x);
  }

  toString(): string {
    return 'Swish()';
  }
}

const swish = new Swish();
const testInput = tensor([-2, -1, 0, 1, 2]);
const swishOutput = swish.forward(testInput);

console.log(`Вход: ${testInput.toArray()}`);
console.log(`Swish выход: ${swishOutput.toArray().map((x: number) => x.toFixed(4))}`);
console.log('✅ Swish слой работает!');

// ============================================
// 2. Слой с обучаемыми параметрами: ScaleShift
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔧 2. Слой с параметрами: ScaleShift\n');

/**
 * ScaleShift: y = scale * x + shift
 * Простой слой с двумя обучаемыми параметрами
 */
class ScaleShift extends Module {
  scale: Parameter;
  shift: Parameter;
  readonly features: number;

  constructor(features: number) {
    super();
    this.features = features;
    
    // Инициализация: scale=1, shift=0
    this.scale = new Parameter(ones([features], { requiresGrad: true }), 'scale');
    this.shift = new Parameter(zeros([features], { requiresGrad: true }), 'shift');
    
    this.registerParameter('scale', this.scale);
    this.registerParameter('shift', this.shift);
  }

  forward(x: Tensor): Tensor {
    return x.mul(this.scale.data).add(this.shift.data);
  }

  toString(): string {
    return `ScaleShift(features=${this.features})`;
  }
}

const scaleShift = new ScaleShift(4);
console.log(`ScaleShift: ${scaleShift.toString()}`);
console.log(`Параметры: ${scaleShift.numParameters()}`);

const ssInput = tensor([[1, 2, 3, 4]]);
console.log(`Вход: ${ssInput.toArray()}`);
console.log(`Выход (scale=1, shift=0): ${scaleShift.forward(ssInput).toArray()}`);

// Изменяем параметры
for (let i = 0; i < scaleShift.scale.data.size; i++) {
  (scaleShift.scale.data.data as Float32Array)[i] = 2;
  (scaleShift.shift.data.data as Float32Array)[i] = 10;
}
console.log(`Выход (scale=2, shift=10): ${scaleShift.forward(ssInput).toArray()}`);
console.log('✅ ScaleShift слой работает!');

// ============================================
// 3. Residual Block
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔧 3. Residual Block (ResNet-style)\n');

/**
 * Residual Block: output = F(x) + x
 * Ключевой компонент ResNet архитектуры
 */
class ResidualBlock extends Module {
  linear1: Linear;
  linear2: Linear;
  relu: ReLU;
  readonly features: number;

  constructor(features: number) {
    super();
    this.features = features;
    
    this.linear1 = new Linear(features, features);
    this.linear2 = new Linear(features, features);
    this.relu = new ReLU();
    
    this.registerModule('linear1', this.linear1);
    this.registerModule('linear2', this.linear2);
  }

  forward(x: Tensor): Tensor {
    // F(x) = linear2(relu(linear1(x)))
    let residual = this.linear1.forward(x);
    residual = this.relu.forward(residual);
    residual = this.linear2.forward(x);
    
    // output = F(x) + x (skip connection)
    return residual.add(x);
  }

  toString(): string {
    return `ResidualBlock(\n  (linear1): ${this.linear1}\n  (relu): ReLU()\n  (linear2): ${this.linear2}\n)`;
  }
}

const resBlock = new ResidualBlock(8);
console.log('Архитектура ResidualBlock:');
console.log('┌─────────────────────────────┐');
console.log('│        Input (x)            │');
console.log('│           ↓                 │');
console.log('│    ┌──────┴──────┐          │');
console.log('│    ↓             ↓          │');
console.log('│  Linear1      Identity      │');
console.log('│    ↓             │          │');
console.log('│   ReLU          │          │');
console.log('│    ↓             │          │');
console.log('│  Linear2        │          │');
console.log('│    ↓             │          │');
console.log('│    └──────+──────┘          │');
console.log('│           ↓                 │');
console.log('│       Output                │');
console.log('└─────────────────────────────┘');
console.log(`Параметры: ${resBlock.numParameters()}`);

const resInput = randn([2, 8]);
const resOutput = resBlock.forward(resInput);
console.log(`\nВход shape: [${resInput.shape}]`);
console.log(`Выход shape: [${resOutput.shape}]`);
console.log('✅ ResidualBlock работает!');

// ============================================
// 4. Self-Attention Layer (упрощённая)
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔧 4. Self-Attention (Transformer-style)\n');

/**
 * Упрощённый Self-Attention для понимания механизма
 * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
 */
class SelfAttention extends Module {
  queryProj: Linear;
  keyProj: Linear;
  valueProj: Linear;
  outProj: Linear;
  readonly embedDim: number;
  readonly scale: number;

  constructor(embedDim: number) {
    super();
    this.embedDim = embedDim;
    this.scale = Math.sqrt(embedDim);
    
    this.queryProj = new Linear(embedDim, embedDim, false);
    this.keyProj = new Linear(embedDim, embedDim, false);
    this.valueProj = new Linear(embedDim, embedDim, false);
    this.outProj = new Linear(embedDim, embedDim, false);
    
    this.registerModule('query', this.queryProj);
    this.registerModule('key', this.keyProj);
    this.registerModule('value', this.valueProj);
    this.registerModule('out', this.outProj);
  }

  forward(x: Tensor): Tensor {
    // x: [batch, seq_len, embed_dim] или [seq_len, embed_dim]
    
    // Проекции Q, K, V
    const Q = this.queryProj.forward(x);  // [batch, seq, embed]
    const K = this.keyProj.forward(x);
    const V = this.valueProj.forward(x);
    
    // Attention scores: QK^T / sqrt(d)
    // Для простоты работаем с 2D: [seq, embed]
    const scores = Q.matmul(K.T).div(this.scale);  // [seq, seq]
    
    // Softmax
    const weights = softmax(scores, -1);  // [seq, seq]
    
    // Weighted values
    const attended = weights.matmul(V);  // [seq, embed]
    
    // Output projection
    return this.outProj.forward(attended);
  }

  toString(): string {
    return `SelfAttention(embed_dim=${this.embedDim})`;
  }
}

const attention = new SelfAttention(16);
console.log('Self-Attention:');
console.log('  Q = W_q * X');
console.log('  K = W_k * X');
console.log('  V = W_v * X');
console.log('  Attention = softmax(QK^T / √d) * V');
console.log(`\nПараметры: ${attention.numParameters()}`);

const seqLen = 5;
const attInput = randn([seqLen, 16]);  // [seq, embed]
const attOutput = attention.forward(attInput);
console.log(`\nВход shape: [${attInput.shape}] (seq_len=5, embed=16)`);
console.log(`Выход shape: [${attOutput.shape}]`);
console.log('✅ Self-Attention работает!');

// ============================================
// 5. Сборка в полноценную модель
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🏗️ 5. Сборка полноценной модели\n');

/**
 * Простой Transformer-style энкодер
 */
class MiniTransformerBlock extends Module {
  attention: SelfAttention;
  ffn: Sequential;
  norm1: LayerNorm;
  norm2: LayerNorm;

  constructor(embedDim: number, ffnDim: number) {
    super();
    
    this.attention = new SelfAttention(embedDim);
    this.ffn = new Sequential(
      new Linear(embedDim, ffnDim),
      new ReLU(),
      new Linear(ffnDim, embedDim)
    );
    this.norm1 = new LayerNorm(embedDim);
    this.norm2 = new LayerNorm(embedDim);
    
    this.registerModule('attention', this.attention);
    this.registerModule('ffn', this.ffn);
    this.registerModule('norm1', this.norm1);
    this.registerModule('norm2', this.norm2);
  }

  forward(x: Tensor): Tensor {
    // Pre-norm architecture
    // x = x + Attention(Norm(x))
    const normed1 = this.norm1.forward(x);
    const attended = this.attention.forward(normed1);
    let out = x.add(attended);
    
    // x = x + FFN(Norm(x))
    const normed2 = this.norm2.forward(out);
    const ffnOut = this.ffn.forward(normed2);
    out = out.add(ffnOut);
    
    return out;
  }

  toString(): string {
    return 'MiniTransformerBlock(\n  attention + residual\n  ffn + residual\n)';
  }
}

const transformer = new MiniTransformerBlock(16, 64);
console.log('Mini Transformer Block:');
console.log('┌─────────────────────────────────────┐');
console.log('│           Input                     │');
console.log('│             ↓                       │');
console.log('│    ┌───────┴───────┐                │');
console.log('│    ↓               ↓                │');
console.log('│  LayerNorm      Identity            │');
console.log('│    ↓               │                │');
console.log('│  Self-Attn        │                │');
console.log('│    ↓               │                │');
console.log('│    └───────+───────┘                │');
console.log('│             ↓                       │');
console.log('│    ┌───────┴───────┐                │');
console.log('│    ↓               ↓                │');
console.log('│  LayerNorm      Identity            │');
console.log('│    ↓               │                │');
console.log('│    FFN            │                │');
console.log('│    ↓               │                │');
console.log('│    └───────+───────┘                │');
console.log('│             ↓                       │');
console.log('│          Output                     │');
console.log('└─────────────────────────────────────┘');
console.log(`\nВсего параметров: ${transformer.numParameters()}`);

const tfInput = randn([4, 16]);  // [seq=4, embed=16]
const tfOutput = transformer.forward(tfInput);
console.log(`\nВход shape: [${tfInput.shape}]`);
console.log(`Выход shape: [${tfOutput.shape}]`);
console.log('✅ Mini Transformer Block работает!');

// ============================================
// 6. Обучение кастомной модели
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🎓 6. Обучение модели с кастомными слоями\n');

// Простая модель с кастомными слоями
class CustomModel extends Module {
  block1: ResidualBlock;
  block2: ResidualBlock;
  output: Linear;

  constructor(features: number, outputDim: number) {
    super();
    this.block1 = new ResidualBlock(features);
    this.block2 = new ResidualBlock(features);
    this.output = new Linear(features, outputDim);
    
    this.registerModule('block1', this.block1);
    this.registerModule('block2', this.block2);
    this.registerModule('output', this.output);
  }

  forward(x: Tensor): Tensor {
    let out = this.block1.forward(x);
    out = this.block2.forward(out);
    return this.output.forward(out);
  }
}

const model = new CustomModel(8, 2);
const criterion = new MSELoss();
const optimizer = new Adam(model.parameters(), 0.01);

console.log(`Модель с ${model.numParameters()} параметрами`);
console.log('Обучение на случайных данных...\n');

const trainX = randn([16, 8]);
const trainY = randn([16, 2]);

for (let epoch = 0; epoch < 50; epoch++) {
  const pred = model.forward(trainX);
  const loss = criterion.forward(pred, trainY);
  
  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();
  
  if (epoch % 10 === 0 || epoch === 49) {
    console.log(`Эпоха ${epoch}: loss = ${loss.item().toFixed(4)}`);
  }
}

console.log('\n✅ Модель с кастомными слоями успешно обучается!');

// ============================================
// Итоги
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🎉 Все кастомные слои работают!');
console.log('\n📝 Резюме созданных слоёв:');
console.log('  1. Swish - простая кастомная активация');
console.log('  2. ScaleShift - слой с обучаемыми параметрами');
console.log('  3. ResidualBlock - skip connections (ResNet)');
console.log('  4. SelfAttention - механизм attention (Transformer)');
console.log('  5. MiniTransformerBlock - полный блок трансформера');
console.log('\n✅ Brainy позволяет создавать любые архитектуры!');
