/**
 * Hugging Face Integration Example
 * Download and use pre-trained models from Hugging Face Hub
 */

import {
  createHuggingFaceHub,
  downloadModel,
  loadWeightsIntoModel,
  TRM,
  createTinyTRM,
  tensor,
} from '../src';

async function main() {
  console.log('='.repeat(60));
  console.log('HUGGING FACE INTEGRATION EXAMPLE');
  console.log('='.repeat(60));

  // Create Hub client
  const hub = createHuggingFaceHub({
    // token: process.env.HF_TOKEN, // Optional: for private models
  });

  // Example 1: Get model info
  console.log('\n1. Getting model info...');
  try {
    const info = await hub.getModelInfo('bert-base-uncased');
    console.log('   Model:', info.modelId || 'bert-base-uncased');
    console.log('   Downloads:', info.downloads);
    console.log('   Tags:', info.tags?.slice(0, 5).join(', '));
  } catch (e: any) {
    console.log('   Error:', e.message);
  }

  // Example 2: List files in a model
  console.log('\n2. Listing files in sentence-transformers/all-MiniLM-L6-v2...');
  try {
    const files = await hub.listFiles('sentence-transformers/all-MiniLM-L6-v2');
    console.log('   Files:', files.slice(0, 10).join(', '));
  } catch (e: any) {
    console.log('   Error:', e.message);
  }

  // Example 3: Download and parse safetensors weights
  console.log('\n3. Downloading small model weights...');
  try {
    // Using a small model for demo
    const weights = await downloadModel('sentence-transformers/all-MiniLM-L6-v2');
    console.log('   Downloaded', weights.size, 'tensors');

    // Show some tensor shapes
    let count = 0;
    for (const [name, tensor] of weights) {
      if (count++ < 5) {
        console.log(`   ${name}: shape=${tensor.shape}`);
      }
    }
    if (weights.size > 5) {
      console.log(`   ... and ${weights.size - 5} more`);
    }
  } catch (e: any) {
    console.log('   Error:', e.message);
  }

  // Example 4: Download config
  console.log('\n4. Downloading model config...');
  try {
    const config = await hub.downloadConfig('bert-base-uncased');
    if (config) {
      console.log('   Hidden size:', config.hidden_size);
      console.log('   Num layers:', config.num_hidden_layers);
      console.log('   Vocab size:', config.vocab_size);
    }
  } catch (e: any) {
    console.log('   Error:', e.message);
  }

  // Example 5: Load weights into a model
  console.log('\n5. Loading weights into local model...');
  try {
    // Create a local model
    const model = createTinyTRM(768, 768, 256, 4);
    console.log('   Created TRM model');

    // Download weights (would need a compatible model)
    // For demo, we just show the API
    console.log('   To load weights:');
    console.log('     const weights = await downloadModel("your-model-id");');
    console.log('     loadWeightsIntoModel(model, weights);');
  } catch (e: any) {
    console.log('   Error:', e.message);
  }

  // Example 6: Download tokenizer
  console.log('\n6. Downloading tokenizer...');
  try {
    const tokenizer = await hub.downloadTokenizer('bert-base-uncased');
    console.log('   Vocab loaded:', !!tokenizer.vocab);
    console.log('   Config loaded:', !!tokenizer.config);
    if (tokenizer.vocab) {
      const vocabSize = Object.keys(tokenizer.vocab).length;
      console.log('   Vocab size:', vocabSize);
    }
  } catch (e: any) {
    console.log('   Error:', e.message);
  }

  console.log('\n' + '='.repeat(60));
  console.log('EXAMPLE COMPLETE');
  console.log('='.repeat(60));
}

main().catch(console.error);
