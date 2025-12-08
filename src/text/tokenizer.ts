/**
 * @fileoverview Токенизаторы для обработки текста
 * @description BPE (Byte Pair Encoding) и WordPiece токенизаторы
 */

import { Tensor, tensor, zeros } from '../core/tensor';
import { DType } from '../core/dtype';

/**
 * Специальные токены
 */
export const SpecialTokens = {
  PAD: '[PAD]',
  UNK: '[UNK]',
  CLS: '[CLS]',
  SEP: '[SEP]',
  MASK: '[MASK]',
  BOS: '<|bos|>',
  EOS: '<|eos|>',
} as const;

/**
 * Конфигурация токенизатора
 */
export interface TokenizerConfig {
  /** Размер словаря */
  vocabSize: number;
  /** Максимальная длина последовательности */
  maxLength?: number;
  /** Специальные токены */
  specialTokens?: Record<string, string>;
  /** Использовать lowercase */
  lowercase?: boolean;
  /** Padding side ('left' или 'right') */
  paddingSide?: 'left' | 'right';
  /** Truncation side */
  truncationSide?: 'left' | 'right';
}

/**
 * Результат токенизации
 */
export interface TokenizerOutput {
  /** ID токенов */
  inputIds: number[];
  /** Маска внимания (1 для реальных токенов, 0 для padding) */
  attentionMask: number[];
  /** Token type IDs (для BERT-подобных моделей) */
  tokenTypeIds?: number[];
}

/**
 * Батч токенов
 */
export interface BatchEncoding {
  inputIds: Tensor;
  attentionMask: Tensor;
  tokenTypeIds?: Tensor;
}

/**
 * Базовый класс токенизатора
 */
export abstract class Tokenizer {
  protected vocab: Map<string, number> = new Map();
  protected reverseVocab: Map<number, string> = new Map();
  protected config: TokenizerConfig;

  constructor(config: TokenizerConfig) {
    this.config = {
      maxLength: 512,
      lowercase: false,
      paddingSide: 'right',
      truncationSide: 'right',
      specialTokens: {
        pad: SpecialTokens.PAD,
        unk: SpecialTokens.UNK,
        bos: SpecialTokens.BOS,
        eos: SpecialTokens.EOS,
      },
      ...config,
    };
  }

  /**
   * Получает ID токена
   */
  getTokenId(token: string): number {
    return this.vocab.get(token) ?? this.vocab.get(this.config.specialTokens!.unk)!;
  }

  /**
   * Получает токен по ID
   */
  getToken(id: number): string {
    return this.reverseVocab.get(id) ?? this.config.specialTokens!.unk;
  }

  /**
   * Размер словаря
   */
  get vocabSize(): number {
    return this.vocab.size;
  }

  /**
   * ID специальных токенов
   */
  get padTokenId(): number {
    return this.vocab.get(this.config.specialTokens!.pad!)!;
  }

  get unkTokenId(): number {
    return this.vocab.get(this.config.specialTokens!.unk!)!;
  }

  get bosTokenId(): number | undefined {
    const token = this.config.specialTokens?.bos;
    return token ? this.vocab.get(token) : undefined;
  }

  get eosTokenId(): number | undefined {
    const token = this.config.specialTokens?.eos;
    return token ? this.vocab.get(token) : undefined;
  }

  /**
   * Базовая предобработка текста
   */
  protected preprocess(text: string): string {
    if (this.config.lowercase) {
      text = text.toLowerCase();
    }
    // Нормализация пробелов
    text = text.replace(/\s+/g, ' ').trim();
    return text;
  }

  /**
   * Абстрактный метод токенизации
   */
  abstract tokenize(text: string): string[];

  /**
   * Кодирует текст в ID токенов
   */
  encode(text: string, addSpecialTokens: boolean = true): TokenizerOutput {
    const preprocessed = this.preprocess(text);
    const tokens = this.tokenize(preprocessed);

    let inputIds: number[] = tokens.map((t) => this.getTokenId(t));

    // Добавляем специальные токены
    if (addSpecialTokens) {
      if (this.bosTokenId !== undefined) {
        inputIds = [this.bosTokenId, ...inputIds];
      }
      if (this.eosTokenId !== undefined) {
        inputIds = [...inputIds, this.eosTokenId];
      }
    }

    // Truncation
    const maxLen = this.config.maxLength!;
    if (inputIds.length > maxLen) {
      if (this.config.truncationSide === 'right') {
        inputIds = inputIds.slice(0, maxLen);
      } else {
        inputIds = inputIds.slice(-maxLen);
      }
    }

    // Padding
    const attentionMask = new Array(inputIds.length).fill(1);
    while (inputIds.length < maxLen) {
      if (this.config.paddingSide === 'right') {
        inputIds.push(this.padTokenId);
        attentionMask.push(0);
      } else {
        inputIds.unshift(this.padTokenId);
        attentionMask.unshift(0);
      }
    }

    return { inputIds, attentionMask };
  }

  /**
   * Декодирует ID токенов обратно в текст
   */
  decode(ids: number[], skipSpecialTokens: boolean = true): string {
    const specialIds = new Set<number>();
    if (skipSpecialTokens) {
      specialIds.add(this.padTokenId);
      if (this.bosTokenId !== undefined) specialIds.add(this.bosTokenId);
      if (this.eosTokenId !== undefined) specialIds.add(this.eosTokenId);
    }

    const tokens = ids
      .filter((id) => !specialIds.has(id))
      .map((id) => this.getToken(id));

    return this.detokenize(tokens);
  }

  /**
   * Объединяет токены в текст
   */
  protected abstract detokenize(tokens: string[]): string;

  /**
   * Кодирует батч текстов
   */
  encodeBatch(texts: string[], addSpecialTokens: boolean = true): BatchEncoding {
    const encodings = texts.map((t) => this.encode(t, addSpecialTokens));

    const inputIds = tensor(
      encodings.map((e) => e.inputIds),
      { dtype: DType.Int32 }
    );
    const attentionMask = tensor(
      encodings.map((e) => e.attentionMask),
      { dtype: DType.Int32 }
    );

    return { inputIds, attentionMask };
  }

  /**
   * Сохраняет токенизатор
   */
  save(): object {
    return {
      type: this.constructor.name,
      config: this.config,
      vocab: Object.fromEntries(this.vocab),
    };
  }

  /**
   * Загружает словарь
   */
  loadVocab(vocab: Record<string, number>): void {
    this.vocab.clear();
    this.reverseVocab.clear();

    for (const [token, id] of Object.entries(vocab)) {
      this.vocab.set(token, id);
      this.reverseVocab.set(id, token);
    }
  }
}

/**
 * BPE (Byte Pair Encoding) токенизатор
 * Используется в GPT-2, GPT-3, и других моделях
 */
export class BPETokenizer extends Tokenizer {
  private merges: Map<string, string> = new Map();
  private mergeRanks: Map<string, number> = new Map();
  private cache: Map<string, string[]> = new Map();

  constructor(config: TokenizerConfig) {
    super(config);
    this.initializeBaseVocab();
  }

  /**
   * Инициализирует базовый словарь (байты + специальные токены)
   */
  private initializeBaseVocab(): void {
    let id = 0;

    // Специальные токены
    for (const token of Object.values(this.config.specialTokens!)) {
      this.vocab.set(token, id);
      this.reverseVocab.set(id, token);
      id++;
    }

    // Базовые символы (ASCII + unicode)
    for (let i = 33; i <= 126; i++) {
      const char = String.fromCharCode(i);
      if (!this.vocab.has(char)) {
        this.vocab.set(char, id);
        this.reverseVocab.set(id, char);
        id++;
      }
    }

    // Пробел и основные символы
    const extraChars = ' \n\t';
    for (const char of extraChars) {
      if (!this.vocab.has(char)) {
        this.vocab.set(char, id);
        this.reverseVocab.set(id, char);
        id++;
      }
    }
  }

  /**
   * Обучает BPE на корпусе текстов
   */
  train(texts: string[], numMerges: number = 1000): void {
    // Подсчитываем частоты слов
    const wordFreqs = new Map<string, number>();

    for (const text of texts) {
      const preprocessed = this.preprocess(text);
      const words = preprocessed.split(/\s+/);

      for (const word of words) {
        if (word.length === 0) continue;
        const key = word.split('').join(' ') + ' </w>';
        wordFreqs.set(key, (wordFreqs.get(key) || 0) + 1);
      }
    }

    // Обучаем BPE
    for (let i = 0; i < numMerges; i++) {
      const pairs = this.getPairFreqs(wordFreqs);
      if (pairs.size === 0) break;

      // Находим наиболее частую пару
      let bestPair = '';
      let bestFreq = 0;
      for (const [pair, freq] of pairs) {
        if (freq > bestFreq) {
          bestFreq = freq;
          bestPair = pair;
        }
      }

      if (bestFreq === 0) break;

      // Добавляем merge
      const [a, b] = bestPair.split(' ');
      const merged = a + b;
      this.merges.set(bestPair, merged);
      this.mergeRanks.set(bestPair, i);

      // Добавляем новый токен в словарь
      if (!this.vocab.has(merged)) {
        const newId = this.vocab.size;
        this.vocab.set(merged, newId);
        this.reverseVocab.set(newId, merged);
      }

      // Обновляем wordFreqs
      const newWordFreqs = new Map<string, number>();
      for (const [word, freq] of wordFreqs) {
        const newWord = word.split(bestPair).join(merged);
        newWordFreqs.set(newWord, (newWordFreqs.get(newWord) || 0) + freq);
      }
      wordFreqs.clear();
      for (const [k, v] of newWordFreqs) {
        wordFreqs.set(k, v);
      }
    }

    this.cache.clear();
  }

  /**
   * Получает частоты пар символов
   */
  private getPairFreqs(wordFreqs: Map<string, number>): Map<string, number> {
    const pairs = new Map<string, number>();

    for (const [word, freq] of wordFreqs) {
      const symbols = word.split(' ');
      for (let i = 0; i < symbols.length - 1; i++) {
        const pair = symbols[i] + ' ' + symbols[i + 1];
        pairs.set(pair, (pairs.get(pair) || 0) + freq);
      }
    }

    return pairs;
  }

  /**
   * Токенизирует текст
   */
  tokenize(text: string): string[] {
    const words = text.split(/(\s+)/);
    const tokens: string[] = [];

    for (const word of words) {
      if (word.length === 0) continue;

      if (/^\s+$/.test(word)) {
        tokens.push(word);
        continue;
      }

      // Проверяем кеш
      if (this.cache.has(word)) {
        tokens.push(...this.cache.get(word)!);
        continue;
      }

      // Применяем BPE
      let symbols = word.split('');
      symbols[symbols.length - 1] += '</w>';

      while (symbols.length > 1) {
        let minRank = Infinity;
        let minPair = '';
        let minIdx = -1;

        for (let i = 0; i < symbols.length - 1; i++) {
          const pair = symbols[i] + ' ' + symbols[i + 1];
          const rank = this.mergeRanks.get(pair);
          if (rank !== undefined && rank < minRank) {
            minRank = rank;
            minPair = pair;
            minIdx = i;
          }
        }

        if (minIdx === -1) break;

        const merged = this.merges.get(minPair)!;
        symbols = [
          ...symbols.slice(0, minIdx),
          merged,
          ...symbols.slice(minIdx + 2),
        ];
      }

      // Удаляем маркер конца слова
      symbols = symbols.map((s) => s.replace('</w>', ''));

      this.cache.set(word, symbols);
      tokens.push(...symbols);
    }

    return tokens;
  }

  /**
   * Объединяет токены в текст
   */
  protected detokenize(tokens: string[]): string {
    return tokens.join('');
  }

  /**
   * Загружает merges
   */
  loadMerges(merges: [string, string][]): void {
    this.merges.clear();
    this.mergeRanks.clear();

    for (let i = 0; i < merges.length; i++) {
      const [pair, merged] = merges[i];
      this.merges.set(pair, merged);
      this.mergeRanks.set(pair, i);

      if (!this.vocab.has(merged)) {
        const newId = this.vocab.size;
        this.vocab.set(merged, newId);
        this.reverseVocab.set(newId, merged);
      }
    }

    this.cache.clear();
  }

  /**
   * Сохраняет токенизатор
   */
  save(): object {
    return {
      ...super.save(),
      merges: Array.from(this.merges.entries()),
    };
  }
}

/**
 * WordPiece токенизатор
 * Используется в BERT и других моделях
 */
export class WordPieceTokenizer extends Tokenizer {
  private prefixToken: string = '##';
  private maxWordLength: number = 100;

  constructor(config: TokenizerConfig) {
    super({
      ...config,
      specialTokens: {
        pad: SpecialTokens.PAD,
        unk: SpecialTokens.UNK,
        cls: SpecialTokens.CLS,
        sep: SpecialTokens.SEP,
        mask: SpecialTokens.MASK,
        ...config.specialTokens,
      },
    });
    this.initializeVocab();
  }

  /**
   * Инициализирует базовый словарь
   */
  private initializeVocab(): void {
    let id = 0;

    // Специальные токены
    for (const token of Object.values(this.config.specialTokens!)) {
      this.vocab.set(token, id);
      this.reverseVocab.set(id, token);
      id++;
    }

    // Базовые символы
    for (let i = 33; i <= 126; i++) {
      const char = String.fromCharCode(i);
      if (!this.vocab.has(char)) {
        this.vocab.set(char, id);
        this.reverseVocab.set(id, char);
        id++;
      }
    }
  }

  /**
   * Обучает WordPiece на корпусе текстов
   */
  train(texts: string[], vocabSize: number = 30000): void {
    // Подсчитываем частоты слов
    const wordFreqs = new Map<string, number>();

    for (const text of texts) {
      const preprocessed = this.preprocess(text);
      const words = preprocessed.split(/\s+/);

      for (const word of words) {
        if (word.length === 0 || word.length > this.maxWordLength) continue;
        wordFreqs.set(word, (wordFreqs.get(word) || 0) + 1);
      }
    }

    // Начальный словарь - все символы
    const charFreqs = new Map<string, number>();
    for (const [word, freq] of wordFreqs) {
      for (const char of word) {
        charFreqs.set(char, (charFreqs.get(char) || 0) + freq);
      }
    }

    // Добавляем символы в словарь
    for (const char of charFreqs.keys()) {
      if (!this.vocab.has(char)) {
        const id = this.vocab.size;
        this.vocab.set(char, id);
        this.reverseVocab.set(id, char);
      }
    }

    // Итеративно добавляем подслова
    while (this.vocab.size < vocabSize) {
      const pairScores = new Map<string, number>();

      for (const [word, wordFreq] of wordFreqs) {
        const subwords = this.tokenizeWord(word);
        if (subwords.length < 2) continue;

        for (let i = 0; i < subwords.length - 1; i++) {
          const pair = subwords[i] + subwords[i + 1].replace(this.prefixToken, '');
          const score = wordFreq; // Можно использовать более сложную формулу
          pairScores.set(pair, (pairScores.get(pair) || 0) + score);
        }
      }

      if (pairScores.size === 0) break;

      // Находим лучшую пару
      let bestPair = '';
      let bestScore = 0;
      for (const [pair, score] of pairScores) {
        if (score > bestScore) {
          bestScore = score;
          bestPair = pair;
        }
      }

      if (bestScore === 0) break;

      // Добавляем в словарь
      if (!this.vocab.has(bestPair)) {
        const id = this.vocab.size;
        this.vocab.set(bestPair, id);
        this.reverseVocab.set(id, bestPair);
      }

      // Добавляем версию с ##
      const prefixedPair = this.prefixToken + bestPair;
      if (!this.vocab.has(prefixedPair)) {
        const id = this.vocab.size;
        this.vocab.set(prefixedPair, id);
        this.reverseVocab.set(id, prefixedPair);
      }
    }
  }

  /**
   * Токенизирует одно слово
   */
  private tokenizeWord(word: string): string[] {
    if (word.length > this.maxWordLength) {
      return [this.config.specialTokens!.unk];
    }

    const tokens: string[] = [];
    let start = 0;

    while (start < word.length) {
      let end = word.length;
      let foundToken: string | null = null;

      while (start < end) {
        let substr = word.slice(start, end);
        if (start > 0) {
          substr = this.prefixToken + substr;
        }

        if (this.vocab.has(substr)) {
          foundToken = substr;
          break;
        }
        end--;
      }

      if (foundToken === null) {
        return [this.config.specialTokens!.unk];
      }

      tokens.push(foundToken);
      start = end;
    }

    return tokens;
  }

  /**
   * Токенизирует текст
   */
  tokenize(text: string): string[] {
    const words = text.split(/\s+/);
    const tokens: string[] = [];

    for (const word of words) {
      if (word.length === 0) continue;
      tokens.push(...this.tokenizeWord(word));
    }

    return tokens;
  }

  /**
   * Объединяет токены в текст
   */
  protected detokenize(tokens: string[]): string {
    let text = '';

    for (const token of tokens) {
      if (token.startsWith(this.prefixToken)) {
        text += token.slice(this.prefixToken.length);
      } else {
        if (text.length > 0) {
          text += ' ';
        }
        text += token;
      }
    }

    return text;
  }

  /**
   * CLS token ID
   */
  get clsTokenId(): number | undefined {
    return this.vocab.get(this.config.specialTokens!.cls!);
  }

  /**
   * SEP token ID
   */
  get sepTokenId(): number | undefined {
    return this.vocab.get(this.config.specialTokens!.sep!);
  }

  /**
   * MASK token ID
   */
  get maskTokenId(): number | undefined {
    return this.vocab.get(this.config.specialTokens!.mask!);
  }
}

/**
 * Простой символьный токенизатор
 * Разбивает текст на отдельные символы
 * Поддерживает Unicode (включая кириллицу)
 */
export class CharTokenizer extends Tokenizer {
  private dynamicVocab: boolean = true;

  constructor(config: Partial<TokenizerConfig> = {}) {
    super({
      vocabSize: 65536, // Support Unicode
      maxLength: 512,
      ...config,
    });
    this.initializeVocab();
  }

  /**
   * Инициализирует словарь символов
   */
  private initializeVocab(): void {
    let id = 0;

    // Специальные токены
    for (const token of Object.values(this.config.specialTokens!)) {
      this.vocab.set(token, id);
      this.reverseVocab.set(id, token);
      id++;
    }

    // ASCII символы (32-126 - printable)
    for (let i = 32; i < 127; i++) {
      const char = String.fromCharCode(i);
      if (!this.vocab.has(char)) {
        this.vocab.set(char, id);
        this.reverseVocab.set(id, char);
        id++;
      }
    }

    // Cyrillic characters (U+0400 - U+04FF)
    for (let i = 0x0400; i <= 0x04ff; i++) {
      const char = String.fromCharCode(i);
      if (!this.vocab.has(char)) {
        this.vocab.set(char, id);
        this.reverseVocab.set(id, char);
        id++;
      }
    }

    // Common punctuation and special chars
    const extraChars = '\n\t\r';
    for (const char of extraChars) {
      if (!this.vocab.has(char)) {
        this.vocab.set(char, id);
        this.reverseVocab.set(id, char);
        id++;
      }
    }
  }

  /**
   * Добавляет символ в словарь если его нет
   */
  private ensureChar(char: string): void {
    if (!this.vocab.has(char) && this.dynamicVocab) {
      const id = this.vocab.size;
      this.vocab.set(char, id);
      this.reverseVocab.set(id, char);
    }
  }

  /**
   * Токенизирует текст
   */
  tokenize(text: string): string[] {
    const chars = [...text]; // Proper Unicode splitting
    // Ensure all characters are in vocab
    for (const char of chars) {
      this.ensureChar(char);
    }
    return chars;
  }

  /**
   * Объединяет токены в текст
   */
  protected detokenize(tokens: string[]): string {
    return tokens.join('');
  }

  /**
   * Обучает токенизатор на корпусе текстов (добавляет все символы)
   */
  train(texts: string[]): void {
    for (const text of texts) {
      for (const char of [...text]) {
        this.ensureChar(char);
      }
    }
  }
}

/**
 * Создаёт токенизатор из сохранённых данных
 */
export function loadTokenizer(data: {
  type: string;
  config: TokenizerConfig;
  vocab: Record<string, number>;
  merges?: [string, string][];
}): Tokenizer {
  let tokenizer: Tokenizer;

  switch (data.type) {
    case 'BPETokenizer':
      tokenizer = new BPETokenizer(data.config);
      if (data.merges) {
        (tokenizer as BPETokenizer).loadMerges(data.merges);
      }
      break;
    case 'WordPieceTokenizer':
      tokenizer = new WordPieceTokenizer(data.config);
      break;
    case 'CharTokenizer':
      tokenizer = new CharTokenizer(data.config);
      break;
    default:
      throw new Error(`Unknown tokenizer type: ${data.type}`);
  }

  tokenizer.loadVocab(data.vocab);
  return tokenizer;
}
