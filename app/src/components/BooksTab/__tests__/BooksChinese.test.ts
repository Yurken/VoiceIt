import { createInstance } from 'i18next';
import { describe, expect, it } from 'vitest';
import en from '@/i18n/locales/en/translation.json';
import zhCN from '@/i18n/locales/zh-CN/translation.json';

function leafPaths(value: Record<string, unknown>, prefix = ''): string[] {
  return Object.entries(value).flatMap(([key, item]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return typeof item === 'object' && item !== null
      ? leafPaths(item as Record<string, unknown>, path)
      : [path];
  });
}

describe('Books Chinese translations', () => {
  it('defines every Books label in Chinese without relying on the English fallback', async () => {
    const instance = createInstance();
    await instance.init({ lng: 'zh-CN', fallbackLng: false, resources: { 'zh-CN': { translation: zhCN } } });
    for (const path of leafPaths(en.books)) {
      expect(instance.exists(`books.${path}`), path).toBe(true);
    }
    expect(instance.t('nav.books')).toBe('有声书');
    expect(instance.t('books.title')).toBe('有声书');
    expect(instance.t('books.import.voiceModePreset')).toBe('Qwen 内置声音（默认）');
    expect(instance.t('books.overview.generateSelected', { count: 3 })).toBe('生成选中章节（3）');
    expect(instance.t('books.chapterEditor.voiceStage')).toBe('年龄阶段');
  });
});
