import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useImportBook, useAnalyzeBook } from '@/lib/hooks/useBooks';
import { cn } from '@/lib/utils/cn';
import { useBooksStore } from '@/stores/booksStore';

// Supported extensions
const ACCEPTED_EXTS = ['.epub', '.fb2', '.txt', '.pdf'];

function isAccepted(file: File): boolean {
  const lower = file.name.toLowerCase();
  return ACCEPTED_EXTS.some((ext) => lower.endsWith(ext));
}

export function BookImport() {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [extError, setExtError] = useState<string | null>(null);
  const [modelSize, setModelSize] = useState<string>('1.7B');
  const [narratorVoice, setNarratorVoice] = useState<string>('auto');
  const [autoAnalyze, setAutoAnalyze] = useState(true);
  const [voiceMode, setVoiceMode] = useState<'preset' | 'designed'>('preset');

  const setView = useBooksStore((s) => s.setView);
  const setSelectedBookId = useBooksStore((s) => s.setSelectedBookId);

  const importMutation = useImportBook();
  const analyzeMutation = useAnalyzeBook();

  const book = importMutation.data;

  // ── File handling ─────────────────────────────────────────────────────────

  function handleFile(file: File) {
    if (!isAccepted(file)) {
      setExtError(t('books.import.unsupportedFormat'));
      return;
    }
    setExtError(null);
    importMutation.mutate({ file, opts: { model_size: modelSize, narrator_voice_id: narratorVoice, voice_mode: voiceMode } }, {
      onSuccess: (imported) => {
        if (autoAnalyze) startAnalysis(imported.id);
      },
    });
  }

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // Reset input so the same file can be re-selected after an error
    e.target.value = '';
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  function handleDragOver(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(true);
  }

  function handleDragLeave() {
    setDragOver(false);
  }

  // ── Analyze ───────────────────────────────────────────────────────────────

  function startAnalysis(bookId: string) {
    analyzeMutation.mutate(
      { bookId, opts: { model_size: modelSize, narrator_voice_id: narratorVoice, voice_mode: voiceMode } },
      {
        onSuccess: () => {
          setSelectedBookId(bookId);
          setView('analysis');
        },
      },
    );
  }

  function handleAnalyze() {
    if (book) startAnalysis(book.id);
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-4 max-w-3xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">{t('books.title')}</h1>
        <p className="text-sm text-muted-foreground mt-0.5">{t('books.subtitle')}</p>
      </div>

      {/* Dropzone card */}
      <Card>
        <CardContent className="p-4">
          {/* Hidden file input — carries the book-dropzone testid so Playwright's
              setInputFiles() can target it directly */}
          <input
            ref={inputRef}
            data-testid="book-dropzone"
            type="file"
            accept={ACCEPTED_EXTS.join(',')}
            className="sr-only"
            onChange={handleInputChange}
            aria-label={t('books.import.dropzoneLabel')}
          />

          {/* Visible drag-drop zone */}
          <div
            role="button"
            tabIndex={0}
            aria-label={t('books.import.dropzoneLabel')}
            className={cn(
              'flex flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed p-8 text-center cursor-pointer transition-colors',
              dragOver ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50',
              (importMutation.isPending || analyzeMutation.isPending) && 'opacity-60 pointer-events-none',
            )}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
            }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <p className="text-sm">
              {t('books.import.dropzoneHint')}
            </p>
            <p className="text-xs text-muted-foreground">
              {t('books.import.pdfNote')}
            </p>
          </div>

          {/* Extension validation error */}
          {extError && (
            <p className="mt-2 text-sm text-destructive" role="alert">
              {extError}
            </p>
          )}

          {/* Import error from server */}
          {importMutation.isError && (
            <p className="mt-2 text-sm text-destructive" role="alert">
              {t('books.import.importError')}
            </p>
          )}

          {/* Loading indicator */}
          {importMutation.isPending && (
            <p className="mt-2 text-sm text-muted-foreground">{t('common.loading')}</p>
          )}
        </CardContent>
      </Card>

      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={autoAnalyze} onChange={(e) => setAutoAnalyze(e.target.checked)} />
        {t('books.import.autoAnalyze')}
      </label>
      {analyzeMutation.isError && <p className="text-sm text-destructive" role="alert">{t('books.import.analysisFailed')}</p>}

      {/* Detected metadata — only shown after a successful import */}
      {book && (
        <>
          <Card data-testid="book-metadata">
            <CardContent className="p-4">
              <h2 className="text-base font-semibold mb-3">{t('books.import.detected')}</h2>
              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
                <dt className="text-muted-foreground">{t('books.import.fieldTitle')}</dt>
                <dd data-testid="meta-title" className="font-medium">{book.title}</dd>

                <dt className="text-muted-foreground">{t('books.import.fieldAuthor')}</dt>
                <dd data-testid="meta-author">{book.author ?? t('common.unknown')}</dd>

                <dt className="text-muted-foreground">{t('books.import.fieldFormat')}</dt>
                <dd>
                  <Badge variant="secondary" className="uppercase">
                    {book.source_format}
                  </Badge>
                </dd>

                <dt className="text-muted-foreground">{t('books.import.fieldChapters')}</dt>
                <dd data-testid="meta-chapters">
                  {t('books.import.chapterCount', { count: book.chapters.length })}
                </dd>
              </dl>
            </CardContent>
          </Card>

        </>
      )}

          {/* Analysis options */}
          <Card>
            <CardContent className="p-4">
              <h2 className="text-base font-semibold mb-3">{t('books.import.analysisOptions')}</h2>
              <div className="mb-4 flex flex-col gap-1.5">
                <label htmlFor="book-voice-mode" className="text-xs text-muted-foreground">{t('books.import.voiceMode')}</label>
                <select id="book-voice-mode" className="rounded border bg-background px-3 py-2 text-sm" value={voiceMode} onChange={(event) => setVoiceMode(event.target.value as 'preset' | 'designed')}>
                  <option value="preset">{t('books.import.voiceModePreset')}</option>
                  <option value="designed">{t('books.import.voiceModeDesigned')}</option>
                </select>
                <p className="text-xs text-muted-foreground">{t(voiceMode === 'designed' ? 'books.import.voiceModeDesignedHint' : 'books.import.qwenDefault')}</p>
              </div>
              <div className="flex flex-wrap gap-4">
                {/* Model size */}
                <div className="flex flex-col gap-1.5 flex-1 min-w-[200px]">
                  <label
                    htmlFor="model-select-trigger"
                    className="text-xs text-muted-foreground"
                  >
                    {t('books.import.modelLabel')}
                  </label>
                  <Select value={modelSize} onValueChange={setModelSize}>
                    <SelectTrigger
                      id="model-select-trigger"
                      data-testid="model-select"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1.7B">
                        {t('books.import.model17b')}
                      </SelectItem>
                      <SelectItem value="0.6B">
                        {t('books.import.model06b')}
                      </SelectItem>
                      <SelectItem value="4B">
                        {t('books.import.model4b')}
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Narrator voice */}
                <div className="flex flex-col gap-1.5 flex-1 min-w-[200px]">
                  <label
                    htmlFor="narrator-select-trigger"
                    className="text-xs text-muted-foreground"
                  >
                    {t('books.import.narratorLabel')}
                  </label>
                  <Select value={narratorVoice} onValueChange={setNarratorVoice}>
                    <SelectTrigger
                      id="narrator-select-trigger"
                      data-testid="narrator-select"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">
                        {t('books.import.narratorAuto')}
                      </SelectItem>

                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Actions */}
              <div className="flex justify-end gap-2 mt-4">
                <Button
                  variant="ghost"
                  onClick={() => setView('library')}
                >
                  {t('common.cancel')}
                </Button>
                {book && <Button
                  data-testid="analyze-btn"
                  onClick={handleAnalyze}
                  disabled={!book || analyzeMutation.isPending}
                >
                  {analyzeMutation.isPending
                    ? t('books.import.analyzing')
                    : t('books.import.analyzeBtn')}
                </Button>}
              </div>
            </CardContent>
          </Card>
    </div>
  );
}
