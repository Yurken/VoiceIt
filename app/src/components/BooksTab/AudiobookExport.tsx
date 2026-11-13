/**
 * AudiobookExport — audiobook export UI (D7).
 *
 * Launched from the hub's export-btn → setView('export').
 * Provides: format selection, quality/bitrate select, metadata (title/author)
 * + cover drop, a Start button, streamed progress, and a Download button
 * enabled on completion.
 *
 * Subscribes via useBookProgress (C3) for export_progress / export_complete.
 * On complete, enables Download which fetches the Blob via useDownloadExport
 * and saves via platform.filesystem.saveFile with the correct extension.
 *
 * data-testids match wireframe-06, consumed by S10 (d7.spec.ts):
 *   export-format, export-metadata (+ cover-drop), export-action (+ export-status,
 *   start-export-btn, download-btn)
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useBookProgress } from '@/lib/hooks/useBookProgress';
import { useBook, useDownloadExport, useStartExport } from '@/lib/hooks/useBooks';
import type {
  BookErrorEvent,
  ExportBitrate,
  ExportCompleteEvent,
  ExportFormat,
  ExportProgressEvent,
} from '@/lib/api/types';
import { cn } from '@/lib/utils/cn';
import { useBooksStore } from '@/stores/booksStore';

// ─── Constants ────────────────────────────────────────────────────────────────

const EXPORT_FORMATS: ExportFormat[] = ['m4b', 'mp3_single', 'mp3_per_chapter'];

const FORMAT_EXT: Record<ExportFormat, string> = {
  m4b: '.m4b',
  mp3_single: '.mp3',
  mp3_per_chapter: '.zip',
};

type ExportPhase = 'idle' | 'running' | 'complete' | 'error';

function exportFormat(filename?: string): ExportFormat | undefined {
  if (filename?.toLowerCase().endsWith('.zip')) return 'mp3_per_chapter';
  if (filename?.toLowerCase().endsWith('.mp3')) return 'mp3_single';
  if (filename?.toLowerCase().endsWith('.m4b')) return 'm4b';
  return undefined;
}

// ─── Component ────────────────────────────────────────────────────────────────

export function AudiobookExport() {
  const { t } = useTranslation();
  const selectedBookId = useBooksStore((s) => s.selectedBookId);
  const setView = useBooksStore((s) => s.setView);

  const { data: book } = useBook(selectedBookId);

  // ── Form state ───────────────────────────────────────────────────────────────
  const [format, setFormat] = useState<ExportFormat>('m4b');
  const [bitrate, setBitrate] = useState<ExportBitrate>('128k');
  const [title, setTitle] = useState(book?.title ?? '');
  const [author, setAuthor] = useState(book?.author ?? '');
  useEffect(() => {
    setTitle(book?.title ?? '');
    setAuthor(book?.author ?? '');
  }, [book?.id, book?.title, book?.author]);

  // ── Export state ─────────────────────────────────────────────────────────────
  const [phase, setPhase] = useState<ExportPhase>('idle');
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [completedExport, setCompletedExport] = useState<{ format: ExportFormat; filename?: string } | null>(null);
  const restoredBookId = useRef<string | null>(null);
  const previouslyExporting = useRef(false);

  useEffect(() => {
    if (!book) return;
    const initialLoad = restoredBookId.current !== book.id;
    restoredBookId.current = book.id;
    const finishedExport = previouslyExporting.current && book.status !== 'exporting';
    previouslyExporting.current = book.status === 'exporting';
    if (book.status === 'exporting') {
      setPhase('running');
      setStatusMessage(t('books.export.exporting'));
    } else if (book.export_filename) {
      const savedFormat = book.export_format ?? exportFormat(book.export_filename) ?? 'm4b';
      setCompletedExport({ format: savedFormat, filename: book.export_filename });
      if (initialLoad || finishedExport) setFormat(savedFormat);
      setProgress(100);
      setPhase('complete');
      setStatusMessage(t('books.export.complete'));
    } else if (initialLoad) {
      setCompletedExport(null);
      setPhase('idle');
      setProgress(0);
      setErrorMessage('');
    }
    // Restore on server state changes, preserving edits to the next export form.
  }, [book?.id, book?.status, book?.export_filename, book?.export_format, t]);

  // ── Hooks ─────────────────────────────────────────────────────────────────────
  const startExport = useStartExport();
  const downloadExport = useDownloadExport();

  // ── SSE handlers ─────────────────────────────────────────────────────────────

  const onExportProgress = useCallback((event: ExportProgressEvent) => {
    setPhase('running');
    setProgress(event.progress);
    if ('message' in event && typeof (event as any).message === 'string') {
      setStatusMessage((event as any).message as string);
    }
  }, []);

  const onExportComplete = useCallback(
    (event: ExportCompleteEvent) => {
      setCompletedExport({ format: exportFormat(event.filename) ?? format, filename: event.filename });
      setProgress(100);
      setPhase('complete');
      setStatusMessage(t('books.export.complete'));
    },
    [t, format],
  );

  const onError = useCallback((event: BookErrorEvent) => {
    if (event.stage === 'connection') return;
    setPhase('error');
    setErrorMessage(event.message);
  }, []);

  useBookProgress(selectedBookId ?? '', {
    onExportProgress,
    onExportComplete,
    onError,
  });

  // ── Handlers ─────────────────────────────────────────────────────────────────

  async function handleStartExport() {
    if (!selectedBookId) return;
    setPhase('running');
    setCompletedExport(null);
    setProgress(0);
    setStatusMessage(t('books.export.starting'));
    setErrorMessage('');

    try {
      await startExport.mutateAsync({
        bookId: selectedBookId,
        data: {
          format,
          bitrate,
          title: title || undefined,
          author: author || undefined,
        },
      });
    } catch (err: unknown) {
      setPhase('error');
      setErrorMessage(err instanceof Error ? err.message : t('books.export.failed'));
    }
  }

  async function handleDownload() {
    if (!selectedBookId || phase !== 'complete') return;
    try {
      await downloadExport.mutateAsync({
        bookId: selectedBookId,
        bookTitle: title || book?.title || 'audiobook',
        format: completedExport?.format ?? format,
        filename: completedExport?.filename,
      });
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : t('books.export.downloadFailed'));
    }
  }

  const isRunning = phase === 'running' || startExport.isPending;
  const isComplete = phase === 'complete';

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">{t('books.export.title')}</h1>
          {book?.title && (
            <p className="text-sm text-muted-foreground mt-0.5">{book.title}</p>
          )}
        </div>
        <Button variant="ghost" onClick={() => setView('overview')}>
          {t('books.export.back')}
        </Button>
      </div>

      <div className="flex flex-col md:flex-row gap-4 items-start">
        {/* Left column: format + quality */}
        <div className="flex flex-col gap-4 flex-1">
          {/* Format selection */}
          <Card>
            <CardContent className="p-4">
              <h2 className="text-lg font-semibold mb-3">{t('books.export.format')}</h2>
              <div
                data-testid="export-format"
                className="flex flex-col gap-2"
                role="radiogroup"
                aria-label={t('books.export.format')}
              >
                {EXPORT_FORMATS.map((fmt) => (
                  <label
                    key={fmt}
                    className={cn(
                      'flex items-center gap-3 p-3 rounded-md border cursor-pointer transition-colors',
                      format === fmt
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:bg-accent/40',
                    )}
                  >
                    <input
                      type="radio"
                      name="export-format"
                      value={fmt}
                      checked={format === fmt}
                      disabled={isRunning}
                      onChange={() => setFormat(fmt)}
                      className="sr-only"
                    />
                    <span
                      className={cn(
                        'w-4 h-4 rounded-full border-2 flex items-center justify-center shrink-0',
                        format === fmt ? 'border-primary' : 'border-muted-foreground',
                      )}
                    >
                      {format === fmt && (
                        <span className="w-2 h-2 rounded-full bg-primary" />
                      )}
                    </span>
                    <div>
                      <div className="font-medium text-sm">{t(`books.export.${fmt}`)}</div>
                      <div className="text-xs text-muted-foreground">{FORMAT_EXT[fmt]}</div>
                    </div>
                  </label>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Quality */}
          <Card>
            <CardContent className="p-4">
              <h2 className="text-lg font-semibold mb-3">{t('books.export.quality')}</h2>
              <div className="flex flex-col gap-2">
                <Label htmlFor="bitrate-select" className="text-sm">
                  {t('books.export.bitrate')}
                </Label>
                <Select
                  value={bitrate}
                  disabled={isRunning}
                  onValueChange={(val) => setBitrate(val as ExportBitrate)}
                >
                  <SelectTrigger id="bitrate-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="64k">64 kbps</SelectItem>
                    <SelectItem value="128k">128 kbps</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right column: metadata + action */}
        <div className="flex flex-col gap-4 flex-1">
          {/* Metadata */}
          <Card>
            <CardContent className="p-4">
              <h2 className="text-lg font-semibold mb-3">{t('books.export.metadata')}</h2>
              <div
                data-testid="export-metadata"
                className="flex flex-col gap-3"
              >
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="export-title" className="text-sm">
                    {t('books.export.bookTitle')}
                  </Label>
                  <Input
                    id="export-title"
                    disabled={isRunning}
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder={book?.title ?? t('books.export.bookTitlePlaceholder')}
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="export-author" className="text-sm">
                    {t('books.export.author')}
                  </Label>
                  <Input
                    id="export-author"
                    disabled={isRunning}
                    value={author}
                    onChange={(e) => setAuthor(e.target.value)}
                    placeholder={book?.author ?? t('books.export.authorPlaceholder')}
                  />
                </div>

              </div>
            </CardContent>
          </Card>

          {/* Action + status */}
          <Card>
            <CardContent className="p-4">
              <div
                data-testid="export-action"
                className="flex flex-col gap-3"
              >
                {/* Status area */}
                <div data-testid="export-status" className="min-h-[56px]">
                  {phase === 'idle' && (
                    <p className="text-sm text-muted-foreground">
                      {t('books.export.ready')}
                    </p>
                  )}

                  {phase === 'running' && (
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center justify-between">
                        <p className="text-sm text-muted-foreground">
                          {statusMessage || t('books.export.exporting')}
                        </p>
                        <Badge variant="secondary">{Math.round(progress)}%</Badge>
                      </div>
                      <Progress value={progress} className="h-2" />
                    </div>
                  )}

                  {phase === 'complete' && (
                    <div className="flex items-center gap-2">
                      <Badge variant="default" className="border-green-700 text-green-400">
                        {t('books.export.done')}
                      </Badge>
                      <p className="text-sm text-muted-foreground">{statusMessage}</p>
                    </div>
                  )}

                  {phase === 'error' && (
                    <p className="text-sm text-destructive">{errorMessage}</p>
                  )}
                </div>

                {phase !== 'error' && errorMessage && <p className="text-sm text-destructive" role="alert">{errorMessage}</p>}
                {/* Buttons */}
                <div className="flex gap-2">
                  <Button
                    data-testid="start-export-btn"
                    onClick={handleStartExport}
                    disabled={isRunning}
                    className="flex-1"
                  >
                    {t(isRunning ? 'books.export.exporting' : 'books.export.start')}
                  </Button>

                  <Button
                    data-testid="download-btn"
                    variant="outline"
                    disabled={!isComplete || downloadExport.isPending}
                    onClick={handleDownload}
                    className="flex-1"
                  >
                    {downloadExport.isPending
                      ? t('books.export.downloading')
                      : t('books.export.download', { ext: FORMAT_EXT[completedExport?.format ?? format] })}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
