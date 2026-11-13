/**
 * SegmentDeliveryControl — emotion pill + delivery/tone popover (D4).
 *
 * Clicking the emotion pill opens a popover with:
 *   - A row of emotion preset buttons (controlled vocabulary)
 *   - An intensity slider (0..1)
 *   - A free-text delivery/instruction input
 *   - A per-segment preview button
 *
 * Changes are persisted via useUpdateSegment (B7 PATCH).
 * Preview synthesizes the segment via usePreviewSegment (non-destructive regenerate).
 *
 * data-testids:
 *   emotion-{segId}   — the pill trigger
 *   delivery-popover  — the open popover content
 *   preview-btn       — the preview play button inside the popover
 */

import { useTranslation } from 'react-i18next';
import { useState } from 'react';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils/cn';
import { apiClient } from '@/lib/api/client';
import { toast } from '@/components/ui/use-toast';
import { useUpdateSegment, usePreviewSegment } from '@/lib/hooks/useBooks';

// ─── Shared emotion vocabulary (matches backend controlled vocabulary) ─────────

export const EMOTION_PRESETS = [
  'neutral',
  'happy',
  'sad',
  'angry',
  'tense',
  'afraid',
  'whisper',
  'urgent',
] as const;

export type EmotionPreset = (typeof EMOTION_PRESETS)[number];

// ─── Props ─────────────────────────────────────────────────────────────────────

interface SegmentDeliveryControlProps {
  segmentId: string;
  bookId: string;
  chapterId: string;
  /** Current emotion from the segment (may be any string including non-preset values) */
  emotion: string;
  emotionIntensity: number;
  delivery?: string;
}

// ─── Component ─────────────────────────────────────────────────────────────────

export function SegmentDeliveryControl({
  segmentId,
  bookId,
  chapterId,
  emotion: initialEmotion,
  emotionIntensity: initialIntensity,
  delivery: initialDelivery = '',
}: SegmentDeliveryControlProps) {
  const { t } = useTranslation();
  const [previewAudioSrc, setPreviewAudioSrc] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [localEmotion, setLocalEmotion] = useState(initialEmotion);
  const [localIntensity, setLocalIntensity] = useState(initialIntensity);
  const [localDelivery, setLocalDelivery] = useState(initialDelivery);

  const { mutate: updateMutate } = useUpdateSegment();
  const { mutate: previewMutate, isPending: isPreviewing } = usePreviewSegment();

  function handleEmotionSelect(emotion: string) {
    setLocalEmotion(emotion);
    updateMutate(
      {
        segmentId,
        bookId,
        chapterId,
        data: {
          emotion,
          emotion_intensity: localIntensity,
          delivery: localDelivery || undefined,
        },
      },
      {},
    );
  }

  function handleIntensityChange(values: number[]) {
    const intensity = values[0];
    setLocalIntensity(intensity);
    updateMutate(
      {
        segmentId,
        bookId,
        chapterId,
        data: {
          emotion: localEmotion,
          emotion_intensity: intensity,
          delivery: localDelivery || undefined,
        },
      },
      {},
    );
  }

  function handleDeliveryChange(value: string) {
    setLocalDelivery(value);
  }

  function handleDeliveryBlur() {
    updateMutate(
      {
        segmentId,
        bookId,
        chapterId,
        data: {
          emotion: localEmotion,
          emotion_intensity: localIntensity,
          delivery: localDelivery || undefined,
        },
      },
      {},
    );
  }

  function handlePreview() {
    // Send only the emotion override; emotion_intensity + delivery are already
    // persisted on the segment (PATCHed on change above), and the backend
    // composes the instruct via the same compose_instruct() generation uses, so
    // the auditioned clip matches what will be rendered. (Do NOT send the raw
    // delivery as `instruct` — that is a full verbatim override that would
    // bypass intensity/emotion composition.)
    previewMutate({
      segmentId,
      data: {
        emotion: localEmotion,
      },
    }, {
      onSuccess: (result) => setPreviewAudioSrc(apiClient.getBookAudioUrl(result.generation_id)),
      onError: (error) => toast({ title: t('books.chapterEditor.previewFailed'), description: error.message, variant: 'destructive' }),
    });
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <span
          data-testid={`emotion-${segmentId}`}
          className="ml-1 cursor-pointer rounded px-1.5 py-0.5 text-xs text-muted-foreground hover:bg-muted"
          title={t('books.chapterEditor.deliveryTitle')}
        >
          {t(`books.chapterEditor.emotions.${localEmotion}`, { defaultValue: localEmotion })} ▾
        </span>
      </PopoverTrigger>
      <PopoverContent
        data-testid="delivery-popover"
        className="w-80 p-3"
        align="start"
        side="bottom"
      >
        {/* Header */}
        <p className="mb-2 text-xs font-medium text-muted-foreground">
          {t('books.chapterEditor.deliveryTitle')}
        </p>

        {/* Emotion preset buttons */}
        <div className="mb-3 flex flex-wrap gap-1">
          {EMOTION_PRESETS.map((preset) => (
            <button
              key={t(`books.chapterEditor.emotions.${preset}`)}
              onClick={() => handleEmotionSelect(preset)}
              className={cn(
                'rounded px-2 py-0.5 text-xs capitalize transition-colors',
                localEmotion === preset
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80',
              )}
            >
              {t(`books.chapterEditor.emotions.${preset}`)}
            </button>
          ))}
        </div>

        {/* Intensity slider */}
        <div className="mb-3">
          <label className="mb-1 block text-xs text-muted-foreground">
            {t('books.chapterEditor.intensity', { value: Math.round(localIntensity * 100) })}
          </label>
          <Slider
            min={0}
            max={1}
            step={0.05}
            value={[localIntensity]}
            onValueChange={handleIntensityChange}
            className="w-full"
          />
        </div>

        {/* Free-text delivery instruction */}
        <div className="mb-3">
          <label className="mb-1 block text-xs text-muted-foreground">
            {t('books.chapterEditor.instruction')}
          </label>
          <Input
            value={localDelivery}
            onChange={(e) => handleDeliveryChange(e.target.value)}
            onBlur={handleDeliveryBlur}
            placeholder={t('books.chapterEditor.instructionPlaceholder')}
            className="text-xs"
          />
        </div>

        {/* Preview button */}
        <Button
          data-testid="preview-btn"
          variant="secondary"
          size="sm"
          onClick={handlePreview}
          disabled={isPreviewing}
          className="w-full"
        >
          {t(isPreviewing ? 'books.chapterEditor.previewing' : 'books.chapterEditor.previewLine')}
        </Button>
        {previewAudioSrc && <audio controls autoPlay src={previewAudioSrc} aria-label={t('books.chapterEditor.previewAudio')} className="mt-2 w-full" />}
      </PopoverContent>
    </Popover>
  );
}
