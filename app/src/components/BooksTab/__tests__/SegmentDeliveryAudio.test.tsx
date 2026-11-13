import '@/i18n';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { SegmentDeliveryControl } from '../SegmentDeliveryControl';

vi.mock('@/lib/hooks/useBooks', () => ({
  useUpdateSegment: () => ({ mutate: vi.fn() }),
  usePreviewSegment: () => ({
    isPending: false,
    mutate: (_payload: unknown, options: { onSuccess: (result: { generation_id: string }) => void }) => options.onSuccess({ generation_id: 'preview-1' }),
  }),
}));
vi.mock('@/lib/api/client', () => ({
  apiClient: { getBookAudioUrl: (id: string) => `/api/audio/${id}` },
}));

describe('Segment delivery audition', () => {
  it('provides playable audio controls after generating the selected line preview', () => {
    render(<SegmentDeliveryControl segmentId="line-1" bookId="book-1" chapterId="chapter-1" emotion="neutral" emotionIntensity={0.5} />);
    fireEvent.click(screen.getByTestId('emotion-line-1'));
    fireEvent.click(screen.getByTestId('preview-btn'));
    const audio = screen.getByLabelText('Line preview audio');
    expect(audio).toHaveAttribute('src', '/api/audio/preview-1');
    expect(audio).toHaveAttribute('controls');
  });
});
