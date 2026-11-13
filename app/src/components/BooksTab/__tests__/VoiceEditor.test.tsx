/// <reference types="@testing-library/jest-dom/vitest" />
//
// VoiceEditor — behaviour tests.
//
// Quality bar (SC8): no spy invocation-count assertions in this file.
// View / character-selection transitions are observed against the real
// booksStore (Zustand) and the re-rendered DOM; hook-layer mutations are
// observed by capturing the request payload that crossed the boundary
// into a module-level variable and reading it as data.
//
import '@/i18n';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { VoiceEditor } from '@/components/BooksTab/VoiceEditor';
import { useBooksStore } from '@/stores/booksStore';

// ─── Hook mock state ──────────────────────────────────────────────────────────
//
// The hook layer is the system's boundary to the server. We capture the payloads
// the component sends through that boundary into module-level variables so the
// tests can inspect them as outcomes (the request shape that crossed the
// boundary) rather than asserting on call counts of internal spies. Where the
// mutation has a success/error callback contract, the mock drives it from the
// requested `outcome` to observe the resulting DOM / store change.

interface PreviewPayload {
  charId: string;
  data: {
    design_prompt?: string;
    prompt_id?: string;
    profile_id?: string;
    preset_voice_id?: string;
  };
}

interface UpdatePayload {
  bookId: string;
  charId: string;
  data: {
    design_prompt?: string;
    prompt_id?: string;
    profile_id?: string;
    preset_voice_id?: string;
  };
}

let lastPreviewPayload: PreviewPayload | null = null;
let lastUpdatePayload: UpdatePayload | null = null;
let updateOutcome: 'success' | 'pending' = 'pending';

let previewData: { generation_id: string; audio_path: string } | undefined = {
  generation_id: 'g1',
  audio_path: '/audio/g1',
};

let characterList = [
  {
    id: 'm',
    name: 'Mira',
    color: '#34d399',
    voice_type: 'designed',
    voice_label: 'designed',
    vocal_description: 'warm, resolute',
    dialogue_count: 142,
    confidence: 0.9,
    archetype: 'determined, weary, protective',
    gender: 'female',
    age_range: '30s',
  },
  {
    id: 'j',
    name: 'Jules',
    color: '#6d8bff',
    voice_type: 'preset',
    voice_label: 'preset',
    vocal_description: 'deep, calm',
    dialogue_count: 80,
    confidence: 0.75,
    archetype: undefined,
    gender: undefined,
    age_range: undefined,
  },
];

vi.mock('@/lib/hooks/useBooks', () => ({
  useCharacters: () => ({
    data: characterList,
  }),
  usePreviewCharacter: () => ({
    mutate: (payload: PreviewPayload) => {
      lastPreviewPayload = payload;
    },
    data: previewData,
    isPending: false,
  }),
  useUpdateCharacter: () => ({
    mutate: (
      payload: UpdatePayload,
      opts?: { onSuccess?: () => void; onError?: (err: Error) => void },
    ) => {
      lastUpdatePayload = payload;
      if (updateOutcome === 'success') opts?.onSuccess?.();
    },
    isPending: false,
  }),
  useVoiceOptions: () => ({ data: { library: [], book: [], presets: [] } }),
  useCloneVoiceForCharacter: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useSaveVoiceToLibrary: () => ({ mutate: vi.fn(), isPending: false }),
}));

// useAudioRecording needs PlatformProvider — mock it for unit tests
vi.mock('@/lib/hooks/useAudioRecording', () => ({
  useAudioRecording: () => ({
    isRecording: false,
    duration: 0,
    error: null,
    startRecording: vi.fn(),
    stopRecording: vi.fn(),
    cancelRecording: vi.fn(),
  }),
}));

// ─── API client mock ──────────────────────────────────────────────────────────

vi.mock('@/lib/api/client', () => ({
  apiClient: {
    listVoicePrompts: async () => [{ id: 'prompt-1', name: 'Gentle scholar', category: 'Narration', description: 'A soft mature scholarly voice', prompt: 'A gentle mature voice with precise articulation', sample_text: 'Hello', source: 'library', language: 'zh' }],
    getBookAudioUrl: (id: string) => `http://localhost/audio/${id}`,
  },
}));

vi.mock('@/lib/hooks/useReferenceTranscript', () => ({
  useReferenceTranscript: () => ({
    status: 'idle',
    isTranscribing: false,
    regeneratePrompt: false,
    retranscribe: vi.fn(),
    acceptRegenerate: vi.fn(),
    keepEdits: vi.fn(),
  }),
}));

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('VoiceEditor (Design)', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Reset captured boundary payloads.
    lastPreviewPayload = null;
    lastUpdatePayload = null;
    updateOutcome = 'pending';

    // Use the real booksStore so view/selection changes become observable
    // state, not spy calls. We seed it to mirror an open VoiceEditor session
    // (a book is selected and Mira is the focused character).
    useBooksStore.getState().reset();
    useBooksStore.getState().setSelectedBookId('b1');
    useBooksStore.getState().setSelectedCharacterId('m');
    useBooksStore.getState().setView('voice-editor');

    previewData = { generation_id: 'g1', audio_path: '/audio/g1' };
    characterList = [
      {
        id: 'm',
        name: 'Mira',
        color: '#34d399',
        voice_type: 'designed',
        voice_label: 'designed',
        vocal_description: 'warm, resolute',
        dialogue_count: 142,
        confidence: 0.9,
        archetype: 'determined, weary, protective',
        gender: 'female',
        age_range: '30s',
      },
      {
        id: 'j',
        name: 'Jules',
        color: '#6d8bff',
        voice_type: 'preset',
        voice_label: 'preset',
        vocal_description: 'deep, calm',
        dialogue_count: 80,
        confidence: 0.75,
        archetype: undefined,
        gender: undefined,
        age_range: undefined,
      },
    ];
  });


  it('opens existing voice choices by default', () => {
    render(<VoiceEditor />);
    expect(screen.getByRole('tab', { name: 'Library' })).toHaveAttribute('data-state', 'active');
    expect(screen.queryByTestId('design-prompt')).not.toBeInTheDocument();
  });

  it('selects a prompt from the library and sends the stable prompt id', async () => {
    const user = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    await user.click(await screen.findByRole('button', { name: /Gentle scholar/ }));
    expect(screen.getByTestId('design-prompt')).toHaveValue('A gentle mature voice with precise articulation');
    await user.click(screen.getByTestId('assign-voice-btn'));
    expect(lastUpdatePayload?.data).toEqual({ prompt_id: 'prompt-1' });
  });

  it('sends edited prompt text instead of the original library id', async () => {
    const user = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    await user.click(await screen.findByRole('button', { name: /Gentle scholar/ }));
    await user.type(screen.getByTestId('design-prompt'), ', slightly older');
    await user.click(screen.getByTestId('preview-voice-btn'));
    expect(lastPreviewPayload?.data).toEqual({ design_prompt: 'A gentle mature voice with precise articulation, slightly older' });
  });

  it('shows character context and an assigned-voice preview control, no generate/export', () => {
    render(<VoiceEditor initialTab="design" />);
    expect(screen.getByTestId('character-context')).toHaveTextContent('Mira');
    expect(screen.getByTestId('preview-player')).toBeInTheDocument();
    expect(screen.getByTestId('assign-voice-btn')).toBeInTheDocument();
    expect(screen.queryByTestId('generate-all-btn')).not.toBeInTheDocument();
    expect(screen.queryByTestId('export-btn')).not.toBeInTheDocument();
  });

  it('clicking preview-voice-btn sends a preview request for the current character', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    await u.click(screen.getByTestId('preview-voice-btn'));
    expect(lastPreviewPayload).not.toBeNull();
    expect(lastPreviewPayload?.charId).toBe('m');
  });

  it('renders the 3-tab scaffold: Library, Clone, Design', () => {
    render(<VoiceEditor initialTab="design" />);
    expect(screen.getByRole('tab', { name: /library/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /clone/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /design/i })).toBeInTheDocument();
  });

  it('renders design-prompt textarea in the Design tab', () => {
    render(<VoiceEditor initialTab="design" />);
    expect(screen.getByTestId('design-prompt')).toBeInTheDocument();
  });

  it('renders save-to-library-btn', () => {
    render(<VoiceEditor initialTab="design" />);
    expect(screen.getByTestId('save-to-library-btn')).toBeInTheDocument();
  });

  it('renders current-voice badge with character voice type', () => {
    render(<VoiceEditor initialTab="design" />);
    expect(screen.getByTestId('current-voice')).toBeInTheDocument();
    expect(screen.getByTestId('current-voice')).toHaveTextContent('designed');
  });

  it('renders back-to-overview and character-switcher', () => {
    render(<VoiceEditor initialTab="design" />);
    expect(screen.getByTestId('back-to-overview')).toBeInTheDocument();
    expect(screen.getByTestId('character-switcher')).toBeInTheDocument();
  });

  it('shows character name in switcher', () => {
    render(<VoiceEditor initialTab="design" />);
    expect(screen.getByTestId('character-switcher')).toHaveTextContent('Mira');
  });

  it('back-to-overview button navigates the books view to overview', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    await u.click(screen.getByTestId('back-to-overview'));
    expect(useBooksStore.getState().view).toBe('overview');
  });

  it('assign-voice-btn submits the design prompt for the current book and character', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    await u.click(screen.getByTestId('assign-voice-btn'));
    expect(lastUpdatePayload).not.toBeNull();
    expect(lastUpdatePayload?.bookId).toBe('b1');
    expect(lastUpdatePayload?.charId).toBe('m');
    expect(lastUpdatePayload?.data.design_prompt).toBe('warm, resolute');
  });

  it('design-prompt textarea is pre-filled with vocal_description', () => {
    render(<VoiceEditor initialTab="design" />);
    const textarea = screen.getByTestId('design-prompt') as HTMLTextAreaElement;
    expect(textarea.value).toBe('warm, resolute');
  });

  it('typing in design-prompt textarea updates value', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    const textarea = screen.getByTestId('design-prompt') as HTMLTextAreaElement;
    await u.clear(textarea);
    await u.type(textarea, 'bold, loud');
    expect(textarea.value).toBe('bold, loud');
  });

  it('character-switcher next button advances to the next character and updates the UI', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    const switcher = screen.getByTestId('character-switcher');
    // The ▶ button is the second button in the switcher
    const btns = switcher.querySelectorAll('button');
    await u.click(btns[1]); // ▶ next
    expect(useBooksStore.getState().selectedCharacterId).toBe('j');
    // The UI re-renders with the new character — observable outcome.
    expect(screen.getByTestId('character-switcher')).toHaveTextContent('Jules');
    expect(screen.getByTestId('character-context')).toHaveTextContent('Jules');
  });

  it('character-switcher prev button wraps from first to last character', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    const switcher = screen.getByTestId('character-switcher');
    const btns = switcher.querySelectorAll('button');
    await u.click(btns[0]); // ◀ prev — wraps from index 0 to last
    expect(useBooksStore.getState().selectedCharacterId).toBe('j');
    expect(screen.getByTestId('character-switcher')).toHaveTextContent('Jules');
  });

  it('shows "1 of 2" position in the switcher', () => {
    render(<VoiceEditor initialTab="design" />);
    expect(screen.getByTestId('character-switcher')).toHaveTextContent('1 of 2');
  });

  it('renders preview-player row with audio ready state', () => {
    render(<VoiceEditor initialTab="design" />);
    const player = screen.getByTestId('preview-player');
    // previewData is set so the player should show cached/ready state
    expect(player).toBeInTheDocument();
  });

  it('renders preview-player with empty state when no preview data', () => {
    previewData = undefined;
    render(<VoiceEditor initialTab="design" />);
    const player = screen.getByTestId('preview-player');
    expect(player).toBeInTheDocument();
    expect(player).toHaveTextContent(/not generated/i);
  });

  it('shows character archetype as traits when present', () => {
    render(<VoiceEditor initialTab="design" />);
    const ctx = screen.getByTestId('character-context');
    expect(ctx).toHaveTextContent('determined, weary, protective');
  });

  it('shows gender and age_range badges when present', () => {
    render(<VoiceEditor initialTab="design" />);
    const ctx = screen.getByTestId('character-context');
    expect(ctx).toHaveTextContent('Female · 30s');
  });

  it('shows "no character selected" message when character list is empty', () => {
    characterList = [];
    useBooksStore.getState().setSelectedCharacterId(null);
    render(<VoiceEditor initialTab="design" />);
    expect(screen.getByText(/no character selected/i)).toBeInTheDocument();
  });

  it('preview-player play button is disabled when no audio src available', () => {
    previewData = undefined;
    render(<VoiceEditor initialTab="design" />);
    const player = screen.getByTestId('preview-player');
    // The play button should be disabled when no audio src
    const playBtn = player.querySelector('button');
    expect(playBtn).toBeDisabled();
  });

  it('Library tab body shows the voice-panel-library section', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    await u.click(screen.getByRole('tab', { name: /library/i }));
    expect(screen.getByTestId('voice-panel-library')).toBeInTheDocument();
  });

  it('Clone tab body shows the clone panel (C12)', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    await u.click(screen.getByRole('tab', { name: /clone/i }));
    expect(screen.getByTestId('voice-panel-clone')).toBeInTheDocument();
    expect(screen.getByTestId('clone-dropzone')).toBeInTheDocument();
  });

  it('preview request carries the edited design_prompt as its payload', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    // clear and type in new prompt
    const textarea = screen.getByTestId('design-prompt') as HTMLTextAreaElement;
    await u.clear(textarea);
    await u.type(textarea, 'gruff old man');
    await u.click(screen.getByTestId('preview-voice-btn'));
    expect(lastPreviewPayload).not.toBeNull();
    expect(lastPreviewPayload?.charId).toBe('m');
    expect(lastPreviewPayload?.data.design_prompt).toBe('gruff old man');
  });

  it('renders "low" confidence badge when character confidence is 0.3', () => {
    characterList = [
      {
        id: 'm',
        name: 'Mira',
        color: '#34d399',
        voice_type: 'designed',
        voice_label: 'designed',
        vocal_description: 'warm, resolute',
        dialogue_count: 142,
        confidence: 0.3,
        archetype: 'determined, weary, protective',
        gender: 'female',
        age_range: '30s',
      },
    ];
    useBooksStore.getState().setSelectedCharacterId('m');
    render(<VoiceEditor initialTab="design" />);
    const ctx = screen.getByTestId('character-context');
    expect(ctx).toHaveTextContent(/low/i);
    expect(ctx).toHaveTextContent(/confidence/i);
  });

  it('renders "medium" confidence badge when character confidence is 0.6', () => {
    characterList = [
      {
        id: 'm',
        name: 'Mira',
        color: '#34d399',
        voice_type: 'designed',
        voice_label: 'designed',
        vocal_description: 'warm, resolute',
        dialogue_count: 142,
        confidence: 0.6,
        archetype: 'determined, weary, protective',
        gender: 'female',
        age_range: '30s',
      },
    ];
    useBooksStore.getState().setSelectedCharacterId('m');
    render(<VoiceEditor initialTab="design" />);
    const ctx = screen.getByTestId('character-context');
    expect(ctx).toHaveTextContent(/medium/i);
    expect(ctx).toHaveTextContent(/confidence/i);
  });

  it('assign-voice-btn navigates back to overview after a successful save', async () => {
    const u = userEvent.setup();
    updateOutcome = 'success';
    render(<VoiceEditor initialTab="design" />);
    await u.click(screen.getByTestId('assign-voice-btn'));
    expect(useBooksStore.getState().view).toBe('overview');
  });

  it('assign-voice-btn keeps the view on voice-editor while the save is pending', async () => {
    const u = userEvent.setup();
    updateOutcome = 'pending';
    render(<VoiceEditor initialTab="design" />);
    await u.click(screen.getByTestId('assign-voice-btn'));
    // No onSuccess was invoked → no navigation should have happened yet.
    expect(useBooksStore.getState().view).toBe('voice-editor');
    // But the request did cross the boundary with the right payload.
    expect(lastUpdatePayload?.charId).toBe('m');
  });

  it('character-switcher cycles forward through both characters and back to the first', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    const nextBtn = screen.getByTestId('character-switcher').querySelectorAll('button')[1];
    await u.click(nextBtn);
    expect(useBooksStore.getState().selectedCharacterId).toBe('j');
    await u.click(screen.getByTestId('character-switcher').querySelectorAll('button')[1]);
    expect(useBooksStore.getState().selectedCharacterId).toBe('m');
    expect(screen.getByTestId('character-context')).toHaveTextContent('Mira');
  });

  it('switching characters refreshes the design prompt to the new character description', async () => {
    const u = userEvent.setup();
    render(<VoiceEditor initialTab="design" />);
    const textareaBefore = screen.getByTestId('design-prompt') as HTMLTextAreaElement;
    expect(textareaBefore.value).toBe('warm, resolute');
    const nextBtn = screen.getByTestId('character-switcher').querySelectorAll('button')[1];
    await u.click(nextBtn);
    const textareaAfter = screen.getByTestId('design-prompt') as HTMLTextAreaElement;
    expect(textareaAfter.value).toBe('deep, calm');
  });
});
