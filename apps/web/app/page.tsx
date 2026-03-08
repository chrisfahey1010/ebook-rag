"use client";

import { FormEvent, useEffect, useState, useTransition } from "react";

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type DocumentSummary = {
  id: string;
  filename: string;
  original_filename: string;
  title: string | null;
  sha256: string;
  page_count: number | null;
  chunk_count: number;
  status: string;
  chunking_config: ChunkingConfig | null;
  created_at: string;
  updated_at: string | null;
};

type ChunkingConfig = {
  target_words: number;
  min_words: number;
  overlap_words: number;
  max_heading_words: number;
};

type UploadResponse = {
  document: DocumentSummary;
  ingestion_job_id: string;
  ingestion_status: string;
  ingestion_error: string | null;
};

type IngestionJobSummary = {
  id: string;
  document_id: string;
  status: string;
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
};

type IngestionStatusResponse = {
  document: DocumentSummary;
  ingestion_job: IngestionJobSummary | null;
};

type PageCharRange = {
  page_number: number;
  start_char: number;
  end_char: number;
};

type ParagraphProvenance = {
  global_index: number;
  page_number: number;
  page_paragraph_index: number;
  char_start: number;
  char_end: number;
  is_heading: boolean;
};

type ChunkProvenance = {
  source_page_numbers?: number[];
  paragraph_count?: number;
  paragraph_range?: {
    start: number;
    end: number;
  };
  page_paragraph_range?: {
    start_page: number;
    start_index: number;
    end_page: number;
    end_index: number;
  };
  char_range?: {
    start_page: number;
    start_char: number;
    end_page: number;
    end_char: number;
  };
  page_char_ranges?: PageCharRange[];
  paragraphs?: ParagraphProvenance[];
};

type Citation = {
  chunk_id: string;
  document_id: string;
  document_title: string | null;
  document_filename: string;
  chunk_index: number;
  page_start: number;
  page_end: number;
  text: string;
  provenance: ChunkProvenance | null;
  dense_score: number;
  lexical_score: number;
  hybrid_score: number;
  rerank_score: number;
  score: number;
};

type QAResponse = {
  normalized_question: string;
  answer: string;
  supported: boolean;
  answer_mode: string;
  confidence: number;
  support_score: number;
  citations: Citation[];
  retrieved_chunk_count: number;
  trace: QATrace | null;
};

type QATimingBreakdown = {
  normalization_ms: number;
  retrieval_ms: number;
  context_assembly_ms: number;
  answer_generation_ms: number;
  total_ms: number;
};

type QATrace = {
  answer_provider: string;
  answer_mode: string;
  question_router: QAQuestionRouter;
  runtime: QARuntimeMetadata;
  retrieved_chunks: RetrievalMatch[];
  selected_contexts: RetrievalMatch[];
  cited_contexts: RetrievalMatch[];
  prompt_snapshot: string;
  timings: QATimingBreakdown;
};

type QAQuestionRouter = {
  answer_mode: string;
  reason: string;
  facet_count: number;
  context_count: number;
  should_use_generative: boolean;
};

type QARuntimeMetadata = {
  embedding_provider: string;
  embedding_model: string | null;
  reranker_provider: string;
  reranker_model: string | null;
  answer_provider: string;
  answer_model: string | null;
};

type RetrievalMatch = {
  chunk_id: string;
  document_id: string;
  document_title: string | null;
  document_filename: string;
  chunk_index: number;
  page_start: number;
  page_end: number;
  text: string;
  provenance: ChunkProvenance | null;
  dense_score: number;
  lexical_score: number;
  hybrid_score: number;
  rerank_score: number;
  score: number;
};

type RetrievalResponse = {
  normalized_query: string;
  matches: RetrievalMatch[];
};

function statusTone(status: string): string {
  if (status === "ready") {
    return "bg-emerald-100 text-emerald-800";
  }
  if (status === "failed") {
    return "bg-rose-100 text-rose-800";
  }
  return "bg-amber-100 text-amber-900";
}

function formatPageRange(pageStart: number, pageEnd: number): string {
  if (pageStart === pageEnd) {
    return `p. ${pageStart}`;
  }
  return `pp. ${pageStart}-${pageEnd}`;
}

function formatDocumentLabel(document: DocumentSummary): string {
  return document.title || document.original_filename;
}

function formatChunkingConfig(config: ChunkingConfig | null): string {
  if (!config) {
    return "Not recorded";
  }
  return `${config.target_words}w target, ${config.overlap_words}w overlap`;
}

function formatCharRangeLabel(
  pageNumber: number,
  startChar: number,
  endChar: number,
): string {
  return `p. ${pageNumber} chars ${startChar}-${endChar}`;
}

function formatScore(value: number): string {
  return value.toFixed(3);
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatModeLabel(mode: string): string {
  return mode.replace(/_/g, " ");
}

function formatRuntimeLabel(provider: string, model: string | null): string {
  return model ? `${provider} · ${model}` : provider;
}

function formatCrossPageSpan(charRange: NonNullable<ChunkProvenance["char_range"]>): string {
  if (charRange.start_page === charRange.end_page) {
    return formatCharRangeLabel(
      charRange.start_page,
      charRange.start_char,
      charRange.end_char,
    );
  }
  return `start p. ${charRange.start_page} char ${charRange.start_char}, end p. ${charRange.end_page} char ${charRange.end_char}`;
}

function splitIntoSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function normalizeForTermMatch(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]+/g, " ")
    .split(/\s+/)
    .filter((term) => term.length > 2);
}

function findBestMatchingAnswerSentence(
  answerText: string,
  citationText: string,
): string {
  const answerSentences = splitIntoSentences(answerText);
  const citationTerms = new Set(normalizeForTermMatch(citationText));
  let bestSentence = answerSentences[0] ?? answerText;
  let bestScore = -1;

  for (const sentence of answerSentences) {
    const sentenceTerms = normalizeForTermMatch(sentence);
    const overlap = sentenceTerms.filter((term) => citationTerms.has(term)).length;
    if (overlap > bestScore) {
      bestSentence = sentence;
      bestScore = overlap;
    }
  }

  return bestSentence;
}

function getCitationInspectorCopy(citation: Citation, answerText: string) {
  const answerSentence = findBestMatchingAnswerSentence(answerText, citation.text);
  const pageCharRanges = citation.provenance?.page_char_ranges ?? [];
  const paragraphRange = citation.provenance?.page_paragraph_range;
  const charRange = citation.provenance?.char_range;

  return {
    answerSentence,
    pageCharRanges,
    paragraphRange,
    charRange,
  };
}

async function parseApiError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail ?? `Request failed with status ${response.status}.`;
  } catch {
    return `Request failed with status ${response.status}.`;
  }
}

async function fetchRetrieval(
  query: string,
  documentId: string,
): Promise<RetrievalResponse> {
  const response = await fetch(`${apiBaseUrl}/api/debug/retrieve`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      document_id: documentId,
      top_k: 5,
    }),
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

  return (await response.json()) as RetrievalResponse;
}

async function streamQaAnswer(
  question: string,
  documentId: string,
  onDelta: (delta: string) => void,
): Promise<QAResponse> {
  const response = await fetch(`${apiBaseUrl}/api/qa/ask-stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      document_id: documentId,
      top_k: 5,
      include_trace: true,
    }),
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

  if (!response.body) {
    throw new Error("Streaming response body was unavailable.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload: QAResponse | null = null;

  function processEventBlock(block: string) {
    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length === 0) {
      return;
    }

    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }

    const data = dataLines.join("\n");
    if (!data) {
      return;
    }

    const payload = JSON.parse(data) as
      | { delta?: string; detail?: string }
      | QAResponse;

    if (eventName === "answer_delta") {
      onDelta((payload as { delta?: string }).delta ?? "");
      return;
    }
    if (eventName === "error") {
      throw new Error(
        (payload as { detail?: string }).detail ?? "Streaming request failed.",
      );
    }
    if (eventName === "complete") {
      finalPayload = payload as QAResponse;
    }
  }

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder
      .decode(value ?? new Uint8Array(), { stream: !done })
      .replace(/\r\n/g, "\n");

    let boundaryIndex = buffer.indexOf("\n\n");
    while (boundaryIndex !== -1) {
      const block = buffer.slice(0, boundaryIndex);
      buffer = buffer.slice(boundaryIndex + 2);
      processEventBlock(block);
      boundaryIndex = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }

  if (buffer.trim()) {
    processEventBlock(buffer);
  }

  if (!finalPayload) {
    throw new Error("Streaming response ended before a completion event.");
  }

  return finalPayload;
}

export default function Home() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string>("");
  const [question, setQuestion] = useState("");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [libraryError, setLibraryError] = useState<string | null>(null);
  const [qaError, setQaError] = useState<string | null>(null);
  const [retrievalError, setRetrievalError] = useState<string | null>(null);
  const [lastUploadMessage, setLastUploadMessage] = useState<string | null>(null);
  const [answer, setAnswer] = useState<QAResponse | null>(null);
  const [retrievalResult, setRetrievalResult] = useState<RetrievalResponse | null>(null);
  const [qaTrace, setQaTrace] = useState<QATrace | null>(null);
  const [selectedCitationChunkId, setSelectedCitationChunkId] = useState<string | null>(
    null,
  );
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatusResponse | null>(null);
  const [ingestionError, setIngestionError] = useState<string | null>(null);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(true);
  const [isRefreshingIngestion, startRefreshIngestionTransition] = useTransition();
  const [isReprocessing, startReprocessTransition] = useTransition();
  const [isUploading, startUploadTransition] = useTransition();
  const [isAnswering, startAnswerTransition] = useTransition();
  const [isInspecting, startInspectTransition] = useTransition();
  const [isDeleting, startDeleteTransition] = useTransition();

  const selectedDocument =
    documents.find((document) => document.id === selectedDocumentId) ?? null;
  const selectedCitation =
    answer?.citations.find((citation) => citation.chunk_id === selectedCitationChunkId) ??
    answer?.citations[0] ??
    null;
  const selectedCitationInspector =
    selectedCitation && answer
      ? getCitationInspectorCopy(selectedCitation, answer.answer)
      : null;

  function syncDocument(nextDocument: DocumentSummary) {
    setDocuments((current) =>
      current.map((document) =>
        document.id === nextDocument.id ? nextDocument : document,
      ),
    );
  }

  useEffect(() => {
    let cancelled = false;

    async function loadDocuments() {
      setIsLoadingDocuments(true);
      setLibraryError(null);

      try {
        const response = await fetch(`${apiBaseUrl}/api/documents`, {
          cache: "no-store",
        });

        if (!response.ok) {
          throw new Error(await parseApiError(response));
        }

        const payload = (await response.json()) as DocumentSummary[];
        if (cancelled) {
          return;
        }

        setDocuments(payload);
        setSelectedDocumentId((current) => {
          if (current && payload.some((document) => document.id === current)) {
            return current;
          }
          return payload[0]?.id ?? "";
        });
      } catch (error) {
        if (cancelled) {
          return;
        }
        setLibraryError(
          error instanceof Error ? error.message : "Failed to load documents.",
        );
      } finally {
        if (!cancelled) {
          setIsLoadingDocuments(false);
        }
      }
    }

    void loadDocuments();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedDocumentId) {
      setIngestionStatus(null);
      setIngestionError(null);
      return;
    }

    let cancelled = false;

    startRefreshIngestionTransition(async () => {
      try {
        const response = await fetch(
          `${apiBaseUrl}/api/ingestion/${selectedDocumentId}/status`,
          {
            cache: "no-store",
          },
        );

        if (!response.ok) {
          throw new Error(await parseApiError(response));
        }

        const payload = (await response.json()) as IngestionStatusResponse;
        if (cancelled) {
          return;
        }

        setIngestionStatus(payload);
        setIngestionError(null);
        syncDocument(payload.document);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setIngestionError(
          error instanceof Error ? error.message : "Failed to load ingestion status.",
        );
      }
    });

    return () => {
      cancelled = true;
    };
  }, [selectedDocumentId]);

  useEffect(() => {
    setSelectedCitationChunkId(answer?.citations[0]?.chunk_id ?? null);
  }, [answer]);

  function handleUploadSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const fileInput = form.elements.namedItem("pdf") as HTMLInputElement | null;
    const file = fileInput?.files?.[0];

    if (!file) {
      setUploadError("Choose a PDF before uploading.");
      return;
    }

    setUploadError(null);
    setLastUploadMessage(null);

    startUploadTransition(async () => {
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch(`${apiBaseUrl}/api/documents/upload`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(await parseApiError(response));
        }

        const payload = (await response.json()) as UploadResponse;
        setDocuments((current) => [payload.document, ...current]);
        setSelectedDocumentId(payload.document.id);
        setIngestionStatus({
          document: payload.document,
          ingestion_job: {
            id: payload.ingestion_job_id,
            document_id: payload.document.id,
            status: payload.ingestion_status,
            error_message: payload.ingestion_error,
            started_at: null,
            finished_at: null,
            created_at: payload.document.created_at,
          },
        });
        setAnswer(null);
        setQaTrace(null);
        setRetrievalResult(null);
        setRetrievalError(null);
        setQuestion("");
        setLastUploadMessage(
          payload.ingestion_status === "completed"
            ? `Indexed ${payload.document.original_filename} with ${payload.document.chunk_count} chunks.`
            : `Upload finished with status ${payload.ingestion_status}.`,
        );
        form.reset();
      } catch (error) {
        setUploadError(
          error instanceof Error ? error.message : "Upload failed.",
        );
      }
    });
  }

  function refreshIngestionStatus() {
    if (!selectedDocumentId) {
      return;
    }

    setIngestionError(null);
    startRefreshIngestionTransition(async () => {
      try {
        const response = await fetch(
          `${apiBaseUrl}/api/ingestion/${selectedDocumentId}/status`,
          {
            cache: "no-store",
          },
        );

        if (!response.ok) {
          throw new Error(await parseApiError(response));
        }

        const payload = (await response.json()) as IngestionStatusResponse;
        setIngestionStatus(payload);
        syncDocument(payload.document);
      } catch (error) {
        setIngestionError(
          error instanceof Error ? error.message : "Failed to refresh ingestion status.",
        );
      }
    });
  }

  function reprocessSelectedDocument() {
    if (!selectedDocumentId || !selectedDocument) {
      return;
    }

    setIngestionError(null);
    setLastUploadMessage(null);

    startReprocessTransition(async () => {
      try {
        const response = await fetch(
          `${apiBaseUrl}/api/ingestion/${selectedDocumentId}/reprocess`,
          {
            method: "POST",
          },
        );

        if (!response.ok) {
          throw new Error(await parseApiError(response));
        }

        const payload = (await response.json()) as IngestionStatusResponse;
        setIngestionStatus(payload);
        syncDocument(payload.document);
        setLastUploadMessage(
          payload.ingestion_job?.status === "completed"
            ? `Reprocessed ${selectedDocument.original_filename}.`
            : `Reprocess finished with status ${payload.ingestion_job?.status ?? payload.document.status}.`,
        );
      } catch (error) {
        setIngestionError(
          error instanceof Error ? error.message : "Reprocess failed.",
        );
      }
    });
  }

  function handleQuestionSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedDocumentId) {
      setQaError("Select a document before asking a question.");
      return;
    }

    if (!question.trim()) {
      setQaError("Enter a question first.");
      return;
    }

    setQaError(null);
    setRetrievalError(null);
    const trimmedQuestion = question.trim();

    startAnswerTransition(async () => {
      try {
        setQaTrace(null);
        setSelectedCitationChunkId(null);
        setAnswer({
          normalized_question: trimmedQuestion,
          answer: "",
          supported: false,
          answer_mode: "unsupported",
          confidence: 0,
          support_score: 0,
          citations: [],
          retrieved_chunk_count: 0,
          trace: null,
        });

        const qaPayload = await streamQaAnswer(
          trimmedQuestion,
          selectedDocumentId,
          (delta) => {
            setAnswer((current) => {
              if (!current) {
                return {
                  normalized_question: trimmedQuestion,
                  answer: delta,
                  supported: false,
                  answer_mode: "unsupported",
                  confidence: 0,
                  support_score: 0,
                  citations: [],
                  retrieved_chunk_count: 0,
                  trace: null,
                };
              }
              return {
                ...current,
                answer: `${current.answer}${delta}`,
              };
            });
          },
        );

        setAnswer(qaPayload);
        setQaTrace(qaPayload.trace);
        setSelectedCitationChunkId(qaPayload.citations[0]?.chunk_id ?? null);
        setRetrievalResult(
          qaPayload.trace
            ? {
                normalized_query: qaPayload.normalized_question,
                matches: qaPayload.trace.retrieved_chunks,
              }
            : null,
        );
      } catch (error) {
        setQaError(error instanceof Error ? error.message : "Question failed.");
      }
    });
  }

  function inspectRetrieval() {
    if (!selectedDocumentId) {
      setRetrievalError("Select a document before inspecting retrieval.");
      return;
    }

    if (!question.trim()) {
      setRetrievalError("Enter a question first.");
      return;
    }

    setQaError(null);
    setRetrievalError(null);
    const trimmedQuestion = question.trim();

    startInspectTransition(async () => {
      try {
        const payload = await fetchRetrieval(trimmedQuestion, selectedDocumentId);
        setRetrievalResult(payload);
      } catch (error) {
        setRetrievalError(
          error instanceof Error ? error.message : "Retrieval inspection failed.",
        );
      }
    });
  }

  function handleDeleteDocument(document: DocumentSummary) {
    if (deletingDocumentId) {
      return;
    }

    const confirmed = window.confirm(
      `Delete ${formatDocumentLabel(document)}? This removes the uploaded PDF and all indexed data.`,
    );
    if (!confirmed) {
      return;
    }

    setLibraryError(null);
    setDeletingDocumentId(document.id);

    startDeleteTransition(async () => {
      try {
        const response = await fetch(
          `${apiBaseUrl}/api/documents/${document.id}`,
          {
            method: "DELETE",
          },
        );

        if (!response.ok) {
          throw new Error(await parseApiError(response));
        }

        setDocuments((current) => {
          const nextDocuments = current.filter(
            (currentDocument) => currentDocument.id !== document.id,
          );
          setSelectedDocumentId((currentSelectedId) => {
            if (currentSelectedId !== document.id) {
              return currentSelectedId;
            }
            return nextDocuments[0]?.id ?? "";
          });
          return nextDocuments;
        });

        if (selectedDocumentId === document.id) {
          setAnswer(null);
          setQaTrace(null);
          setSelectedCitationChunkId(null);
          setRetrievalResult(null);
          setQaError(null);
          setRetrievalError(null);
          setQuestion("");
        }

        setLastUploadMessage(`Deleted ${document.original_filename}.`);
      } catch (error) {
        setLibraryError(
          error instanceof Error ? error.message : "Delete failed.",
        );
      } finally {
        setDeletingDocumentId(null);
      }
    });
  }

  return (
    <main className="min-h-screen px-4 py-6 md:px-8 md:py-8 lg:px-12">
      <section className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="overflow-hidden rounded-[2rem] border border-[var(--border)] bg-[var(--surface)] shadow-[0_28px_90px_rgba(63,42,29,0.08)]">
          <div className="grid gap-6 px-6 py-8 md:px-8 lg:grid-cols-[1.15fr_0.85fr] lg:px-10">
            <div className="flex flex-col gap-4">
              <p className="text-xs uppercase tracking-[0.35em] text-[var(--muted)]">
                Local-first PDF QA
              </p>
              <div>
                <h1 className="text-4xl leading-none tracking-[-0.05em] md:text-6xl">
                  Ask ebooks and manuals like a serious retrieval system.
                </h1>
                <p className="mt-4 max-w-3xl text-base leading-7 text-[var(--muted)] md:text-lg">
                  Upload a PDF, inspect the indexed library, and ask grounded
                  questions with chunk-level citations from the current backend.
                </p>
              </div>
            </div>

            <div className="rounded-[1.75rem] bg-[var(--accent)]/94 p-6 text-[#fff8f1]">
              <p className="text-sm uppercase tracking-[0.22em] text-white/70">
                Environment
              </p>
              <dl className="mt-4 grid gap-4 text-sm">
                <div>
                  <dt className="text-white/60">API</dt>
                  <dd className="mt-1 break-all text-base">{apiBaseUrl}</dd>
                </div>
                <div>
                  <dt className="text-white/60">Library</dt>
                  <dd className="mt-1 text-base">{documents.length} documents</dd>
                </div>
                <div>
                  <dt className="text-white/60">Selected</dt>
                  <dd className="mt-1 text-base">
                    {selectedDocument
                      ? formatDocumentLabel(selectedDocument)
                      : "No document selected"}
                  </dd>
                </div>
              </dl>
            </div>
          </div>
        </header>

        <section className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
          <aside className="flex flex-col gap-6">
            <section className="rounded-[1.6rem] border border-[var(--border)] bg-[var(--surface)] p-5">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted)]">
                    Upload
                  </p>
                  <h2 className="mt-2 text-2xl">Add a PDF</h2>
                </div>
                <div className="rounded-full border border-[var(--border)] px-3 py-1 text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                  Phase 1
                </div>
              </div>

              <form className="mt-5 flex flex-col gap-4" onSubmit={handleUploadSubmit}>
                <label className="rounded-[1.2rem] border border-dashed border-[var(--border)] bg-white/65 p-4">
                  <span className="block text-sm text-[var(--muted)]">
                    Choose a digital PDF with selectable text.
                  </span>
                  <input
                    className="mt-3 block w-full text-sm file:mr-4 file:rounded-full file:border-0 file:bg-[var(--accent)] file:px-4 file:py-2 file:text-sm file:font-medium file:text-white"
                    type="file"
                    name="pdf"
                    accept="application/pdf"
                    disabled={isUploading}
                  />
                </label>

                <button
                  type="submit"
                  disabled={isUploading}
                  className="rounded-full bg-[var(--accent)] px-5 py-3 text-sm font-medium text-white transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isUploading ? "Uploading..." : "Upload and index"}
                </button>
              </form>

              {uploadError ? (
                <p className="mt-4 rounded-[1rem] bg-rose-50 px-4 py-3 text-sm text-rose-800">
                  {uploadError}
                </p>
              ) : null}
              {lastUploadMessage ? (
                <p className="mt-4 rounded-[1rem] bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
                  {lastUploadMessage}
                </p>
              ) : null}
            </section>

            <section className="rounded-[1.6rem] border border-[var(--border)] bg-[var(--surface)] p-5">
              <div className="flex items-end justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted)]">
                    Library
                  </p>
                  <h2 className="mt-2 text-2xl">Indexed documents</h2>
                </div>
                <span className="text-sm text-[var(--muted)]">
                  {isLoadingDocuments ? "Loading..." : `${documents.length} total`}
                </span>
              </div>

              {libraryError ? (
                <p className="mt-4 rounded-[1rem] bg-rose-50 px-4 py-3 text-sm text-rose-800">
                  {libraryError}
                </p>
              ) : null}

              <div className="mt-5 flex max-h-[32rem] flex-col gap-3 overflow-y-auto pr-1">
                {!isLoadingDocuments && documents.length === 0 ? (
                  <div className="rounded-[1.2rem] border border-dashed border-[var(--border)] px-4 py-5 text-sm text-[var(--muted)]">
                    No PDFs uploaded yet.
                  </div>
                ) : null}

                {documents.map((document) => {
                  const selected = document.id === selectedDocumentId;
                  const isDeletingThisDocument =
                    deletingDocumentId === document.id && isDeleting;

                  return (
                    <article
                      key={document.id}
                      className={`rounded-[1.2rem] border px-4 py-4 text-left transition ${
                        selected
                          ? "border-[var(--accent)] bg-white shadow-[0_16px_32px_rgba(63,42,29,0.08)]"
                          : "border-[var(--border)] bg-white/60 hover:bg-white"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedDocumentId(document.id);
                            setAnswer(null);
                            setQaTrace(null);
                            setQaError(null);
                            setRetrievalResult(null);
                            setRetrievalError(null);
                          }}
                          className="min-w-0 flex-1 text-left"
                        >
                          <p className="truncate text-base font-medium">
                            {formatDocumentLabel(document)}
                          </p>
                          <p className="mt-1 truncate text-sm text-[var(--muted)]">
                            {document.original_filename}
                          </p>
                        </button>

                        <div className="flex items-start gap-2">
                          <span
                            className={`rounded-full px-3 py-1 text-xs font-medium uppercase tracking-[0.16em] ${statusTone(document.status)}`}
                          >
                            {document.status}
                          </span>
                          <button
                            type="button"
                            onClick={() => handleDeleteDocument(document)}
                            disabled={isDeletingThisDocument}
                            className="rounded-full border border-rose-200 bg-rose-50 px-3 py-1 text-xs font-medium uppercase tracking-[0.16em] text-rose-700 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-60"
                          >
                            {isDeletingThisDocument ? "Deleting..." : "Delete"}
                          </button>
                        </div>
                      </div>

                      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm text-[var(--muted)]">
                        <div>
                          <dt>Pages</dt>
                          <dd className="mt-1 text-[var(--foreground)]">
                            {document.page_count ?? "-"}
                          </dd>
                        </div>
                        <div>
                          <dt>Chunks</dt>
                          <dd className="mt-1 text-[var(--foreground)]">
                            {document.chunk_count}
                          </dd>
                        </div>
                      </dl>
                    </article>
                  );
                })}
              </div>
            </section>
          </aside>

          <section className="grid gap-6">
            <section className="rounded-[1.6rem] border border-[var(--border)] bg-[var(--surface)] p-6">
              <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted)]">
                    Workspace
                  </p>
                  <h2 className="mt-2 text-3xl">
                    {selectedDocument
                      ? formatDocumentLabel(selectedDocument)
                      : "Select a document"}
                  </h2>
                  <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--muted)]">
                    Ask a focused question against one indexed PDF. Answers come
                    from `/api/qa/ask-stream` and expose the supporting chunk text.
                  </p>
                </div>
                {selectedDocument ? (
                  <div className="rounded-[1rem] border border-[var(--border)] bg-white/70 px-4 py-3 text-sm text-[var(--muted)]">
                    {selectedDocument.page_count ?? "-"} pages,{" "}
                    {selectedDocument.chunk_count} chunks,{" "}
                    {formatChunkingConfig(selectedDocument.chunking_config)}
                  </div>
                ) : null}
              </div>

              {selectedDocument ? (
                <div className="mt-6 rounded-[1.25rem] border border-[var(--border)] bg-white/65 p-4">
                  <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                        Ingestion
                      </p>
                      <p className="mt-2 text-sm text-[var(--foreground)]">
                        Status:{" "}
                        <span className="font-medium">
                          {ingestionStatus?.ingestion_job?.status ?? selectedDocument.status}
                        </span>
                        {ingestionStatus?.ingestion_job?.error_message
                          ? ` · ${ingestionStatus.ingestion_job.error_message}`
                          : ""}
                      </p>
                      <p className="mt-1 text-sm text-[var(--muted)]">
                        Chunking: {formatChunkingConfig(selectedDocument.chunking_config)}
                      </p>
                    </div>
                    <div className="flex flex-col gap-3 sm:flex-row">
                      <button
                        type="button"
                        onClick={refreshIngestionStatus}
                        disabled={isRefreshingIngestion}
                        className="rounded-full border border-[var(--border)] bg-white/80 px-4 py-2 text-sm font-medium text-[var(--foreground)] transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {isRefreshingIngestion ? "Refreshing..." : "Refresh status"}
                      </button>
                      <button
                        type="button"
                        onClick={reprocessSelectedDocument}
                        disabled={isReprocessing}
                        className="rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {isReprocessing ? "Reprocessing..." : "Reprocess document"}
                      </button>
                    </div>
                  </div>
                  {ingestionError ? (
                    <p className="mt-4 rounded-[1rem] bg-rose-50 px-4 py-3 text-sm text-rose-800">
                      {ingestionError}
                    </p>
                  ) : null}
                </div>
              ) : null}

              <form className="mt-6 flex flex-col gap-4" onSubmit={handleQuestionSubmit}>
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  rows={4}
                  disabled={
                    !selectedDocument ||
                    selectedDocument.status !== "ready" ||
                    isAnswering ||
                    isInspecting
                  }
                  placeholder="What does this document say about battery care, architecture decisions, or a specific topic?"
                  className="w-full rounded-[1.25rem] border border-[var(--border)] bg-white/80 px-4 py-4 text-base outline-none transition placeholder:text-[var(--muted)] focus:border-[var(--accent)]"
                />
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <p className="text-sm text-[var(--muted)]">
                    {selectedDocument?.status === "failed"
                      ? "This document failed ingestion and cannot be queried."
                      : "Questions are limited to the currently selected document."}
                  </p>
                  <div className="flex flex-col gap-3 sm:flex-row">
                    <button
                      type="button"
                      onClick={inspectRetrieval}
                      disabled={
                        !selectedDocument ||
                        selectedDocument.status !== "ready" ||
                        isAnswering ||
                        isInspecting
                      }
                      className="rounded-full border border-[var(--border)] bg-white/80 px-5 py-3 text-sm font-medium text-[var(--foreground)] transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {isInspecting ? "Inspecting..." : "Inspect retrieval"}
                    </button>
                    <button
                      type="submit"
                      disabled={
                        !selectedDocument ||
                        selectedDocument.status !== "ready" ||
                        isAnswering ||
                        isInspecting
                      }
                      className="rounded-full bg-[var(--foreground)] px-5 py-3 text-sm font-medium text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {isAnswering ? "Searching and answering..." : "Ask question"}
                    </button>
                  </div>
                </div>
              </form>

              {qaError ? (
                <p className="mt-4 rounded-[1rem] bg-rose-50 px-4 py-3 text-sm text-rose-800">
                  {qaError}
                </p>
              ) : null}
              {retrievalError ? (
                <p className="mt-4 rounded-[1rem] bg-rose-50 px-4 py-3 text-sm text-rose-800">
                  {retrievalError}
                </p>
              ) : null}
            </section>

            <section className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.8fr)_minmax(320px,0.95fr)]">
              <article className="rounded-[1.6rem] border border-[var(--border)] bg-[var(--surface)] p-6">
                <div className="flex items-end justify-between gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted)]">
                      Answer
                    </p>
                    <h3 className="mt-2 text-2xl">Grounded response</h3>
                  </div>
                  {answer ? (
                    <span
                      className={`rounded-full px-3 py-1 text-xs font-medium uppercase tracking-[0.16em] ${
                        answer.supported
                          ? "bg-emerald-100 text-emerald-800"
                          : "bg-amber-100 text-amber-900"
                      }`}
                    >
                      {answer.supported ? "Supported" : "Insufficient support"}
                    </span>
                  ) : null}
                </div>

                {!answer ? (
                  <p className="mt-5 rounded-[1.2rem] border border-dashed border-[var(--border)] px-4 py-5 text-sm leading-6 text-[var(--muted)]">
                    Ask a question to populate the answer pane and citation
                    trail.
                  </p>
                ) : (
                  <div className="mt-5 space-y-4">
                    <p className="rounded-[1.25rem] bg-white/75 px-5 py-5 text-base leading-8 shadow-[0_14px_34px_rgba(63,42,29,0.05)]">
                      {answer.answer || (isAnswering ? "Streaming answer..." : "")}
                    </p>
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                        <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                          Normalized question
                        </p>
                        <p className="mt-2 text-sm leading-6">
                          {answer.normalized_question}
                        </p>
                      </div>
                      <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                        <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                          Retrieved chunks
                        </p>
                        <p className="mt-2 text-sm leading-6">
                          {answer.retrieved_chunk_count} candidates considered
                        </p>
                      </div>
                      <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                        <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                          Answer mode
                        </p>
                        <p className="mt-2 text-sm capitalize leading-6">
                          {formatModeLabel(answer.answer_mode)}
                        </p>
                      </div>
                      <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                        <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                          Support confidence
                        </p>
                        <p className="mt-2 text-sm leading-6">
                          {formatPercent(answer.confidence)} confidence ·{" "}
                          {formatPercent(answer.support_score)} evidence score
                        </p>
                      </div>
                    </div>
                    {qaTrace ? (
                      <div className="space-y-4">
                        <div className="grid gap-4 md:grid-cols-3">
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Answer provider
                            </p>
                            <p className="mt-2 text-sm leading-6">
                              {qaTrace.answer_provider}
                            </p>
                          </div>
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Context window
                            </p>
                            <p className="mt-2 text-sm leading-6">
                              {qaTrace.selected_contexts.length} chunks selected
                            </p>
                          </div>
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Total latency
                            </p>
                            <p className="mt-2 text-sm leading-6">
                              {qaTrace.timings.total_ms.toFixed(1)} ms
                            </p>
                          </div>
                        </div>
                        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Router decision
                            </p>
                            <p className="mt-2 text-sm capitalize leading-6">
                              {formatModeLabel(qaTrace.question_router.answer_mode)}
                            </p>
                            <p className="mt-1 text-sm leading-6 text-[var(--muted)]">
                              {qaTrace.question_router.reason}
                            </p>
                          </div>
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Embeddings
                            </p>
                            <p className="mt-2 text-sm leading-6">
                              {formatRuntimeLabel(
                                qaTrace.runtime.embedding_provider,
                                qaTrace.runtime.embedding_model,
                              )}
                            </p>
                          </div>
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Reranker
                            </p>
                            <p className="mt-2 text-sm leading-6">
                              {formatRuntimeLabel(
                                qaTrace.runtime.reranker_provider,
                                qaTrace.runtime.reranker_model,
                              )}
                            </p>
                          </div>
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Answer runtime
                            </p>
                            <p className="mt-2 text-sm leading-6">
                              {formatRuntimeLabel(
                                qaTrace.runtime.answer_provider,
                                qaTrace.runtime.answer_model,
                              )}
                            </p>
                          </div>
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Router inputs
                            </p>
                            <p className="mt-2 text-sm leading-6">
                              {qaTrace.question_router.facet_count} facets ·{" "}
                              {qaTrace.question_router.context_count} contexts
                            </p>
                          </div>
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Generative path
                            </p>
                            <p className="mt-2 text-sm leading-6">
                              {qaTrace.question_router.should_use_generative
                                ? "Enabled"
                                : "Not used"}
                            </p>
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                )}
              </article>

              <aside className="rounded-[1.6rem] border border-[var(--border)] bg-[var(--surface)] p-6">
                <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted)]">
                  Citations
                </p>
                <h3 className="mt-2 text-2xl">Source preview</h3>

                {!answer || answer.citations.length === 0 ? (
                  <p className="mt-5 rounded-[1.2rem] border border-dashed border-[var(--border)] px-4 py-5 text-sm leading-6 text-[var(--muted)]">
                    Supporting chunks will appear here with page spans and
                    similarity scores.
                  </p>
                ) : (
                  <div className="mt-5 space-y-4">
                    <div className="flex max-h-[20rem] flex-col gap-4 overflow-y-auto pr-1">
                      {answer.citations.map((citation, index) => {
                        const selected = selectedCitation?.chunk_id === citation.chunk_id;

                        return (
                          <button
                            key={`${citation.chunk_id}-${index}`}
                            type="button"
                            onClick={() => setSelectedCitationChunkId(citation.chunk_id)}
                            className={`rounded-[1.2rem] border p-4 text-left transition ${
                              selected
                                ? "border-[var(--accent)] bg-white shadow-[0_16px_32px_rgba(63,42,29,0.08)]"
                                : "border-[var(--border)] bg-white/70 hover:bg-white"
                            }`}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <p className="text-sm font-medium">
                                  {citation.document_title || citation.document_filename}
                                </p>
                                <p className="mt-1 text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                                  Citation {index + 1}
                                </p>
                              </div>
                              <span className="rounded-full bg-stone-100 px-3 py-1 text-xs font-medium text-stone-700">
                                {formatScore(citation.score)}
                              </span>
                            </div>

                            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm text-[var(--muted)]">
                              <div>
                                <dt>Pages</dt>
                                <dd className="mt-1 text-[var(--foreground)]">
                                  {formatPageRange(
                                    citation.page_start,
                                    citation.page_end,
                                  )}
                                </dd>
                              </div>
                              <div>
                                <dt>Chunk</dt>
                                <dd className="mt-1 text-[var(--foreground)]">
                                  #{citation.chunk_index}
                                </dd>
                              </div>
                            </dl>

                            <p className="mt-4 line-clamp-5 text-sm leading-7 text-[var(--foreground)]">
                              {citation.text}
                            </p>
                          </button>
                        );
                      })}
                    </div>

                    {selectedCitation && answer && selectedCitationInspector ? (
                      <article className="rounded-[1.2rem] border border-[var(--accent)] bg-amber-50/70 p-4">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Citation inspector
                            </p>
                            <h4 className="mt-2 text-lg">Page-local evidence trace</h4>
                          </div>
                          <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-900">
                            chunk #{selectedCitation.chunk_index}
                          </span>
                        </div>

                        <div className="mt-4 grid gap-4">
                          <div className="rounded-[1rem] border border-[var(--border)] bg-white/80 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                              Answer sentence
                            </p>
                            <p className="mt-2 text-sm leading-7">
                              {selectedCitationInspector.answerSentence}
                            </p>
                          </div>

                          <div className="rounded-[1rem] border border-[var(--border)] bg-white/80 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                              Cited excerpt
                            </p>
                            <p className="mt-2 text-sm leading-7">{selectedCitation.text}</p>
                          </div>

                          <div className="rounded-[1rem] border border-[var(--border)] bg-white/80 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                              Normalized source span
                            </p>
                            <div className="mt-3 space-y-3 text-sm">
                              {selectedCitationInspector.charRange ? (
                                <p className="text-[var(--foreground)]">
                                  Source span:{" "}
                                  {formatCrossPageSpan(selectedCitationInspector.charRange)}
                                </p>
                              ) : (
                                <p className="text-[var(--muted)]">
                                  No normalized-text span metadata recorded.
                                </p>
                              )}

                              {selectedCitationInspector.pageCharRanges.length > 0 ? (
                                <div className="flex flex-wrap gap-2">
                                  {selectedCitationInspector.pageCharRanges.map((range) => (
                                    <span
                                      key={`${selectedCitation.chunk_id}-${range.page_number}-${range.start_char}`}
                                      className="rounded-full bg-stone-100 px-3 py-1 text-xs font-medium text-stone-700"
                                    >
                                      {formatCharRangeLabel(
                                        range.page_number,
                                        range.start_char,
                                        range.end_char,
                                      )}
                                    </span>
                                  ))}
                                </div>
                              ) : null}

                              {selectedCitationInspector.paragraphRange ? (
                                <p className="text-[var(--foreground)]">
                                  Paragraph provenance: p.{" "}
                                  {selectedCitationInspector.paragraphRange.start_page}
                                  /
                                  {selectedCitationInspector.paragraphRange.start_index} to p.{" "}
                                  {selectedCitationInspector.paragraphRange.end_page}
                                  /
                                  {selectedCitationInspector.paragraphRange.end_index}
                                </p>
                              ) : null}
                            </div>
                          </div>

                          <div className="grid gap-3 sm:grid-cols-2">
                            <div className="rounded-[1rem] border border-[var(--border)] bg-white/80 px-4 py-4 text-sm">
                              <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                                Citation scoring
                              </p>
                              <dl className="mt-3 grid grid-cols-2 gap-2 text-[var(--foreground)]">
                                <div>
                                  <dt className="text-[var(--muted)]">Final</dt>
                                  <dd>{formatScore(selectedCitation.score)}</dd>
                                </div>
                                <div>
                                  <dt className="text-[var(--muted)]">Rerank</dt>
                                  <dd>{formatScore(selectedCitation.rerank_score)}</dd>
                                </div>
                                <div>
                                  <dt className="text-[var(--muted)]">Hybrid</dt>
                                  <dd>{formatScore(selectedCitation.hybrid_score)}</dd>
                                </div>
                                <div>
                                  <dt className="text-[var(--muted)]">Lexical</dt>
                                  <dd>{formatScore(selectedCitation.lexical_score)}</dd>
                                </div>
                                <div>
                                  <dt className="text-[var(--muted)]">Dense</dt>
                                  <dd>{formatScore(selectedCitation.dense_score)}</dd>
                                </div>
                              </dl>
                            </div>

                            <div className="rounded-[1rem] border border-[var(--border)] bg-white/80 px-4 py-4 text-sm">
                              <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                                Inspection intent
                              </p>
                              <p className="mt-3 leading-7 text-[var(--foreground)]">
                                Use the answer sentence, excerpt, and page-local offsets
                                together to spot right-page versus wrong-sentence citation
                                failures quickly.
                              </p>
                            </div>
                          </div>
                        </div>
                      </article>
                    ) : null}
                  </div>
                )}
              </aside>

              <aside className="rounded-[1.6rem] border border-[var(--border)] bg-[var(--surface)] p-6">
                <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted)]">
                  Retrieval debug
                </p>
                <h3 className="mt-2 text-2xl">Candidate ranking</h3>

                {!retrievalResult ? (
                  <p className="mt-5 rounded-[1.2rem] border border-dashed border-[var(--border)] px-4 py-5 text-sm leading-6 text-[var(--muted)]">
                    Run retrieval inspection to see the ranked candidates that
                    the current retriever returns for this question.
                  </p>
                ) : (
                  <div className="mt-5 space-y-4">
                    <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                      <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                        Normalized query
                      </p>
                      <p className="mt-2 text-sm leading-6">
                        {retrievalResult.normalized_query}
                      </p>
                    </div>

                    {qaTrace ? (
                      <>
                        <div className="grid gap-4 md:grid-cols-2">
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Selected context
                            </p>
                            <p className="mt-2 text-sm leading-6">
                              {qaTrace.selected_contexts.length} chunks sent to answer generation
                            </p>
                          </div>
                          <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                              Timings
                            </p>
                            <p className="mt-2 text-sm leading-6">
                              retrieval {qaTrace.timings.retrieval_ms.toFixed(1)} ms, answer{" "}
                              {qaTrace.timings.answer_generation_ms.toFixed(1)} ms
                            </p>
                          </div>
                        </div>

                        <div className="rounded-[1.1rem] border border-[var(--border)] bg-white/60 px-4 py-4">
                          <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                            Prompt snapshot
                          </p>
                          <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-[0.9rem] bg-stone-950 px-4 py-4 text-xs leading-6 text-stone-100">
                            {qaTrace.prompt_snapshot}
                          </pre>
                        </div>
                      </>
                    ) : null}

                    <div className="flex max-h-[42rem] flex-col gap-4 overflow-y-auto pr-1">
                      {retrievalResult.matches.map((match, index) => (
                        <article
                          key={match.chunk_id}
                          className="rounded-[1.2rem] border border-[var(--border)] bg-white/70 p-4"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="text-sm font-medium">
                                {match.document_title || match.document_filename}
                              </p>
                              <p className="mt-1 text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                                Rank {index + 1}
                              </p>
                            </div>
                            <span className="rounded-full bg-stone-100 px-3 py-1 text-xs font-medium text-stone-700">
                              {match.score.toFixed(3)}
                            </span>
                          </div>

                          <dl className="mt-4 grid grid-cols-2 gap-3 text-sm text-[var(--muted)]">
                            <div>
                              <dt>Pages</dt>
                              <dd className="mt-1 text-[var(--foreground)]">
                                {formatPageRange(match.page_start, match.page_end)}
                              </dd>
                            </div>
                            <div>
                              <dt>Chunk</dt>
                              <dd className="mt-1 text-[var(--foreground)]">
                                #{match.chunk_index}
                              </dd>
                            </div>
                          </dl>

                          {match.provenance?.page_char_ranges?.length ? (
                            <div className="mt-4 flex flex-wrap gap-2">
                              {match.provenance.page_char_ranges.map((range) => (
                                <span
                                  key={`${match.chunk_id}-${range.page_number}-${range.start_char}`}
                                  className="rounded-full bg-stone-100 px-3 py-1 text-xs font-medium text-stone-700"
                                >
                                  {formatCharRangeLabel(
                                    range.page_number,
                                    range.start_char,
                                    range.end_char,
                                  )}
                                </span>
                              ))}
                            </div>
                          ) : null}

                          <p className="mt-4 text-sm leading-7 text-[var(--foreground)]">
                            {match.text}
                          </p>
                        </article>
                      ))}
                    </div>

                    {qaTrace && qaTrace.selected_contexts.length > 0 ? (
                      <div className="space-y-4">
                        <div>
                          <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted)]">
                            Final context
                          </p>
                          <h4 className="mt-2 text-lg">Chunks passed to answer generation</h4>
                        </div>

                        <div className="flex max-h-[32rem] flex-col gap-4 overflow-y-auto pr-1">
                          {qaTrace.selected_contexts.map((match, index) => (
                            <article
                              key={`${match.chunk_id}-selected`}
                              className="rounded-[1.2rem] border border-[var(--border)] bg-amber-50/70 p-4"
                            >
                              <div className="flex items-start justify-between gap-3">
                                <div>
                                  <p className="text-sm font-medium">
                                    {match.document_title || match.document_filename}
                                  </p>
                                  <p className="mt-1 text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                                    Context {index + 1}
                                  </p>
                                </div>
                                <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-900">
                                  {match.score.toFixed(3)}
                                </span>
                              </div>

                              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm text-[var(--muted)]">
                                <div>
                                  <dt>Pages</dt>
                                  <dd className="mt-1 text-[var(--foreground)]">
                                    {formatPageRange(match.page_start, match.page_end)}
                                  </dd>
                                </div>
                                <div>
                                  <dt>Chunk</dt>
                                  <dd className="mt-1 text-[var(--foreground)]">
                                    #{match.chunk_index}
                                  </dd>
                                </div>
                              </dl>

                              {match.provenance?.page_char_ranges?.length ? (
                                <div className="mt-4 flex flex-wrap gap-2">
                                  {match.provenance.page_char_ranges.map((range) => (
                                    <span
                                      key={`${match.chunk_id}-selected-${range.page_number}-${range.start_char}`}
                                      className="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-900"
                                    >
                                      {formatCharRangeLabel(
                                        range.page_number,
                                        range.start_char,
                                        range.end_char,
                                      )}
                                    </span>
                                  ))}
                                </div>
                              ) : null}

                              <p className="mt-4 text-sm leading-7 text-[var(--foreground)]">
                                {match.text}
                              </p>
                            </article>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                )}
              </aside>
            </section>
          </section>
        </section>
      </section>
    </main>
  );
}
