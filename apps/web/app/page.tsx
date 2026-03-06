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
  created_at: string;
  updated_at: string | null;
};

type UploadResponse = {
  document: DocumentSummary;
  ingestion_job_id: string;
  ingestion_status: string;
  ingestion_error: string | null;
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
  score: number;
};

type QAResponse = {
  normalized_question: string;
  answer: string;
  supported: boolean;
  citations: Citation[];
  retrieved_chunk_count: number;
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
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(true);
  const [isUploading, startUploadTransition] = useTransition();
  const [isAnswering, startAnswerTransition] = useTransition();
  const [isInspecting, startInspectTransition] = useTransition();

  const selectedDocument =
    documents.find((document) => document.id === selectedDocumentId) ?? null;

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
        setAnswer(null);
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
        const [retrievalPayload, qaResponse] = await Promise.all([
          fetchRetrieval(trimmedQuestion, selectedDocumentId),
          fetch(`${apiBaseUrl}/api/qa/ask`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              question: trimmedQuestion,
              document_id: selectedDocumentId,
              top_k: 5,
            }),
          }),
        ]);

        if (!qaResponse.ok) {
          throw new Error(await parseApiError(qaResponse));
        }
        const qaPayload = (await qaResponse.json()) as QAResponse;

        setRetrievalResult(retrievalPayload);
        setAnswer(qaPayload);
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

                  return (
                    <button
                      key={document.id}
                      type="button"
                      onClick={() => {
                        setSelectedDocumentId(document.id);
                        setAnswer(null);
                        setQaError(null);
                        setRetrievalResult(null);
                        setRetrievalError(null);
                      }}
                      className={`rounded-[1.2rem] border px-4 py-4 text-left transition ${
                        selected
                          ? "border-[var(--accent)] bg-white shadow-[0_16px_32px_rgba(63,42,29,0.08)]"
                          : "border-[var(--border)] bg-white/60 hover:bg-white"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate text-base font-medium">
                            {formatDocumentLabel(document)}
                          </p>
                          <p className="mt-1 truncate text-sm text-[var(--muted)]">
                            {document.original_filename}
                          </p>
                        </div>
                        <span
                          className={`rounded-full px-3 py-1 text-xs font-medium uppercase tracking-[0.16em] ${statusTone(document.status)}`}
                        >
                          {document.status}
                        </span>
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
                    </button>
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
                    from `/api/qa/ask` and expose the supporting chunk text.
                  </p>
                </div>
                {selectedDocument ? (
                  <div className="rounded-[1rem] border border-[var(--border)] bg-white/70 px-4 py-3 text-sm text-[var(--muted)]">
                    {selectedDocument.page_count ?? "-"} pages,{" "}
                    {selectedDocument.chunk_count} chunks
                  </div>
                ) : null}
              </div>

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
                      {answer.answer}
                    </p>
                    <div className="grid gap-4 md:grid-cols-2">
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
                    </div>
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
                  <div className="mt-5 flex max-h-[42rem] flex-col gap-4 overflow-y-auto pr-1">
                    {answer.citations.map((citation, index) => (
                      <article
                        key={citation.chunk_id}
                        className="rounded-[1.2rem] border border-[var(--border)] bg-white/70 p-4"
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
                            {citation.score.toFixed(3)}
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

                        <p className="mt-4 text-sm leading-7 text-[var(--foreground)]">
                          {citation.text}
                        </p>
                      </article>
                    ))}
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

                          <p className="mt-4 text-sm leading-7 text-[var(--foreground)]">
                            {match.text}
                          </p>
                        </article>
                      ))}
                    </div>
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
