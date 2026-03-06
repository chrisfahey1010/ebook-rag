const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const milestones = [
  "Upload and register PDF ebooks",
  "Extract text with PyMuPDF",
  "Index chunked content in PostgreSQL + pgvector",
  "Answer questions with source citations",
];

export default function Home() {
  return (
    <main className="min-h-screen px-6 py-10 md:px-10 lg:px-16">
      <section className="mx-auto flex max-w-6xl flex-col gap-8 rounded-[2rem] border border-[var(--border)] bg-[var(--surface)] p-8 shadow-[0_20px_80px_rgba(63,42,29,0.08)] backdrop-blur md:p-12">
        <div className="flex flex-col gap-4 border-b border-[var(--border)] pb-8">
          <p className="text-sm uppercase tracking-[0.3em] text-[var(--muted)]">
            Production-style portfolio project
          </p>
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <h1 className="text-5xl leading-none tracking-[-0.04em] md:text-7xl">
                ebook-rag
              </h1>
              <p className="mt-4 max-w-2xl text-lg leading-8 text-[var(--muted)]">
                A local-first PDF QA platform for ebooks and manuals. The focus
                is grounded retrieval, readable architecture, and inspectable
                citations.
              </p>
            </div>
            <div className="rounded-full border border-[var(--border)] px-5 py-3 text-sm text-[var(--muted)]">
              API target: {apiBaseUrl}
            </div>
          </div>
        </div>

        <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <article className="rounded-[1.5rem] border border-[var(--border)] bg-white/70 p-6">
            <p className="text-sm uppercase tracking-[0.2em] text-[var(--muted)]">
              Scope
            </p>
            <h2 className="mt-3 text-2xl">First scaffold milestone</h2>
            <p className="mt-4 max-w-2xl text-base leading-7 text-[var(--muted)]">
              This shell establishes the frontend surface area while the backend
              grows into ingestion, retrieval, provider routing, and citation
              tracing. The UI intentionally starts simple and product-oriented.
            </p>
          </article>

          <aside className="rounded-[1.5rem] bg-[var(--accent)] p-6 text-[#fff7ef]">
            <p className="text-sm uppercase tracking-[0.2em] text-white/70">
              Next up
            </p>
            <ul className="mt-4 space-y-4 text-base leading-7">
              {milestones.map((milestone) => (
                <li
                  key={milestone}
                  className="border-b border-white/15 pb-4 last:border-b-0 last:pb-0"
                >
                  {milestone}
                </li>
              ))}
            </ul>
          </aside>
        </section>
      </section>
    </main>
  );
}
