import React from "react";

/**
 * High-fidelity Markdown & Rich Text Renderer for Statutory RAG Copilot Messages
 * Transforms raw markdown (###, **, `, •, ---, alerts) into beautifully styled dark-mode JSX
 */
export default function MarkdownMessage({ content }) {
  if (!content) return null;

  // Split into lines
  const lines = content.split("\n");

  const renderInline = (text) => {
    // Process bold (**...**) and inline code (`...`)
    const parts = [];
    let remaining = text;
    let keyIdx = 0;

    // Regex matching either **bold** or `code`
    const regex = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    const splitArr = remaining.split(regex);

    return splitArr.map((segment, idx) => {
      if (segment.startsWith("**") && segment.endsWith("**")) {
        return (
          <strong key={idx} className="text-white font-bold">
            {segment.slice(2, -2)}
          </strong>
        );
      } else if (segment.startsWith("`") && segment.endsWith("`")) {
        return (
          <code
            key={idx}
            className="px-1.5 py-0.5 rounded bg-slate-900/90 border border-slate-700 font-mono text-[11px] text-cyan-300 font-semibold"
          >
            {segment.slice(1, -1)}
          </code>
        );
      }
      return segment;
    });
  };

  return (
    <div className="space-y-1.5 text-xs text-slate-200 font-sans leading-relaxed">
      {lines.map((line, idx) => {
        const trimmed = line.trim();

        // Empty line
        if (!trimmed) {
          return <div key={idx} className="h-1.5" />;
        }

        // Horizontal Rule ---
        if (trimmed === "---" || trimmed === "***" || trimmed === "___") {
          return <hr key={idx} className="border-slate-700/80 my-2" />;
        }

        // H2: ## Heading
        if (trimmed.startsWith("## ")) {
          return (
            <h2
              key={idx}
              className="text-sm font-extrabold text-cyan-300 mt-3 mb-1 tracking-tight flex items-center gap-1.5"
            >
              {renderInline(trimmed.slice(3))}
            </h2>
          );
        }

        // H3: ### Heading
        if (trimmed.startsWith("### ")) {
          return (
            <h3
              key={idx}
              className="text-xs font-bold text-cyan-300 uppercase tracking-wider mt-2.5 mb-1 pb-1 border-b border-slate-800 flex items-center gap-1.5"
            >
              {renderInline(trimmed.slice(4))}
            </h3>
          );
        }

        // H4: #### Heading
        if (trimmed.startsWith("#### ")) {
          return (
            <h4
              key={idx}
              className="text-xs font-bold text-slate-100 mt-2 mb-0.5"
            >
              {renderInline(trimmed.slice(5))}
            </h4>
          );
        }

        // Bullet point: •, -, *
        if (trimmed.startsWith("• ") || trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
          const bulletText = trimmed.slice(2);
          const isWarning = bulletText.includes("⚠️");
          const isSuccess = bulletText.includes("✅");
          const isStop = bulletText.includes("⛔");

          return (
            <div
              key={idx}
              className={`flex items-start gap-2 py-0.5 pl-1 rounded-lg ${
                isWarning
                  ? "bg-amber-500/10 border border-amber-500/20 px-2 py-1 text-amber-200 my-1"
                  : isSuccess
                  ? "bg-emerald-500/10 border border-emerald-500/20 px-2 py-1 text-emerald-200 my-1"
                  : isStop
                  ? "bg-red-500/10 border border-red-500/20 px-2 py-1 text-red-200 my-1"
                  : ""
              }`}
            >
              {!isWarning && !isSuccess && !isStop && (
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0" />
              )}
              <span className="flex-1">{renderInline(bulletText)}</span>
            </div>
          );
        }

        // Regular paragraph text
        return (
          <p key={idx} className="text-slate-300">
            {renderInline(trimmed)}
          </p>
        );
      })}
    </div>
  );
}

