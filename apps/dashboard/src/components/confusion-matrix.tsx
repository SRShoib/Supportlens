// Confusion matrix heatmap (SPEC M9: "confusion matrices"). Dataviz skill:
// magnitude job -> sequential color, one hue light->dark, normalized by the
// matrix's own global max cell value (not per-row) so the darkest cell
// anywhere in the grid is always the single largest count. Color is a
// reinforcing secondary channel only -- every cell also renders its raw
// count as text, so the matrix is fully legible with color removed (CVD,
// print, forced-colors), matching the skill's "never color alone" rule.
// Plain HTML table, not SVG: this is tabular data (row=true label,
// column=predicted label), and a table lets a screen reader read real cell
// values instead of an unlabeled shape. Wrapped in its own horizontal
// scroll container (large label counts, e.g. intent's 27 classes, would
// otherwise force the whole page to scroll sideways).

const SEQUENTIAL_STEPS = [
  "bg-blue-50 dark:bg-blue-950",
  "bg-blue-100 dark:bg-blue-900",
  "bg-blue-200 dark:bg-blue-800/80",
  "bg-blue-300 dark:bg-blue-700/80",
  "bg-blue-400 dark:bg-blue-600",
  "bg-blue-500 dark:bg-blue-500",
];
const LIGHT_TEXT_STEP_INDEX = 4; // steps at/after this index need light text for contrast

interface ConfusionMatrixProps {
  labels: string[];
  matrix: number[][];
}

function bucketIndex(value: number, max: number): number {
  if (max <= 0) return 0;
  const fraction = value / max;
  return Math.min(SEQUENTIAL_STEPS.length - 1, Math.floor(fraction * SEQUENTIAL_STEPS.length));
}

export function ConfusionMatrix({ labels, matrix }: ConfusionMatrixProps) {
  if (labels.length === 0 || matrix.length === 0) {
    return <p className="text-sm text-zinc-500 dark:text-zinc-400">No confusion matrix yet.</p>;
  }

  const max = Math.max(1, ...matrix.flat());

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-0.5 text-[11px]">
          <thead>
            <tr>
              <th className="p-0" />
              <th
                colSpan={labels.length}
                className="pb-1 text-center font-medium text-zinc-500 dark:text-zinc-400"
              >
                predicted
              </th>
            </tr>
            <tr>
              <th className="p-0" />
              {labels.map((label) => (
                <th
                  key={label}
                  title={label}
                  className="max-w-8 truncate p-1 text-center font-normal text-zinc-500 dark:text-zinc-400"
                >
                  {label.length > 6 ? `${label.slice(0, 5)}…` : label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={labels[rowIndex]}>
                {rowIndex === 0 && (
                  <th
                    rowSpan={matrix.length}
                    className="pr-1 text-right align-middle font-medium text-zinc-500 [writing-mode:vertical-rl] dark:text-zinc-400"
                  >
                    actual
                  </th>
                )}
                <th
                  title={labels[rowIndex]}
                  className="max-w-16 truncate p-1 text-right font-normal text-zinc-500 dark:text-zinc-400"
                >
                  {labels[rowIndex]}
                </th>
                {row.map((value, colIndex) => {
                  const step = bucketIndex(value, max);
                  const isDiagonal = rowIndex === colIndex;
                  return (
                    <td
                      key={colIndex}
                      title={`actual ${labels[rowIndex]}, predicted ${labels[colIndex]}: ${value}`}
                      className={`relative h-8 w-8 rounded-sm text-center tabular-nums transition-transform duration-150 hover:z-10 hover:scale-125 hover:shadow-md ${SEQUENTIAL_STEPS[step]} ${
                        step >= LIGHT_TEXT_STEP_INDEX
                          ? "text-white"
                          : "text-zinc-700 dark:text-zinc-200"
                      } ${isDiagonal ? "ring-1 ring-inset ring-indigo-400 dark:ring-indigo-400/70" : ""}`}
                    >
                      {value}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-zinc-400 dark:text-zinc-500">
        Rows are the true label, columns the predicted label. Outlined cells are the diagonal
        (correct predictions).
      </p>
    </div>
  );
}
