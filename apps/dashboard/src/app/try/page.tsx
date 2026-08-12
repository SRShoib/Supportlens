import { FlaskIcon } from "@/components/icons";
import { TryItForm } from "@/components/try-it-form";

export const metadata = { title: "Try it live — supportlens" };

export default function TryItPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="animate-fade-in-up">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-pink-100 text-pink-600 dark:bg-pink-500/15 dark:text-pink-400">
            <FlaskIcon className="h-5 w-5" />
          </span>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Try it live
          </h1>
        </div>
        <p className="mt-3 max-w-2xl text-sm text-zinc-500 dark:text-zinc-400">
          Paste any customer message and run it through the real, already-trained models --
          baseline vs. fine-tuned transformer, side by side, plus entity extraction. This is
          read-only inference: nothing you type here is saved as a ticket.
        </p>
      </div>

      <div className="animate-fade-in-up mt-8" style={{ animationDelay: "80ms" }}>
        <TryItForm />
      </div>
    </div>
  );
}
