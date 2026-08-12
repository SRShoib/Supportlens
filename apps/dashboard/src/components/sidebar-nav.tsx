"use client";

// Left sidebar shell, replacing the old top nav bar. Each section gets a
// fixed, permanent identity color (never reused as a data-encoding color
// within that section's own charts -- see confusion-matrix.tsx's blue,
// drift-panel.tsx's status palette, topics-over-time-chart.tsx's categorical
// set -- this is chrome/wayfinding color, a different context, deliberately
// picked to avoid same-page collisions, e.g. Metrics is cyan rather than
// emerald because /metrics itself already uses emerald for "stable/ok"
// status badges). Active-route detection needs the current pathname, which
// is only available client-side in the App Router, hence "use client" here
// while RootLayout (src/app/layout.tsx) stays a server component.

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { MenuIcon, XIcon } from "@/components/icons";
import { COLOR_STYLES, NAV_ITEMS } from "@/lib/nav";

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function SidebarNav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label="Open navigation"
        className="fixed top-4 left-4 z-40 flex h-10 w-10 items-center justify-center rounded-xl border border-zinc-200/70 bg-white/80 text-zinc-600 shadow-sm backdrop-blur-md md:hidden dark:border-white/10 dark:bg-zinc-900/80 dark:text-zinc-300"
      >
        <MenuIcon className="h-5 w-5" />
      </button>

      {open && (
        <div
          role="presentation"
          onClick={() => setOpen(false)}
          className="animate-fade-in fixed inset-0 z-40 bg-zinc-950/40 backdrop-blur-sm md:hidden"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-64 shrink-0 -translate-x-full flex-col border-r border-zinc-200/70 bg-white/85 backdrop-blur-xl transition-transform duration-300 ease-out md:sticky md:top-0 md:h-screen md:translate-x-0 dark:border-white/10 dark:bg-zinc-950/70 ${
          open ? "translate-x-0" : ""
        }`}
      >
        <div className="flex items-center justify-between gap-2 px-5 py-5">
          <Link href="/" className="group flex items-center gap-2.5" onClick={() => setOpen(false)}>
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-xs font-bold text-white shadow-sm shadow-indigo-500/30 transition-transform duration-200 group-hover:scale-105">
              SL
            </span>
            <span className="bg-gradient-to-r from-zinc-900 to-zinc-600 bg-clip-text text-sm font-semibold tracking-tight text-transparent dark:from-zinc-50 dark:to-zinc-400">
              supportlens
            </span>
          </Link>
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
            className="flex h-8 w-8 items-center justify-center rounded-lg text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 md:hidden dark:hover:bg-white/10 dark:hover:text-zinc-200"
          >
            <XIcon className="h-4 w-4" />
          </button>
        </div>

        <nav className="flex-1 space-y-1 px-3">
          {NAV_ITEMS.map((item) => {
            const active = isActive(pathname, item.href);
            const styles = COLOR_STYLES[item.color];
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                className={`group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors duration-150 ${
                  active
                    ? `${styles.activeBg} ${styles.activeText}`
                    : "text-zinc-600 hover:bg-zinc-100/80 dark:text-zinc-400 dark:hover:bg-white/5"
                }`}
              >
                {active && (
                  <span className={`absolute top-1/2 left-0 h-5 w-1 -translate-y-1/2 rounded-r-full ${styles.bar}`} />
                )}
                <span
                  className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-transform duration-150 group-hover:scale-105 ${styles.iconBg} ${styles.iconText}`}
                >
                  <Icon className="h-4 w-4" />
                </span>
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="px-5 py-5 text-xs text-zinc-400 dark:text-zinc-500">
          <p className="font-medium text-zinc-500 dark:text-zinc-400">Customer support intel</p>
          <p className="mt-0.5">Baselines → transformers → RAG</p>
        </div>
      </aside>
    </>
  );
}
