// Plain data module (no "use client") shared by src/components/sidebar-nav.tsx
// (a client component -- it needs usePathname() for active-route
// highlighting) and src/app/page.tsx (a Server Component). A Server
// Component can only import *components* from a "use client" module --
// every other export of a client module becomes an opaque reference across
// that boundary -- so the plain NAV_ITEMS/COLOR_STYLES data has to live
// here instead, in a module neither side is a client boundary for.

import type { ComponentType } from "react";

import { ActivityIcon, FlaskIcon, HomeIcon, InboxIcon, SearchIcon, TagIcon } from "@/components/icons";

export type NavColor = "indigo" | "blue" | "violet" | "amber" | "cyan" | "pink";

export interface NavItem {
  href: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
  color: NavColor;
}

export const NAV_ITEMS: NavItem[] = [
  { href: "/", label: "Overview", icon: HomeIcon, color: "indigo" },
  { href: "/tickets", label: "Tickets", icon: InboxIcon, color: "blue" },
  { href: "/topics", label: "Topics", icon: TagIcon, color: "violet" },
  { href: "/search", label: "Search", icon: SearchIcon, color: "amber" },
  { href: "/metrics", label: "Metrics", icon: ActivityIcon, color: "cyan" },
  { href: "/try", label: "Try it live", icon: FlaskIcon, color: "pink" },
];

// Tailwind's v4 scanner needs full literal class strings in source, not
// `bg-${color}-500` interpolation -- same "static lookup record" convention
// entity-highlighted-text.tsx's LABEL_STYLES and drift-panel.tsx's
// STATUS_BADGE_CLASS already use for the identical reason.
export const COLOR_STYLES: Record<
  NavColor,
  { iconBg: string; iconText: string; activeBg: string; activeText: string; bar: string }
> = {
  indigo: {
    iconBg: "bg-indigo-100 dark:bg-indigo-500/15",
    iconText: "text-indigo-600 dark:text-indigo-400",
    activeBg: "bg-indigo-50 dark:bg-indigo-500/10",
    activeText: "text-indigo-700 dark:text-indigo-300",
    bar: "bg-indigo-500",
  },
  blue: {
    iconBg: "bg-blue-100 dark:bg-blue-500/15",
    iconText: "text-blue-600 dark:text-blue-400",
    activeBg: "bg-blue-50 dark:bg-blue-500/10",
    activeText: "text-blue-700 dark:text-blue-300",
    bar: "bg-blue-500",
  },
  violet: {
    iconBg: "bg-violet-100 dark:bg-violet-500/15",
    iconText: "text-violet-600 dark:text-violet-400",
    activeBg: "bg-violet-50 dark:bg-violet-500/10",
    activeText: "text-violet-700 dark:text-violet-300",
    bar: "bg-violet-500",
  },
  amber: {
    iconBg: "bg-amber-100 dark:bg-amber-500/15",
    iconText: "text-amber-600 dark:text-amber-400",
    activeBg: "bg-amber-50 dark:bg-amber-500/10",
    activeText: "text-amber-700 dark:text-amber-300",
    bar: "bg-amber-500",
  },
  cyan: {
    iconBg: "bg-cyan-100 dark:bg-cyan-500/15",
    iconText: "text-cyan-600 dark:text-cyan-400",
    activeBg: "bg-cyan-50 dark:bg-cyan-500/10",
    activeText: "text-cyan-700 dark:text-cyan-300",
    bar: "bg-cyan-500",
  },
  pink: {
    iconBg: "bg-pink-100 dark:bg-pink-500/15",
    iconText: "text-pink-600 dark:text-pink-400",
    activeBg: "bg-pink-50 dark:bg-pink-500/10",
    activeText: "text-pink-700 dark:text-pink-300",
    bar: "bg-pink-500",
  },
};
