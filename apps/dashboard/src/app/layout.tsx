import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "supportlens",
  description: "Customer support intelligence dashboard",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col bg-zinc-50 dark:bg-black">
        <header className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
          <div className="mx-auto flex max-w-4xl items-center gap-6 px-6 py-4">
            <Link
              href="/tickets"
              className="text-sm font-semibold text-zinc-900 dark:text-zinc-50"
            >
              supportlens
            </Link>
            <nav className="flex gap-4 text-sm text-zinc-500 dark:text-zinc-400">
              <Link href="/tickets" className="hover:text-zinc-900 dark:hover:text-zinc-50">
                Tickets
              </Link>
              <Link href="/topics" className="hover:text-zinc-900 dark:hover:text-zinc-50">
                Topics
              </Link>
            </nav>
          </div>
        </header>
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
