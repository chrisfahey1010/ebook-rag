import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ebook-rag",
  description: "Local-first PDF question answering with grounded citations.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

