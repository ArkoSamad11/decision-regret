import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "xDR — expected decision regret",
  description:
    "Every on-ball football action has alternatives the player declined. xDR measures the gap, and tests whether the measurement survives transfer to a competition the model never saw.",
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
