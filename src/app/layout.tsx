import type { Metadata } from "next";
import { Heebo } from "next/font/google";
import "./globals.css";

const heebo = Heebo({ subsets: ["hebrew"], weight: ["300", "400", "500", "600", "700"] });

export const metadata: Metadata = {
  title: "סימולטור אופטיקה | Fourier Optics Simulator",
  description: "סינון תמונות במישור פורייה - מערכת 4f אופטית",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="he" dir="rtl">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebApplication",
              name: "Fourier Optics Simulator",
              applicationCategory: "EducationalApplication",
              operatingSystem: "Any",
              description: "Interactive simulation for Fourier optics filtering.",
              offers: {
                "@type": "Offer",
                price: "0",
              },
            }),
          }}
        />
      </head>
      <body className={`${heebo.className} min-h-screen antialiased`}>
        {children}
      </body>
    </html>
  );
}
