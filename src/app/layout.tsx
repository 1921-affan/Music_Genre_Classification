import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Header from '../components/Header';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Music Analysis Hub',
  description: 'Classify music genres and identify songs using AI and Spotify API',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className} suppressHydrationWarning={true}>
        <Header />
        <main className="container mx-auto px-4 py-8 bg-gray-50 min-h-screen">
          {children}
        </main>
        <footer className="bg-purple-700 text-white text-center py-4 mt-8">
          <p>&copy; 2025 Music Analysis Hub. All rights reserved.</p>
        </footer>
      </body>
    </html>
  );
}
