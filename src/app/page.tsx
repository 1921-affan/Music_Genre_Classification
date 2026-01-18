'use client';

import Link from 'next/link';
import Image from 'next/image';
import { MusicalNoteIcon, BookOpenIcon } from '@heroicons/react/24/solid';
import { Typewriter } from 'react-simple-typewriter';
import { useInView } from 'react-intersection-observer';

export default function Home() {
  const phrases = [
    "Discover the genre of your favorite tunes!",
    "Identify songs with just a few clicks!",
    "Explore the world of music analysis",
    "Your journey into AI-powered music insights starts here"
  ];

  const sectionColor = "rgb(130, 60, 255)";

  const { ref: ref1, inView: inView1 } = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  const { ref: ref2, inView: inView2 } = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  const { ref: ref3, inView: inView3 } = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  return (
    <div className="flex flex-col items-center justify-start min-h-screen bg-gradient-to-b from-purple-100 to-white">
      {/* Hero Section */}
      <div className="w-full h-[400px] relative mb-16">
        <Image
          src="/background.jpg?height=1080&width=1920"
          alt="Music background"
          fill
          className="object-cover brightness-75"
          priority
        />
        <div className="absolute inset-0 bg-purple-900/30" />

        {/* Centered Content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <h1 className="text-6xl font-bold mb-4 text-center text-white">
            Music Analysis Hub
          </h1>
          <div className="text-center text-xl text-white/90 h-20">
            <Typewriter
              words={phrases}
              loop={0}
              cursor
              cursorStyle="|"
              typeSpeed={70}
              deleteSpeed={50}
              delaySpeed={2000}
            />
          </div>
        </div>
      </div>

      {/* Content Sections */}
      <div className="flex flex-col gap-12 max-w-5xl w-full px-4 pb-16">
        <div ref={ref1}>
          <Link href="/genre-classification" className="block">
            <div
              className={`p-8 rounded-lg shadow-lg hover:shadow-xl transition-all duration-500 relative overflow-hidden group ${inView1 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
                }`}
              style={{ backgroundColor: sectionColor }}
            >
              <div className="relative z-10 flex items-center justify-between">
                <div>
                  <MusicalNoteIcon className="h-16 w-16 text-white mb-4" />
                  <h2 className="text-2xl font-bold text-white mb-2">Genre Classification</h2>
                  <p className="text-purple-100 mb-4">
                    Upload your audio and discover its genre instantly using our AI-powered classification system.
                  </p>
                  <ul className="space-y-2 text-white/90">
                    <li className="flex items-center">
                      <span className="mr-2">•</span>
                      Supports multiple audio formats (MP3, WAV, FLAC)
                    </li>
                    <li className="flex items-center">
                      <span className="mr-2">•</span>
                      Real-time analysis with high accuracy
                    </li>
                    <li className="flex items-center">
                      <span className="mr-2">•</span>
                      Detailed genre breakdown and confidence scores
                    </li>
                  </ul>
                </div>
                <div className="flex items-center justify-center h-8 w-8 bg-white rounded-full">
                  <span className="text-purple-800 text-lg font-bold">→</span>
                </div>
              </div>
            </div>
          </Link>
        </div>

        <div ref={ref3}>
          <Link href="/chatbot" className="block">
            <div
              className={`p-8 rounded-lg shadow-lg hover:shadow-xl transition-all duration-500 relative overflow-hidden group ${inView3 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
                }`}
              style={{ backgroundColor: sectionColor }}
            >
              <div className="relative z-10 flex items-center justify-between">
                <div>
                  <BookOpenIcon className="h-16 w-16 text-white mb-4" />
                  <h2 className="text-2xl font-bold text-white mb-2">Music Chatbot</h2>
                  <p className="text-purple-100 mb-4">
                    Engage with our AI-powered chatbot for music-related queries and insights.
                  </p>
                  <ul className="space-y-2 text-white/90">
                    <li className="flex items-center">
                      <span className="mr-2">•</span>
                      Ask questions about music theory, history, and artists
                    </li>
                    <li className="flex items-center">
                      <span className="mr-2">•</span>
                      Get personalized music recommendations
                    </li>
                    <li className="flex items-center">
                      <span className="mr-2">•</span>
                      Explore our extensive knowledge base on music topics
                    </li>
                  </ul>
                </div>
                <div className="flex items-center justify-center h-8 w-8 bg-white rounded-full">
                  <span className="text-purple-800 text-lg font-bold">→</span>
                </div>
              </div>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}

