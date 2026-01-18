'use client';

import { useState, useEffect } from 'react';
import { MusicalNoteIcon } from '@heroicons/react/24/solid';
import io from 'socket.io-client';
import Link from 'next/link'

// Define the shape of the classification result
interface ClassificationResult {
  status: string;
  predicted_class: string;
  confidence: number;
}

// Define the workflow steps
const workflowSteps = ['upload', 'preprocessing', 'feature_extraction', 'inference', 'complete'];

export default function GenreClassification() {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [progressMessage, setProgressMessage] = useState('');

  useEffect(() => {
    const socket = io('http://localhost:5000');
    socket.on('progress', (data) => {
      // Find index of step that matches data.step exactly or is included
      const stepIndex = workflowSteps.findIndex(step => step === data.step);
      if (stepIndex !== -1) {
        setCurrentStep(stepIndex);
      }
      setProgressMessage(data.message);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setCurrentStep(0);
    }
  };

  const handleDragEnter = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const droppedFile = event.dataTransfer.files[0];
    setFile(droppedFile);
    setCurrentStep(0);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!file) {
      alert("Please upload an audio file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setClassificationResult(null);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data: ClassificationResult = await response.json();
      if (data.status === 'success') {
        setClassificationResult(data);
      } else {
        throw new Error(data.status);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while classifying the music. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start justify-center min-h-screen bg-gradient-to-b from-purple-50 to-white relative overflow-hidden p-8">
      { }
      <svg className="absolute top-0 left-0 w-full h-full" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="music-pattern" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
            <path d="M25 50 Q 40 40, 50 50 T 75 50" fill="none" stroke="rgba(107, 33, 168, 0.1)" strokeWidth="2" />
            <circle cx="25" cy="50" r="3" fill="rgba(107, 33, 168, 0.1)" />
            <circle cx="75" cy="50" r="3" fill="rgba(107, 33, 168, 0.1)" />
          </pattern>
        </defs>
        <rect x="0" y="0" width="100%" height="100%" fill="url(#music-pattern)" />
      </svg>

      <div className="col-span-full text-center mb-8">
        <h1 className="text-4xl font-bold mb-8 text-center text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600 animate-pulse relative z-10">
          Classify Your Music
        </h1>

      </div>
      <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-xl relative overflow-hidden z-10">
        <div className="relative z-10">
          <div className="mb-6 text-center">
            <MusicalNoteIcon className="h-16 w-16 text-purple-600 mx-auto mb-4 animate-bounce" />
            <p className="text-gray-600">Upload your audio file to classify its genre</p>
          </div>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div
              className="flex items-center justify-center w-full"
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
            >
              <label
                htmlFor="dropzone-file"
                className={`flex flex-col items-center justify-center w-full h-64 border-2 border-purple-300 border-dashed rounded-lg cursor-pointer bg-purple-50 hover:bg-purple-100 transition-all duration-300 ${isDragging ? 'border-purple-500 bg-purple-100' : ''
                  }`}
              >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <svg
                    className="w-10 h-10 mb-3 text-purple-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
                    />
                  </svg>
                  <p className="mb-2 text-sm text-purple-500">
                    <span className="font-semibold">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-purple-500">MP3, WAV, or OGG (MAX. 10MB)</p>
                </div>
                <input
                  id="dropzone-file"
                  type="file"
                  className="hidden"
                  onChange={handleFileChange}
                  accept=".mp3,.wav,.ogg"
                />
              </label>
            </div>
            {file && (
              <p className="text-sm text-purple-600">
                Selected file: {file.name}
              </p>
            )}
            <button
              type="submit"
              className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-3 px-4 rounded-md hover:from-purple-700 hover:to-indigo-700 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!file || loading}
            >
              {loading ? 'Classifying...' : 'Classify Music'}
            </button>
          </form>
        </div>
      </div>
      <div className="space-y-8">
        {/* Workflow Steps */}
        <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-xl relative z-10">
          <h2 className="text-2xl font-semibold text-center text-purple-700 mb-4">Workflow Progress</h2>
          <div className="flex justify-between mb-4">
            {workflowSteps.map((step, index) => (
              <div key={step} className="flex flex-col items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center ${index <= currentStep ? 'bg-purple-600 text-white' : 'bg-gray-200 text-gray-600'
                    }`}
                >
                  {index + 1}
                </div>
                <p className="text-xs mt-2">{step}</p>
              </div>
            ))}
          </div>
          <p className="text-sm text-center text-gray-600">{progressMessage}</p>
        </div>

        {/* Display Classification Result */}
        {classificationResult && (
          <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-xl relative z-10">
            <h2 className="text-2xl font-semibold text-center text-purple-700">Prediction Result</h2>
            <p className="mt-4 text-center text-gray-600">
              Predicted Genre: <span className="font-bold text-purple-600">{classificationResult.predicted_class}</span>
            </p>
            <p className="mt-2 text-center text-gray-600">
              Confidence: <span className="font-bold text-purple-600">{classificationResult.confidence}%</span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

