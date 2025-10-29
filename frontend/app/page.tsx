"use client";

import { CopilotKit } from "copilotkit";

export default function Home() {
  return (
    <CopilotKit
      backendUrl="http://localhost:8000/copilot"
      instructions="You are a mechanical engineering assistant for Code_Aster."
    >
      <main className="flex flex-col items-center justify-center min-h-screen p-10 font-sans bg-gray-50 dark:bg-black">
        <h1 className="text-3xl font-bold mb-6 text-gray-800 dark:text-white">
          🧠 Code_Aster Assistant
        </h1>
        <p className="text-gray-600 dark:text-gray-300 text-center max-w-xl mb-4">
          Ask me to simulate or visualize a beam.
        </p>
        <p className="text-sm text-gray-400 text-center">
          Example: <code>Simulate a 3-meter beam with 15 elements</code>
        </p>
      </main>
    </CopilotKit>
  );
}
