// src/App.js
import React from 'react';
import { ChatBot } from 'react-chatbotify';

const App = () => {
  const config = {
    botName: "EduBot",
    theme: { primaryColor: "#4CAF50" },
  };

  const handleMessage = async (userMessage) => {
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userMessage, lang: 'hi' }),  // e.g., Hindi
      });
      const data = await response.json();
      return data.response;
    } catch (error) {
      return "Error: Try again!";
    }
  };

  return (
    <ChatBot
      config={config}
      onUserMessage={handleMessage}
    />
  );
};

export default App;
