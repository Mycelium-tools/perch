'use client'

import { FolderSearch, PencilLine, FileText, Workflow, Gavel, Users, CalendarClock, ChartLine, ArrowUp, Copy } from "lucide-react";
import { FormEvent, JSX } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from 'remark-gfm';
import { useState, useRef, useEffect } from "react";
import BirdLoader from "../components/BirdLoader";
import { useChat } from "../components/ClientLayout";
import Image from "next/image";

type Chat = { id: string; title: string; history: { question: string; answer: string; context?: any; pending?: boolean }[] };

export default function Home() {
  const { chats, setChats, activeChatId, setActiveChatId, chatHistory, setChatHistory } = useChat();
  const [showContext, setShowContext] = useState<number | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [showSnackbar, setShowSnackbar] = useState(false);

  // Sync chatHistory changes back to the active chat
  useEffect(() => {
    if (activeChatId && chatHistory.length > 0) {
      setChats(prevChats =>
        prevChats.map(chat =>
          chat.id === activeChatId
            ? { ...chat, history: chatHistory }
            : chat
        )
      );
    }
  }, [chatHistory, activeChatId, setChats]);

  function getSessionId() {
    let id = localStorage.getItem("session_id");
    if (!id) {
      id = Math.random().toString(36).substring(2, 15);
      localStorage.setItem("session_id", id);
    }
    return id;
  }

  const handleCopyResponse = async (responseText: string, index: number) => {
    try {
      await navigator.clipboard.writeText(responseText);
      setCopiedIndex(index);
      setShowSnackbar(true);

      setTimeout(() => setCopiedIndex(null), 2000); // reset copied state after 2 seconds
      setTimeout(() => setShowSnackbar(false), 3000); // hide snackbar after 3 seconds
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const iconMap: Record<string, JSX.Element> = {
    "magnifying-glass-chart": <FolderSearch className="w-8 h-8 inline mr-2" />,
    "pencil-paper": <PencilLine className="w-8 h-8 inline mr-2" />,
    "document-review": <FileText className="w-8 h-8 inline mr-2" />,
    "roadmap": <Workflow className="w-8 h-8 inline mr-2" />,
    "gavel-search": <Gavel className="w-8 h-8 inline mr-2" />,
    "group-mobilize": <Users className="w-8 h-8 inline mr-2" />,
    "calendar-clock": <CalendarClock className="w-8 h-8 inline mr-2" />,
    "bar-chart-forecast": <ChartLine className="w-8 h-8 inline mr-2" />,
  };
  const promptSuggestions = [
    {
      id: 1,
      icon: "magnifying-glass-chart",
      text: "I'm a student at Boston University and I want to push my dining hall to offer more plant-based food options. How do I get started? Are there existing programs or organizations that support this kind of campaign?"
    },
    {
      id: 2,
      icon: "pencil-paper",
      text: "I work for the humane league (THL) and we're planning to launch a cage-free campaign targeting Crumbl cookies. What strategies have worked in other cage-free campaigns, and what are the key steps to getting a company to make a public commitment?"
    },
    {
      id: 3,
      icon: "document-review",
      text: "I want to start a fur-free campaign in my city to push for a local ban on fur sales. What should I know before getting started? Are there any pitfalls or lessons from previous campaigns I should be aware of?"
    },
    {
      id: 4,
      icon: "roadmap",
      text: "placeholder text"
    },
    {
      id: 5,
      icon: "gavel-search",
      text: "placeholder text"
    },
    {
      id: 6,
      icon: "group-mobilize",
      text: "placeholder text"
    },
    {
      id: 7,
      icon: "calendar-clock",
      text: "placeholder text"
    },
    {
      id: 8,
      icon: "bar-chart-forecast",
      text: "placeholder text"
    }
  ];

  const [question, setQuestion] = useState<string>("");
  // const [lastQuestion, setLastQuestion] = useState<string>(""); // NEW
  const [context, setContext] = useState<any>("");  // mirrors last response context for legacy use

  const lastMsgRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null); // dummy element at bottom for auto-scroll
  const chatScrollRef = useRef<HTMLDivElement | null>(null);
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true);

  // Scroll to bottom during updates only when user is already near the bottom.
  useEffect(() => {
    if (autoScrollEnabled && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "auto" });
    }
  }, [chatHistory, autoScrollEnabled]);

  const handleChatScroll = () => {
    const el = chatScrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setAutoScrollEnabled(distanceFromBottom < 80);
  };

  const [sessionId, setSessionId] = useState<string | null>(null);

  // Set sessionId only on client
  useEffect(() => {
    setSessionId(getSessionId());
  }, []);

  const submitQuestion = async (questionText: string) => {
    if (!sessionId) return; // Don't send request until sessionId is set
    setAutoScrollEnabled(true);

    if (!activeChatId) {
      // Create a new chat
      const newId = Math.random().toString(36).substring(2, 15);
      const chatTitle = questionText.length > 30
        ? questionText.substring(0, 30) + "..."
        : questionText;

      const newChat = {
        id: newId,
        title: chatTitle, // Use first question as title
        history: [],
      };
      setChats((prev) => [...prev, newChat]);
      setActiveChatId(newId);
    }

    // Add pending message
    setChatHistory((prev) => [
      ...prev,
      { question: questionText, answer: "Thinking...", context: null, pending: true },
    ]);
    setQuestion(""); // Clear input after submit

    // if the environment variable exists, use it; otherwise, default to hardcoded local or production URL
    const localServer = "http://localhost:8000";
    const prodServer = "https://pawlicy-gpt-production.up.railway.app";

    const baseUrl = process.env.NEXT_PUBLIC_API_URL
      ? process.env.NEXT_PUBLIC_API_URL
      : process.env.NODE_ENV === 'production' ? prodServer : localServer;

    // Use streaming endpoint for token-by-token display
    const res = await fetch(`${baseUrl}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: questionText, session_id: sessionId }),
    });
    if (!res.ok || !res.body) {
      const text = await res.text();
      console.error("Backend error:", res.status, text);
      return;
    }

    // Read SSE stream and update chat state incrementally
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const event = JSON.parse(line.slice(6));

        if (event.type === "text") {
          // Append each token; replace "Thinking..." on first chunk
          setChatHistory((prev) => {
            const updated = [...prev];
            const last = { ...updated[updated.length - 1] };
            last.answer = last.pending ? event.content : last.answer + event.content;
            last.pending = false;
            updated[updated.length - 1] = last;
            return updated;
          });
        } else if (event.type === "sources") {
          // Attach source documents once streaming is complete
          setChatHistory((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { ...updated[updated.length - 1], context: event.context };
            return updated;
          });
          setContext(event.context);
        }
      }
    }
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    await submitQuestion(question);
  };

  const onPromptClick = async (promptText: string) => {
    setQuestion(promptText);
    await submitQuestion(promptText);
  };


  return (
    <div className="h-full flex flex-col">
      {/* HEADER */}
      {chatHistory.length === 0 && (
        <div className="w-full flex justify-center items-center flex-shrink-0">
          <h1 className="text-[40px] text-pawlicy-green p-4 flex justify-center items-center w-full text-center pt-22 pb-8">
            Perch
          </h1>
        </div>
      )}

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* INPUT FIELD (top, only if no answer) */}
        {chatHistory.length === 0 && (
          <div className="flex-shrink-0 w-full max-w-5xl mx-auto px-4">
            <form onSubmit={handleSubmit} className="w-full space-y-4">
              <div className="w-full relative">
                <input
                  className="w-full min-w-0 px-4 py-4 pb-26 pr-12 text-md border border-[#D7E8CD] shadow-md rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask Perch anything about animal welfare advocacy, ordinances, legislation, and more."
                />
                <button
                  type="submit"
                  className="absolute bottom-2 right-2 bg-black rounded-full p-2 flex items-center justify-center hover:bg-gray-700 transition cursor-pointer disabled:bg-gray-300 disabled:cursor-default"
                  aria-label="Send"
                  disabled={!question.trim()}
                >
                  <ArrowUp className="w-5 h-5 text-white" />
                </button>
              </div>
            </form>
          </div>
        )}

        {/* CHAT HISTORY - This should be the scrollable area */}
        {chatHistory.length > 0 && (
          <div
            ref={chatScrollRef}
            onScroll={handleChatScroll}
            className="flex-1 overflow-y-auto px-4 pb-60"
          >
            <div className="flex flex-col gap-6 max-w-5xl mx-auto">
              {chatHistory.map((msg, idx) => (
                <div key={idx} ref={idx === chatHistory.length - 1 ? lastMsgRef : null}>
                  {/* User query bubble */}
                  <div className="flex justify-end my-8">
                    <div className="bg-pawlicy-lightgreen text-gray-900 rounded-3xl px-6 py-4 max-w-lg shadow-md">
                      <div className="text-left">
                        <ReactMarkdown>{msg.question}</ReactMarkdown>
                      </div>
                    </div>
                  </div>
                  {/* AI answer bubble */}
                  <div className="mt-2 flex justify-start">
                    <div className="px-4 prose max-w-3xl">
                      {msg.pending ? (
                        <span>
                          <BirdLoader /> Thinking...
                        </span>
                      ) : (
                        <div>
                          {(() => {
                            console.log('Raw answer:', msg.answer);
                            return (                              
                              <ReactMarkdown remarkPlugins={[remarkGfm]}
                              components={{
                                blockquote: ({node, ...props}) => (
                                  <blockquote className="border-l-4 border-gray-300 pl-4 italic my-4 text-gray-600" {...props} />
                                ),
                                // --- TABLE FIXES ---
                                table: ({node, ...props}) => (
                                  <div className="my-6 overflow-x-auto rounded-lg border border-gray-200">
                                    <table className="min-w-full divide-y divide-gray-200" {...props} />
                                  </div>
                                ),
                                thead: ({node, ...props}) => <thead className="bg-gray-50" {...props} />,
                                th: ({node, ...props}) => (
                                  <th 
                                    className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-900 !text-left" 
                                    {...props} 
                                  />
                                ),
                                td: ({node, ...props}) => (
                                  <td 
                                    className="whitespace-normal px-4 py-3 text-sm text-gray-700 border-t border-gray-200 !text-left" 
                                    {...props} 
                                  />
                                ),
                                // -------------------
                                
                                // --- HEADER HIERARCHY ---
                                h1: ({node, ...props}) => <h1 className="text-3xl font-bold mt-8 mb-4 text-gray-900" {...props} />,
                                h2: ({node, ...props}) => <h2 className="text-2xl font-bold mt-6 mb-3 text-gray-800 border-b pb-2" {...props} />,
                                h3: ({node, ...props}) => <h3 className="text-xl font-semibold mt-5 mb-2 text-gray-800" {...props} />,
                                h4: ({node, ...props}) => <h4 className="text-lg font-medium mt-4 mb-1 text-gray-700 italic" {...props} />,
                                
                                // --- CONTENT STYLING ---
                                p: ({node, ...props}) => <p className="mb-4 leading-relaxed text-gray-700" {...props} />,
                                ul: ({node, ...props}) => <ul className="list-disc list-inside mb-4 ml-4 space-y-1" {...props} />,
                                ol: ({node, ...props}) => <ol className="list-decimal list-inside mb-4 ml-4 space-y-1" {...props} />,
                                li: ({node, ...props}) => <li className="mb-1 [&>p]:inline" {...props} />,
                                strong: ({node, ...props}) => <strong className="font-bold text-gray-900" {...props} />,
                              }}>
                              {msg.answer}

                              </ReactMarkdown>
                            );
                          })()}
                          <div className="mt-4 pt-4 border-t text-sm text-gray-800 italic">
                            Prepared by Perch. Please consult City Counsel before filing to confirm compliance with state pre‑emption rules and charter procedures.
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {msg.context && (
                    <div className="flex items-center gap-2 mb-2 mt-4 pl-6 max-w-3xl w-full">
                      {/* COPY */}
                      <button
                        className="text-[#66991D] hover:text-green-900 cursor-pointer transition-colors rounded"
                        title={copiedIndex === idx ? "Copied!" : "Copy response"}
                        onClick={() => handleCopyResponse(msg.answer, idx)}
                      >
                        <Copy className={`w-6 h-6 ${copiedIndex === idx ? 'text-green-600' : ''}`} />
                      </button>

                      {/* EDIT */}
                      <button
                        className="text-[#66991D] hover:text-green-900 cursor-pointer transition-colors p-1 rounded"
                        title="Edit"
                      >
                        <PencilLine className="w-6 h-6" />
                      </button>

                      {/* SOURCES */}
                      <button
                        className="text-[#66991D] hover:text-green-900 cursor-pointer transition-colors p-1 rounded"
                        title="Sources"
                        onClick={() => setShowContext(showContext === idx ? null : idx)}
                      >
                        Sources
                      </button>

                      <div className="ml-auto text-xs text-gray-400 italic">
                        Generated by Perch — verify before using in policy or legal contexts.
                      </div>
                    </div>
                  )}

                  {showContext === idx && msg.context && (
                    <div>
                        <h3 className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
                        <FolderSearch className="w-4 h-4" />
                        Retrieved Research & Policy Documents
                        </h3>
                        <ul className="space-y-4">
                          {Array.from(
                            new Map(
                              msg.context.map((doc: any) => [
                                doc.metadata?.source_name || "Untitled Document",
                                doc
                              ])
                            ).values()
                          ).map((doc: any, cidx: number) => {
                            const url = doc.metadata?.source_url || "";
                            const isUrl = url.startsWith("http"); // Check if it's a web link or a file path
                            const content = (
                              <div className="font-bold text-gray-900">
                                {doc.metadata?.source_name || "Untitled"} | {doc.metadata?.source_organization || ""}
                              </div>
                            );return (
                              <li key={cidx} className="text-sm border-l-4 border-green-500 pl-4 py-1">
                                {isUrl ? (
                                  <a href={url} target="_blank" rel="noopener noreferrer">
                                    {content}
                                  </a>
                                ) : (
                                  content
                                )}
                              </li>
                            );
                          }
                          )}
                        </ul>
                      </div>
                  )}
                </div>
              ))}
            </div>
            <div ref={bottomRef} /> {/* scroll anchor — keeps view at bottom during streaming */}
          </div>
        )}

        {/* PROMPT SUGGESTIONS (only when no chat history) */}
        {chatHistory.length === 0 && (
          <div className="flex-shrink-0 px-4">
            {/* INSTRUCTIONS */}
            <div className="w-full flex justify-center pt-12 pb-4 text-gray-500 font-semibold text-md">
              Don't know where to start? Here are some examples of things you can ask me:
            </div>

            {/* PROMPT SUGGESTIONS */}
            <div className="grid grid-cols-2 gap-4 max-w-4xl mx-auto mt-4 mb-32">
              {promptSuggestions.map((p) => (
                <div
                  key={p.id}
                  className="border-[#D7E8CD] border-2 rounded-4xl p-3 text-gray-500 text-sm cursor-pointer hover:bg-gray-100 transition"
                  onClick={() => onPromptClick(p.text.replace(/\*\*/g, ""))}
                >
                  <div className="flex items-center justify-center gap-2">
                    {iconMap[p.icon]}
                    <ReactMarkdown>{p.text}</ReactMarkdown>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* FIXED INPUT FIELD (bottom, only if answer exists) */}
      {chatHistory.length > 0 && (
        <div
          className="fixed bottom-8 bg-white border-[#D7E8CD] flex-shrink-0"
          style={{ left: "15.5rem", width: "calc(100% - 15.5rem)" }}
        >
          <div className="max-w-5xl mx-auto px-4">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="w-full relative">
                <input
                  className="w-full min-w-0 px-4 py-4 pb-26 pr-12 text-md border border-[#D7E8CD] shadow-md rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask me anything"
                />
                <button
                  type="submit"
                  className="absolute bottom-2 right-2 bg-black rounded-full p-2 flex items-center justify-center hover:bg-gray-700 transition cursor-pointer disabled:bg-gray-300 disabled:cursor-default"
                  aria-label="Send"
                  disabled={!question.trim()}
                >
                  <ArrowUp className="w-5 h-5 text-white" />
                </button>
              </div>
            </form>
            <div className="text-sm text-center text-gray-500 font-medium pt-4">
              Perch can make mistakes. Check important information.
            </div>
          </div>
        </div>
      )}

      {/* Snackbar for copy notification */}
      {showSnackbar && (
        <div className="fixed bottom-24 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg z-50 transition-all duration-300 ease-in-out">
          <div className="flex items-center gap-2">
            <Copy className="w-4 h-4" />
            <span className="text-sm font-medium">Copied to clipboard!</span>
          </div>
        </div>
      )}

    </div>
  );
}
