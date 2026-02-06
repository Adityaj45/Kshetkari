import React, { useState, useRef, useEffect } from 'react';
import { FaTimes, FaPaperPlane, FaCommentDots } from 'react-icons/fa';

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: "Namaste! I am U.D.A.Y. How can I help you today?", sender: "ai" },
  ]);
  const [input, setInput] = useState("");
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isOpen]);

  const sendMessage = () => {
    if (!input.trim()) return;
    const newMsg = { id: messages.length + 1, text: input, sender: "user" };
    setMessages((prev) => [...prev, newMsg]);
    setInput("");
    
    // Simulate AI Response
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        { 
          id: prev.length + 1, 
          text: "I have analyzed the parameters. U.D.A.Y. suggests using organic mulch to retain soil moisture.", 
          sender: "ai" 
        },
      ]);
    }, 1200);
  };

  return (
    <>
      {isOpen && (
        <div className="chat-popup">
          <div className="chat-header" onClick={() => setIsOpen(false)}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <div style={{ background: "white", padding: "4px", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <img src="https://img.icons8.com/color/96/robot-2.png" alt="Bot" style={{ width: "32px", height: "32px", objectFit: "contain" }} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: "1rem" }}>U.D.A.Y.</h3>
                <div style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "0.75rem", opacity: 0.9 }}>
                  <div className="online-dot" style={{ width: "8px", height: "8px" }}></div> Online
                </div>
              </div>
            </div>
            <FaTimes />
          </div>
          
          <div className="chat-messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`message-wrapper ${msg.sender}`}>
                {msg.sender === "ai" && (
                  <div className="chat-avatar">
                    <img src="https://img.icons8.com/color/96/robot-2.png" alt="Bot" />
                  </div>
                )}
                <div className={`message ${msg.sender}`}>{msg.text}</div>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          <div className="chat-input-area">
            <input
              type="text" 
              className="chat-input" 
              placeholder="Ask anything..."
              value={input} 
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && sendMessage()}
            />
            <button className="send-btn" onClick={sendMessage}>
              <FaPaperPlane size={14} />
            </button>
          </div>
        </div>
      )}
      
      <div className="floating-chat-btn" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? <FaTimes size={24} /> : <FaCommentDots size={28} />}
        {!isOpen && <div className="notification-dot"></div>}
      </div>
    </>
  );
};

export default ChatWidget;