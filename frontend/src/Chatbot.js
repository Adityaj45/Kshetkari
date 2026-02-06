import React, { useState, useRef, useEffect } from 'react';
import { FaRobot, FaPaperPlane, FaTimes, FaCommentDots } from 'react-icons/fa';
import './Chatbot.css';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: "Namaste! I am U.D.A.Y., your personal agriculture assistant. How can I help you improve your yield today?", sender: "ai" }
  ]);
  const [chatInput, setChatInput] = useState("");
  const chatEndRef = useRef(null);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const sendMessage = () => {
    if (!chatInput.trim()) return;
    const newMsg = { id: messages.length + 1, text: chatInput, sender: "user" };
    setMessages([...messages, newMsg]);
    setChatInput("");

    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: prev.length + 1,
        text: "I have analyzed the parameters. U.D.A.Y. suggests using organic mulch to retain soil moisture during this heatwave.",
        sender: "ai"
      }]);
    }, 1200);
  };

  useEffect(() => {
    if (isOpen) {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isOpen]);

  return (
    <div className="chatbot-container">
      {/* Floating Action Button */}
      <div className={`chat-fab ${isOpen ? 'open' : ''}`} onClick={toggleChat}>
        {isOpen ? <FaTimes /> : <FaCommentDots />}
      </div>

      {/* Chat Window Popup */}
      {isOpen && (
        <div className="chat-popup">
          <div className="chat-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ background: 'white', padding: '6px', borderRadius: '50%', color: '#AEB877', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <FaRobot size={18} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1rem', color: 'white' }}>U.D.A.Y.</h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.7rem', opacity: 0.9, color: 'white' }}>
                  <div className="online-dot" style={{ background: '#D8E983' }}></div> Online
                </div>
              </div>
            </div>
            <div className="close-btn" onClick={toggleChat}><FaTimes /></div>
          </div>

          <div className="chat-messages">
            {messages.map(msg => (
              <div key={msg.id} className={`message ${msg.sender}`}>
                {msg.text}
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          <div className="chat-input-area">
            <input
              type="text"
              className="chat-input"
              placeholder="Ask U.D.A.Y..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button className="send-btn" onClick={sendMessage}><FaPaperPlane /></button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
