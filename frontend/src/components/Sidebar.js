import React from 'react';
import { 
  FaSeedling, FaThLarge, FaLeaf, FaChartLine, 
  FaBug, FaInfoCircle, FaQuestionCircle, FaTimes 
} from 'react-icons/fa';

/**
 * Sidebar Component for Navigation
 * @param {boolean} isOpen - Controls visibility on mobile
 * @param {function} onClose - Closes sidebar (mobile)
 * @param {string} activeTab - Currently selected tab id
 * @param {function} setActiveTab - Navigation handler
 * @param {string} userName - Displayed in user profile section
 */
const Sidebar = ({ isOpen, onClose, activeTab, setActiveTab, userName }) => {
  const menuItems = [
    { id: "dashboard", label: "Dashboard", icon: <FaThLarge /> },
    { id: "disease", label: "Disease Detect", icon: <FaLeaf /> },
    { id: "pest", label: "Pest Detection", icon: <FaBug /> },
    { id: "yield", label: "Yield Forecast", icon: <FaChartLine /> },
    { id: "about", label: "About Us", icon: <FaInfoCircle /> },
    { id: "faq", label: "FAQ", icon: <FaQuestionCircle /> },
  ];

  return (
    <>
      {/* Overlay for mobile view when sidebar is open */}
      <div 
        className={`sidebar-overlay ${isOpen ? "visible" : ""}`} 
        onClick={onClose}
      ></div>

      <div className={`sidebar ${isOpen ? "open" : ""}`}>
        {/* Logo and Mobile Close Button */}
        <div className="logo-row" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div className="logo">
            <FaSeedling /> FarmGuard
          </div>
          <FaTimes className="mobile-close-btn" onClick={onClose} />
        </div>

        {/* Navigation Links */}
        <div className="nav-menu">
          {menuItems.map((item) => (
            <div
              key={item.id}
              className={`nav-item ${activeTab === item.id ? "active" : ""}`}
              onClick={() => { 
                setActiveTab(item.id); 
                onClose(); // Automatically close sidebar on mobile after clicking
              }}
            >
              {item.icon} {item.label}
            </div>
          ))}
        </div>

        {/* User Profile Section */}
        <div className="user-profile">
          <div className="avatar-circle">
            {userName ? userName.charAt(0).toUpperCase() : 'U'}
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: "0.95rem" }}>{userName}</div>
            <div style={{ fontSize: "0.75rem", opacity: 0.8 }}>Premium Plan</div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;