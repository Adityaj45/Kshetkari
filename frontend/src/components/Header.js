import React from 'react';
import { FaBars, FaCalendarAlt } from 'react-icons/fa';

/**
 * Header Component
 * @param {string} activeTab - The current page ID
 * @param {string} userName - The logged-in user's name
 * @param {function} onOpenSidebar - Function to trigger mobile sidebar
 */
const Header = ({ activeTab, userName, onOpenSidebar }) => {
  const getTodayDate = () => {
    return new Date().toLocaleDateString("en-US", { 
      year: "numeric", 
      month: "short", 
      day: "numeric" 
    });
  };

  const getTitle = () => {
    const titles = {
      dashboard: "Dashboard",
      yield: "Yield Forecast",
      disease: "Disease Detection",
      pest: "Pest Detection",
      about: "About Us",
      faq: "FAQ"
    };
    return titles[activeTab] || activeTab;
  };

  return (
    <div className="header">
      <div className="header-left">
        <button className="mobile-menu-btn" onClick={onOpenSidebar}>
          <FaBars />
        </button>
        <div>
          <h1>{getTitle()}</h1>
          <p style={{ color: "#64748b" }}>Welcome back, {userName}</p>
        </div>
      </div>
      <button className="date-btn">
        <FaCalendarAlt /> {getTodayDate()}
      </button>
    </div>
  );
};

export default Header;