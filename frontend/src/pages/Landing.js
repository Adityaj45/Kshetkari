import React from 'react';
import { 
  FaSeedling, 
  FaArrowRight, 
  FaCamera, 
  FaChartLine, 
  FaSpider, 
  FaRobot 
} from 'react-icons/fa';

/**
 * Landing Page Component
 * @param {Function} onLoginClick - Function to switch from Landing to Login view
 */
const Landing = ({ onLoginClick }) => {
  return (
    <div className="landing-container">
      {/* Navigation Bar */}
      <nav className="landing-nav">
        <div className="nav-logo">
          <FaSeedling /> FarmGuard
        </div>
        <button className="nav-btn" onClick={onLoginClick}>
          Login
        </button>
      </nav>

      {/* Hero Section */}
      <header className="hero-section">
        <h1 className="hero-title">
          Smart AI for <br /> Smarter Farming
        </h1>
        <p className="hero-subtitle">
          Empower your farm with real-time disease detection, pest prediction, 
          and AI-driven yield forecasts. The future of agriculture is here.
        </p>
        <button className="cta-btn" onClick={onLoginClick}>
          Get Started 
          <FaArrowRight style={{ marginLeft: '10px', fontSize: '0.9em' }} />
        </button>
      </header>

      {/* Features Grid */}
      <section className="features-section">
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">
              <FaCamera />
            </div>
            <h3>Disease Detection</h3>
            <p>Instantly diagnose plant diseases by simply uploading a photo of a leaf.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">
              <FaChartLine />
            </div>
            <h3>Yield Forecasting</h3>
            <p>Predict your harvest yield with high accuracy using soil and weather data.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">
              <FaSpider />
            </div>
            <h3>Pest Prediction</h3>
            <p>Get alerts about potential pest outbreaks before they damage your crops.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">
              <FaRobot />
            </div>
            <h3>AI Assistant</h3>
            <p>Chat with U.D.A.Y., your personal 24/7 agricultural expert.</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <p>&copy; {new Date().getFullYear()} FarmGuard. Built for the Future of Farming.</p>
      </footer>
    </div>
  );
};

export default Landing;