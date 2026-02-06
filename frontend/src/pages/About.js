import React from 'react';
import { FaHandsHelping, FaRocket, FaRobot, FaGlobe } from 'react-icons/fa';

const About = () => {
  return (
    <div className="about-container">
      <div className="about-hero">
        <h1>About FarmGuard</h1>
        <p>Empowering Farmers with Smart Technology</p>
      </div>
      <div className="about-grid">
        <div className="about-card">
          <div className="about-icon-box"><FaHandsHelping /></div>
          <h3>Who We Are</h3>
          <p>We are a technology-driven initiative focused on empowering farmers. Our platform combines artificial intelligence and real-time data to help farmers make better decisions for healthier crops.</p>
        </div>
        <div className="about-card">
          <div className="about-icon-box"><FaRocket /></div>
          <h3>How It Works</h3>
          <p>Simply upload an image of a leaf. Our AI analyzes it to detect pests and diseases. We provide recommended actions, preventive measures, and care tips in simple language.</p>
        </div>
        <div className="about-card">
          <div className="about-icon-box"><FaRobot /></div>
          <h3>AI Support</h3>
          <p>Our built-in U.D.A.Y. chatbot answers farming questions 24/7. It offers guidance on crop care, weather planning, and assists with follow-up concerns instantly.</p>
        </div>
        <div className="about-card">
          <div className="about-icon-box"><FaGlobe /></div>
          <h3>Our Vision</h3>
          <p>We believe technology should work with farmers, for farmers. We aim to reduce crop loss and make modern agricultural knowledge available to everyone.</p>
        </div>
      </div>
    </div>
  );
};

export default About;