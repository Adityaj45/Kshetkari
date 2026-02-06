import React from 'react';
import { FaChevronDown } from 'react-icons/fa';

const FAQ = ({ faqData, expandedFaq, toggleFaq }) => {
  return (
    <div className="faq-container">
      <div className="faq-header">
        <h1>Frequently Asked Questions</h1>
        <p style={{ color: '#64748b' }}>Get answers to your farming queries</p>
      </div>
      {faqData.map((item, index) => (
        <div key={index} className={`faq-item ${expandedFaq === index ? 'active' : ''}`}>
          <div className="faq-question" onClick={() => toggleFaq(index)}>
            {index + 1}. {item.q}
            <FaChevronDown className="faq-icon" />
          </div>
          <div className="faq-answer">
            {item.a}
          </div>
        </div>
      ))}
    </div>
  );
};

export default FAQ;