import React from 'react';
import { FaSpider, FaBug, FaExclamationTriangle } from 'react-icons/fa';

const PestDetection = () => {
  return (
    <div className="pest-layout">
      <div className="card">
        <div className="card-header" style={{ color: '#d97706' }}>
          AI Pest Identification
        </div>
        <div 
          className="upload-box pest-upload-box" 
          style={{ height: '100%', background: '#fffbeb', borderColor: '#fcd34d' }}
        >
          <div style={{ 
            width: '80px', height: '80px', background: '#fef3c7', 
            borderRadius: '50%', display: 'flex', alignItems: 'center', 
            justifyContent: 'center', marginBottom: '20px' 
          }}>
            <FaSpider size={40} color="#d97706" />
          </div>
          <p style={{ fontSize: '1.4rem', color: '#b45309', fontWeight: '700' }}>Identify Pests</p>
          <p style={{ color: '#92400e', marginBottom: '15px' }}>
            Upload a photo of the insect or damage
          </p>
          <button style={{ 
            padding: '10px 20px', background: '#d97706', color: 'white', 
            border: 'none', borderRadius: '10px', fontWeight: 600, cursor: 'pointer' 
          }}>
            Browse Image
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-header">Common Pests</div>
        <div className="encyclopedia-item">
          <div className="icon-circle" style={{ background: '#d97706', width: '30px', height: '30px', fontSize: '0.8rem' }}>
            <FaBug />
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: '0.9rem', color: '#1e1b4b' }}>Locusts</div>
            <span style={{ fontSize: '0.75rem', color: '#ef4444' }}>High Threat</span>
          </div>
        </div>
        
        <div className="encyclopedia-item">
          <div className="icon-circle" style={{ background: '#d97706', width: '30px', height: '30px', fontSize: '0.8rem' }}>
            <FaBug />
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: '0.9rem', color: '#1e1b4b' }}>Aphids</div>
            <span style={{ fontSize: '0.75rem', color: '#f59e0b' }}>Medium Threat</span>
          </div>
        </div>

        <div className="encyclopedia-item">
          <div className="icon-circle" style={{ background: '#d97706', width: '30px', height: '30px', fontSize: '0.8rem' }}>
            <FaBug />
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: '0.9rem', color: '#1e1b4b' }}>Fall Armyworm</div>
            <span style={{ fontSize: '0.75rem', color: '#ef4444' }}>High Threat</span>
          </div>
        </div>

        <div style={{ 
          marginTop: 'auto', padding: '10px', background: '#fffbeb', 
          borderRadius: '8px', fontSize: '0.85rem', color: '#92400e' 
        }}>
          <FaExclamationTriangle style={{ marginRight: '5px' }} /> 
          Tip: Check underside of leaves.
        </div>
      </div>
    </div>
  );
};

export default PestDetection;