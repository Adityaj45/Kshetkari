import React, { useState, useRef } from 'react';
import { 
  FaSearch, FaLeaf, FaCamera, FaClipboardList, FaTag, 
  FaExclamationTriangle, FaCheckCircle, FaClock, FaChartBar 
} from 'react-icons/fa';

// 1. Define the Class Names Array (mapped to indices 0-16)
const CLASS_NAMES = [
  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
  "Corn_(maize)___Common_rust_",
  "Corn_(maize)___healthy",
  "Corn_(maize)___Northern_Leaf_Blight",
  "Potato___Early_blight",
  "Potato___healthy",
  "Potato___Late_blight",
  "Tomato___Bacterial_spot",
  "Tomato___Early_blight",
  "Tomato___healthy",
  "Tomato___Late_blight",
  "Tomato___Leaf_Mold",
  "Tomato___Septoria_leaf_spot",
  "Tomato___Spider_mites Two-spotted_spider_mite",
  "Tomato___Target_Spot",
  "Tomato___Tomato_mosaic_virus",
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
];

const DiseaseDetection = () => {
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const fileInputRef = useRef(null);

  // Helper: Format the messy class name into readable text
  const formatClassName = (name) => {
    return name.replace(/_/g, ' ').replace(/   /g, ' - ').trim();
  };

  // Helper: Format Timestamp
  const formatTime = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleString(); // e.g. "2/6/2026, 2:31:20 AM"
  };

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const previewUrl = URL.createObjectURL(file);
    setSelectedImage(previewUrl);
    await uploadImage(file);
  };

  const uploadImage = async (file) => {
    setIsScanning(true);
    setScanResult(null);

    const formData = new FormData();
    formData.append('file', file); 

    try {
      const response = await fetch('http://127.0.0.1:8000/PlantDetection', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.status === 'success') {
        const pred = data.prediction;
        
        // 2. Map the Index to the Name
        const detectedName = CLASS_NAMES[pred.class_index] || "Unknown Class";
        const formattedName = formatClassName(detectedName);
        const confidencePct = (pred.confidence * 100).toFixed(2);

        // 3. Create the Result Object with ALL data
        const resultObject = {
          displayName: formattedName,
          rawLabel: pred.label,
          index: pred.class_index,
          confidence: confidencePct,
          timestamp: formatTime(pred.timestamp),
          treatment: data.recommendation
        };
        
        setScanResult(resultObject);
      } else {
        alert("Detection failed. Please try again.");
      }
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Error connecting to server. Is the backend running?");
    } finally {
      setIsScanning(false);
    }
  };

  const resetScan = () => {
    setScanResult(null);
    setSelectedImage(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const triggerFileUpload = () => {
    fileInputRef.current.click();
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', gap: '30px', flexWrap: 'wrap' }}>
      
      <input type="file" ref={fileInputRef} style={{ display: 'none' }} accept="image/*" onChange={handleFileChange} />

      {/* LEFT COLUMN: Upload/Scan Action */}
      <div className="card" style={{ flex: 1, minWidth: '350px', height: 'auto' }}>
        <div className="card-header" style={{ fontSize: '1.3rem' }}>Upload Leaf Image</div>
        
        <div 
          className="upload-box" 
          onClick={!isScanning ? triggerFileUpload : undefined} 
          style={{ height: '400px', cursor: isScanning ? 'default' : 'pointer', transition: '0.3s', position: 'relative', overflow: 'hidden' }}
        >
          {selectedImage && !isScanning && !scanResult && (
             <img src={selectedImage} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'cover', position: 'absolute', top: 0, left: 0, opacity: 0.3 }} />
          )}

          {isScanning ? (
            <div style={{ textAlign: 'center', animation: 'fadeIn 0.5s', position: 'relative', zIndex: 2 }}>
              <div style={{ marginBottom: '20px', color: '#AEB877' }}><FaSearch size={50} className="pulse-icon" /></div>
              <h3 style={{ color: '#1e293b' }}>Analyzing Image...</h3>
              <p style={{ color: '#64748b' }}>Processing AI Prediction...</p>
            </div>
          ) : scanResult ? (
            <div style={{ textAlign: 'center', width: '100%', animation: 'fadeIn 0.5s', position: 'relative', zIndex: 2 }}>
              <div style={{ width: '150px', height: '150px', background: '#ecfdf5', borderRadius: '15px', margin: '0 auto 20px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px solid #AEB877' }}>
                <FaLeaf size={70} color="#AEB877" />
              </div>
              <h3 style={{ color: '#AEB877' }}>Analysis Complete!</h3>
              <p style={{ color: '#64748b' }}>Results generated successfully.</p>
              <button onClick={(e) => { e.stopPropagation(); resetScan(); }} style={{ marginTop: '25px', padding: '10px 25px', background: '#e2e8f0', color: '#4b5563', border: 'none', borderRadius: '12px', fontWeight: '600', cursor: 'pointer' }}>
                Scan Another
              </button>
            </div>
          ) : (
            <div style={{ position: 'relative', zIndex: 2, textAlign: 'center' }}>
              <div style={{ width: '80px', height: '80px', background: '#f0fdf4', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '20px', margin: '0 auto 20px' }}><FaCamera size={40} color="#AEB877" /></div>
              <p style={{ fontSize: '1.4rem', color: '#064e3b', fontWeight: '700' }}>Upload Leaf Image</p>
              <p style={{ color: '#64748b' }}>Click to browse and scan</p>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT COLUMN: Analysis Report */}
      <div className="card" style={{ flex: 1.2, minWidth: '350px', maxHeight: '600px', overflowY: 'auto' }}>
        <div className="card-header" style={{ fontSize: '1.3rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '15px' }}><FaClipboardList /> Analysis Report</div>
        
        {scanResult ? (
          <div style={{ padding: '10px', animation: 'slideUp 0.5s' }}>
            
            {/* 1. Identification Section */}
            <div style={{ marginBottom: '25px' }}>
              <label style={{ display: 'block', color: '#64748b', fontSize: '0.9rem', marginBottom: '10px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                <FaTag style={{ marginRight: '5px' }} /> Identification
              </label>
              <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                {/* Main Result Name */}
                <div style={{ fontSize: '1.3rem', fontWeight: '800', color: '#1e293b', marginBottom: '5px', lineHeight: '1.4' }}>
                  {scanResult.displayName}
                </div>
                {/* Metadata Row */}
                <div style={{ display: 'flex', gap: '15px', fontSize: '0.85rem', color: '#64748b', marginTop: '10px', flexWrap: 'wrap' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                    <FaExclamationTriangle color="#f59e0b" /> Index: {scanResult.index}
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                    <FaClock color="#3b82f6" /> {scanResult.timestamp}
                  </span>
                </div>
              </div>
            </div>

            {/* 2. Confidence Meter */}
            <div style={{ marginBottom: '25px' }}>
               <label style={{ display: 'block', color: '#64748b', fontSize: '0.9rem', marginBottom: '10px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                <FaChartBar style={{ marginRight: '5px' }} /> Confidence Analysis
              </label>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontWeight: '600', color: '#1e293b' }}>AI Certainty</span>
                <span style={{ fontWeight: '700', color: '#AEB877' }}>{scanResult.confidence}%</span>
              </div>
              <div style={{ width: '100%', height: '10px', background: '#e2e8f0', borderRadius: '5px' }}>
                <div style={{ width: `${scanResult.confidence}%`, height: '100%', background: '#AEB877', borderRadius: '5px', transition: 'width 1s ease-out' }}></div>
              </div>
              <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '8px' }}>
                Raw Prediction Label: "{scanResult.rawLabel}"
              </p>
            </div>

            {/* 3. Recommendation Section */}
            <div style={{ background: '#ecfdf5', padding: '20px', borderRadius: '15px', border: '1px solid #A5C89E' }}>
              <label style={{ display: 'block', color: '#064e3b', fontSize: '1rem', marginBottom: '10px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FaCheckCircle color="#059669" /> Recommendation
              </label>
              <p style={{ color: '#064e3b', lineHeight: '1.6', fontSize: '1rem', fontWeight: '500' }}>
                {scanResult.treatment}
              </p>
            </div>

          </div>
        ) : (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', minHeight: '300px' }}>
            <FaClipboardList size={50} style={{ marginBottom: '15px', opacity: 0.3 }} />
            <p>No data available.</p>
            <p style={{ fontSize: '0.9rem' }}>Upload an image to see the report.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DiseaseDetection;