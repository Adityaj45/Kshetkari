import React, { useState, useEffect } from 'react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

const YieldForecast = () => {
  // 1. State for Form Inputs
  const [soilData, setSoilData] = useState({ N: 80, P: 45, K: 30, ph: 6.5 });
  const [prediction, setPrediction] = useState(0.63); // Default value from your example
  const [loading, setLoading] = useState(false);

  // 2. Fetch Weather & Predict Function
  const handleAnalyze = async () => {
    setLoading(true);
    try {
      // First, get the current weather for rainfall/temp/humidity
      const weatherRes = await fetch(
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Mumbai?unitGroup=metric&key=EXCNRQ7ZZ6XGB5KGN7HGNHGWT&contentType=json"
      );
      const weather = await weatherRes.json();
      const current = weather.currentConditions;

      // Second, POST to your local model
      const response = await fetch('http://127.0.0.1:8000/cropYieldPrediction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...soilData,
          temperature: current.temp,
          rainfall: weather.days[0].precip || 0, // Taking precip from today's forecast
          humidity: current.humidity,
          year: 2026 // Current Year
        }),
      });

      const result = await response.json();
      setPrediction(result.predicted_yield);
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setLoading(false);
    }
  };

  // 3. Dynamic Chart Data
  // Normalizing the yield for the visual chart (assuming a max scale of 10 tons for the 100% fill)
  const yieldChartData = {
    labels: ['Yield Potential', 'Gap'],
    datasets: [{
      data: [prediction, Math.max(0, 10 - prediction)], 
      backgroundColor: ['#AEB877', '#f1f5f9'],
      borderWidth: 0,
      cutout: '75%',
    }],
  };

  return (
    <div className="yield-container" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', padding: '20px' }}>
      <div className="card">
        <div className="card-header">Soil Parameters</div>
        <form className="form-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginTop: '15px' }}>
          {Object.keys(soilData).map((key) => (
            <div key={key}>
              <label className="login-label" style={{ display: 'block', marginBottom: '5px', textTransform: 'capitalize' }}>
                {key === 'ph' ? 'pH Level' : key}
              </label>
              <input 
                type="number" 
                className="yield-input" 
                value={soilData[key]}
                onChange={(e) => setSoilData({...soilData, [key]: parseFloat(e.target.value)})}
                style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #e2e8f0' }}
              />
            </div>
          ))}
          <button 
            type="button" 
            className="predict-btn" 
            onClick={handleAnalyze}
            disabled={loading}
            style={{ gridColumn: 'span 2', background: '#AEB877', color: 'white', padding: '12px', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}
          >
            {loading ? 'Analyzing...' : 'Analyze Yield'}
          </button>
        </form>
      </div>

      <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center' }}>
        <div style={{ width: '220px', height: '220px', marginBottom: '20px' }}>
          <Doughnut data={yieldChartData} options={{ plugins: { legend: { display: false } } }} />
        </div>
        <h2 style={{ color: '#1e1b4b', margin: '0' }}>Projected Yield</h2>
        <div style={{ fontSize: '3.5rem', fontWeight: '800', color: '#AEB877', lineHeight: 1, margin: '10px 0' }}>
          {prediction.toFixed(2)} <span style={{ fontSize: '1.2rem', color: '#94a3b8' }}>tons</span>
        </div>
        <p style={{ color: '#64748b' }}>Target achieved based on current soil health</p>
      </div>
    </div>
  );
};

export default YieldForecast;