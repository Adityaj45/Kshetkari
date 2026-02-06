import React, { useState, useEffect, useCallback } from 'react';
import { Line } from 'react-chartjs-2';
import { 
  FaCamera, FaBell, FaTemperatureHigh, FaBug, FaFileAlt, 
  FaCloudSun, FaSun, FaChartArea, FaCloudUploadAlt 
} from 'react-icons/fa';

// --- API UTILS ---
export function getKey(){
  return "EXCNRQ7ZZ6XGB5KGN7HGNHGWT";
}

export function getUrl(){
  return "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/";
}

const Dashboard = ({ userName, setActiveTab }) => {
  const [weatherData, setWeatherData] = useState(null);
  const [pestRisk, setPestRisk] = useState({ risk: "Loading...", probabilities: [] });

  // 1. Fetch Weather Data
  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const response = await fetch(`${getUrl()}Mumbai?unitGroup=metric&key=${getKey()}&contentType=json`);
        const data = await response.json();
        setWeatherData(data.currentConditions);
      } catch (error) {
        console.error("Weather fetch failed:", error);
      }
    };
    fetchWeather();
  }, []);

  // 2. Fetch Pest Risk from LSTM Model
  const fetchPestRisk = useCallback(async () => {
    if (!weatherData) return;

    try {
      const response = await fetch('http://127.0.0.1:8000/PestRiskModel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          temperature: weatherData.temp,
          humidity: weatherData.humidity,
          wind_speed: weatherData.windspeed,
          pressure: weatherData.pressure || 1013 // Fallback if pressure is missing
        }),
      });
      const riskResult = await response.json();
      setPestRisk(riskResult);
    } catch (error) {
      console.error("Pest API failed:", error);
    }
  }, [weatherData]);

  useEffect(() => {
    fetchPestRisk();
  }, [fetchPestRisk]);

  // --- CHART CONFIGURATION ---
  // Using the probabilities from the API to populate the chart
  const dashboardChartData = {
    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
    datasets: [{
      label: 'LSTM Probability Distribution',
      data: pestRisk.probabilities.length > 0 
            ? pestRisk.probabilities.map(p => (p * 100).toFixed(2)) 
            : [0, 0, 0],
      borderColor: '#AEB877',
      backgroundColor: 'rgba(174, 184, 119, 0.2)',
      borderWidth: 3,
      pointBackgroundColor: '#ffffff',
      pointBorderColor: '#AEB877',
      pointRadius: 6,
      fill: true,
      tension: 0.4
    }]
  };

  const dashboardChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { backgroundColor: 'rgba(30, 41, 59, 0.9)' }
    },
    scales: {
      y: { beginAtZero: true, ticks: { callback: (val) => `${val}%` } }
    }
  };

  return (
    <div className="dashboard-grid">
      <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
        {/* Upload Card */}
        <div className="card" style={{ height: '380px' }}>
          <div className="card-header">Scan Crop <FaCamera /></div>
          <div className="upload-box" onClick={() => setActiveTab('disease')}>
            <FaCloudUploadAlt size={50} style={{ marginBottom: '15px' }} />
            <p style={{ fontSize: '1.1rem', fontWeight: 600 }}>Upload Leaf Image</p>
          </div>
        </div>

        {/* Live Alerts Card */}
        <div className="card">
          <div className="card-header">Live Alerts <FaBell /></div>
          <div className="alert-item pest">
            <div className="icon-circle" style={{ 
              background: pestRisk.risk === 'High' ? '#ef4444' : '#f59e0b' 
            }}><FaBug /></div>
            <div>
              <div style={{ fontWeight: 700, color: '#1e1b4b' }}>
                Pest Risk: {pestRisk.risk}
              </div>
              <div style={{ fontSize: '0.85rem', color: '#64748b' }}>
                LSTM Model suggests {pestRisk.risk.toLowerCase()} monitoring.
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '30px', gridColumn: 'span 2' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
          <div className="card">
            <div className="card-header">Yield Forecast <FaFileAlt /></div>
            <div className="metric-value">6.4 <span style={{ fontSize: '1.2rem', color: '#94a3b8' }}>tons</span></div>
            <span className={`badge-risk ${pestRisk.risk === 'High' ? 'high' : ''}`}>
              RISK: {pestRisk.risk.toUpperCase()}
            </span>
          </div>

          <div className="card">
            <div className="card-header">Field Conditions <FaCloudSun /></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '2.5rem', fontWeight: 800 }}>
                  {weatherData ? `${Math.round(weatherData.temp)}°C` : '--°C'}
                </div>
                <div style={{ color: '#ef4444', fontWeight: 600 }}>{weatherData?.conditions}</div>
              </div>
              <FaSun size={45} color="#D8E983" />
            </div>
          </div>
        </div>

        {/* Chart Card */}
        <div className="card" style={{ flex: 1, minHeight: '300px' }}>
          <div className="card-header">LSTM Pest Risk Probability <FaChartArea /></div>
          <div style={{ height: '100%', width: '100%' }}>
            <Line data={dashboardChartData} options={dashboardChartOptions} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;