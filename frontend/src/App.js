import React, { useState } from "react";
import "./App.css";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  ArcElement,
} from "chart.js";

// Global Components
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import ChatWidget from "./components/ChatWidget";

// Pages
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import DiseaseDetection from "./pages/DiseaseDetection";
import YieldForecast from "./pages/YieldForecast";
import PestDetection from "./pages/PestDetection";
import About from "./pages/About";
import FAQ from "./pages/FAQ";

// --- CHART REGISTRATION ---
ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Filler, ArcElement
);

// --- STATIC DATA ---
const FAQ_DATA = [
  { q: "How does the crop disease detection work?", a: "You upload a clear image of a leaf or crop. Our AI system analyzes the image using trained models to identify possible pests, diseases, or deficiencies." },
  { q: "What kind of problems can the system detect?", a: "The system can identify common leaf diseases, pest damage, fungal infections, and nutrient-related issues." },
  { q: "How accurate are the results?", a: "Our system is trained on agricultural datasets and provides highly reliable predictions. However, results should be used as guidance." },
  { q: "What type of images should I upload?", a: "Upload a clear, well-lit image of the affected leaf or crop." },
  { q: "Does the platform provide treatment recommendations?", a: "Yes. After detection, we provide suggested actions, preventive measures, and care tips." },
  { q: "How is weather information useful?", a: "Weather data helps farmers plan watering, pesticide spraying, fertilizer use, and harvesting." },
  { q: "What does the chatbot help with?", a: "Our chatbot can answer farming questions, explain disease reports, and suggest best practices." },
  { q: "Is this platform free?", a: "Yes, basic features are free so farmers can easily access essential crop support." },
  { q: "Do I need technical knowledge?", a: "No. The system is designed to be simple and user-friendly." },
  { q: "Can this platform replace experts?", a: "The platform is a support tool. For serious crop damage, consulting an agricultural specialist is recommended." },
];

// --- CUSTOM HOOKS ---
const useScanner = () => {
  const [scanResult, setScanResult] = useState(null);
  const [isScanning, setIsScanning] = useState(false);

  const handleSimulateScan = () => {
    if (scanResult) return;
    setIsScanning(true);
    setTimeout(() => {
      setIsScanning(false);
      setScanResult({
        plantType: "Tomato (Solanum lycopersicum)",
        matches: [
          { name: "Tomato", confidence: 94.2 },
          { name: "Potato", confidence: 3.5 },
          { name: "Bell Pepper", confidence: 2.3 },
        ],
        diseaseName: "Early Blight (Alternaria solani)",
        treatment: "Apply copper-based fungicides immediately. Ensure proper spacing between plants to improve air circulation and reduce humidity. Remove affected leaves to prevent spread.",
      });
    }, 2500);
  };

  const resetScan = (e) => {
    if (e) e.stopPropagation();
    setScanResult(null);
    setIsScanning(false);
  };

  return { scanResult, isScanning, handleSimulateScan, resetScan };
};

// --- MAIN APP ---
function App() {
  // Navigation & Auth State
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [showLanding, setShowLanding] = useState(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [userName, setUserName] = useState("");
  
  // Login Form State
  const [inputName, setInputName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // FAQ State
  const [expandedFaq, setExpandedFaq] = useState(null);

  // Scanner Logic
  const scanner = useScanner();

  // Handlers
  const toggleFaq = (index) => setExpandedFaq(expandedFaq === index ? null : index);
  
  const renderContent = () => {
    switch (activeTab) {
      case "dashboard": return <Dashboard userName={userName} setActiveTab={setActiveTab} />;
      case "disease":   return <DiseaseDetection {...scanner} />;
      case "pest":      return <PestDetection />;
      case "yield":     return <YieldForecast />;
      case "about":     return <About />;
      case "faq":       return <FAQ faqData={FAQ_DATA} expandedFaq={expandedFaq} toggleFaq={toggleFaq} />;
      default:          return <div>Page not found</div>;
    }
  };

  // --- VIEW LOGIC ---
  if (showLanding && !isLoggedIn) {
    return <Landing onLoginClick={() => setShowLanding(false)} />;
  }

  if (!isLoggedIn) {
    return (
      <Login
        setIsLoggedIn={setIsLoggedIn} 
        setUserName={setUserName} 
        onBack={() => setShowLanding(true)}
        inputName={inputName} 
        setInputName={setInputName}
        email={email} 
        setEmail={setEmail}
        password={password} 
        setPassword={setPassword}
      />
    );
  }

  return (
    <div className="app-container">
      <Sidebar 
        isOpen={isSidebarOpen} 
        onClose={() => setIsSidebarOpen(false)} 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        userName={userName} 
      />

      <div className="main-content">
        <Header 
          activeTab={activeTab} 
          userName={userName} 
          onOpenSidebar={() => setIsSidebarOpen(true)} 
        />
        {renderContent()}
      </div>

      <ChatWidget />
    </div>
  );
}

export default App;