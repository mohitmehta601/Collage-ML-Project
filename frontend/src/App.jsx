import React, { useState, useEffect } from "react";
import {
  Activity,
  TrendingUp,
  AlertTriangle,
  BarChart3,
  Shield,
  Database,
} from "lucide-react";
import Dashboard from "./components/Dashboard";
import PredictionForm from "./components/PredictionForm";
import TopPredictions from "./components/TopPredictions";
import Analytics from "./components/Analytics";
import ModelInfo from "./components/ModelInfo";
import { checkHealth } from "./services/api";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const health = await checkHealth();
        setApiStatus(health.model_loaded ? "connected" : "degraded");
      } catch (error) {
        setApiStatus("error");
      }
    };

    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const renderTabContent = () => {
    switch (activeTab) {
      case "dashboard":
        return <Dashboard />;
      case "predict":
        return <PredictionForm />;
      case "top-predictions":
        return <TopPredictions />;
      case "analytics":
        return <Analytics />;
      case "model-info":
        return <ModelInfo />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="container">
          <div className="header-content">
            <div className="header-left">
              <Shield className="logo-icon" size={32} />
              <div>
                <h1 className="header-title">Insider Trading Risk Detector</h1>
                <p className="header-subtitle">
                  ML-Powered Risk Assessment Platform
                </p>
              </div>
            </div>
            <div className="header-right">
              <div className={`api-status ${apiStatus}`}>
                <div className="status-dot"></div>
                <span>
                  {apiStatus === "connected"
                    ? "API Connected"
                    : apiStatus === "degraded"
                    ? "API Degraded"
                    : apiStatus === "error"
                    ? "API Error"
                    : "Checking..."}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="nav">
        <div className="container">
          <div className="tabs">
            <button
              className={`tab ${activeTab === "dashboard" ? "active" : ""}`}
              onClick={() => setActiveTab("dashboard")}
            >
              <Activity size={18} />
              Dashboard
            </button>
            <button
              className={`tab ${activeTab === "predict" ? "active" : ""}`}
              onClick={() => setActiveTab("predict")}
            >
              <TrendingUp size={18} />
              New Prediction
            </button>
            <button
              className={`tab ${
                activeTab === "top-predictions" ? "active" : ""
              }`}
              onClick={() => setActiveTab("top-predictions")}
            >
              <AlertTriangle size={18} />
              High Risk Cases
            </button>
            <button
              className={`tab ${activeTab === "analytics" ? "active" : ""}`}
              onClick={() => setActiveTab("analytics")}
            >
              <BarChart3 size={18} />
              Analytics
            </button>
            <button
              className={`tab ${activeTab === "model-info" ? "active" : ""}`}
              onClick={() => setActiveTab("model-info")}
            >
              <Database size={18} />
              Model Info
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="main">
        <div className="container">{renderTabContent()}</div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>
            &copy; 2025 Insider Trading Risk Detector v2.0 | ML Engineering Team
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
