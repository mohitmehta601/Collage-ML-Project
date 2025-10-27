import React, { useState } from "react";
import { Send, RefreshCw, CheckCircle } from "lucide-react";
import { predictSingle } from "../services/api";

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    ticker_symbol: "",
    company_name: "",
    insider_role: "Chief Executive Officer",
    under_schedule: false,
    aggregated_signal: "sell",
    aggregated_shares: "",
    aggregated_value_usd: "",
    aggregated_percent_of_shares: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Convert numeric fields
      const payload = {
        ...formData,
        aggregated_shares: parseFloat(formData.aggregated_shares),
        aggregated_value_usd: parseFloat(formData.aggregated_value_usd),
        aggregated_percent_of_shares: parseFloat(
          formData.aggregated_percent_of_shares
        ),
      };

      const result = await predictSingle(payload);
      setPrediction(result);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to get prediction. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      ticker_symbol: "",
      company_name: "",
      insider_role: "Chief Executive Officer",
      under_schedule: false,
      aggregated_signal: "sell",
      aggregated_shares: "",
      aggregated_value_usd: "",
      aggregated_percent_of_shares: "",
    });
    setPrediction(null);
    setError(null);
  };

  const getRiskBadgeClass = (level) => {
    const classes = {
      Low: "risk-low",
      Medium: "risk-medium",
      High: "risk-high",
      Critical: "risk-critical",
    };
    return classes[level] || "risk-low";
  };

  return (
    <div>
      <h2 className="page-title">New Risk Prediction</h2>

      {error && <div className="error">{error}</div>}

      {prediction && (
        <div
          className="success"
          style={{ display: "flex", alignItems: "center", gap: "12px" }}
        >
          <CheckCircle size={24} />
          <div>
            <strong>Prediction Complete!</strong>
            <div style={{ marginTop: "8px" }}>
              Risk Level:{" "}
              <span
                className={`risk-badge ${getRiskBadgeClass(
                  prediction.risk_level
                )}`}
              >
                {prediction.risk_level}
              </span>
              {" | "}
              Probability:{" "}
              <strong>{(prediction.risk_probability * 100).toFixed(2)}%</strong>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Trade Information</h3>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="grid grid-2">
            <div className="form-group">
              <label>Ticker Symbol *</label>
              <input
                type="text"
                name="ticker_symbol"
                value={formData.ticker_symbol}
                onChange={handleChange}
                placeholder="e.g., AAPL"
                required
              />
            </div>

            <div className="form-group">
              <label>Company Name *</label>
              <input
                type="text"
                name="company_name"
                value={formData.company_name}
                onChange={handleChange}
                placeholder="e.g., Apple Inc."
                required
              />
            </div>
          </div>

          <div className="grid grid-2">
            <div className="form-group">
              <label>Insider Role *</label>
              <select
                name="insider_role"
                value={formData.insider_role}
                onChange={handleChange}
                required
              >
                <option value="Chief Executive Officer">
                  Chief Executive Officer
                </option>
                <option value="Chief Financial Officer">
                  Chief Financial Officer
                </option>
                <option value="Chief Operating Officer">
                  Chief Operating Officer
                </option>
                <option value="Chairman">Chairman</option>
                <option value="President">President</option>
                <option value="Vice President">Vice President</option>
                <option value="Director">Director</option>
                <option value="General Counsel">General Counsel</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div className="form-group">
              <label>Transaction Type *</label>
              <select
                name="aggregated_signal"
                value={formData.aggregated_signal}
                onChange={handleChange}
                required
              >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </select>
            </div>
          </div>

          <div className="grid grid-3">
            <div className="form-group">
              <label>Number of Shares *</label>
              <input
                type="number"
                name="aggregated_shares"
                value={formData.aggregated_shares}
                onChange={handleChange}
                placeholder="e.g., 100000"
                min="1"
                step="1"
                required
              />
            </div>

            <div className="form-group">
              <label>Transaction Value (USD) *</label>
              <input
                type="number"
                name="aggregated_value_usd"
                value={formData.aggregated_value_usd}
                onChange={handleChange}
                placeholder="e.g., 15000000"
                min="0"
                step="0.01"
                required
              />
            </div>

            <div className="form-group">
              <label>Percent of Shares *</label>
              <input
                type="number"
                name="aggregated_percent_of_shares"
                value={formData.aggregated_percent_of_shares}
                onChange={handleChange}
                placeholder="e.g., 0.05"
                min="0"
                max="1"
                step="0.0001"
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                name="under_schedule"
                checked={formData.under_schedule}
                onChange={handleChange}
                style={{ width: "auto", margin: 0 }}
              />
              Under 10b5-1 Trading Plan
            </label>
          </div>

          <div style={{ display: "flex", gap: "12px", marginTop: "24px" }}>
            <button
              type="submit"
              className="btn-primary"
              disabled={loading}
              style={{ flex: 1 }}
            >
              {loading ? (
                <>
                  <div
                    className="spinner"
                    style={{
                      width: "16px",
                      height: "16px",
                      borderWidth: "2px",
                      display: "inline-block",
                    }}
                  ></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Send size={18} style={{ marginRight: "8px" }} />
                  Predict Risk
                </>
              )}
            </button>

            <button
              type="button"
              className="btn-secondary"
              onClick={handleReset}
            >
              <RefreshCw size={18} style={{ marginRight: "8px" }} />
              Reset
            </button>
          </div>
        </form>
      </div>

      {prediction && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Prediction Results</h3>
          </div>

          <div style={{ padding: "20px" }}>
            <div className="grid grid-2">
              <div>
                <h4 style={{ marginBottom: "12px", color: "#555" }}>
                  Trade Details
                </h4>
                <p>
                  <strong>Company:</strong> {prediction.company_name}
                </p>
                <p>
                  <strong>Ticker:</strong> {prediction.ticker_symbol}
                </p>
                <p>
                  <strong>Insider:</strong> {prediction.insider_role}
                </p>
              </div>

              <div>
                <h4 style={{ marginBottom: "12px", color: "#555" }}>
                  Risk Assessment
                </h4>
                <p>
                  <strong>Risk Level:</strong>{" "}
                  <span
                    className={`risk-badge ${getRiskBadgeClass(
                      prediction.risk_level
                    )}`}
                  >
                    {prediction.risk_level}
                  </span>
                </p>
                <p>
                  <strong>Probability:</strong>{" "}
                  {(prediction.risk_probability * 100).toFixed(2)}%
                </p>
                <p>
                  <strong>Classification:</strong>{" "}
                  {prediction.predicted_high_risk === 1
                    ? "High Risk"
                    : "Normal"}
                </p>
              </div>
            </div>

            <div
              style={{
                marginTop: "24px",
                padding: "16px",
                background: "#f9fafb",
                borderRadius: "8px",
              }}
            >
              <div className="progress-bar" style={{ height: "24px" }}>
                <div
                  className="progress-fill"
                  style={{
                    width: `${prediction.risk_probability * 100}%`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "white",
                    fontSize: "12px",
                    fontWeight: "600",
                  }}
                >
                  {(prediction.risk_probability * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
