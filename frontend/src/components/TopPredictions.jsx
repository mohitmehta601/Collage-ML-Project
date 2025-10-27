import React, { useState, useEffect } from "react";
import { Download, Filter } from "lucide-react";
import { getTopPredictions } from "../services/api";

const TopPredictions = () => {
  const [predictions, setPredictions] = useState([]);
  const [filteredPredictions, setFilteredPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterRisk, setFilterRisk] = useState("all");
  const [limit, setLimit] = useState(100);

  useEffect(() => {
    fetchPredictions();
  }, [limit]);

  useEffect(() => {
    applyFilter();
  }, [predictions, filterRisk]);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      const data = await getTopPredictions(limit);
      setPredictions(data.predictions || []);
      setError(null);
    } catch (err) {
      setError("Failed to load predictions. Please ensure the API is running.");
    } finally {
      setLoading(false);
    }
  };

  const applyFilter = () => {
    if (filterRisk === "all") {
      setFilteredPredictions(predictions);
    } else {
      const filtered = predictions.filter((pred) => {
        const prob = pred.risk_probability;
        if (filterRisk === "critical") return prob >= 0.8;
        if (filterRisk === "high") return prob >= 0.6 && prob < 0.8;
        if (filterRisk === "medium") return prob >= 0.4 && prob < 0.6;
        if (filterRisk === "low") return prob < 0.4;
        return true;
      });
      setFilteredPredictions(filtered);
    }
  };

  const getRiskBadge = (probability) => {
    if (probability >= 0.8)
      return <span className="risk-badge risk-critical">Critical</span>;
    if (probability >= 0.6)
      return <span className="risk-badge risk-high">High</span>;
    if (probability >= 0.4)
      return <span className="risk-badge risk-medium">Medium</span>;
    return <span className="risk-badge risk-low">Low</span>;
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
    }).format(value);
  };

  const formatDate = (dateString) => {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  const exportToCSV = () => {
    const headers = [
      "Ticker",
      "Company",
      "Insider Role",
      "Signal",
      "Shares",
      "Value (USD)",
      "Execution Date",
      "Risk Probability",
      "Risk Level",
    ];

    const rows = filteredPredictions.map((pred) => [
      pred.ticker_symbol,
      pred.company_name,
      pred.insider_role,
      pred.aggregated_signal,
      pred.aggregated_shares,
      pred.aggregated_value_usd,
      pred.earliest_execution_date,
      pred.risk_probability,
      pred.risk_probability >= 0.8
        ? "Critical"
        : pred.risk_probability >= 0.6
        ? "High"
        : pred.risk_probability >= 0.4
        ? "Medium"
        : "Low",
    ]);

    const csv = [headers, ...rows].map((row) => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `high_risk_predictions_${
      new Date().toISOString().split("T")[0]
    }.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
      </div>
    );
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div>
      <h2 className="page-title">High Risk Cases</h2>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <Filter size={20} />
            Filters & Actions
          </h3>
        </div>

        <div className="grid grid-3">
          <div className="form-group">
            <label>Risk Level Filter</label>
            <select
              value={filterRisk}
              onChange={(e) => setFilterRisk(e.target.value)}
            >
              <option value="all">All Levels</option>
              <option value="critical">Critical Only</option>
              <option value="high">High Only</option>
              <option value="medium">Medium Only</option>
              <option value="low">Low Only</option>
            </select>
          </div>

          <div className="form-group">
            <label>Number of Records</label>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
            >
              <option value="50">Top 50</option>
              <option value="100">Top 100</option>
              <option value="200">Top 200</option>
              <option value="500">Top 500</option>
            </select>
          </div>

          <div className="form-group">
            <label style={{ opacity: 0 }}>Action</label>
            <button
              className="btn-primary"
              onClick={exportToCSV}
              style={{ width: "100%" }}
            >
              <Download size={18} style={{ marginRight: "8px" }} />
              Export to CSV
            </button>
          </div>
        </div>

        <div
          style={{
            marginTop: "16px",
            padding: "12px",
            background: "#f9fafb",
            borderRadius: "8px",
          }}
        >
          <strong>Showing {filteredPredictions.length}</strong> of{" "}
          {predictions.length} total predictions
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Prediction Results</h3>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Ticker</th>
                <th>Company</th>
                <th>Insider Role</th>
                <th>Signal</th>
                <th>Value (USD)</th>
                <th>% of Shares</th>
                <th>Execution Date</th>
                <th>Risk Probability</th>
                <th>Risk Level</th>
              </tr>
            </thead>
            <tbody>
              {filteredPredictions.length === 0 ? (
                <tr>
                  <td
                    colSpan="10"
                    style={{ textAlign: "center", padding: "40px" }}
                  >
                    No predictions found matching the selected filters.
                  </td>
                </tr>
              ) : (
                filteredPredictions.map((pred, index) => (
                  <tr key={index}>
                    <td style={{ fontWeight: "600" }}>{index + 1}</td>
                    <td style={{ fontWeight: "600", color: "#667eea" }}>
                      {pred.ticker_symbol}
                    </td>
                    <td>{pred.company_name}</td>
                    <td>{pred.insider_role}</td>
                    <td>
                      <span
                        style={{
                          padding: "4px 8px",
                          borderRadius: "4px",
                          fontSize: "12px",
                          fontWeight: "600",
                          background:
                            pred.aggregated_signal?.toLowerCase() === "buy"
                              ? "#dcfce7"
                              : "#fee2e2",
                          color:
                            pred.aggregated_signal?.toLowerCase() === "buy"
                              ? "#166534"
                              : "#991b1b",
                        }}
                      >
                        {pred.aggregated_signal}
                      </span>
                    </td>
                    <td>{formatCurrency(pred.aggregated_value_usd)}</td>
                    <td>
                      {(pred.aggregated_percent_of_shares * 100).toFixed(3)}%
                    </td>
                    <td>{formatDate(pred.earliest_execution_date)}</td>
                    <td>
                      <strong style={{ color: "#667eea" }}>
                        {(pred.risk_probability * 100).toFixed(2)}%
                      </strong>
                    </td>
                    <td>{getRiskBadge(pred.risk_probability)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default TopPredictions;
