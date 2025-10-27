import React, { useState, useEffect } from "react";
import { TrendingUp, AlertTriangle, Activity, Users } from "lucide-react";
import { getStatistics, getMetrics } from "../services/api";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [statsData, metricsData] = await Promise.all([
          getStatistics(),
          getMetrics(),
        ]);
        setStats(statsData);
        setMetrics(metricsData);
        setError(null);
      } catch (err) {
        setError(
          "Failed to load dashboard data. Please ensure the API is running."
        );
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

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

  const riskData = stats?.risk_distribution
    ? [
        {
          name: "Low Risk",
          value: stats.risk_distribution.low,
          color: "#22c55e",
        },
        {
          name: "Medium Risk",
          value: stats.risk_distribution.medium,
          color: "#f59e0b",
        },
        {
          name: "High Risk",
          value: stats.risk_distribution.high,
          color: "#fb923c",
        },
        {
          name: "Critical Risk",
          value: stats.risk_distribution.critical,
          color: "#ef4444",
        },
      ]
    : [];

  return (
    <div>
      <h2 className="page-title">Dashboard Overview</h2>

      {/* v3.0 Enhancement Banner */}
      {metrics?.model_version === "3.0" && (
        <div
          style={{
            padding: "20px",
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            borderRadius: "12px",
            color: "white",
            marginBottom: "24px",
            boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            <TrendingUp size={32} />
            <div>
              <h3 style={{ margin: 0, fontSize: "20px", fontWeight: "700" }}>
                🎉 Model v3.0 Enhanced - Now Active!
              </h3>
              <p
                style={{ margin: "8px 0 0 0", fontSize: "14px", opacity: 0.95 }}
              >
                <strong>New Features:</strong> Optimal threshold tuning (0.914),
                6 new detection features, precision-focused training, and
                business rules post-processing. Precision:{" "}
                {(metrics?.test_precision * 100).toFixed(1)}% | Recall:{" "}
                {(metrics?.test_recall * 100).toFixed(1)}% | ROC-AUC:{" "}
                {(metrics?.roc_auc * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-4">
        <div
          className="stat-card"
          style={{
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          }}
        >
          <Activity size={32} style={{ marginBottom: "12px" }} />
          <div className="stat-value">
            {stats?.total_records?.toLocaleString() || 0}
          </div>
          <div className="stat-label">Total Records</div>
        </div>

        <div
          className="stat-card"
          style={{
            background: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
          }}
        >
          <AlertTriangle size={32} style={{ marginBottom: "12px" }} />
          <div className="stat-value">
            {stats?.risk_distribution?.critical || 0}
          </div>
          <div className="stat-label">Critical Risk Cases</div>
        </div>

        <div
          className="stat-card"
          style={{
            background: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
          }}
        >
          <TrendingUp size={32} style={{ marginBottom: "12px" }} />
          <div className="stat-value">
            {(metrics?.roc_auc * 100).toFixed(1) || 0}%
          </div>
          <div className="stat-label">Model ROC-AUC</div>
        </div>

        <div
          className="stat-card"
          style={{
            background: "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
          }}
        >
          <Users size={32} style={{ marginBottom: "12px" }} />
          <div className="stat-value">
            {(stats?.average_risk_probability * 100).toFixed(1) || 0}%
          </div>
          <div className="stat-label">Avg Risk Score</div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-2">
        {/* Risk Distribution Chart */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Risk Distribution</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) =>
                  `${name}: ${(percent * 100).toFixed(1)}%`
                }
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {riskData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Model Performance */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Model Performance</h3>
          </div>
          <div style={{ padding: "20px" }}>
            <div style={{ marginBottom: "24px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: "8px",
                }}
              >
                <span style={{ fontWeight: "500" }}>Training Accuracy</span>
                <span style={{ fontWeight: "600", color: "#667eea" }}>
                  {(metrics?.training_accuracy * 100).toFixed(2)}%
                </span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{
                    width: `${(metrics?.training_accuracy || 0) * 100}%`,
                  }}
                ></div>
              </div>
            </div>

            <div style={{ marginBottom: "24px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: "8px",
                }}
              >
                <span style={{ fontWeight: "500" }}>Test Accuracy</span>
                <span style={{ fontWeight: "600", color: "#667eea" }}>
                  {(metrics?.test_accuracy * 100).toFixed(2)}%
                </span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${(metrics?.test_accuracy || 0) * 100}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: "8px",
                }}
              >
                <span style={{ fontWeight: "500" }}>ROC-AUC Score</span>
                <span style={{ fontWeight: "600", color: "#667eea" }}>
                  {(metrics?.roc_auc * 100).toFixed(2)}%
                </span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${(metrics?.roc_auc || 0) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Top Companies */}
      {stats?.top_companies_by_volume &&
        Object.keys(stats.top_companies_by_volume).length > 0 && (
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Top Companies by Trading Volume</h3>
            </div>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Ticker</th>
                    <th>Number of Trades</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats.top_companies_by_volume)
                    .slice(0, 10)
                    .map(([ticker, count], index) => (
                      <tr key={ticker}>
                        <td>{index + 1}</td>
                        <td style={{ fontWeight: "600", color: "#667eea" }}>
                          {ticker}
                        </td>
                        <td>{count.toLocaleString()}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
    </div>
  );
};

export default Dashboard;
