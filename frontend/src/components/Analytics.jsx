import React, { useState, useEffect } from "react";
import { BarChart3 } from "lucide-react";
import { getFeatureImportance, getStatistics } from "../services/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const Analytics = () => {
  const [featureImportance, setFeatureImportance] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [fiData, statsData] = await Promise.all([
          getFeatureImportance(),
          getStatistics(),
        ]);
        setFeatureImportance(fiData.features || []);
        setStats(statsData);
        setError(null);
      } catch (err) {
        setError(
          "Failed to load analytics data. Please ensure the API is running."
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

  const topFeatures = featureImportance.slice(0, 15);

  const COLORS = [
    "#667eea",
    "#764ba2",
    "#f093fb",
    "#f5576c",
    "#4facfe",
    "#00f2fe",
    "#43e97b",
    "#38f9d7",
    "#fa709a",
    "#fee140",
  ];

  return (
    <div>
      <h2 className="page-title">Analytics Dashboard</h2>

      {/* Risk Distribution */}
      {stats?.risk_distribution && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Risk Distribution Breakdown</h3>
          </div>

          <div className="grid grid-4">
            <div style={{ textAlign: "center", padding: "20px" }}>
              <div
                style={{
                  fontSize: "48px",
                  fontWeight: "700",
                  color: "#22c55e",
                  marginBottom: "8px",
                }}
              >
                {stats.risk_distribution.low.toLocaleString()}
              </div>
              <div
                style={{ fontSize: "14px", color: "#666", marginBottom: "8px" }}
              >
                Low Risk Cases
              </div>
              <div className="progress-bar" style={{ height: "8px" }}>
                <div
                  style={{
                    height: "100%",
                    width: `${
                      (stats.risk_distribution.low / stats.total_records) * 100
                    }%`,
                    background: "#22c55e",
                  }}
                ></div>
              </div>
              <div
                style={{ fontSize: "12px", color: "#999", marginTop: "4px" }}
              >
                {(
                  (stats.risk_distribution.low / stats.total_records) *
                  100
                ).toFixed(1)}
                % of total
              </div>
            </div>

            <div style={{ textAlign: "center", padding: "20px" }}>
              <div
                style={{
                  fontSize: "48px",
                  fontWeight: "700",
                  color: "#f59e0b",
                  marginBottom: "8px",
                }}
              >
                {stats.risk_distribution.medium.toLocaleString()}
              </div>
              <div
                style={{ fontSize: "14px", color: "#666", marginBottom: "8px" }}
              >
                Medium Risk Cases
              </div>
              <div className="progress-bar" style={{ height: "8px" }}>
                <div
                  style={{
                    height: "100%",
                    width: `${
                      (stats.risk_distribution.medium / stats.total_records) *
                      100
                    }%`,
                    background: "#f59e0b",
                  }}
                ></div>
              </div>
              <div
                style={{ fontSize: "12px", color: "#999", marginTop: "4px" }}
              >
                {(
                  (stats.risk_distribution.medium / stats.total_records) *
                  100
                ).toFixed(1)}
                % of total
              </div>
            </div>

            <div style={{ textAlign: "center", padding: "20px" }}>
              <div
                style={{
                  fontSize: "48px",
                  fontWeight: "700",
                  color: "#fb923c",
                  marginBottom: "8px",
                }}
              >
                {stats.risk_distribution.high.toLocaleString()}
              </div>
              <div
                style={{ fontSize: "14px", color: "#666", marginBottom: "8px" }}
              >
                High Risk Cases
              </div>
              <div className="progress-bar" style={{ height: "8px" }}>
                <div
                  style={{
                    height: "100%",
                    width: `${
                      (stats.risk_distribution.high / stats.total_records) * 100
                    }%`,
                    background: "#fb923c",
                  }}
                ></div>
              </div>
              <div
                style={{ fontSize: "12px", color: "#999", marginTop: "4px" }}
              >
                {(
                  (stats.risk_distribution.high / stats.total_records) *
                  100
                ).toFixed(1)}
                % of total
              </div>
            </div>

            <div style={{ textAlign: "center", padding: "20px" }}>
              <div
                style={{
                  fontSize: "48px",
                  fontWeight: "700",
                  color: "#ef4444",
                  marginBottom: "8px",
                }}
              >
                {stats.risk_distribution.critical.toLocaleString()}
              </div>
              <div
                style={{ fontSize: "14px", color: "#666", marginBottom: "8px" }}
              >
                Critical Risk Cases
              </div>
              <div className="progress-bar" style={{ height: "8px" }}>
                <div
                  style={{
                    height: "100%",
                    width: `${
                      (stats.risk_distribution.critical / stats.total_records) *
                      100
                    }%`,
                    background: "#ef4444",
                  }}
                ></div>
              </div>
              <div
                style={{ fontSize: "12px", color: "#999", marginTop: "4px" }}
              >
                {(
                  (stats.risk_distribution.critical / stats.total_records) *
                  100
                ).toFixed(1)}
                % of total
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Feature Importance Chart */}
      {topFeatures.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <BarChart3 size={20} />
              Top 15 Feature Importance
            </h3>
          </div>

          <ResponsiveContainer width="100%" height={500}>
            <BarChart
              data={topFeatures}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="feature" />
              <Tooltip />
              <Bar dataKey="importance" fill="#667eea">
                {topFeatures.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Feature Importance Table */}
      {featureImportance.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Feature Importance Rankings</h3>
          </div>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Feature Name</th>
                  <th>Importance Score</th>
                  <th>Relative Importance</th>
                </tr>
              </thead>
              <tbody>
                {featureImportance.map((feat, index) => {
                  const maxImportance = featureImportance[0].importance;
                  const relativeImportance =
                    (feat.importance / maxImportance) * 100;

                  return (
                    <tr key={index}>
                      <td style={{ fontWeight: "600" }}>{index + 1}</td>
                      <td style={{ fontWeight: "600", color: "#667eea" }}>
                        {feat.feature}
                      </td>
                      <td>{feat.importance.toFixed(6)}</td>
                      <td>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                          }}
                        >
                          <div className="progress-bar" style={{ flex: 1 }}>
                            <div
                              className="progress-fill"
                              style={{ width: `${relativeImportance}%` }}
                            ></div>
                          </div>
                          <span
                            style={{
                              fontSize: "12px",
                              color: "#666",
                              minWidth: "50px",
                            }}
                          >
                            {relativeImportance.toFixed(1)}%
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Analytics;
