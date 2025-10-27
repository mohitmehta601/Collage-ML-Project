import React, { useState, useEffect } from "react";
import { Database, Calendar, TrendingUp, CheckCircle } from "lucide-react";
import { getMetrics } from "../services/api";

const ModelInfo = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const data = await getMetrics();
        setMetrics(data);
        setError(null);
      } catch (err) {
        setError(
          "Failed to load model information. Please ensure the API is running."
        );
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
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

  const formatDate = (isoString) => {
    if (!isoString) return "N/A";
    return new Date(isoString).toLocaleString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div>
      <h2 className="page-title">Model Information</h2>

      {/* Model Overview */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <Database size={20} />
            Model Overview
          </h3>
        </div>

        <div className="grid grid-2">
          <div>
            <p style={{ marginBottom: "16px" }}>
              <strong
                style={{ display: "block", color: "#666", marginBottom: "4px" }}
              >
                Model Version
              </strong>
              <span
                style={{
                  fontSize: "18px",
                  color: "#667eea",
                  fontWeight: "600",
                }}
              >
                {metrics?.model_version || "N/A"}
              </span>
            </p>

            <p style={{ marginBottom: "16px" }}>
              <strong
                style={{ display: "block", color: "#666", marginBottom: "4px" }}
              >
                Model Type
              </strong>
              <span
                style={{
                  fontSize: "18px",
                  color: "#667eea",
                  fontWeight: "600",
                }}
              >
                {metrics?.model_type || "N/A"}
              </span>
            </p>
          </div>

          <div>
            <p style={{ marginBottom: "16px" }}>
              <strong
                style={{ display: "block", color: "#666", marginBottom: "4px" }}
              >
                <Calendar
                  size={16}
                  style={{ display: "inline", marginRight: "4px" }}
                />
                Training Date
              </strong>
              <span style={{ fontSize: "14px" }}>
                {formatDate(metrics?.train_date)}
              </span>
            </p>

            <p style={{ marginBottom: "16px" }}>
              <strong
                style={{ display: "block", color: "#666", marginBottom: "4px" }}
              >
                <CheckCircle
                  size={16}
                  style={{ display: "inline", marginRight: "4px" }}
                />
                Status
              </strong>
              <span
                style={{
                  display: "inline-block",
                  padding: "4px 12px",
                  borderRadius: "6px",
                  background: "#dcfce7",
                  color: "#166534",
                  fontSize: "14px",
                  fontWeight: "600",
                }}
              >
                Production Ready
              </span>
            </p>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <TrendingUp size={20} />
            Performance Metrics
          </h3>
        </div>

        <div className="grid grid-4">
          <div
            style={{
              textAlign: "center",
              padding: "24px",
              background: "#f9fafb",
              borderRadius: "8px",
            }}
          >
            <div
              style={{
                fontSize: "42px",
                fontWeight: "700",
                color: "#667eea",
                marginBottom: "8px",
              }}
            >
              {(metrics?.test_accuracy * 100).toFixed(2)}%
            </div>
            <div style={{ fontSize: "14px", color: "#666", fontWeight: "500" }}>
              Test Accuracy
            </div>
          </div>

          <div
            style={{
              textAlign: "center",
              padding: "24px",
              background: "#f9fafb",
              borderRadius: "8px",
            }}
          >
            <div
              style={{
                fontSize: "42px",
                fontWeight: "700",
                color: "#667eea",
                marginBottom: "8px",
              }}
            >
              {(metrics?.roc_auc * 100).toFixed(2)}%
            </div>
            <div style={{ fontSize: "14px", color: "#666", fontWeight: "500" }}>
              ROC-AUC Score
            </div>
          </div>

          <div
            style={{
              textAlign: "center",
              padding: "24px",
              background: "#f9fafb",
              borderRadius: "8px",
            }}
          >
            <div
              style={{
                fontSize: "42px",
                fontWeight: "700",
                color: "#22c55e",
                marginBottom: "8px",
              }}
            >
              {(metrics?.test_precision * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: "14px", color: "#666", fontWeight: "500" }}>
              Precision
            </div>
          </div>

          <div
            style={{
              textAlign: "center",
              padding: "24px",
              background: "#f9fafb",
              borderRadius: "8px",
            }}
          >
            <div
              style={{
                fontSize: "42px",
                fontWeight: "700",
                color: "#22c55e",
                marginBottom: "8px",
              }}
            >
              {(metrics?.test_recall * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: "14px", color: "#666", fontWeight: "500" }}>
              Recall
            </div>
          </div>
        </div>

        <div
          style={{
            marginTop: "24px",
            padding: "20px",
            background: "#eff6ff",
            borderRadius: "8px",
          }}
        >
          <h4 style={{ marginBottom: "12px", color: "#1e40af" }}>
            Performance Analysis (v3.0 Enhanced)
          </h4>
          <ul style={{ paddingLeft: "20px", lineHeight: "1.8" }}>
            <li>
              <strong>ROC-AUC of {(metrics?.roc_auc * 100).toFixed(1)}%</strong>{" "}
              indicates excellent discrimination ability between normal and
              high-risk cases.
            </li>
            <li>
              <strong>
                High Recall ({(metrics?.test_recall * 100).toFixed(1)}%)
              </strong>{" "}
              ensures most high-risk cases are detected with optimal threshold
              tuning.
            </li>
            <li>
              Model uses <strong>temporal validation</strong> to prevent
              look-ahead bias and ensure realistic performance.
            </li>
            <li>
              <strong>Enhanced features</strong> include director-level trades,
              unusual patterns, and company-specific anomalies.
            </li>
            <li>
              <strong>Business rules</strong> post-processing reduces false
              positives by downgrading small trades and upgrading director large
              trades.
            </li>
          </ul>
        </div>
      </div>

      {/* Features Used */}
      {metrics?.features_used && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              Features Used ({metrics.features_used.length})
            </h3>
          </div>

          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "8px",
              padding: "20px",
            }}
          >
            {metrics.features_used.map((feature, index) => (
              <div
                key={index}
                style={{
                  padding: "8px 16px",
                  background:
                    "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                  color: "white",
                  borderRadius: "6px",
                  fontSize: "14px",
                  fontWeight: "500",
                }}
              >
                {feature}
              </div>
            ))}
          </div>

          <div
            style={{
              marginTop: "20px",
              padding: "20px",
              background: "#f9fafb",
              borderRadius: "8px",
            }}
          >
            <h4 style={{ marginBottom: "12px", color: "#555" }}>
              Feature Categories (v3.0)
            </h4>
            <div className="grid grid-2">
              <div>
                <p style={{ marginBottom: "8px" }}>
                  <strong>Role-Based (2):</strong> Executive weight, director
                  indicator
                </p>
                <p style={{ marginBottom: "8px" }}>
                  <strong>Transaction Size (7):</strong> Dollar value, shares,
                  percentiles
                </p>
                <p style={{ marginBottom: "8px" }}>
                  <strong>Trading Plan (1):</strong> 10b5-1 plan status
                </p>
                <p style={{ marginBottom: "8px" }}>
                  <strong>Signal Type (2):</strong> Buy/sell indicators
                </p>
              </div>
              <div>
                <p style={{ marginBottom: "8px" }}>
                  <strong>Interactions (4):</strong> Combined features
                </p>
                <p style={{ marginBottom: "8px" }}>
                  <strong>Log Transforms (2):</strong> Reduced skew in large
                  values
                </p>
                <p style={{ marginBottom: "8px" }}>
                  <strong>NEW - Patterns (2):</strong> Size deviation, company
                  z-score
                </p>
                <p style={{ marginBottom: "8px" }}>
                  <strong>NEW - Context (2):</strong> Quarter end, concurrent
                  trades
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Architecture */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Model Architecture & Improvements</h3>
        </div>

        <div style={{ padding: "20px" }}>
          <h4 style={{ marginBottom: "16px", color: "#667eea" }}>
            XGBoost Hyperparameters (v3.0)
          </h4>
          <div className="grid grid-2" style={{ marginBottom: "24px" }}>
            <div>
              <p style={{ marginBottom: "8px" }}>
                <strong>n_estimators:</strong> 500 trees
              </p>
              <p style={{ marginBottom: "8px" }}>
                <strong>max_depth:</strong> 6 levels (reduced from 8)
              </p>
              <p style={{ marginBottom: "8px" }}>
                <strong>learning_rate:</strong> 0.05 (slow learning)
              </p>
              <p style={{ marginBottom: "8px" }}>
                <strong>subsample:</strong> 0.7 (70% row sampling)
              </p>
            </div>
            <div>
              <p style={{ marginBottom: "8px" }}>
                <strong>colsample_bytree:</strong> 0.5 (50% column sampling)
              </p>
              <p style={{ marginBottom: "8px" }}>
                <strong>reg_alpha (L1):</strong> 10.0 (stronger)
              </p>
              <p style={{ marginBottom: "8px" }}>
                <strong>reg_lambda (L2):</strong> 15.0 (stronger)
              </p>
              <p style={{ marginBottom: "8px" }}>
                <strong>scale_pos_weight:</strong> 54.22 (1.5x balanced)
              </p>
            </div>
          </div>

          <h4 style={{ marginBottom: "16px", color: "#667eea" }}>
            v3.0 Enhancements
          </h4>
          <div
            style={{
              padding: "16px",
              background: "#dcfce7",
              borderRadius: "8px",
              borderLeft: "4px solid #22c55e",
            }}
          >
            <ul style={{ paddingLeft: "20px", lineHeight: "1.8" }}>
              <li>
                <strong>✓ Optimal Threshold Tuning:</strong> Uses 0.914 instead
                of 0.5 for better F1-score
              </li>
              <li>
                <strong>✓ 6 New Features:</strong> Director indicator, size
                deviation, company z-score, quarter-end, concurrent trades,
                director large trade
              </li>
              <li>
                <strong>✓ Precision-Focused Training:</strong> 1.5x class weight
                to reduce false positives
              </li>
              <li>
                <strong>✓ Stronger Regularization:</strong> L1=10, L2=15 to
                prevent overfitting
              </li>
              <li>
                <strong>✓ Business Rules Post-Processing:</strong> Smart
                filtering of predictions
              </li>
            </ul>
          </div>

          <h4
            style={{
              marginBottom: "16px",
              color: "#667eea",
              marginTop: "24px",
            }}
          >
            Key Improvements from v1.0 & v2.0
          </h4>
          <div
            style={{
              padding: "16px",
              background: "#dbeafe",
              borderRadius: "8px",
              borderLeft: "4px solid #3b82f6",
            }}
          >
            <ul style={{ paddingLeft: "20px", lineHeight: "1.8" }}>
              <li>
                <strong>✓ Fixed Data Leakage:</strong> Removed features derived
                from target variable
              </li>
              <li>
                <strong>✓ Temporal Validation:</strong> Train on 2020-2024, test
                on 2025+ data
              </li>
              <li>
                <strong>✓ Enhanced Feature Engineering:</strong> Added pattern
                detection and contextual features
              </li>
              <li>
                <strong>✓ Class Imbalance Handling:</strong> Weighted sampling
                for minority class
              </li>
              <li>
                <strong>✓ Leakage-Free Features:</strong> Only pre-trade
                observable information
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Usage Guidelines */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Usage Guidelines</h3>
        </div>

        <div style={{ padding: "20px" }}>
          <h4 style={{ marginBottom: "16px", color: "#667eea" }}>
            Classification Thresholds (Optimal: 0.914)
          </h4>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Threshold</th>
                  <th>Use Case</th>
                  <th>Expected Recall</th>
                  <th>Expected Precision</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    <strong>0.3</strong>
                  </td>
                  <td>Aggressive screening - catch most violations</td>
                  <td>~98%</td>
                  <td>~15%</td>
                </tr>
                <tr>
                  <td>
                    <strong>0.5</strong>
                  </td>
                  <td>High recall approach</td>
                  <td>~95%</td>
                  <td>~25%</td>
                </tr>
                <tr style={{ background: "#dcfce7" }}>
                  <td>
                    <strong>0.914 ⭐</strong>
                  </td>
                  <td>Optimal F1-Score (Default)</td>
                  <td>~93%</td>
                  <td>~31%</td>
                </tr>
                <tr>
                  <td>
                    <strong>0.95</strong>
                  </td>
                  <td>Very conservative - high confidence only</td>
                  <td>~60%</td>
                  <td>~50%</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div
            style={{
              marginTop: "24px",
              padding: "16px",
              background: "#fef3c7",
              borderRadius: "8px",
              borderLeft: "4px solid #f59e0b",
            }}
          >
            <h4 style={{ marginBottom: "12px", color: "#92400e" }}>
              ⚠️ Important Notes
            </h4>
            <ul style={{ paddingLeft: "20px", lineHeight: "1.8" }}>
              <li>
                This is a <strong>screening tool</strong>, not a definitive
                judgment of wrongdoing.
              </li>
              <li>
                All high-risk predictions should be reviewed by compliance
                professionals.
              </li>
              <li>
                False positives are expected and acceptable for initial
                screening.
              </li>
              <li>
                Model performance may degrade over time - periodic retraining
                recommended.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelInfo;
