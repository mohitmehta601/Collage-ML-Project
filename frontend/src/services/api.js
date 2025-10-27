import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

// Get model metrics
export const getMetrics = async () => {
  const response = await api.get('/metrics');
  return response.data;
};

// Predict single trade
export const predictSingle = async (trade) => {
  const response = await api.post('/predict', trade);
  return response.data;
};

// Predict bulk trades
export const predictBulk = async (trades) => {
  const response = await api.post('/predict/bulk', { trades });
  return response.data;
};

// Get top predictions
export const getTopPredictions = async (n = 100) => {
  const response = await api.get(`/predictions/top/${n}`);
  return response.data;
};

// Get statistics
export const getStatistics = async () => {
  const response = await api.get('/statistics');
  return response.data;
};

// Get feature importance
export const getFeatureImportance = async () => {
  const response = await api.get('/feature-importance');
  return response.data;
};

export default api;
