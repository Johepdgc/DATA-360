/**
 * Application configuration
 * Centralizes environment-specific settings and API endpoints
 */

// API base URL - fallback to localhost if not defined
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:4000/api';

// API endpoints
export const API_ENDPOINTS = {
  // Data endpoints
  COMPLAINTS_TOP10: `${API_BASE_URL}/complaints/top10`,
  COMPLAINTS_BY_CATEGORY: `${API_BASE_URL}/complaints/category`,
  COMPLAINTS_TRENDS: `${API_BASE_URL}/complaints/trends`,
  COMPLAINTS_SANKEY: `${API_BASE_URL}/sankey`,
  TEST_CONNECTION: `${API_BASE_URL}/test`,
  
  // ML endpoints
  ML_RESULTS: `${API_BASE_URL}/ml/results`,
  ML_ANALYZE_SENTIMENT: `${API_BASE_URL}/ml/analyze/sentiment`,
  ML_ANALYZE_TOPICS: `${API_BASE_URL}/ml/analyze/topics`
};

// Chart colors - consistent color scheme
export const CHART_COLORS = [
  '#3b82f6', // blue-500
  '#8b5cf6', // violet-500
  '#ec4899', // pink-500
  '#f97316', // orange-500
  '#22c55e', // green-500
  '#14b8a6', // teal-500
  '#06b6d4', // cyan-500
  '#6366f1', // indigo-500
  '#f59e0b', // amber-500
  '#ef4444', // red-500
  '#84cc16', // lime-500
  '#a855f7', // purple-500
];