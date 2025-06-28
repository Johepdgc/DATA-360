import { useState, useEffect } from "react";
import axios from "axios";
import { API_ENDPOINTS } from "../config";
import BarChart from "../components/BarChart";
import DonutChart from "../components/DonutChart";
import ComplaintsTable from "../components/ComplaintsTable";
import TopicAnalysisPanel from "../components/TopicAnalysisPanel";
import SentimentAnalysisPanel from "../components/SentimentAnalysisPanel";

export default function Dashboard() {
  const [chartData, setChartData] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Add states for ML results
  const [mlResults, setMlResults] = useState(null);
  const [mlLoading, setMlLoading] = useState(false);
  const [mlError, setMlError] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      console.log("Fetching top complaints from:", API_ENDPOINTS.COMPLAINTS_TOP10);
      const response = await axios.get(API_ENDPOINTS.COMPLAINTS_TOP10);

      if (response.status !== 200) {
        throw new Error(`Failed to fetch data: ${response.status} ${response.statusText}`);
      }

      console.log("API Response:", response.data);

      if (Array.isArray(response.data)) {
        setChartData(
          response.data.map((item) => ({
            label: item.label || "Unknown",
            count: item.count || 0,
          }))
        );
      } else {
        console.warn("Unexpected API response format:", response.data);
        throw new Error("Received invalid data format from API");
      }
    } catch (err) {
      console.error("Error fetching chart data:", err);
      setError(err.message || "Failed to load chart data. Please try again later.");
      setChartData([]);
    } finally {
      setLoading(false);
    }
  };

  // Check for ML results on load
  useEffect(() => {
    fetchData();
    checkMlResults();
  }, []);

  const checkMlResults = async () => {
    try {
      console.log("Checking for ML results");
      // Use relative path instead of API_ENDPOINTS.ML_RESULTS
      const response = await axios.get("/api/ml/results");
      
      if (response.data.status === "success") {
        console.log("ML results found:", response.data);
        setMlResults(response.data);
      }
    } catch (err) {
      console.log("No ML results available yet:", err.message);
      // Not setting error state as this is expected when no analysis has been run
    }
  };

  const handleCategorySelect = (category) => {
    setSelectedCategory(category);
  };

  const runMLAnalysis = async (type = 'topics') => {
    try {
      setMlLoading(true);
      setMlError(null);
      
      const endpoint = type === 'sentiment' 
        ? "/api/ml/analyze/sentiment"
        : "/api/ml/analyze/topics";
      
      console.log(`Running ${type} analysis from:`, endpoint);
      const response = await axios.get(endpoint);
      
      if (response.data.status === "success") {
        console.log(`${type} analysis completed successfully:`, response.data);
        await checkMlResults(); // Refresh results
      } else {
        const errorMsg = `${type} analysis failed: ${response.data.message || 'Unknown error'}`;
        console.error(errorMsg);
        setMlError(errorMsg);
      }
    } catch (err) {
      const errorMsg = `Error running ${type} analysis: ${err.message}`;
      console.error(errorMsg, err);
      setMlError(errorMsg);
    } finally {
      setMlLoading(false);
    }
  };

  // Loading State
  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        <p className="ml-3 text-gray-600">Cargando datos…</p>
      </div>
    );
  }

  // Error State
  if (error) {
    return (
      <div className="p-8">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
        <button
          onClick={fetchData}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          Intentar nuevamente
        </button>
      </div>
    );
  }

  // No Data State
  if (chartData.length === 0) {
    return (
      <div className="p-8">
        <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded mb-4">
          No hay datos disponibles para mostrar.
        </div>
        <button
          onClick={fetchData}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          Intentar nuevamente
        </button>
      </div>
    );
  }

  // Main Dashboard
  return (
    <div className="p-8 min-h-screen bg-gray-50">
      <h1 className="text-2xl font-bold mb-6">Dashboard de Quejas</h1>

      {/* Charts Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white p-4 rounded-lg shadow">
          <BarChart data={chartData} onCategorySelect={handleCategorySelect} />
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <DonutChart data={chartData} onCategorySelect={handleCategorySelect} />
        </div>
      </div>

      {/* ML Analysis Controls */}
      <div className="mb-8 bg-white p-4 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Análisis de Machine Learning</h2>

        <div className="flex space-x-4 mb-4">
          <button
            onClick={() => runMLAnalysis('topics')}
            disabled={mlLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 transition"
          >
            {mlLoading && filterSentiment === 'topics' ? (
              <span className="flex items-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Procesando...
              </span>
            ) : 'Analizar Temas'}
          </button>

          <button
            onClick={() => runMLAnalysis('sentiment')}
            disabled={mlLoading}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 transition"
          >
            {mlLoading && filterSentiment === 'sentiment' ? (
              <span className="flex items-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Procesando...
              </span>
            ) : 'Analizar Sentimientos'}
          </button>
        </div>

        {mlError && (
          <div className="bg-red-100 border border-red-400 text-red-700 p-3 rounded">
            {mlError}
          </div>
        )}

        {mlLoading && (
          <div className="flex items-center text-gray-600">
            <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-blue-500 mr-2"></div>
            Procesando datos con ML (esto puede tomar varios minutos)...
          </div>
        )}
      </div>

      {/* ML Results Section */}
      {mlResults && (
        <div className="mb-8">
          {mlResults.topicAnalysis && (
            <TopicAnalysisPanel data={mlResults.topicAnalysis} />
          )}

          {mlResults.sentimentAnalysis && (
            <SentimentAnalysisPanel data={mlResults.sentimentAnalysis} />
          )}
        </div>
      )}

      {/* Selected Category Table */}
      {selectedCategory && (
        <div className="mb-8 bg-white p-4 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">
            Quejas de la categoría: {selectedCategory}
          </h2>
          <ComplaintsTable category={selectedCategory} />
        </div>
      )}
    </div>
  );
}
