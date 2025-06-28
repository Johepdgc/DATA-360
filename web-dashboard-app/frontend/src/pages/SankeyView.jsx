import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import SankeyChart from "../components/SankeyChart";
import DateFilters from "../components/DateFilters";

export default function SankeyView() {
  const [sankeyData, setSankeyData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dateRange, setDateRange] = useState({
    from: "2025-01-01",
    to: "2025-01-31",
  });

  // Use useCallback to prevent infinite loop
  const fetchSankeyData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams();
      if (dateRange.from) params.append("dateFrom", dateRange.from);
      if (dateRange.to) params.append("dateTo", dateRange.to);

      console.log("Fetching Sankey data with params:", params.toString());

      const response = await axios.get(
        `http://localhost:4000/api/sankey?${params.toString()}`
      );

      if (
        response.data &&
        response.data.nodes &&
        response.data.links &&
        response.data.nodes.length > 0
      ) {
        setSankeyData(response.data);
      } else {
        setError("No hay suficientes datos para generar el diagrama Sankey.");
      }
    } catch (err) {
      console.error("Error fetching Sankey data:", err);
      if (err.response && err.response.data && err.response.data.error) {
        // Use the specific error message from the backend
        setError(err.response.data.error);
      } else {
        setError(`Error al cargar datos: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  }, [dateRange.from, dateRange.to]);

  // Generate seed data for January-February if needed
  const generateSeedData = async () => {
    try {
      setLoading(true);
      setError(null);

      await axios.get("http://localhost:4000/api/seed-early-2025-data");

      // Fetch data again after seeding
      await fetchSankeyData();
    } catch (err) {
      console.error("Error seeding data:", err);
      setError(`Error al generar datos de prueba: ${err.message}`);
      setLoading(false);
    }
  };

  // Handle date range changes
  const handleDateRangeChange = useCallback((newRange) => {
    setDateRange(newRange);
  }, []);

  // Fetch data when date range changes
  useEffect(() => {
    fetchSankeyData();
  }, [fetchSankeyData]);

  // Set dimensions based on viewport
  const width = Math.min(window.innerWidth - 40, 1200);
  const height = 600;

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Diagrama de Sankey</h1>
      <p className="text-gray-600 mb-4">
        Este diagrama muestra cómo las categorías de quejas evolucionan a lo
        largo del tiempo.
      </p>

      <div className="mb-6">
        <DateFilters
          onRangeChange={handleDateRangeChange}
          initialRange={dateRange}
        />
      </div>

      <div className="bg-white rounded-xl shadow p-4">
        {loading ? (
          <div className="flex justify-center items-center h-80">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : error ? (
          <div className="text-center p-8 bg-red-50 rounded-lg">
            <p className="text-red-500">{error}</p>
            <p className="text-sm text-gray-500 mt-4">
              Intente ajustar el rango de fechas o generar datos de prueba.
            </p>
            <button
              onClick={generateSeedData}
              className="mt-4 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
            >
              Generar datos de prueba para Enero-Febrero 2025
            </button>
          </div>
        ) : sankeyData.nodes && sankeyData.nodes.length > 0 ? (
          <SankeyChart data={sankeyData} width={width} height={height} />
        ) : (
          <div className="text-center p-8 bg-gray-50 rounded-lg">
            <p className="text-gray-500">
              No hay suficientes datos para generar el diagrama Sankey.
            </p>
            <p className="text-sm text-gray-400 mt-2">
              Intente ajustar el rango de fechas o verificar que haya datos
              disponibles.
            </p>
            <button
              onClick={generateSeedData}
              className="mt-4 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
            >
              Generar datos de prueba para Enero-Febrero 2025
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
