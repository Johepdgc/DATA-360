import { useState, useEffect } from "react";
import axios from "axios";

import DateFilters from "./components/DateFilters";
import KeywordSearch from "./components/KeywordSearch";
import BarChart from "./components/BarChart";
import DonutChart from "./components/DonutChart";
import SankeyChart from "./components/SankeyChart";

export default function App() {
  const [range, setRange] = useState({ from: "", to: "" });
  const [keyword, setKeyword] = useState("");
  const [top10, setTop10] = useState([]);
  const [sankeyData, setSankey] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchAll = async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      if (range.from) params.append("dateFrom", range.from);
      if (range.to) params.append("dateTo", range.to);
      if (keyword) params.append("search", keyword);

      const [{ data: t10 }, { data: sk }] = await Promise.all([
        axios.get(`http://localhost:5000/api/top10?${params}`),
        axios.get(`http://localhost:5000/api/sankey?${params}`),
      ]);

      setTop10(t10);
      setSankey(sk);
    } catch (err) {
      console.error(err);
      setError("Error cargando datos del dashboard");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAll();
  }, [range, keyword]);

  return (
    <div className="p-8 min-h-screen bg-gray-50">
      <h1 className="text-2xl font-bold mb-6">Dashboard de Quejas</h1>

      <DateFilters onRangeChange={setRange} />
      <KeywordSearch onSearch={setKeyword} />

      {loading && (
        <p className="text-center text-gray-600 py-4">Cargando datos…</p>
      )}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {!loading && !error && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <BarChart data={top10} />
            <DonutChart data={top10} />
          </div>
          <div className="mt-8">
            <SankeyChart data={sankeyData} />
          </div>
        </>
      )}
    </div>
  );
}
