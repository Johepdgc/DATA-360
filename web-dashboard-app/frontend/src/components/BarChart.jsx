import { useRef, useEffect } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";
import PropTypes from "prop-types";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

export default function BarChart({ data }) {
  const chartData = {
    labels: data.map((d) => d.label),
    datasets: [
      {
        label: "Cantidad",
        data: data.map((d) => d.count),
        backgroundColor: "#3b82f6",
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow h-80">
      <h3 className="text-lg mb-2 font-semibold">Top 10 Quejas (Barras)</h3>
      <Bar data={chartData} options={options} />
    </div>
  );
}

// PropTypes for type checking
BarChart.propTypes = {
  data: PropTypes.arrayOf(
    PropTypes.shape({
      label: PropTypes.string.isRequired,
      count: PropTypes.number.isRequired,
    })
  ).isRequired,
};
