import { useRef } from "react";
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

export default function BarChart({ data, onCategorySelect }) {
  const chartRef = useRef();

  const chartData = {
    labels: data.map((d) => d.label),
    datasets: [
      {
        label: "Cantidad",
        data: data.map((d) => d.count),
        backgroundColor: [
          "#3b82f6",
          "#ef4444",
          "#f59e0b",
          "#10b981",
          "#8b5cf6",
          "#ec4899",
          "#22d3ee",
          "#eab308",
          "#a3e635",
          "#f43f5e",
        ].slice(0, data.length),
        borderWidth: 1,
        borderColor: "#e5e7eb",
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (context) => `Cantidad: ${context.raw} quejas`,
        },
      },
    },
    onClick: (e, elements) => {
      if (elements.length > 0 && onCategorySelect) {
        const index = elements[0].index;
        onCategorySelect(data[index].label);
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          precision: 0,
        },
        title: {
          display: true,
          text: "Cantidad de quejas",
        },
      },
      x: {
        ticks: {
          maxRotation: 45,
          minRotation: 45,
        },
      },
    },
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow h-80">
      <h3 className="text-lg mb-2 font-semibold">Top 10 Quejas (Barras)</h3>
      <p className="text-sm text-gray-500 mb-2">
        Haga clic en una barra para ver detalles
      </p>
      <Bar ref={chartRef} data={chartData} options={options} />
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
  onCategorySelect: PropTypes.func,
};
