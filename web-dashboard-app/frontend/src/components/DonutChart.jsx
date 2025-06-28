import { useRef } from "react";
import { Doughnut } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import PropTypes from "prop-types";

ChartJS.register(ArcElement, Tooltip, Legend);

const COLORS = [
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
];

export default function DonutChart({ data, onCategorySelect }) {
  const chartRef = useRef();

  const chartData = {
    labels: data.map((d) => d.label),
    datasets: [
      {
        data: data.map((d) => d.count),
        backgroundColor: COLORS.slice(0, data.length),
        borderWidth: 1,
        borderColor: "#e5e7eb",
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "right",
        labels: {
          boxWidth: 15,
          padding: 15,
          font: {
            size: 11,
          },
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const label = context.label || "";
            const value = context.raw || 0;
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = Math.round((value / total) * 100);
            return `${label}: ${value} (${percentage}%)`;
          },
        },
      },
    },
    onClick: (e, elements) => {
      if (elements.length > 0 && onCategorySelect) {
        const index = elements[0].index;
        onCategorySelect(data[index].label);
      }
    },
    cutout: "65%",
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow h-80">
      <h3 className="text-lg mb-2 font-semibold">Distribución de Quejas</h3>
      <p className="text-sm text-gray-500 mb-2">
        Haga clic en un segmento para ver detalles
      </p>
      <Doughnut ref={chartRef} data={chartData} options={options} />
    </div>
  );
}

// PropTypes for type checking
DonutChart.propTypes = {
  data: PropTypes.arrayOf(
    PropTypes.shape({
      label: PropTypes.string.isRequired,
      count: PropTypes.number.isRequired,
    })
  ).isRequired,
  onCategorySelect: PropTypes.func,
};
