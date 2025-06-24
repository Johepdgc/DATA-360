import { Doughnut } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import PropTypes from "prop-types";

ChartJS.register(ArcElement, Tooltip, Legend);

const COLORS = [
  "#ef4444",
  "#f59e0b",
  "#10b981",
  "#3b82f6",
  "#8b5cf6",
  "#ec4899",
  "#22d3ee",
  "#eab308",
  "#a3e635",
  "#f43f5e",
];

export default function DonutChart({ data }) {
  const chartData = {
    labels: data.map((d) => d.label),
    datasets: [
      {
        data: data.map((d) => d.count),
        backgroundColor: COLORS.slice(0, data.length),
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: "right" } },
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow h-80">
      <h3 className="text-lg mb-2 font-semibold">Top 10 Quejas (Dona)</h3>
      <Doughnut data={chartData} options={options} />
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
};
