import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import PropTypes from "prop-types";

export default function CategoryChart({ data, onSelect }) {
  const formatted = Object.entries(data).map(([name, count]) => ({
    name,
    count,
  }));
  return (
    <div className="bg-white rounded-2xl shadow p-6 mb-8">
      <h2 className="text-xl mb-4">Quejas por Motivo</h2>
      <ResponsiveContainer width="100%" height={300} />
      <BarChart
        data={formatted}
        onClick={({ activeLabel }) => onSelect(activeLabel)}
      >
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="count" fill="#3B82F6" className="cursor-pointer" />
      </BarChart>
    </div>
  );
}

// PropTypes for type checking
CategoryChart.propTypes = {
  data: PropTypes.object.isRequired,
  onSelect: PropTypes.func.isRequired,
};
