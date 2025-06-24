import { useState, useEffect } from "react";
import PropTypes from "prop-types";

export default function DateFilters({ onRangeChange }) {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");

  useEffect(() => {
    onRangeChange({ from, to });
  }, [from, to, onRangeChange]);

  return (
    <div className="flex gap-4 mb-6">
      <input
        type="date"
        value={from}
        onChange={(e) => setFrom(e.target.value)}
        className="border rounded px-3 py-1"
      />
      <input
        type="date"
        value={to}
        onChange={(e) => setTo(e.target.value)}
        className="border rounded px-3 py-1"
      />
    </div>
  );
}
// PropTypes for type checking
DateFilters.propTypes = {
  onRangeChange: PropTypes.func.isRequired,
};
