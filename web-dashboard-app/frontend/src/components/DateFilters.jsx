import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";

export default function DateFilters({ onRangeChange, initialRange }) {
  // Use proper date format with zero-padded month and day
  const [from, setFrom] = useState(initialRange?.from || "2025-01-01");
  const [to, setTo] = useState(initialRange?.to || "2025-01-31");
  const [error, setError] = useState("");

  // Validate date range
  const validateDateRange = useCallback(() => {
    if (from && to) {
      const fromDate = new Date(from);
      const toDate = new Date(to);

      // Calculate difference in days
      const diffTime = Math.abs(toDate - fromDate);
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

      if (diffDays > 60) {
        setError(
          "Para un mejor rendimiento, seleccione un rango máximo de 60 días"
        );
        return false;
      }

      if (diffDays < 7) {
        setError(
          "Para el diagrama Sankey, seleccione un rango mínimo de 7 días"
        );
        return false;
      }

      setError("");
      return true;
    }
    return true;
  }, [from, to]);

  useEffect(() => {
    if (validateDateRange()) {
      onRangeChange({ from, to });
    }
  }, [from, to, onRangeChange, validateDateRange]);

  const handleReset = () => {
    // Use proper date format with zero-padded month and day
    setFrom("2025-01-01");
    setTo("2025-01-31");
    setError("");
  };

  return (
    <div className="bg-gray-50 p-4 rounded-lg mb-4">
      <h3 className="text-md font-medium mb-2">Filtros de fecha</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label
            htmlFor="date-from"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Desde:
          </label>
          <input
            id="date-from"
            type="date"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
            className="border border-gray-300 rounded px-3 py-2 w-full"
          />
        </div>
        <div>
          <label
            htmlFor="date-to"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Hasta:
          </label>
          <input
            id="date-to"
            type="date"
            value={to}
            onChange={(e) => setTo(e.target.value)}
            className="border border-gray-300 rounded px-3 py-2 w-full"
          />
        </div>
      </div>

      {error && <div className="mt-2 text-red-500 text-sm">{error}</div>}

      <button
        onClick={handleReset}
        className="mt-3 bg-gray-200 hover:bg-gray-300 text-gray-800 px-4 py-2 rounded text-sm"
      >
        Restablecer fechas
      </button>

      <p className="text-xs text-gray-500 mt-2">
        Para el diagrama Sankey, seleccione fechas que abarquen al menos 7 días
        para tener datos suficientes.
      </p>
    </div>
  );
}

// PropTypes for type checking
DateFilters.propTypes = {
  onRangeChange: PropTypes.func.isRequired,
  initialRange: PropTypes.shape({
    from: PropTypes.string,
    to: PropTypes.string,
  }),
};
