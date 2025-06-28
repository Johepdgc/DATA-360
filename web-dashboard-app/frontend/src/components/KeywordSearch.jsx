import { useState, useMemo } from "react";
import debounce from "lodash.debounce";
import PropTypes from "prop-types";

export default function KeywordSearch({ onSearch }) {
  const [term, setTerm] = useState("");

  // Create a debounced version of onSearch
  const debounced = useMemo(
    () => debounce((value) => onSearch(value), 300),
    [onSearch]
  );

  const handleChange = (e) => {
    const v = e.target.value;
    setTerm(v);
    debounced(v.trim());
  };

  const handleClear = () => {
    setTerm("");
    onSearch("");
  };

  return (
    <div className="relative">
      <label
        htmlFor="keyword-search"
        className="block text-sm font-medium text-gray-700 mb-1"
      >
        Buscar por palabra clave:
      </label>
      <div className="relative">
        <input
          id="keyword-search"
          type="text"
          placeholder="Ingrese término de búsqueda..."
          value={term}
          onChange={handleChange}
          className="border rounded px-3 py-1 w-full pr-10"
        />
        {term && (
          <button
            onClick={handleClear}
            className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600"
            aria-label="Limpiar búsqueda"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        )}
      </div>
      <p className="text-xs text-gray-500 mt-1">
        Busca quejas que contengan palabras específicas en los comentarios
      </p>
    </div>
  );
}

// PropTypes for type checking
KeywordSearch.propTypes = {
  onSearch: PropTypes.func.isRequired,
};
