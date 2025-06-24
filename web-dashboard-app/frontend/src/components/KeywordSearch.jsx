import { useState, useMemo } from "react";
import debounce from "lodash.debounce";
import PropTypes from "prop-types";

export default function KeywordSearch({ onSearch }) {
  const [term, setTerm] = useState("");

  // creamos una versión debounceada de onSearch
  const debounced = useMemo(
    () => debounce((value) => onSearch(value), 300),
    [onSearch]
  );

  const handleChange = (e) => {
    const v = e.target.value;
    setTerm(v);
    debounced(v.trim());
  };

  return (
    <div className="mb-6">
      <input
        type="text"
        placeholder="Buscar por palabra clave…"
        value={term}
        onChange={handleChange}
        className="border rounded px-3 py-1 w-full"
      />
    </div>
  );
}
// PropTypes for type checking
KeywordSearch.propTypes = {
  onSearch: PropTypes.func.isRequired,
};
