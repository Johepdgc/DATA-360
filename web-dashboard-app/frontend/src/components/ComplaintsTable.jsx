import { useEffect, useState } from "react";
import axios from "axios";
import PropTypes from "prop-types";

export default function ComplaintsTable({ category }) {
  // 1) Estado de la lista de quejas
  const [list, setList] = useState([]);
  // 2) Indicador de carga
  const [loading, setLoading] = useState(true);
  // 3) Mensaje de error
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  useEffect(() => {
    // Se dispara cada vez que cambie 'category'
    setLoading(true);
    setError(null);

    const params = new URLSearchParams();
    params.append("category", category);
    if (searchTerm) params.append("search", searchTerm);
    if (dateFrom) params.append("dateFrom", dateFrom);
    if (dateTo) params.append("dateTo", dateTo);

    axios
      .get(`/api/complaints?${params.toString()}`)
      .then((res) => {
        setList(res.data); // guardamos el arreglo de quejas
        setLoading(false); // detenemos spinner
      })
      .catch((err) => {
        console.error("Error loading complaints:", err);
        setError("Error cargando las quejas"); // texto para el usuario
        setLoading(false);
      });
  }, [category]);

  return (
    <div className="bg-white rounded-2xl shadow p-6">
      <h2 className="text-xl mb-4">Detalle: {category}</h2>

      {/* 4) Spinner mientras cargan datos */}
      {loading && (
        <div className="text-center py-4">
          <p className="text-gray-600">Cargando detalles...</p>
        </div>
      )}

      {/* 5) Alerta si ocurre un error */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {/* Filtros de búsqueda */}
      <div className="mb-4">
        <input
          type="text"
          placeholder="Buscar comentarios..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="border border-gray-300 rounded px-3 py-2 w-full mb-2"
        />
        <div className="flex space-x-2">
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            className="border border-gray-300 rounded px-3 py-2 w-full"
          />
          <input
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            className="border border-gray-300 rounded px-3 py-2 w-full"
          />
        </div>
      </div>
      {/* Botón para limpiar filtros */}
      <button
        onClick={() => {
          setSearchTerm("");
          setDateFrom("");
          setDateTo("");
        }}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Limpiar Filtros
      </button>
      <hr className="my-4" />
      {/* 6) Tabla de resultados */}
      {!loading && !error && (
        <table className="w-full text-left border-collapse">
          <thead>
            <tr>
              <th className="py-2 border-b">Cliente</th>
              <th className="py-2 border-b">Comentarios</th>
              <th className="py-2 border-b">Fecha</th>
            </tr>
          </thead>
          <tbody>
            {list.map((row, i) => (
              <tr
                key={`${row["Nombre del Cliente"]}-${row["Fecha de interacción"]}-${i}`}
                className="border-t"
              >
                <td className="py-2">{row["Nombre del Cliente"]}</td>
                <td className="py-2">{row["Comentarios"]}</td>
                <td className="py-2">{row["Fecha de interacción"]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// 7) Validación de props: category debe ser string
ComplaintsTable.propTypes = {
  category: PropTypes.string.isRequired,
};
