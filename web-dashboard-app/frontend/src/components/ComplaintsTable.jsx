import { useEffect, useState } from "react";
import axios from "axios";
import PropTypes from "prop-types";

export default function ComplaintsTable({ category }) {
  // 1) State for complaints list
  const [list, setList] = useState([]);
  // 2) Loading indicator
  const [loading, setLoading] = useState(true);
  // 3) Error message
  const [error, setError] = useState(null);
  // 4) Filters
  const [searchTerm, setSearchTerm] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  // 5) Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);

  useEffect(() => {
    // Triggered whenever category changes
    fetchData();
  }, [category, searchTerm, dateFrom, dateTo]);

  const fetchData = () => {
    setLoading(true);
    setError(null);
    setCurrentPage(1);

    const params = new URLSearchParams();
    params.append("category", category);
    if (searchTerm) params.append("search", searchTerm);
    if (dateFrom) params.append("dateFrom", dateFrom);
    if (dateTo) params.append("dateTo", dateTo);

    axios
      .get(`/api/complaints?${params.toString()}`)
      .then((res) => {
        setList(res.data); // Store complaints array
        setLoading(false); // Stop spinner
      })
      .catch((err) => {
        console.error("Error loading complaints:", err);
        setError("Error cargando las quejas"); // User-friendly text
        setLoading(false);
      });
  };

  const clearFilters = () => {
    setSearchTerm("");
    setDateFrom("");
    setDateTo("");
  };

  // Pagination logic
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = list.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(list.length / itemsPerPage);

  // Handle page changes
  const paginate = (pageNumber) => setCurrentPage(pageNumber);

  return (
    <div className="bg-white rounded-xl shadow p-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4">
        <h2 className="text-xl font-semibold">Detalle: {category}</h2>
        <p className="text-sm text-gray-500">
          {list.length} quejas encontradas
        </p>
      </div>

      {/* Filters */}
      <div className="bg-gray-50 p-4 rounded-lg mb-4">
        <h3 className="text-md font-medium mb-2">Filtros</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
          <div>
            <label
              htmlFor="comment-search"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Buscar en comentarios:
            </label>
            <input
              id="comment-search"
              type="text"
              placeholder="Buscar..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="border border-gray-300 rounded px-3 py-2 w-full"
            />
          </div>
          <div>
            <label
              htmlFor="date-from-filter"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Desde:
            </label>
            <input
              id="date-from-filter"
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="border border-gray-300 rounded px-3 py-2 w-full"
            />
          </div>
          <div>
            <label
              htmlFor="date-to-filter"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Hasta:
            </label>
            <input
              id="date-to-filter"
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="border border-gray-300 rounded px-3 py-2 w-full"
            />
          </div>
        </div>
        <button
          onClick={clearFilters}
          className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-4 py-2 rounded text-sm"
        >
          Limpiar Filtros
        </button>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mb-2"></div>
          <p className="text-gray-600">Cargando detalles...</p>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {/* Results table */}
      {!loading && !error && (
        <>
          {list.length === 0 ? (
            <div className="text-center py-8 bg-gray-50 rounded-lg">
              <p className="text-gray-600">
                No se encontraron quejas con los filtros seleccionados.
              </p>
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="px-4 py-3 border-b-2 border-gray-200">
                        Cliente
                      </th>
                      <th className="px-4 py-3 border-b-2 border-gray-200">
                        Comentarios
                      </th>
                      <th className="px-4 py-3 border-b-2 border-gray-200">
                        Fecha
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentItems.map((row, i) => (
                      <tr
                        key={`${row["Nombre del Cliente"] || 'unknown'}-${row["Fecha de interacción"] || 'unknown'}-${i}`}
                        className="hover:bg-gray-50 border-b"
                      >
                        <td className="px-4 py-3">
                          {row["Nombre del Cliente"] || 'N/A'}
                        </td>
                        <td className="px-4 py-3">{row["Comentarios"] || 'N/A'}</td>
                        <td className="px-4 py-3">
                          {row["Fecha de interacción"] ? new Date(row["Fecha de interacción"]).toLocaleDateString() : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex justify-center mt-4">
                  <nav className="inline-flex rounded-md shadow">
                    <button
                      onClick={() =>
                        paginate(currentPage > 1 ? currentPage - 1 : 1)
                      }
                      disabled={currentPage === 1}
                      className="px-3 py-1 rounded-l-md border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-50"
                    >
                      &laquo; Anterior
                    </button>

                    <div className="px-4 py-1 border-t border-b border-gray-300 bg-white text-gray-700">
                      Página {currentPage} de {totalPages}
                    </div>

                    <button
                      onClick={() =>
                        paginate(
                          currentPage < totalPages
                            ? currentPage + 1
                            : totalPages
                        )
                      }
                      disabled={currentPage === totalPages}
                      className="px-3 py-1 rounded-r-md border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-50"
                    >
                      Siguiente &raquo;
                    </button>
                  </nav>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}

// PropTypes for type checking
ComplaintsTable.propTypes = {
  category: PropTypes.string.isRequired,
};
