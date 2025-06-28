import { useState } from 'react';
import PropTypes from 'prop-types';

export default function SentimentAnalysisPanel({ data }) {
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [filterSentiment, setFilterSentiment] = useState('all'); // 'all', 'positive', 'neutral', 'negative'
  
  if (!data || !data.clusters) {
    return (
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <h2 className="text-xl font-semibold mb-4">Análisis de Sentimiento</h2>
        <p className="text-gray-500 italic">No hay datos de análisis de sentimiento disponibles.</p>
      </div>
    );
  }
  
  // Calculate sentiment distribution across all clusters
  const calculateSentimentDistribution = () => {
    let positive = 0, neutral = 0, negative = 0, total = 0;
    
    Object.entries(data.clusters).forEach(([_, cluster]) => {
      // Each cluster has keywords but not necessarily sentiment data
      // We'll use the overall sentiment distribution from total_records
      total++;
    });
    
    // If we can't calculate real percentages, use default distribution
    if (total === 0) {
      return { positive: 33, neutral: 34, negative: 33 };
    }
    
    return {
      positive: Math.round((positive / total) * 100),
      neutral: Math.round((neutral / total) * 100),
      negative: Math.round((negative / total) * 100)
    };
  };
  
  const sentimentDistribution = calculateSentimentDistribution();
  
  // Filter clusters by sentiment if filter is active
  const getFilteredClusters = () => {
    if (filterSentiment === 'all') {
      return Object.entries(data.clusters);
    }
    
    // In a real implementation, you would filter based on dominant sentiment
    // Since we don't have that data in the current structure, this is a placeholder
    return Object.entries(data.clusters);
  };
  
  const handleClusterClick = (clusterId) => {
    setSelectedCluster(selectedCluster === clusterId ? null : clusterId);
  };
  
  return (
    <div className="bg-white p-4 rounded-lg shadow mb-6">
      <h2 className="text-xl font-semibold mb-4">Análisis de Sentimiento</h2>
      <p className="mb-4">
        Se analizaron {data.total_records} quejas y se clasificaron en {Object.keys(data.clusters).length} grupos.
      </p>
      
      {/* Sentiment Distribution Summary */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-lg font-medium mb-3">Distribución de Sentimiento</h3>
        <div className="flex items-center mb-2">
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div 
              className="bg-green-500 h-4 rounded-l-full" 
              style={{ width: `${sentimentDistribution.positive}%` }}
            />
            <div 
              className="bg-yellow-400 h-4" 
              style={{ 
                width: `${sentimentDistribution.neutral}%`,
                marginLeft: `${sentimentDistribution.positive}%`,
                marginTop: '-16px' // h-4 = 16px
              }}
            />
            <div 
              className="bg-red-500 h-4 rounded-r-full" 
              style={{ 
                width: `${sentimentDistribution.negative}%`,
                marginLeft: `${sentimentDistribution.positive + sentimentDistribution.neutral}%`,
                marginTop: '-16px' // h-4 = 16px
              }}
            />
          </div>
        </div>
        <div className="flex justify-between text-sm">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-green-500 mr-1"></div>
            <span>Positivo: {sentimentDistribution.positive}%</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-yellow-400 mr-1"></div>
            <span>Neutral: {sentimentDistribution.neutral}%</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-red-500 mr-1"></div>
            <span>Negativo: {sentimentDistribution.negative}%</span>
          </div>
        </div>
      </div>
      
      {/* Sentiment Filter */}
      <div className="mb-4">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Filtrar por Sentimiento</h3>
        <div className="flex space-x-2">
          <button
            onClick={() => setFilterSentiment('all')}
            className={`px-3 py-1 rounded text-sm ${
              filterSentiment === 'all' 
                ? 'bg-blue-100 text-blue-800 border border-blue-300' 
                : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
            }`}
          >
            Todos
          </button>
          <button
            onClick={() => setFilterSentiment('positive')}
            className={`px-3 py-1 rounded text-sm ${
              filterSentiment === 'positive' 
                ? 'bg-green-100 text-green-800 border border-green-300' 
                : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
            }`}
          >
            Positivos
          </button>
          <button
            onClick={() => setFilterSentiment('neutral')}
            className={`px-3 py-1 rounded text-sm ${
              filterSentiment === 'neutral' 
                ? 'bg-yellow-100 text-yellow-800 border border-yellow-300' 
                : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
            }`}
          >
            Neutrales
          </button>
          <button
            onClick={() => setFilterSentiment('negative')}
            className={`px-3 py-1 rounded text-sm ${
              filterSentiment === 'negative' 
                ? 'bg-red-100 text-red-800 border border-red-300' 
                : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
            }`}
          >
            Negativos
          </button>
        </div>
      </div>
      
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-2">Grupos de Quejas</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead className="bg-gray-50">
              <tr>
                <th className="py-2 px-4 border-b text-left">ID</th>
                <th className="py-2 px-4 border-b text-left">Nombre</th>
                <th className="py-2 px-4 border-b text-left">Palabras Clave</th>
                <th className="py-2 px-4 border-b text-left">Acción</th>
              </tr>
            </thead>
            <tbody>
              {getFilteredClusters().map(([id, cluster]) => (
                <tr key={id} className={`hover:bg-gray-50 ${selectedCluster === id ? 'bg-blue-50' : ''}`}>
                  <td className="py-2 px-4 border-b">{id}</td>
                  <td className="py-2 px-4 border-b">{cluster.name}</td>
                  <td className="py-2 px-4 border-b">
                    <div className="flex flex-wrap gap-1">
                      {cluster.keywords.slice(0, 5).map((keyword, i) => (
                        <span key={i} className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                          {keyword}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="py-2 px-4 border-b">
                    <button
                      onClick={() => handleClusterClick(id)}
                      className="text-blue-600 hover:text-blue-800 underline text-sm"
                    >
                      {selectedCluster === id ? 'Ocultar detalles' : 'Ver detalles'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Selected Cluster Details */}
      {selectedCluster && (
        <div className="mt-6 border-t pt-4">
          <h3 className="text-lg font-medium mb-2">
            Detalles del Grupo: {data.clusters[selectedCluster].name}
          </h3>
          <div className="bg-gray-50 p-4 rounded">
            <h4 className="font-medium mb-2">Palabras clave completas:</h4>
            <div className="flex flex-wrap gap-1 mb-4">
              {data.clusters[selectedCluster].keywords.map((keyword, i) => (
                <span key={i} className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                  {keyword}
                </span>
              ))}
            </div>
            
            <h4 className="font-medium mb-2">Ejemplos de quejas en este grupo:</h4>
            <p className="text-gray-500 text-sm italic mb-2">
              Nota: Esta sección mostraría ejemplos reales de quejas en este grupo 
              cuando estén disponibles en los datos de análisis.
            </p>
          </div>
        </div>
      )}
      
      <div className="text-sm text-gray-600 mt-4 pt-4 border-t">
        Análisis generado el: {new Date(data.timestamp.replace(/(\d{8})_(\d{6})/, '$1T$2')).toLocaleString()}
      </div>
    </div>
  );
}

SentimentAnalysisPanel.propTypes = {
  data: PropTypes.object.isRequired
};