import { useState } from 'react';
import PropTypes from 'prop-types';

export default function TopicAnalysisPanel({ data }) {
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview', 'trends', 'documents'
  
  if (!data || !data.topics) {
    return (
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <h2 className="text-xl font-semibold mb-4">Análisis de Temas</h2>
        <p className="text-gray-500 italic">No hay datos de análisis de temas disponibles.</p>
      </div>
    );
  }
  
  const handleTopicClick = (topicId) => {
    setSelectedTopic(selectedTopic === topicId ? null : topicId);
  };
  
  // Get sentiment distribution across all topics
  const getSentimentSummary = () => {
    const summary = { positive: 0, neutral: 0, negative: 0, total: 0 };
    
    Object.values(data.topics).forEach(topic => {
      if (topic.sentiment) {
        summary.positive += topic.sentiment.positive || 0;
        summary.neutral += topic.sentiment.neutral || 0;
        summary.negative += topic.sentiment.negative || 0;
        
        // Assuming percentages add up to 100 for each topic
        summary.total += 1;
      }
    });
    
    // Convert to averages
    if (summary.total > 0) {
      summary.positive = Math.round(summary.positive / summary.total);
      summary.neutral = Math.round(summary.neutral / summary.total);
      summary.negative = Math.round(summary.negative / summary.total);
    }
    
    return summary;
  };
  
  const sentimentSummary = getSentimentSummary();
  
  return (
    <div className="bg-white p-4 rounded-lg shadow mb-6">
      <h2 className="text-xl font-semibold mb-4">Análisis de Temas</h2>
      <p className="mb-4">
        Se analizaron {data.total_complaints || '?'} quejas y se identificaron {data.topics_count || Object.keys(data.topics).length} temas principales.
      </p>
      
      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex -mb-px">
          <button
            onClick={() => setActiveTab('overview')}
            className={`py-2 px-4 text-center border-b-2 font-medium text-sm ${
              activeTab === 'overview'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Resumen
          </button>
          {data.trends && (
            <button
              onClick={() => setActiveTab('trends')}
              className={`py-2 px-4 text-center border-b-2 font-medium text-sm ${
                activeTab === 'trends'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Tendencias
            </button>
          )}
          <button
            onClick={() => setActiveTab('documents')}
            className={`py-2 px-4 text-center border-b-2 font-medium text-sm ${
              activeTab === 'documents'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Documentos
          </button>
        </nav>
      </div>
      
      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <>
          {/* Sentiment Summary */}
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="text-lg font-medium mb-3">Distribución de Sentimiento</h3>
            <div className="flex items-center mb-2">
              <div className="w-full bg-gray-200 rounded-full h-4">
                <div 
                  className="bg-green-500 h-4 rounded-l-full" 
                  style={{ width: `${sentimentSummary.positive}%` }}
                />
                <div 
                  className="bg-yellow-400 h-4" 
                  style={{ 
                    width: `${sentimentSummary.neutral}%`,
                    marginLeft: `${sentimentSummary.positive}%`,
                    marginTop: '-16px' // h-4 = 16px
                  }}
                />
                <div 
                  className="bg-red-500 h-4 rounded-r-full" 
                  style={{ 
                    width: `${sentimentSummary.negative}%`,
                    marginLeft: `${sentimentSummary.positive + sentimentSummary.neutral}%`,
                    marginTop: '-16px' // h-4 = 16px
                  }}
                />
              </div>
            </div>
            <div className="flex justify-between text-sm">
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-green-500 mr-1"></div>
                <span>Positivo: {sentimentSummary.positive}%</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-yellow-400 mr-1"></div>
                <span>Neutral: {sentimentSummary.neutral}%</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-red-500 mr-1"></div>
                <span>Negativo: {sentimentSummary.negative}%</span>
              </div>
            </div>
          </div>
          
          {/* Topics Grid */}
          <div className="mb-4">
            <h3 className="text-lg font-medium mb-2">Temas Identificados</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(data.topics).map(([topicId, topicData]) => (
                <div 
                  key={topicId}
                  className={`border rounded-lg p-4 cursor-pointer hover:bg-blue-50 ${
                    selectedTopic === topicId ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                  }`}
                  onClick={() => handleTopicClick(topicId)}
                >
                  <h4 className="font-bold mb-2">
                    {topicData.name || `Tema ${topicId}`}
                  </h4>
                  <div className="mb-2">
                    <div className="text-sm text-gray-700">Palabras clave:</div>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {topicData.keywords.slice(0, 5).map((keyword, i) => (
                        <span key={i} className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                          {keyword}
                        </span>
                      ))}
                    </div>
                  </div>
                  {topicData.sentiment && (
                    <div className="text-sm text-gray-600 mt-2">
                      <div className="flex items-center">
                        <div className="w-2 h-2 rounded-full bg-green-500 mr-1"></div>
                        <span>Positivo: {topicData.sentiment.positive || 0}%</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 rounded-full bg-yellow-400 mr-1"></div>
                        <span>Neutral: {topicData.sentiment.neutral || 0}%</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 rounded-full bg-red-500 mr-1"></div>
                        <span>Negativo: {topicData.sentiment.negative || 0}%</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </>
      )}
      
      {/* Trends Tab */}
      {activeTab === 'trends' && data.trends && (
        <div className="mb-4">
          <h3 className="text-lg font-medium mb-2">Tendencias de Temas por Mes</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white">
              <thead className="bg-gray-50">
                <tr>
                  <th className="py-2 px-4 border-b text-left">Mes</th>
                  {Object.entries(data.topics).map(([id, topic]) => (
                    <th key={id} className="py-2 px-4 border-b text-left">
                      {topic.name || `Tema ${id}`}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.trends).map(([month, topics]) => (
                  <tr key={month} className="hover:bg-gray-50">
                    <td className="py-2 px-4 border-b font-medium">{month}</td>
                    {Object.entries(data.topics).map(([id, topic]) => (
                      <td key={id} className="py-2 px-4 border-b">
                        {topics[topic.name] || 0}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      
      {/* Documents Tab */}
      {activeTab === 'documents' && (
        <div className="mb-4">
          <h3 className="text-lg font-medium mb-2">Ejemplos de Documentos por Tema</h3>
          <p className="text-sm text-gray-600 mb-4">
            Seleccione un tema para ver ejemplos de quejas clasificadas en ese tema.
          </p>
          
          {selectedTopic ? (
            <div className="mt-4 border-t pt-4">
              <h3 className="text-lg font-medium mb-2">
                Ejemplos del Tema: {data.topics[selectedTopic].name || `Tema ${selectedTopic}`}
              </h3>
              <div className="bg-gray-50 p-4 rounded">
                {data.topics[selectedTopic].representative_docs.length > 0 ? (
                  data.topics[selectedTopic].representative_docs.map((doc, i) => (
                    <div key={i} className="mb-2 pb-2 border-b border-gray-200 last:border-0">
                      <p className="text-gray-800">{doc}</p>
                    </div>
                  ))
                ) : (
                  <p className="text-gray-500 italic">No hay ejemplos disponibles para este tema.</p>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              Seleccione un tema de la lista para ver ejemplos de documentos.
            </div>
          )}
        </div>
      )}
      
      {/* Timestamp Footer */}
      <div className="text-sm text-gray-600 mt-4 pt-4 border-t">
        Análisis generado el: {data.timestamp ? new Date(data.timestamp.replace(/(\d{8})_(\d{6})/, '$1T$2')).toLocaleString() : 'Fecha desconocida'}
      </div>
    </div>
  );
}

TopicAnalysisPanel.propTypes = {
  data: PropTypes.object
};