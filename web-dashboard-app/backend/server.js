require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const errorHandler = require('./middlewares/errorHandler');
const csv = require('csv-parser');

// Initialize Supabase client
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;

// Check if environment variables are set
if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('Missing SUPABASE_URL or SUPABASE_KEY environment variables');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const app = express();
app.use(cors());
app.use(express.json());

// Debug connection info (remove in production)
console.log('Supabase Connection Setup:', {
  url: SUPABASE_URL ? 'URL configured' : 'Missing URL',
  key: SUPABASE_KEY ? 'Key configured' : 'Missing key'
});

// Enable detailed logging for debugging
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// Make sure CORS is properly configured
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5173'], // Add your frontend origins
  credentials: true
}));

// Test connection with detailed diagnostics
app.get('/api/test', async (req, res) => {
  try {
    console.log('Test endpoint called with Supabase URL:', SUPABASE_URL);

    // First check if we can connect to Supabase at all
    const { data: connectionTest, error: connectionError } = await supabase
      .from('cx_quejas')
      .select('*', { count: 'exact', head: true })
      .limit(0);

    if (connectionError) {
      console.error('Connection test failed:', connectionError);
      return res.status(500).json({
        connection: 'failed',
        error: connectionError.message,
        code: connectionError.code,
        details: connectionError.details,
        hint: connectionError.hint
      });
    }

    // If connection succeeded, get a few rows for display
    const { data, error } = await supabase
      .from('cx_quejas')
      .select('*')
      .limit(5);

    if (error) {
      console.error('Data fetch error:', error);
      return res.status(500).json({
        connection: 'success',
        data_fetch: 'failed',
        error: error.message
      });
    }

    console.log('Successfully fetched test data:',
      data ? `${data.length} rows` : 'No data');

    res.json({
      status: 'success',
      connection: 'ok',
      count: connectionTest?.[0]?.count || 0,
      sample_data: data
    });
  } catch (err) {
    console.error('Unexpected error in test endpoint:', err);
    res.status(500).json({
      status: 'error',
      error: err.message,
      stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
    });
  }
});

// Test Supabase connection
app.get('/api/test-supabase', async (req, res) => {
  try {
    console.log('Testing Supabase connection...');

    // Test the connection
    const { data, error, count } = await supabase
      .from('cx_quejas')
      .select('*', { count: 'exact', head: true })
      .limit(0);

    if (error) {
      console.error('Supabase connection error:', error);
      return res.status(500).json({
        status: 'error',
        message: 'Failed to connect to Supabase',
        error: error.message
      });
    }

    // Get some sample data
    const { data: sampleData, error: sampleError } = await supabase
      .from('cx_quejas')
      .select('*')
      .limit(2);

    if (sampleError) {
      console.error('Error fetching sample data:', sampleError);
    }

    res.json({
      status: 'success',
      message: 'Successfully connected to Supabase',
      count: count || 0,
      sample: sampleData || []
    });
  } catch (err) {
    console.error('Unexpected error testing Supabase:', err);
    res.status(500).json({
      status: 'error',
      message: 'Unexpected error testing Supabase connection',
      error: err.message
    });
  }
});

// GET /api/complaints/top10 - Get top 10 complaint categories with counts
app.get('/api/complaints/top10', async (req, res) => {
  const { dateFrom, dateTo, search } = req.query;

  try {
    console.log('Received request for /api/complaints/top10:', { dateFrom, dateTo, search });

    // Get data from CSV instead of Supabase
    const data = await getCsvData();
    let filteredData = [...data];

    // Apply filters
    if (dateFrom) {
      filteredData = filteredData.filter(c => new Date(c["Fecha de interacción"]) >= new Date(dateFrom));
    }
    if (dateTo) {
      filteredData = filteredData.filter(c => new Date(c["Fecha de interacción"]) <= new Date(dateTo));
    }
    if (search) {
      const term = search.toLowerCase();
      filteredData = filteredData.filter(c =>
        (c["Comentarios"] && c["Comentarios"].toLowerCase().includes(term)) ||
        (c["Motivo de su solicitud"] && c["Motivo de su solicitud"].toLowerCase().includes(term))
      );
    }

    // Group by category
    const counts = {};
    filteredData.forEach(row => {
      const cat = row["Motivo de su solicitud"];
      if (cat) {
        counts[cat] = (counts[cat] || 0) + 1;
      }
    });

    // Build top-10 array
    const top10 = Object.entries(counts)
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    console.log(`Returning ${top10.length} categories`);
    res.json(top10);
  } catch (err) {
    console.error('Unexpected error in /api/complaints/top10:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/complaints/by-category - Get percentage breakdown of all complaint categories
app.get('/api/complaints/by-category', async (req, res) => {
  const { dateFrom, dateTo, search } = req.query;

  try {
    // Get data from CSV
    const data = await getCsvData();
    let filteredData = [...data];

    // Apply filters
    if (dateFrom) {
      filteredData = filteredData.filter(c => new Date(c["Fecha de interacción"]) >= new Date(dateFrom));
    }
    if (dateTo) {
      filteredData = filteredData.filter(c => new Date(c["Fecha de interacción"]) <= new Date(dateTo));
    }
    if (search) {
      const term = search.toLowerCase();
      filteredData = filteredData.filter(c => c["Motivo de su solicitud"] && c["Motivo de su solicitud"].toLowerCase().includes(term));
    }

    // Group and calculate percentages
    const counts = {};
    let total = 0;

    filteredData.forEach(row => {
      const cat = row["Motivo de su solicitud"];
      if (cat) {
        counts[cat] = (counts[cat] || 0) + 1;
        total++;
      }
    });

    const result = Object.entries(counts).map(([label, count]) => ({
      label,
      count,
      percentage: Math.round((count / total) * 100)
    }));

    res.json(result);
  } catch (err) {
    console.error('Unexpected error in /api/complaints/by-category:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/complaints - Get detailed complaints with filtering options
app.get('/api/complaints', async (req, res) => {
  const { category, search, dateFrom, dateTo, page = 1, limit = 20 } = req.query;

  try {
    // Get data from CSV
    const data = await getCsvData();
    let filteredData = [...data];

    // Apply filters
    if (category) {
      filteredData = filteredData.filter(c => c["Motivo de su solicitud"] === category);
    }
    if (search) {
      filteredData = filteredData.filter(c => c["Comentarios"] && c["Comentarios"].toLowerCase().includes(search.toLowerCase()));
    }
    if (dateFrom) {
      filteredData = filteredData.filter(c => new Date(c["Fecha de interacción"]) >= new Date(dateFrom));
    }
    if (dateTo) {
      filteredData = filteredData.filter(c => new Date(c["Fecha de interacción"]) <= new Date(dateTo));
    }

    // Add pagination (manually)
    const totalCount = filteredData.length;
    const from = (parseInt(page) - 1) * parseInt(limit);
    const to = from + parseInt(limit);
    const paginatedData = filteredData.slice(from, to);

    res.json({
      data: paginatedData,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total: totalCount
      }
    });
  } catch (err) {
    console.error('Unexpected error in /api/complaints:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/complaints/trends - Get complaint trends over time
app.get('/api/complaints/trends', async (req, res) => {
  const { dateFrom, dateTo, category } = req.query;

  try {
    // Get data from CSV
    const data = await getCsvData();
    let filteredData = [...data];

    // Apply filters
    if (dateFrom) {
      filteredData = filteredData.filter(c => new Date(c["Fecha de interacción"]) >= new Date(dateFrom));
    }
    if (dateTo) {
      filteredData = filteredData.filter(c => new Date(c["Fecha de interacción"]) <= new Date(dateTo));
    }
    if (category) {
      filteredData = filteredData.filter(c => c["Motivo de su solicitud"] === category);
    }

    // Group by month and category
    const trends = {};

    filteredData.forEach(row => {
      if (row["Fecha de interacción"]) {
        const date = new Date(row["Fecha de interacción"]);
        const month = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        const category = row["Motivo de su solicitud"];

        if (category) {
          if (!trends[month]) {
            trends[month] = {};
          }

          trends[month][category] = (trends[month][category] || 0) + 1;
        }
      }
    });

    // Convert to array format for the frontend
    const result = Object.entries(trends).map(([month, categories]) => {
      return {
        month,
        ...categories
      };
    }).sort((a, b) => a.month.localeCompare(b.month));

    res.json(result);
  } catch (err) {
    console.error('Unexpected error in /api/complaints/trends:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/sankey - Get data for Sankey diagram
app.get('/api/sankey', async (req, res) => {
  const { dateFrom, dateTo } = req.query;
  const MAX_PERIODS = 2; // Months max for performance
  const MAX_CATEGORIES = 10; 
  const MAX_LINKS = 100;

  try {
    console.log('Received request for /api/sankey:', { dateFrom, dateTo });

    // Get data from CSV
    const data = await getCsvData();
    
    // Reset cache if we've previously had no data
    if (!data || data.length === 0) {
      csvDataCache = null;
      console.log('CSV data cache was empty, clearing cache');
    }
    
    let filteredData = [...data];
    console.log(`Starting with ${filteredData.length} total records`);

    // Apply date filters
    if (dateFrom) {
      filteredData = filteredData.filter(c => {
        const interactionDate = new Date(c["Fecha de interacción"]);
        const fromDate = new Date(dateFrom);
        return interactionDate >= fromDate;
      });
    }
    
    if (dateTo) {
      filteredData = filteredData.filter(c => {
        const interactionDate = new Date(c["Fecha de interacción"]);
        const toDate = new Date(dateTo);
        return interactionDate <= toDate;
      });
    }

    console.log(`After date filtering: ${filteredData.length} records`);

    // If there's no data, return empty structure with a more helpful message
    if (!filteredData.length) {
      console.log('No data remains after filtering');
      return res.status(404).json({ 
        error: "No hay datos disponibles para el rango de fechas seleccionado.",
        nodes: [], 
        links: [] 
      });
    }

    // Analyze date range for cross-month requests
    const dates = filteredData.map(row => new Date(row["Fecha de interacción"]));
    const minDate = new Date(Math.min(...dates));
    const maxDate = new Date(Math.max(...dates));
    
    const monthDiff = (maxDate.getFullYear() - minDate.getFullYear()) * 12 + 
                     (maxDate.getMonth() - minDate.getMonth());
    
    console.log(`Date range spans ${monthDiff} months from ${minDate.toISOString()} to ${maxDate.toISOString()}`);

    // Verify date range is reasonable
    if (monthDiff > MAX_PERIODS) {
      return res.status(400).json({
        error: `Por favor seleccione un rango máximo de ${MAX_PERIODS} meses para el diagrama Sankey.`
      });
    }

    // Performance optimization for large datasets
    if (filteredData.length > 1000) {
      console.log(`Sampling data from ${filteredData.length} to 1000 records`);
      const sampleRate = Math.ceil(filteredData.length / 1000);
      filteredData = filteredData.filter((_, index) => index % sampleRate === 0);
    }

    // Determine whether to use weeks or months based on date range
    const useWeeks = monthDiff < 1;
    console.log(`Using ${useWeeks ? 'weekly' : 'monthly'} periods for visualization`);

    // Group by period and category
    const byPeriod = {};
    filteredData.forEach(row => {
      if (row["Fecha de interacción"] && row["Motivo de su solicitud"]) {
        const date = new Date(row["Fecha de interacción"]);
        const cat = row["Motivo de su solicitud"];
        let period;
        
        if (useWeeks) {
          // Group by week within month
          const weekNumber = Math.floor(date.getDate() / 7) + 1;
          period = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-W${weekNumber}`;
        } else {
          // Group by month
          period = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        }
        
        byPeriod[period] ??= {};
        byPeriod[period][cat] = (byPeriod[period][cat] || 0) + 1;
      }
    });

    // Get periods and check if we have enough
    const periods = Object.keys(byPeriod).sort();
    console.log(`Found ${periods.length} time periods: ${periods.join(', ')}`);
    
    if (periods.length < 2) {
      return res.status(400).json({ 
        error: "No hay suficientes periodos para generar el diagrama Sankey. Seleccione un rango más amplio."
      });
    }

    // Limit to most common categories
    const allCategories = {};
    Object.values(byPeriod).forEach(periodData => {
      Object.entries(periodData).forEach(([cat, count]) => {
        allCategories[cat] = (allCategories[cat] || 0) + count;
      });
    });

    // Get top categories
    const topCategories = Object.entries(allCategories)
      .sort((a, b) => b[1] - a[1])
      .slice(0, MAX_CATEGORIES)
      .map(([cat]) => cat);
    
    console.log(`Selected top ${topCategories.length} categories`);

    // Filter periods to only include top categories
    Object.keys(byPeriod).forEach(period => {
      const filteredPeriodData = {};
      Object.entries(byPeriod[period])
        .filter(([cat]) => topCategories.includes(cat))
        .forEach(([cat, count]) => {
          filteredPeriodData[cat] = count;
        });
      byPeriod[period] = filteredPeriodData;
    });

    // Build nodes and links with the filtered data
    const nodes = [];
    const index = {};
    let links = [];
    let idx = 0;

    // Create nodes for each category in each period
    periods.forEach(p => {
      Object.keys(byPeriod[p]).forEach(cat => {
        const key = `${p}:${cat}`;
        index[key] = idx;
        
        // Format the period name for display
        let displayPeriod;
        if (p.includes('-W')) {
          // For weeks: "Semana X de Mes"
          const [yearMonth, week] = p.split('-W');
          const [year, month] = yearMonth.split('-');
          const monthName = new Date(year, parseInt(month) - 1, 1).toLocaleString('es-ES', { month: 'long' });
          displayPeriod = `Semana ${week} de ${monthName}`;
        } else {
          // For months: "Enero 2025"
          const [year, month] = p.split('-');
          displayPeriod = new Date(year, parseInt(month) - 1, 1).toLocaleString('es-ES', { month: 'long', year: 'numeric' });
        }
        
        nodes.push({
          id: idx.toString(),
          name: `${displayPeriod} - ${cat}`
        });
        idx++;
      });
    });

    // Create links between periods
    for (let i = 0; i < periods.length - 1; i++) {
      const p0 = periods[i], p1 = periods[i + 1];
      Object.entries(byPeriod[p0]).forEach(([cat0, cnt0]) => {
        Object.entries(byPeriod[p1]).forEach(([cat1, cnt1]) => {
          // Use Math.max(1, ...) to ensure there's at least a minimal connection
          const value = Math.max(1, Math.min(cnt0, cnt1));
          links.push({
            source: index[`${p0}:${cat0}`].toString(),
            target: index[`${p1}:${cat1}`].toString(),
            value
          });
        });
      });
    }

    // Limit the total number of links
    if (links.length > MAX_LINKS) {
      console.log(`Limiting links from ${links.length} to ${MAX_LINKS}`);
      links.sort((a, b) => b.value - a.value);
      links = links.slice(0, MAX_LINKS);
    }

    console.log(`Returning Sankey with ${nodes.length} nodes and ${links.length} links`);
    res.json({ nodes, links });
  } catch (err) {
    console.error('Unexpected error in /api/sankey:', err);
    res.status(500).json({ error: err.message });
  }
});

// Add to server.js
app.get('/api/test-ml-dir', (req, res) => {
  const mlDir = path.join(__dirname, '..', 'ml');
  const outputDir = path.join(mlDir, 'output');

  try {
    // Check if ML directory exists
    const mlDirExists = fs.existsSync(mlDir);

    // Create output directory if it doesn't exist
    let outputDirExists = fs.existsSync(outputDir);
    if (!outputDirExists) {
      fs.mkdirSync(outputDir, { recursive: true });
      outputDirExists = true;
    }

    // Try to write a test file
    const testFile = path.join(outputDir, 'test.json');
    fs.writeFileSync(testFile, JSON.stringify({ test: 'success', date: new Date().toISOString() }));

    res.json({
      status: 'success',
      ml_dir: { exists: mlDirExists, path: mlDir },
      output_dir: { exists: outputDirExists, path: outputDir },
      test_file: { written: true, path: testFile }
    });
  } catch (err) {
    res.status(500).json({
      status: 'error',
      message: 'Error testing ML directories',
      error: err.message,
      stack: err.stack
    });
  }
});

// Add this to your server.js
app.get('/api/seed-sample-data', async (req, res) => {
  try {
    // Check if we already have data
    const { count, error: countError } = await supabase
      .from('cx_quejas')
      .select('*', { count: 'exact', head: true })
      .limit(0);

    if (countError) {
      return res.status(500).json({ error: countError.message });
    }

    if (count > 0) {
      return res.json({ message: `Database already contains ${count} records`, count });
    }

    // Sample data to insert
    const sampleData = [
      {
        "Motivo de su solicitud": "Problema técnico",
        "Comentarios": "La aplicación no funciona correctamente en mi dispositivo",
        "Fecha de interacción": "2025-06-20"
      },
      {
        "Motivo de su solicitud": "Facturación",
        "Comentarios": "Me cobraron dos veces por el mismo servicio",
        "Fecha de interacción": "2025-06-21"
      },
      {
        "Motivo de su solicitud": "Problema técnico",
        "Comentarios": "No puedo acceder a mi cuenta",
        "Fecha de interacción": "2025-06-22"
      },
      {
        "Motivo de su solicitud": "Servicio al cliente",
        "Comentarios": "Excelente atención, gracias por resolver mi problema",
        "Fecha de interacción": "2025-06-23"
      },
      {
        "Motivo de su solicitud": "Facturación",
        "Comentarios": "Error en el monto de mi factura",
        "Fecha de interacción": "2025-06-24"
      }
    ];

    // Insert the sample data
    const { data, error } = await supabase
      .from('cx_quejas')
      .insert(sampleData);

    if (error) {
      return res.status(500).json({ error: error.message });
    }

    res.json({ success: true, message: 'Sample data inserted', count: sampleData.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/data-status', async (req, res) => {
  try {
    // Get data from CSV
    const data = await getCsvData();

    // Get date range
    const dates = data
      .map(item => new Date(item["Fecha de interacción"]))
      .filter(date => !isNaN(date.getTime()));

    const earliest = dates.length > 0 ? new Date(Math.min(...dates)) : null;
    const latest = dates.length > 0 ? new Date(Math.max(...dates)) : null;

    // Get categories
    const categories = [...new Set(data
      .map(item => item["Motivo de su solicitud"])
      .filter(cat => cat) // Remove undefined/null
    )];

    res.json({
      recordCount: data.length,
      dateRange: {
        earliest: earliest ? earliest.toISOString() : null,
        latest: latest ? latest.toISOString() : null
      },
      categoryCount: categories.length,
      sampleCategories: categories.slice(0, 5), // Just show a few
      status: data.length > 0 ? 'data_available' : 'no_data'
    });
  } catch (err) {
    console.error('Unexpected error checking data status:', err);
    res.status(500).json({ error: err.message });
  }
});

// Replace the loadCsvData function with this version
function loadCsvData() {
  return new Promise((resolve, reject) => {
    const results = [];
    const csvPath = path.join(__dirname, 'docs', 'Tracking de solicitudes (Responses) - Form Responses 1.csv');

    fs.createReadStream(csvPath)
      .pipe(csv())
      .on('data', (data) => {
        // Include all data from 2025 without month filtering
        if (data["Fecha de interacción"]) {
          const date = new Date(data["Fecha de interacción"]);
          // Include all data from 2025 (including January and February)
          if (date.getFullYear() === 2025) {
            results.push(data);
          }
        } else {
          results.push(data);
        }
      })
      .on('end', () => {
        console.log(`Loaded ${results.length} records from CSV`);
        resolve(results);
      })
      .on('error', (error) => {
        reject(error);
      });
  });
}

// Cache the CSV data to avoid repeated reads
let csvDataCache = null;
async function getCsvData() {
  if (!csvDataCache) {
    csvDataCache = await loadCsvData();
  }
  return csvDataCache;
}

// Add this to your server.js
app.get('/api/seed-test-data', async (req, res) => {
  try {
    // Generate sample data for 2025
    const categories = [
      "Compra de entradas",
      "Información general sobre concierto",
      "Solicitud de información general",
      "Validación de tickets",
      "Reclamo por cobro indebido",
      "Devolución",
      "Error en la plataforma",
      "Traslado de entradas",
      "Información sobre promociones",
      "Solicitud de afiliación"
    ];

    const comments = [
      "Cliente consulta sobre horarios del evento",
      "Problemas para visualizar entradas en la plataforma",
      "Solicita reintegro por doble cargo",
      "No puede completar la compra por errores",
      "Desea información sobre las promociones disponibles",
      "Consulta sobre lugares de estacionamiento",
      "Pregunta por la edad mínima para ingresar al evento",
      "No recibió confirmación de su compra",
      "Desea transferir entradas a otra persona",
      "Excelente atención, gracias por resolver mi problema"
    ];

    const clients = [
      "María Rodríguez",
      "Juan Pérez",
      "Ana García",
      "Carlos López",
      "Laura Martínez",
      "Pedro González",
      "Sofía Ramírez",
      "Diego Hernández",
      "Valentina Torres",
      "Alejandro Castro"
    ];

    // Generate 6 months of data (July-December 2025)
    const testData = [];
    for (let month = 6; month < 12; month++) {
      // Generate 30-50 entries per month
      const entriesCount = 30 + Math.floor(Math.random() * 20);

      for (let i = 0; i < entriesCount; i++) {
        const day = 1 + Math.floor(Math.random() * 28); // 1-28
        const categoryIndex = Math.floor(Math.random() * categories.length);
        const commentIndex = Math.floor(Math.random() * comments.length);
        const clientIndex = Math.floor(Math.random() * clients.length);

        testData.push({
          "Nombre del Cliente": clients[clientIndex],
          "Motivo de su solicitud": categories[categoryIndex],
          "Comentarios": comments[commentIndex],
          "Fecha de interacción": `2025-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
        });
      }
    }

    // Override the cache with this test data
    csvDataCache = testData;

    res.json({
      success: true,
      message: `Generated ${testData.length} test records for July-December 2025`,
      count: testData.length,
      sample: testData.slice(0, 3) // Show a few samples
    });
  } catch (err) {
    console.error('Error generating test data:', err);
    res.status(500).json({ error: err.message });
  }
});

// Add this endpoint to generate data for January and February 2025
app.get('/api/seed-early-2025-data', async (req, res) => {
  try {
    // Categories and comments from existing seed data
    const categories = [
      "Compra de entradas",
      "Información general sobre concierto",
      "Solicitud de información general",
      "Validación de tickets",
      "Reclamo por cobro indebido",
      "Devolución",
      "Error en la plataforma",
      "Traslado de entradas",
      "Información sobre promociones",
      "Solicitud de afiliación"
    ];

    const comments = [
      "Cliente consulta sobre horarios del evento",
      "Problemas para visualizar entradas en la plataforma",
      "Solicita reintegro por doble cargo",
      "No puede completar la compra por errores",
      "Desea información sobre las promociones disponibles",
      "Consulta sobre lugares de estacionamiento",
      "Pregunta por la edad mínima para ingresar al evento",
      "No recibió confirmación de su compra",
      "Desea transferir entradas a otra persona",
      "Excelente atención, gracias por resolver mi problema"
    ];

    const clients = [
      "María Rodríguez",
      "Juan Pérez",
      "Ana García",
      "Carlos López",
      "Laura Martínez",
      "Pedro González",
      "Sofía Ramírez",
      "Diego Hernández",
      "Valentina Torres",
      "Alejandro Castro"
    ];

    // Generate January and February 2025 data
    const earlyData = [];
    
    // Generate data for each week to ensure good Sankey visualization
    for (let month = 0; month < 2; month++) { // January and February
      for (let week = 1; week <= 4; week++) {
        // Generate 10-15 entries per week
        const entriesCount = 10 + Math.floor(Math.random() * 6);
        
        for (let i = 0; i < entriesCount; i++) {
          // Calculate day based on week (1-7, 8-14, 15-21, 22-28)
          const day = ((week - 1) * 7) + 1 + Math.floor(Math.random() * 7);
          const categoryIndex = Math.floor(Math.random() * categories.length);
          const commentIndex = Math.floor(Math.random() * comments.length);
          const clientIndex = Math.floor(Math.random() * clients.length);

          earlyData.push({
            "Nombre del Cliente": clients[clientIndex],
            "Motivo de su solicitud": categories[categoryIndex],
            "Comentarios": comments[commentIndex],
            "Fecha de interacción": `2025-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
          });
        }
      }
    }

    // Add the early data to the existing cache
    if (!csvDataCache) {
      csvDataCache = earlyData;
    } else {
      // Filter out any existing January/February data
      csvDataCache = csvDataCache.filter(item => {
        if (!item["Fecha de interacción"]) return true;
        const date = new Date(item["Fecha de interacción"]);
        return !(date.getFullYear() === 2025 && (date.getMonth() === 0 || date.getMonth() === 1));
      });
      
      // Add the new data
      csvDataCache = [...csvDataCache, ...earlyData];
    }

    res.json({
      success: true,
      message: `Added ${earlyData.length} records for January-February 2025`,
      count: earlyData.length,
      sampleData: earlyData.slice(0, 3)
    });
  } catch (err) {
    console.error('Error generating early 2025 data:', err);
    res.status(500).json({ error: err.message });
  }
});

app.use(errorHandler);

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
