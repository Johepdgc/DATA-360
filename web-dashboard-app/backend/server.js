const express = require('express');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const cors = require('cors');

const app = express();
app.use(cors());

const DATA_PATH = path.join(__dirname, './docs/Tracking de solicitudes (Responses) - Form Responses 1.csv');

// Load and parse CSV into memory
let complaints = [];
fs.createReadStream(DATA_PATH)
  .pipe(csv())
  .on('data', row => complaints.push(row))
  .on('end', () => console.log('CSV file loaded:', complaints.length, 'records'));

// Endpoint: get all categories counts
app.get('/api/summary', (req, res) => {
  const { dateFrom, dateTo } = req.query;
  let data = complaints;

  if (dateFrom) {
    data = data.filter(c => new Date(c['Fecha de interacción']) >= new Date(dateFrom));
  }
  if (dateTo) {
    data = data.filter(c => new Date(c['Fecha de interacción']) <= new Date(dateTo));
  }

  const summary = data.reduce((acc, { 'Motivo de su solicitud': cat }) => {
    acc[cat] = (acc[cat] || 0) + 1;
    return acc;
  }, {});

  res.json(summary);
});

// Endpoint: get complaints, with optional filters
app.get('/api/complaints', (req, res) => {
  const { category, search, dateFrom, dateTo } = req.query;
  let data = complaints;
  if (category) {
    data = data.filter(c => c['Motivo de su solicitud'] === category);
  }
  if (search) {
    const term = search.toLowerCase();
    data = data.filter(c => c['Comentarios'] && c['Comentarios'].toLowerCase().includes(term));
  }
  if (dateFrom) {
    data = data.filter(c => new Date(c['Fecha de interacción']) >= new Date(dateFrom));
  }
  if (dateTo) {
    data = data.filter(c => new Date(c['Fecha de interacción']) <= new Date(dateTo));
  }
  res.json(data);
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`Backend listening on http://localhost:${PORT}`));
