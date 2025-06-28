#!/bin/bash

# Start frontend and backend concurrently
echo "Starting Dashboard Application..."

# Start backend server
cd backend
echo "Starting backend server..."
npm install
npm start &
BACKEND_PID=$!

# Start frontend
cd ../frontend
echo "Starting frontend server..."
npm install
npm run dev &
FRONTEND_PID=$!

# Handle termination
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT SIGTERM

# Keep script running
wait
