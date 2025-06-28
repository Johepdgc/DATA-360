/**
 * Global error handler middleware
 */

const errorHandler = (err, req, res, next) => {
  console.error('Error caught by global handler:', err);
  
  // Determine if we're in development mode
  const isDev = process.env.NODE_ENV !== 'production';
  
  // Format the error response
  const errorResponse = {
    status: 'error',
    message: err.message || 'Unexpected server error',
    code: err.code || 'INTERNAL_ERROR'
  };
  
  // Add stack trace in development mode
  if (isDev) {
    errorResponse.stack = err.stack;
    errorResponse.details = err.details || err.data;
  }
  
  // Set the appropriate status code
  const statusCode = err.statusCode || 500;
  
  // Send the error response
  res.status(statusCode).json(errorResponse);
};

module.exports = errorHandler;