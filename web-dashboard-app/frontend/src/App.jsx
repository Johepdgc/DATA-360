import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navigation from "./components/Navigation";
import Dashboard from "./pages/Dashboard";
import SankeyView from "./pages/SankeyView";

export default function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen">
        <Navigation />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/sankey" element={<SankeyView />} />
          </Routes>
        </main>
        <footer className="bg-gray-800 text-white text-center p-4 mt-auto">
          <p>© {new Date().getFullYear()} DATA 360 - Dashboard de Quejas</p>
        </footer>
      </div>
    </Router>
  );
}
