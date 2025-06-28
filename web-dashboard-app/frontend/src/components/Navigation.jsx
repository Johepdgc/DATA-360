import { NavLink } from "react-router-dom";

export default function Navigation() {
  return (
    <nav className="bg-blue-600 text-white shadow-md">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 font-bold text-xl">DATA 360</div>
          </div>

          <div className="flex">
            <NavLink
              to="/"
              className={({ isActive }) =>
                `px-4 py-2 rounded-md text-sm font-medium ${
                  isActive
                    ? "bg-blue-700 text-white"
                    : "text-blue-100 hover:bg-blue-500"
                }`
              }
              end
            >
              Dashboard
            </NavLink>

            <NavLink
              to="/sankey"
              className={({ isActive }) =>
                `ml-4 px-4 py-2 rounded-md text-sm font-medium ${
                  isActive
                    ? "bg-blue-700 text-white"
                    : "text-blue-100 hover:bg-blue-500"
                }`
              }
            >
              Análisis de Flujo
            </NavLink>
          </div>
        </div>
      </div>
    </nav>
  );
}
