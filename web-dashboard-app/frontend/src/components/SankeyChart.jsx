import { useRef, useEffect } from "react";
import * as d3 from "d3";
import { sankey, sankeyLinkHorizontal } from "d3-sankey";
import PropTypes from "prop-types";

export default function SankeyChart({ data }) {
  const svgRef = useRef();

  useEffect(() => {
    if (!data.nodes || !data.links) return;
    const svg = d3.select(svgRef.current);
    const { width, height } = svgRef.current.getBoundingClientRect();

    svg.selectAll("*").remove(); // limpia dibujo previo

    // configura layout sankey
    const { nodes, links } = sankey()
      .nodeId((d) => d.id)
      .nodeWidth(15)
      .nodePadding(10)
      .extent([
        [1, 1],
        [width - 1, height - 1],
      ])({
      nodes: data.nodes.map((d) => ({ ...d })),
      links: data.links.map((d) => ({ ...d })),
    });

    // nodos
    svg
      .append("g")
      .selectAll("rect")
      .data(nodes)
      .join("rect")
      .attr("x", (d) => d.x0)
      .attr("y", (d) => d.y0)
      .attr("width", (d) => d.x1 - d.x0)
      .attr("height", (d) => d.y1 - d.y0)
      .attr("fill", "#3b82f6");

    // enlaces
    svg
      .append("g")
      .attr("fill", "none")
      .selectAll("path")
      .data(links)
      .join("path")
      .attr("d", sankeyLinkHorizontal())
      .attr("stroke", "#888")
      .attr("stroke-width", (d) => Math.max(1, d.width));

    // etiquetas
    svg
      .append("g")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .attr("x", (d) => d.x0 - 6)
      .attr("y", (d) => (d.y1 + d.y0) / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .style("font-size", "10px")
      .text((d) => d.id);
  }, [data]);

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg mb-2 font-semibold">Flujo de Quejas (Sankey)</h3>
      <svg ref={svgRef} className="w-full h-64"></svg>
    </div>
  );
}

// PropTypes for type checking
SankeyChart.propTypes = {
  data: PropTypes.shape({
    nodes: PropTypes.arrayOf(
      PropTypes.shape({
        id: PropTypes.string.isRequired,
        name: PropTypes.string,
      })
    ).isRequired,
    links: PropTypes.arrayOf(
      PropTypes.shape({
        source: PropTypes.string.isRequired,
        target: PropTypes.string.isRequired,
        value: PropTypes.number.isRequired,
      })
    ).isRequired,
  }).isRequired,
};
