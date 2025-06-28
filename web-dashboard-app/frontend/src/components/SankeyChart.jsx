import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import { sankey, sankeyLinkHorizontal } from "d3-sankey";
import PropTypes from "prop-types";

export default function SankeyChart({ data }) {
  const svgRef = useRef();
  const [tooltipData, setTooltipData] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!data.nodes || !data.links || data.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const { width, height } = svgRef.current.getBoundingClientRect();

    svg.selectAll("*").remove(); // clear previous drawing

    // Setup Sankey layout
    const { nodes, links } = sankey()
      .nodeId((d) => d.id)
      .nodeWidth(20)
      .nodePadding(15)
      .extent([
        [20, 20],
        [width - 20, height - 20],
      ])({
      nodes: data.nodes.map((d) => ({ ...d })),
      links: data.links.map((d) => ({ ...d })),
    });

    // Calculate node periods for coloring
    const periods = [...new Set(nodes.map((n) => n.id.split(":")[0]))];
    const colorScale = d3
      .scaleOrdinal()
      .domain(periods)
      .range(d3.schemeCategory10);

    // Add links
    svg
      .append("g")
      .attr("fill", "none")
      .selectAll("path")
      .data(links)
      .join("path")
      .attr("d", sankeyLinkHorizontal())
      .attr("stroke", "#aaa")
      .attr("stroke-opacity", 0.5)
      .attr("stroke-width", (d) => Math.max(1, d.width))
      .style("mix-blend-mode", "multiply")
      .on("mouseover", function (event, d) {
        d3.select(this).attr("stroke", "#000").attr("stroke-opacity", 0.8);

        const sourceNode = nodes.find((n) => n.id === d.source.id);
        const targetNode = nodes.find((n) => n.id === d.target.id);

        const sourceName = sourceNode.id.split(":")[1];
        const targetName = targetNode.id.split(":")[1];

        setTooltipData({
          source: sourceName,
          target: targetName,
          value: d.value,
          sourceDate: sourceNode.id.split(":")[0],
          targetDate: targetNode.id.split(":")[0],
        });

        setTooltipPos({
          x: event.pageX,
          y: event.pageY,
        });
      })
      .on("mouseout", function () {
        d3.select(this).attr("stroke", "#aaa").attr("stroke-opacity", 0.5);

        setTooltipData(null);
      });

    // Add nodes
    const nodeGroup = svg.append("g").selectAll("g").data(nodes).join("g");

    // Node rectangles
    nodeGroup
      .append("rect")
      .attr("x", (d) => d.x0)
      .attr("y", (d) => d.y0)
      .attr("width", (d) => d.x1 - d.x0)
      .attr("height", (d) => d.y1 - d.y0)
      .attr("fill", (d) => colorScale(d.id.split(":")[0]))
      .attr("opacity", 0.8)
      .attr("stroke", "#555");

    // Node labels
    nodeGroup
      .append("text")
      .attr("x", (d) => (d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6))
      .attr("y", (d) => (d.y1 + d.y0) / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", (d) => (d.x0 < width / 2 ? "start" : "end"))
      .attr("font-size", "10px")
      .attr("font-weight", "bold")
      .attr("pointer-events", "none")
      .text((d) => d.id.split(":")[1]);

    // Add period labels at the top
    svg
      .append("g")
      .selectAll("text")
      .data(periods)
      .join("text")
      .attr("x", (d, i) => (width / periods.length) * (i + 0.5))
      .attr("y", 10)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .attr("font-weight", "bold")
      .text((d) => d);
  }, [data]);

  return (
    <div className="relative">
      <svg ref={svgRef} className="w-full h-[600px]"></svg>

      {tooltipData && (
        <div
          className="absolute bg-white p-2 rounded shadow-lg border text-sm z-10"
          style={{
            left: `${tooltipPos.x + 10}px`,
            top: `${tooltipPos.y + 10}px`,
            transform: "translate(-50%, -100%)",
          }}
        >
          <p className="font-semibold">
            {tooltipData.sourceDate} → {tooltipData.targetDate}
          </p>
          <p>
            {tooltipData.source} → {tooltipData.target}
          </p>
          <p className="text-blue-600 font-bold">{tooltipData.value} quejas</p>
        </div>
      )}
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
    ),
    links: PropTypes.arrayOf(
      PropTypes.shape({
        source: PropTypes.oneOfType([
          PropTypes.string,
          PropTypes.number,
          PropTypes.object,
        ]).isRequired,
        target: PropTypes.oneOfType([
          PropTypes.string,
          PropTypes.number,
          PropTypes.object,
        ]).isRequired,
        value: PropTypes.number.isRequired,
      })
    ),
  }).isRequired,
};
