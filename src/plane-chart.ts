import * as THREE from "three";
import type { NeuralNetwork } from "./network.ts";
import type { NetworkVisualization } from "./visualization.ts";
import type { ReadoutCharts } from "./charts.ts";
import {
  computeExtent,
  computePlaneReadout,
  type LayoutExtent,
  type PlaneReadoutParams,
} from "./plane-readout.ts";

const CHART_SIZE = 256;
const CHART_GAP = 8;
const READOUT_CHART_HEIGHT = 48;
const READOUT_GAP = 8;
const TRAIL_LEN = 192;
const MARKER_RADIUS = 4;
const OPACITY_REF = 4;
const EPS_EXTENT = 1e-6;

const POS_COLOR_HEX = 0xff6b5e;
const NEG_COLOR_HEX = 0x4fc3f7;
const SINGLE_COLOR_HEX = 0xf5c249;

const PALETTE = {
  paper: new THREE.Color(0xffffff),
  paperDark: new THREE.Color(0x00002b),
};

interface TrailState {
  line: THREE.Line;
  positions: Float32Array;
  colors: Float32Array;
  initialized: boolean;
  color: THREE.Color;
}

interface MarkerState {
  mesh: THREE.Mesh;
  material: THREE.MeshBasicMaterial;
}

export interface PlaneChart {
  extent: LayoutExtent;
  borderMaterials: THREE.LineBasicMaterial[];
  planeOutlineMaterial: THREE.LineBasicMaterial;
  planeFillMaterial: THREE.MeshBasicMaterial;
  update(
    net: NeuralNetwork,
    nodePositions: Float32Array,
    params: PlaneReadoutParams,
    playing: boolean,
    darkMode: boolean,
  ): void;
  onLayoutChanged(nodePositions: Float32Array, n: number): void;
  setPlaneZ(z: number): void;
  getExtent(): LayoutExtent;
  dispose(): void;
}

function planeTopY(numReadouts: number): number {
  return -numReadouts * (READOUT_CHART_HEIGHT + READOUT_GAP) - CHART_GAP;
}

function makeTrail(colorHex: number, group: THREE.Group): TrailState {
  const positions = new Float32Array(TRAIL_LEN * 3);
  const colors = new Float32Array(TRAIL_LEN * 3);
  const geo = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(positions, 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("position", posAttr);
  const colAttr = new THREE.BufferAttribute(colors, 3);
  colAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("color", colAttr);
  const mat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
  });
  const line = new THREE.Line(geo, mat);
  line.renderOrder = 1;
  group.add(line);
  return {
    line,
    positions,
    colors,
    initialized: false,
    color: new THREE.Color(colorHex),
  };
}

function disposeTrail(trail: TrailState, group: THREE.Group): void {
  group.remove(trail.line);
  trail.line.geometry.dispose();
  (trail.line.material as THREE.Material).dispose();
}

function makeMarker(colorHex: number, group: THREE.Group): MarkerState {
  const geo = new THREE.CircleGeometry(MARKER_RADIUS, 24);
  const mat = new THREE.MeshBasicMaterial({
    color: colorHex,
    transparent: true,
    opacity: 0,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = 2;
  group.add(mesh);
  return { mesh, material: mat };
}

function disposeMarker(marker: MarkerState, group: THREE.Group): void {
  group.remove(marker.mesh);
  marker.mesh.geometry.dispose();
  marker.material.dispose();
}

function fillTrailColors(trail: TrailState, bg: THREE.Color): void {
  const { colors, color } = trail;
  const last = TRAIL_LEN - 1;
  for (let i = 0; i < TRAIL_LEN; i++) {
    const t = last === 0 ? 1 : i / last;
    const r = bg.r + (color.r - bg.r) * t;
    const g = bg.g + (color.g - bg.g) * t;
    const b = bg.b + (color.b - bg.b) * t;
    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  (trail.line.geometry.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
}

function shiftAndPushTrail(
  trail: TrailState,
  chartX: number,
  chartY: number,
): void {
  const { positions } = trail;
  if (!trail.initialized) {
    for (let i = 0; i < TRAIL_LEN; i++) {
      positions[i * 3] = chartX;
      positions[i * 3 + 1] = chartY;
      positions[i * 3 + 2] = 0;
    }
    trail.initialized = true;
  } else {
    for (let i = 0; i < TRAIL_LEN - 1; i++) {
      positions[i * 3] = positions[(i + 1) * 3];
      positions[i * 3 + 1] = positions[(i + 1) * 3 + 1];
    }
    positions[(TRAIL_LEN - 1) * 3] = chartX;
    positions[(TRAIL_LEN - 1) * 3 + 1] = chartY;
  }
  (trail.line.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
}

export function createPlaneChart(
  net: NeuralNetwork,
  viz: NetworkVisualization,
  charts: ReadoutCharts,
): PlaneChart {
  const topY = planeTopY(net.numReadouts);
  const bottomY = topY - CHART_SIZE;
  const group = charts.group;

  const borderVerts = new Float32Array([
    0, topY, 0,
    CHART_SIZE, topY, 0,
    CHART_SIZE, bottomY, 0,
    0, bottomY, 0,
    0, topY, 0,
  ]);
  const borderGeo = new THREE.BufferGeometry();
  borderGeo.setAttribute("position", new THREE.BufferAttribute(borderVerts, 3));
  const borderMat = new THREE.LineBasicMaterial({
    color: 0x444444,
    transparent: true,
    opacity: 0.5,
  });
  const border = new THREE.Line(borderGeo, borderMat);
  group.add(border);

  const posTrail = makeTrail(POS_COLOR_HEX, group);
  const negTrail = makeTrail(NEG_COLOR_HEX, group);
  const singleTrail = makeTrail(SINGLE_COLOR_HEX, group);
  fillTrailColors(posTrail, PALETTE.paper);
  fillTrailColors(negTrail, PALETTE.paper);
  fillTrailColors(singleTrail, PALETTE.paper);

  const posMarker = makeMarker(POS_COLOR_HEX, group);
  const negMarker = makeMarker(NEG_COLOR_HEX, group);
  const singleMarker = makeMarker(SINGLE_COLOR_HEX, group);

  let extent = computeExtent(viz.nodePositions, net.numNeurons);

  const planeGeo = new THREE.PlaneGeometry(1, 1);
  const planeFillMat = new THREE.MeshBasicMaterial({
    color: 0x888888,
    transparent: true,
    opacity: 0.08,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const planeMesh = new THREE.Mesh(planeGeo, planeFillMat);
  viz.scene.add(planeMesh);

  const planeOutlineGeo = new THREE.BufferGeometry();
  const outlineVerts = new Float32Array(15);
  planeOutlineGeo.setAttribute("position", new THREE.BufferAttribute(outlineVerts, 3));
  const planeOutlineMat = new THREE.LineBasicMaterial({
    color: 0x444444,
    transparent: true,
    opacity: 0.4,
  });
  const planeOutline = new THREE.Line(planeOutlineGeo, planeOutlineMat);
  viz.scene.add(planeOutline);

  let currentZ = 0;

  function updatePlaneGeometry(): void {
    const cx = (extent.minX + extent.maxX) / 2;
    const cy = (extent.minY + extent.maxY) / 2;
    const wx = Math.max(2, extent.maxX - extent.minX);
    const wy = Math.max(2, extent.maxY - extent.minY);

    planeMesh.position.set(cx, cy, currentZ);
    planeMesh.scale.set(wx, wy, 1);

    const hx = wx / 2;
    const hy = wy / 2;
    const v = outlineVerts;
    v[0] = cx - hx; v[1] = cy - hy; v[2] = currentZ;
    v[3] = cx + hx; v[4] = cy - hy; v[5] = currentZ;
    v[6] = cx + hx; v[7] = cy + hy; v[8] = currentZ;
    v[9] = cx - hx; v[10] = cy + hy; v[11] = currentZ;
    v[12] = cx - hx; v[13] = cy - hy; v[14] = currentZ;
    (planeOutlineGeo.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
  }

  updatePlaneGeometry();

  function worldToChart(wx: number, wy: number): { cx: number; cy: number } {
    const nx = (wx - extent.minX) / Math.max(EPS_EXTENT, extent.maxX - extent.minX);
    const ny = (wy - extent.minY) / Math.max(EPS_EXTENT, extent.maxY - extent.minY);
    const cx = nx * CHART_SIZE;
    const cy = bottomY + ny * CHART_SIZE;
    return { cx, cy };
  }

  let lastBg: THREE.Color | null = null;

  const planeChart: PlaneChart = {
    extent,
    borderMaterials: [borderMat],
    planeOutlineMaterial: planeOutlineMat,
    planeFillMaterial: planeFillMat,

    update(net, nodePositions, params, playing, darkMode) {
      if (params.z !== currentZ) {
        currentZ = params.z;
        updatePlaneGeometry();
      }

      const result = computePlaneReadout(net, nodePositions, params);
      const isDipole = result.isDipole;

      const bg = darkMode ? PALETTE.paperDark : PALETTE.paper;

      function handle(
        centroid: typeof result.pos,
        trail: TrailState,
        marker: MarkerState,
        visible: boolean,
      ) {
        trail.line.visible = visible;
        marker.mesh.visible = visible;
        if (!visible) {
          marker.material.opacity = 0;
          return;
        }

        let targetOpacity = 0;
        if (centroid.valid) {
          const chart = worldToChart(centroid.x, centroid.y);
          marker.mesh.position.set(chart.cx, chart.cy, 0);
          targetOpacity = 1 - Math.exp(-centroid.mag / OPACITY_REF);
          if (playing) shiftAndPushTrail(trail, chart.cx, chart.cy);
        } else {
          if (playing && trail.initialized) {
            const lastX = trail.positions[(TRAIL_LEN - 1) * 3];
            const lastY = trail.positions[(TRAIL_LEN - 1) * 3 + 1];
            shiftAndPushTrail(trail, lastX, lastY);
          }
        }
        marker.material.opacity = targetOpacity;
      }

      if (isDipole) {
        handle(result.pos, posTrail, posMarker, true);
        handle(result.neg, negTrail, negMarker, true);
        singleTrail.line.visible = false;
        singleMarker.mesh.visible = false;
        singleMarker.material.opacity = 0;
      } else {
        handle(result.pos, singleTrail, singleMarker, true);
        posTrail.line.visible = false;
        posMarker.mesh.visible = false;
        posMarker.material.opacity = 0;
        negTrail.line.visible = false;
        negMarker.mesh.visible = false;
        negMarker.material.opacity = 0;
      }

      if (bg !== lastBg) {
        fillTrailColors(posTrail, bg);
        fillTrailColors(negTrail, bg);
        fillTrailColors(singleTrail, bg);
        lastBg = bg;
      }
    },

    onLayoutChanged(nodePositions, n) {
      extent = computeExtent(nodePositions, n);
      planeChart.extent = extent;
      updatePlaneGeometry();
      posTrail.initialized = false;
      negTrail.initialized = false;
      singleTrail.initialized = false;
    },

    setPlaneZ(z) {
      if (z === currentZ) return;
      currentZ = z;
      updatePlaneGeometry();
    },

    getExtent() {
      return extent;
    },

    dispose() {
      group.remove(border);
      borderGeo.dispose();
      borderMat.dispose();

      disposeTrail(posTrail, group);
      disposeTrail(negTrail, group);
      disposeTrail(singleTrail, group);

      disposeMarker(posMarker, group);
      disposeMarker(negMarker, group);
      disposeMarker(singleMarker, group);

      viz.scene.remove(planeMesh);
      viz.scene.remove(planeOutline);
      planeGeo.dispose();
      planeFillMat.dispose();
      planeOutlineGeo.dispose();
      planeOutlineMat.dispose();
    },
  };

  return planeChart;
}
