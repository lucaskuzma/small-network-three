import * as THREE from "three";
import type { NeuralNetwork } from "./network.ts";

// ---------------------------------------------------------------------------
// Constants (pixel units, modular scale: 2,4,8,16,24,32,48,72,96,128,192,256)
// ---------------------------------------------------------------------------

const CHART_WIDTH = 256;
const CHART_HEIGHT = 48;
const CHART_GAP = 8;
const BUFFER_LEN = 300;
const MARGIN = 16;
const ANCHOR_OFFSET = 16; // px to the right of chart edge

const TRACE_COLORS = [
  0x4fc3f7, // light blue
  0x81c784, // green
  0xfff176, // yellow
  0xff8a65, // orange
  0xba68c8, // purple
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TraceData {
  line: THREE.Line;
  positions: Float32Array;
}

interface ConnectorInfo {
  neuronIdx: number;
  readout: number;
  outputIdx: number;
  weight: number; // absolute output weight, normalized [0,1]
  line: THREE.Line;
}

/** Anchor-to-chart "write line" per output channel */
interface WriteLineInfo {
  readout: number;
  outputIdx: number;
  line: THREE.Line;
}

export interface ReadoutCharts {
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  connectors: ConnectorInfo[];
  writeLines: WriteLineInfo[];
  traces: TraceData[][];
  yBases: number[];
  group: THREE.Group;
  nPerReadout: number;
  resize(w: number, h: number): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Find all per-output neuron connections
// ---------------------------------------------------------------------------

function findOutputConnections(net: NeuralNetwork): {
  neuronIdx: number;
  readout: number;
  outputIdx: number;
  weight: number;
}[] {
  const N = net.numNeurons;
  const numReadouts = net.numReadouts;
  const nPer = net.nOutputsPerReadout;
  const numOut = net.numOutputs;
  const ow = net.outputWeights;

  const raw: { neuronIdx: number; readout: number; outputIdx: number; weight: number }[] = [];
  let maxWeight = 0;
  for (let r = 0; r < numReadouts; r++) {
    for (let o = 0; o < nPer; o++) {
      const col = r * nPer + o;
      for (let i = 0; i < N; i++) {
        const w = Math.abs(ow[i * numOut + col]);
        if (w > 0) {
          raw.push({ neuronIdx: i, readout: r, outputIdx: o, weight: w });
          if (w > maxWeight) maxWeight = w;
        }
      }
    }
  }
  if (maxWeight > 0) {
    for (const entry of raw) entry.weight /= maxWeight;
  }
  return raw;
}

// ---------------------------------------------------------------------------
// Anchor Y position for a given output within a chart (group-local coords)
// ---------------------------------------------------------------------------

function anchorLocalY(yBase: number, outputIdx: number, nPer: number): number {
  return yBase - CHART_HEIGHT * ((outputIdx + 0.5) / nPer);
}

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

export function createReadoutCharts(net: NeuralNetwork): ReadoutCharts {
  const numReadouts = net.numReadouts;
  const nPer = net.nOutputsPerReadout;

  const hudScene = new THREE.Scene();
  const w = window.innerWidth;
  const h = window.innerHeight;
  const ortho = new THREE.OrthographicCamera(0, w, h, 0, -1, 1);

  const group = new THREE.Group();
  group.position.set(MARGIN, h - MARGIN, 0);
  hudScene.add(group);

  const traces: TraceData[][] = [];
  const yBases: number[] = [];
  const disposables: { geo: THREE.BufferGeometry; mat: THREE.Material }[] = [];

  for (let r = 0; r < numReadouts; r++) {
    const yBase = -r * (CHART_HEIGHT + CHART_GAP);
    yBases.push(yBase);

    // Border
    const borderVerts = new Float32Array([
      0, yBase, 0,
      CHART_WIDTH, yBase, 0,
      CHART_WIDTH, yBase - CHART_HEIGHT, 0,
      0, yBase - CHART_HEIGHT, 0,
      0, yBase, 0,
    ]);
    const borderGeo = new THREE.BufferGeometry();
    borderGeo.setAttribute("position", new THREE.BufferAttribute(borderVerts, 3));
    const borderMat = new THREE.LineBasicMaterial({
      color: 0x444444,
      transparent: true,
      opacity: 0.5,
    });
    group.add(new THREE.Line(borderGeo, borderMat));
    disposables.push({ geo: borderGeo, mat: borderMat });

    // Traces
    const readoutTraces: TraceData[] = [];
    for (let o = 0; o < nPer; o++) {
      const positions = new Float32Array(BUFFER_LEN * 3);
      for (let p = 0; p < BUFFER_LEN; p++) {
        positions[p * 3] = (p / (BUFFER_LEN - 1)) * CHART_WIDTH;
        positions[p * 3 + 1] = yBase - CHART_HEIGHT;
        positions[p * 3 + 2] = 0;
      }
      const geo = new THREE.BufferGeometry();
      const attr = new THREE.BufferAttribute(positions, 3);
      attr.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute("position", attr);
      const mat = new THREE.LineBasicMaterial({
        color: TRACE_COLORS[o % TRACE_COLORS.length],
        transparent: true,
        opacity: 0.85,
      });
      const line = new THREE.Line(geo, mat);
      group.add(line);
      disposables.push({ geo, mat });
      readoutTraces.push({ line, positions });
    }
    traces.push(readoutTraces);
  }

  // --- Neuron → anchor connectors (per individual output weight) ---
  const outputConns = findOutputConnections(net);

  const connectors: ConnectorInfo[] = [];
  for (const { neuronIdx, readout, outputIdx, weight } of outputConns) {
    const geo = new THREE.BufferGeometry();
    const posArr = new Float32Array(6);
    const posAttr = new THREE.BufferAttribute(posArr, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", posAttr);

    const colArr = new Float32Array(6);
    const colAttr = new THREE.BufferAttribute(colArr, 3);
    colAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("color", colAttr);

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.15,
    });
    const line = new THREE.Line(geo, mat);
    hudScene.add(line);
    connectors.push({ neuronIdx, readout, outputIdx, weight, line });
    disposables.push({ geo, mat });
  }

  // --- Anchor → chart-edge "write lines" (one per output channel, 15 total) ---
  const writeLines: WriteLineInfo[] = [];
  for (let r = 0; r < numReadouts; r++) {
    for (let o = 0; o < nPer; o++) {
      const geo = new THREE.BufferGeometry();
      const posArr = new Float32Array(6);
      const posAttr = new THREE.BufferAttribute(posArr, 3);
      posAttr.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute("position", posAttr);
      const mat = new THREE.LineBasicMaterial({
        color: TRACE_COLORS[o % TRACE_COLORS.length],
        transparent: true,
        opacity: 0.6,
      });
      const line = new THREE.Line(geo, mat);
      hudScene.add(line);
      writeLines.push({ readout: r, outputIdx: o, line });
      disposables.push({ geo, mat });
    }
  }

  function resize(newW: number, newH: number) {
    ortho.right = newW;
    ortho.top = newH;
    ortho.updateProjectionMatrix();
    group.position.set(MARGIN, newH - MARGIN, 0);
  }

  return {
    scene: hudScene,
    camera: ortho,
    connectors,
    writeLines,
    traces,
    yBases,
    group,
    nPerReadout: nPer,
    resize,
    dispose() {
      for (const { geo, mat } of disposables) {
        geo.dispose();
        mat.dispose();
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Update each frame
// ---------------------------------------------------------------------------

const _projected = new THREE.Vector3();

export function updateReadoutCharts(
  charts: ReadoutCharts,
  net: NeuralNetwork,
  worldCamera: THREE.PerspectiveCamera,
  nodePositions: Float32Array,
  pushData: boolean,
): void {
  const numReadouts = net.numReadouts;
  const nPer = charts.nPerReadout;
  const outputs = net.outputs;
  const refCounters = net.refractoryCounters;
  const refPeriods = net.refractionPeriods;
  const h = charts.camera.top;
  const w = charts.camera.right;
  const groupY = h - MARGIN; // group's screen-space Y origin

  // --- Push trace data ---
  if (pushData) {
    for (let r = 0; r < numReadouts; r++) {
      const yBase = charts.yBases[r];
      for (let o = 0; o < nPer; o++) {
        const { positions, line } = charts.traces[r][o];
        const val = outputs[r * nPer + o];

        for (let p = 0; p < BUFFER_LEN - 1; p++) {
          positions[p * 3 + 1] = positions[(p + 1) * 3 + 1];
        }
        positions[(BUFFER_LEN - 1) * 3 + 1] =
          yBase - CHART_HEIGHT + val * CHART_HEIGHT;

        (line.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      }
    }
  }

  // Screen-space x positions
  const chartRightX = MARGIN + CHART_WIDTH;
  const anchorX = chartRightX + ANCHOR_OFFSET;

  // --- Update neuron → anchor connectors ---
  for (const conn of charts.connectors) {
    const { neuronIdx, readout, outputIdx, weight, line } = conn;
    const yBase = charts.yBases[readout];

    const period = refPeriods[neuronIdx];
    const c = period > 0 ? refCounters[neuronIdx] / period : 0;
    const y = weight;

    // RGB = (1-C, 1-M, 1-Y)  C=refractory flash, M=source activation, Y=weight
    const cr = 1 - c;
    const cg = 1 - net.activations[neuronIdx];
    const cb = 1 - y;

    const colAttr = line.geometry.getAttribute("color") as THREE.BufferAttribute;
    const colors = colAttr.array as Float32Array;
    colors[0] = cr; colors[1] = cg; colors[2] = cb;
    colors[3] = cr; colors[4] = cg; colors[5] = cb;
    colAttr.needsUpdate = true;

    // Projected neuron position
    _projected.set(
      nodePositions[neuronIdx * 3],
      nodePositions[neuronIdx * 3 + 1],
      nodePositions[neuronIdx * 3 + 2],
    );
    _projected.project(worldCamera);
    const sx = (_projected.x + 1) / 2 * w;
    const sy = (_projected.y + 1) / 2 * h;

    // Anchor position (fixed per output)
    const ay = groupY + anchorLocalY(yBase, outputIdx, nPer);

    const posAttr = line.geometry.getAttribute("position") as THREE.BufferAttribute;
    const pos = posAttr.array as Float32Array;
    pos[0] = sx;  pos[1] = sy;  pos[2] = 0;
    pos[3] = anchorX; pos[4] = ay; pos[5] = 0;
    posAttr.needsUpdate = true;
  }

  // --- Update anchor → chart-edge write lines ---
  for (const { readout, outputIdx, line } of charts.writeLines) {
    const yBase = charts.yBases[readout];

    // Anchor (fixed Y)
    const ay = groupY + anchorLocalY(yBase, outputIdx, nPer);

    // Chart edge at current trace Y value
    const tracePositions = charts.traces[readout][outputIdx].positions;
    const traceY = tracePositions[(BUFFER_LEN - 1) * 3 + 1]; // group-local
    const edgeY = groupY + traceY;

    const posAttr = line.geometry.getAttribute("position") as THREE.BufferAttribute;
    const pos = posAttr.array as Float32Array;
    pos[0] = anchorX;     pos[1] = ay;    pos[2] = 0;
    pos[3] = chartRightX; pos[4] = edgeY; pos[5] = 0;
    posAttr.needsUpdate = true;
  }
}
