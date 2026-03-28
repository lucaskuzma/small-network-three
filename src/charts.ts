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
  weight: number; // absolute output weight sum (Y channel)
  line: THREE.Line;
  flash: number; // cyan flash, decays per frame
}

export interface ReadoutCharts {
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  connectors: ConnectorInfo[];
  traces: TraceData[][];
  yBases: number[];
  group: THREE.Group;
  resize(w: number, h: number): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Find ALL neurons with non-zero output weights per readout
// ---------------------------------------------------------------------------

function findAllReadoutNeurons(net: NeuralNetwork): { neuronIdx: number; readout: number; weight: number }[] {
  const N = net.state.numNeurons;
  const numReadouts = net.state.numReadouts;
  const nPer = net.state.nOutputsPerReadout;
  const numOut = net.state.numOutputs;
  const ow = net.state.outputWeights;

  const raw: { neuronIdx: number; readout: number; weight: number }[] = [];
  let maxWeight = 0;
  for (let r = 0; r < numReadouts; r++) {
    const colStart = r * nPer;
    const colEnd = colStart + nPer;
    for (let i = 0; i < N; i++) {
      let sum = 0;
      for (let c = colStart; c < colEnd; c++) {
        sum += Math.abs(ow[i * numOut + c]);
      }
      if (sum > 0) {
        raw.push({ neuronIdx: i, readout: r, weight: sum });
        if (sum > maxWeight) maxWeight = sum;
      }
    }
  }
  // Normalize weights to [0, 1]
  if (maxWeight > 0) {
    for (const entry of raw) entry.weight /= maxWeight;
  }
  return raw;
}

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

export function createReadoutCharts(net: NeuralNetwork): ReadoutCharts {
  const numReadouts = net.state.numReadouts;
  const nPer = net.state.nOutputsPerReadout;

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

  // All readout neurons with non-zero output weights
  const neuronReadoutPairs = findAllReadoutNeurons(net);

  const connectors: ConnectorInfo[] = [];
  for (const { neuronIdx, readout, weight } of neuronReadoutPairs) {
    const geo = new THREE.BufferGeometry();
    const posArr = new Float32Array(6);
    const posAttr = new THREE.BufferAttribute(posArr, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", posAttr);

    // Per-vertex color (same CMYK scheme as network edges)
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
    connectors.push({ neuronIdx, readout, weight, line, flash: 0 });
    disposables.push({ geo, mat });
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
    traces,
    yBases,
    group,
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
  flashDecay: number = 0.85,
): void {
  const numReadouts = net.state.numReadouts;
  const nPer = net.state.nOutputsPerReadout;
  const outputs = net.state.outputs;
  const firing = net.state.firing;
  const h = charts.camera.top;

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

  // Update all connector lines (position + CMYK color)
  const w = charts.camera.right;
  for (const conn of charts.connectors) {
    const { neuronIdx, readout, weight, line } = conn;
    const yBase = charts.yBases[readout];

    // Fire flash: set to 1 when source neuron fires, decay each frame
    if (firing[neuronIdx]) conn.flash = 1.0;
    const c = conn.flash;  // cyan = activation flowing
    const y = weight;       // yellow = output weight magnitude

    // CMYK→RGB with M=0, K=0
    const cr = 1 - c;
    const cg = 1;
    const cb = 1 - y;

    const colAttr = line.geometry.getAttribute("color") as THREE.BufferAttribute;
    const colors = colAttr.array as Float32Array;
    colors[0] = cr; colors[1] = cg; colors[2] = cb;
    colors[3] = cr; colors[4] = cg; colors[5] = cb;
    colAttr.needsUpdate = true;

    conn.flash *= flashDecay;
    if (conn.flash < 0.01) conn.flash = 0;

    // Position: projected neuron -> chart right edge
    _projected.set(
      nodePositions[neuronIdx * 3],
      nodePositions[neuronIdx * 3 + 1],
      nodePositions[neuronIdx * 3 + 2],
    );
    _projected.project(worldCamera);

    const sx = (_projected.x + 1) / 2 * w;
    const sy = (_projected.y + 1) / 2 * h;

    const anchorX = MARGIN + CHART_WIDTH;
    const anchorY = (h - MARGIN) + yBase - CHART_HEIGHT / 2;

    const posAttr = line.geometry.getAttribute("position") as THREE.BufferAttribute;
    const pos = posAttr.array as Float32Array;
    pos[0] = sx;
    pos[1] = sy;
    pos[2] = 0;
    pos[3] = anchorX;
    pos[4] = anchorY;
    pos[5] = 0;
    posAttr.needsUpdate = true;
  }
}
