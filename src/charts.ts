import * as THREE from "three";
import type { NeuralNetwork } from "./network.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CHART_WIDTH = 80;
const CHART_HEIGHT = 15;
const CHART_GAP = 6;
const BUFFER_LEN = 300;

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

export interface ReadoutCharts {
  group: THREE.Group;
  connectorLines: THREE.Line[];
  readoutNeurons: number[];
  traces: TraceData[][];
  /** Y-base (top edge) per readout in local coords */
  yBases: number[];
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Find representative neuron per readout
// ---------------------------------------------------------------------------

function findReadoutNeurons(net: NeuralNetwork): number[] {
  const N = net.state.numNeurons;
  const numReadouts = net.state.numReadouts;
  const nPer = net.state.nOutputsPerReadout;
  const numOut = net.state.numOutputs;
  const ow = net.state.outputWeights;

  const result: number[] = [];
  for (let r = 0; r < numReadouts; r++) {
    const colStart = r * nPer;
    const colEnd = colStart + nPer;
    let bestIdx = 0;
    let bestSum = -Infinity;
    for (let i = 0; i < N; i++) {
      let sum = 0;
      for (let c = colStart; c < colEnd; c++) {
        sum += Math.abs(ow[i * numOut + c]);
      }
      if (sum > bestSum) {
        bestSum = sum;
        bestIdx = i;
      }
    }
    result.push(bestIdx);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

export function createReadoutCharts(
  scene: THREE.Scene,
  net: NeuralNetwork,
  nodePositions: Float32Array,
): ReadoutCharts {
  const N = net.state.numNeurons;
  const numReadouts = net.state.numReadouts;
  const nPer = net.state.nOutputsPerReadout;

  // Compute network bounding box to place charts to the left
  let minX = Infinity;
  for (let i = 0; i < N; i++) {
    const x = nodePositions[i * 3];
    if (x < minX) minX = x;
  }

  const group = new THREE.Group();
  group.position.set(minX - 40, 0, 0);
  scene.add(group);

  const totalHeight = numReadouts * CHART_HEIGHT + (numReadouts - 1) * CHART_GAP;
  const traces: TraceData[][] = [];
  const yBases: number[] = [];
  const disposables: { geo: THREE.BufferGeometry; mat: THREE.Material }[] = [];

  for (let r = 0; r < numReadouts; r++) {
    const yBase = totalHeight / 2 - r * (CHART_HEIGHT + CHART_GAP);
    yBases.push(yBase);

    // Border
    const borderPositions = new Float32Array([
      0, yBase, 0,
      CHART_WIDTH, yBase, 0,
      CHART_WIDTH, yBase - CHART_HEIGHT, 0,
      0, yBase - CHART_HEIGHT, 0,
      0, yBase, 0,
    ]);
    const borderGeo = new THREE.BufferGeometry();
    borderGeo.setAttribute("position", new THREE.BufferAttribute(borderPositions, 3));
    const borderMat = new THREE.LineBasicMaterial({
      color: 0x444444,
      transparent: true,
      opacity: 0.5,
    });
    group.add(new THREE.Line(borderGeo, borderMat));
    disposables.push({ geo: borderGeo, mat: borderMat });

    // Readout label line at left edge (small tick marks per output)
    const readoutTraces: TraceData[] = [];

    for (let o = 0; o < nPer; o++) {
      const positions = new Float32Array(BUFFER_LEN * 3);
      // Initialize: x spread across chart width, y at bottom (value 0)
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

  // Representative neurons per readout
  const readoutNeurons = findReadoutNeurons(net);

  // Connector lines (world space, outside the billboard group)
  const connectorLines: THREE.Line[] = [];
  for (let r = 0; r < numReadouts; r++) {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(6);
    const attr = new THREE.BufferAttribute(pos, 3);
    attr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", attr);
    const mat = new THREE.LineBasicMaterial({
      color: 0x666666,
      transparent: true,
      opacity: 0.3,
    });
    const line = new THREE.Line(geo, mat);
    scene.add(line);
    connectorLines.push(line);
    disposables.push({ geo, mat });
  }

  return {
    group,
    connectorLines,
    readoutNeurons,
    traces,
    yBases,
    dispose() {
      scene.remove(group);
      for (const { geo, mat } of disposables) {
        geo.dispose();
        mat.dispose();
      }
      for (const line of connectorLines) {
        scene.remove(line);
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Update each frame
// ---------------------------------------------------------------------------

const _anchor = new THREE.Vector3();

export function updateReadoutCharts(
  charts: ReadoutCharts,
  net: NeuralNetwork,
  camera: THREE.Camera,
  nodePositions: Float32Array,
): void {
  // Billboard: face the camera
  charts.group.quaternion.copy(camera.quaternion);

  const numReadouts = net.state.numReadouts;
  const nPer = net.state.nOutputsPerReadout;
  const outputs = net.state.outputs;

  // Push new output values into each trace
  for (let r = 0; r < numReadouts; r++) {
    const yBase = charts.yBases[r];

    for (let o = 0; o < nPer; o++) {
      const { positions, line } = charts.traces[r][o];
      const val = outputs[r * nPer + o];

      // Shift y values left by one sample (x and z stay fixed)
      for (let p = 0; p < BUFFER_LEN - 1; p++) {
        positions[p * 3 + 1] = positions[(p + 1) * 3 + 1];
      }

      // Write new y value at the right edge
      positions[(BUFFER_LEN - 1) * 3 + 1] =
        yBase - CHART_HEIGHT + val * CHART_HEIGHT;

      (line.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    }
  }

  // Update connector lines (neuron world pos -> chart anchor world pos)
  for (let r = 0; r < numReadouts; r++) {
    const neuronIdx = charts.readoutNeurons[r];
    const yBase = charts.yBases[r];

    // Chart midpoint in local coords -> world
    _anchor.set(0, yBase - CHART_HEIGHT / 2, 0);
    charts.group.localToWorld(_anchor);

    const attr = charts.connectorLines[r].geometry.getAttribute("position") as THREE.BufferAttribute;
    const pos = attr.array as Float32Array;

    pos[0] = nodePositions[neuronIdx * 3];
    pos[1] = nodePositions[neuronIdx * 3 + 1];
    pos[2] = nodePositions[neuronIdx * 3 + 2];
    pos[3] = _anchor.x;
    pos[4] = _anchor.y;
    pos[5] = _anchor.z;

    attr.needsUpdate = true;
  }
}
