import * as THREE from "three";
import type { NeuralNetwork } from "./network.ts";

// ---------------------------------------------------------------------------
// Constants (pixel units)
// ---------------------------------------------------------------------------

const CHART_WIDTH = 250;
const CHART_HEIGHT = 50;
const CHART_GAP = 8;
const BUFFER_LEN = 300;
const LEFT_MARGIN = 15;
const TOP_MARGIN = 15;

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
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  connectorLines: THREE.Line[];
  readoutNeurons: number[];
  traces: TraceData[][];
  /** Y-base (top edge) per readout in local group coords */
  yBases: number[];
  group: THREE.Group;
  resize(w: number, h: number): void;
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
  net: NeuralNetwork,
): ReadoutCharts {
  const numReadouts = net.state.numReadouts;
  const nPer = net.state.nOutputsPerReadout;

  // Separate scene + orthographic camera for screen-space HUD
  const hudScene = new THREE.Scene();
  const w = window.innerWidth;
  const h = window.innerHeight;
  const ortho = new THREE.OrthographicCamera(0, w, h, 0, -1, 1);

  // Group anchored at top-left; charts grow downward (negative local Y)
  const group = new THREE.Group();
  group.position.set(LEFT_MARGIN, h - TOP_MARGIN, 0);
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
        positions[p * 3 + 1] = yBase - CHART_HEIGHT; // bottom = value 0
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

  // Representative neurons
  const readoutNeurons = findReadoutNeurons(net);

  // Connector lines live in the HUD scene (screen-space endpoints)
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
    hudScene.add(line); // in the HUD scene, not in the group
    connectorLines.push(line);
    disposables.push({ geo, mat });
  }

  function resize(newW: number, newH: number) {
    ortho.right = newW;
    ortho.top = newH;
    ortho.updateProjectionMatrix();
    group.position.set(LEFT_MARGIN, newH - TOP_MARGIN, 0);
  }

  return {
    scene: hudScene,
    camera: ortho,
    connectorLines,
    readoutNeurons,
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
): void {
  const numReadouts = net.state.numReadouts;
  const nPer = net.state.nOutputsPerReadout;
  const outputs = net.state.outputs;
  const w = charts.camera.right;
  const h = charts.camera.top;

  // Push new output values only when simulation is running
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

  // Always update connector lines (they follow camera movement)
  for (let r = 0; r < numReadouts; r++) {
    const neuronIdx = charts.readoutNeurons[r];
    const yBase = charts.yBases[r];

    _projected.set(
      nodePositions[neuronIdx * 3],
      nodePositions[neuronIdx * 3 + 1],
      nodePositions[neuronIdx * 3 + 2],
    );
    _projected.project(worldCamera);

    const sx = (_projected.x + 1) / 2 * w;
    const sy = (_projected.y + 1) / 2 * h;

    const anchorX = LEFT_MARGIN;
    const anchorY = (h - TOP_MARGIN) + yBase - CHART_HEIGHT / 2;

    const attr = charts.connectorLines[r].geometry.getAttribute("position") as THREE.BufferAttribute;
    const pos = attr.array as Float32Array;

    pos[0] = sx;
    pos[1] = sy;
    pos[2] = 0;
    pos[3] = anchorX;
    pos[4] = anchorY;
    pos[5] = 0;

    attr.needsUpdate = true;
  }
}
