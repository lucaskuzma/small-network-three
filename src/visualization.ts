import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import createGraph from "ngraph.graph";
import createLayout from "ngraph.forcelayout";
import type { NeuralNetwork } from "./network.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface NetworkVisualization {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  nodePositions: Float32Array; // 3 floats per neuron (x, y, z)
  nodeMesh: THREE.InstancedMesh;
  edgeLineSegments: THREE.LineSegments;
  /** Per-neuron magenta flash channel, decays each frame. */
  fireFlash: Float64Array;
  /** Per-edge: source neuron index */
  edgeSources: Uint32Array;
  /** Per-edge: absolute weight (used as Y channel) */
  edgeWeights: Float32Array;
  /** Per-edge activation flash, decays each frame */
  edgeFlash: Float64Array;
  resize(): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Build the ngraph and run force-directed layout
// ---------------------------------------------------------------------------

interface LayoutResult {
  positions: Float32Array;
  edges: [number, number][];
  edgeWeights: Float32Array;
}

function computeLayout(
  net: NeuralNetwork,
  weightThreshold: number,
): LayoutResult {
  const N = net.state.numNeurons;
  const w = net.state.networkWeights;

  const graph = createGraph();
  for (let i = 0; i < N; i++) graph.addNode(i);

  const edges: [number, number][] = [];
  const weightList: number[] = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const val = w[i * N + j];
      if (Math.abs(val) > weightThreshold) {
        graph.addLink(i, j);
        edges.push([i, j]);
        weightList.push(Math.abs(val));
      }
    }
  }

  const layout = createLayout(graph, {
    dimensions: 3,
    springLength: 15,
    springCoefficient: 0.0002,
    gravity: -1.5,
    theta: 0.8,
    dragCoefficient: 0.02,
    timeStep: 5,
  });

  for (let i = 0; i < 200; i++) layout.step();

  const positions = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const p = layout.getNodePosition(i);
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z ?? 0;
  }

  layout.dispose();
  return { positions, edges, edgeWeights: Float32Array.from(weightList) };
}

// ---------------------------------------------------------------------------
// Create visualization
// ---------------------------------------------------------------------------

export function createVisualization(
  net: NeuralNetwork,
  container: HTMLElement = document.body,
  edgeWeightThreshold = 0.05,
): NetworkVisualization {
  const N = net.state.numNeurons;

  // --- Renderer, scene, camera ---
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  const camera = new THREE.PerspectiveCamera(
    50,
    window.innerWidth / window.innerHeight,
    0.1,
    5000,
  );
  camera.position.set(0, 0, 200);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;

  // Subtle ambient + directional light so spheres have some shading
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(100, 200, 150);
  scene.add(dirLight);

  // --- Layout ---
  const { positions, edges, edgeWeights: edgeWeightsArr } = computeLayout(net, edgeWeightThreshold);

  // --- Nodes (InstancedMesh) ---
  const sphereGeo = new THREE.SphereGeometry(1.0, 12, 8);
  const sphereMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
  const nodeMesh = new THREE.InstancedMesh(sphereGeo, sphereMat, N);
  nodeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  const dummy = new THREE.Object3D();
  for (let i = 0; i < N; i++) {
    dummy.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    dummy.updateMatrix();
    nodeMesh.setMatrixAt(i, dummy.matrix);
    nodeMesh.setColorAt(i, new THREE.Color(1, 1, 1));
  }
  nodeMesh.instanceMatrix.needsUpdate = true;
  if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true;
  scene.add(nodeMesh);

  // --- Edges (LineSegments with per-vertex color) ---
  const numEdges = edges.length;
  const edgePositionArr = new Float32Array(numEdges * 2 * 3);
  const edgeColorArr = new Float32Array(numEdges * 2 * 3);
  const edgeSources = new Uint32Array(numEdges);

  for (let e = 0; e < numEdges; e++) {
    const [a, b] = edges[e];
    edgeSources[e] = a;
    edgePositionArr[e * 6 + 0] = positions[a * 3];
    edgePositionArr[e * 6 + 1] = positions[a * 3 + 1];
    edgePositionArr[e * 6 + 2] = positions[a * 3 + 2];
    edgePositionArr[e * 6 + 3] = positions[b * 3];
    edgePositionArr[e * 6 + 4] = positions[b * 3 + 1];
    edgePositionArr[e * 6 + 5] = positions[b * 3 + 2];
    // Initial color: based on weight only (Y channel)
    const y = edgeWeightsArr[e];
    const r = 1, g = 1, bl = 1 - y;
    edgeColorArr[e * 6 + 0] = r; edgeColorArr[e * 6 + 1] = g; edgeColorArr[e * 6 + 2] = bl;
    edgeColorArr[e * 6 + 3] = r; edgeColorArr[e * 6 + 4] = g; edgeColorArr[e * 6 + 5] = bl;
  }
  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute("position", new THREE.BufferAttribute(edgePositionArr, 3));
  const edgeColorAttr = new THREE.BufferAttribute(edgeColorArr, 3);
  edgeColorAttr.setUsage(THREE.DynamicDrawUsage);
  edgeGeo.setAttribute("color", edgeColorAttr);
  const edgeMat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.15,
  });
  const edgeLineSegments = new THREE.LineSegments(edgeGeo, edgeMat);
  scene.add(edgeLineSegments);

  const edgeFlash = new Float64Array(numEdges);

  // --- Fire flash state ---
  const fireFlash = new Float64Array(N);

  // --- Resize handler ---
  function resize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }
  window.addEventListener("resize", resize);

  return {
    renderer,
    scene,
    camera,
    controls,
    nodePositions: positions,
    nodeMesh,
    edgeLineSegments,
    fireFlash,
    edgeSources,
    edgeWeights: edgeWeightsArr,
    edgeFlash,
    resize,
    dispose() {
      window.removeEventListener("resize", resize);
      renderer.domElement.remove();
      renderer.dispose();
      sphereGeo.dispose();
      sphereMat.dispose();
      edgeGeo.dispose();
      edgeMat.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// Update colors each frame (nodes + edges)
// ---------------------------------------------------------------------------

const _tmpColor = new THREE.Color();

export function updateColors(
  viz: NetworkVisualization,
  net: NeuralNetwork,
  flashDecay: number = 0.85,
): void {
  const N = net.state.numNeurons;
  const activations = net.state.activations;
  const firing = net.state.firing;
  const flash = viz.fireFlash;

  // --- Nodes ---
  for (let i = 0; i < N; i++) {
    if (firing[i]) flash[i] = 1.0;

    const c = activations[i]; // cyan channel
    const m = flash[i]; // magenta channel

    // CMYK→RGB with Y=0, K=0:  R = 1-C,  G = 1-M,  B = 1
    _tmpColor.setRGB(1 - c, 1 - m, 1);
    viz.nodeMesh.setColorAt(i, _tmpColor);

    flash[i] *= flashDecay;
    if (flash[i] < 0.01) flash[i] = 0;
  }

  if (viz.nodeMesh.instanceColor) {
    viz.nodeMesh.instanceColor.needsUpdate = true;
  }

  // --- Edges ---
  const colorAttr = viz.edgeLineSegments.geometry.getAttribute("color") as THREE.BufferAttribute;
  const colors = colorAttr.array as Float32Array;
  const eSources = viz.edgeSources;
  const eWeights = viz.edgeWeights;
  const eFlash = viz.edgeFlash;
  const numEdges = eSources.length;

  for (let e = 0; e < numEdges; e++) {
    const src = eSources[e];
    if (firing[src]) eFlash[e] = 1.0;

    const c = eFlash[e];  // cyan = activation flowing through edge
    const y = eWeights[e]; // yellow = weight magnitude

    // CMYK→RGB with M=0, K=0:  R = 1-C,  G = 1,  B = 1-Y
    const r = 1 - c;
    const g = 1;
    const b = 1 - y;

    // Both vertices of the line segment get the same color
    colors[e * 6 + 0] = r; colors[e * 6 + 1] = g; colors[e * 6 + 2] = b;
    colors[e * 6 + 3] = r; colors[e * 6 + 4] = g; colors[e * 6 + 5] = b;

    eFlash[e] *= flashDecay;
    if (eFlash[e] < 0.01) eFlash[e] = 0;
  }

  colorAttr.needsUpdate = true;
}
