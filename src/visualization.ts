import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import createGraph from "ngraph.graph";
import createLayout from "ngraph.forcelayout";
import type { NeuralNetwork } from "./network.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

const CURVE_SEGMENTS = 8;
const CURVE_AMOUNT = 0.3; // control point offset as fraction of edge length

export interface NetworkVisualization {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  composer: EffectComposer;
  bloomPass: UnrealBloomPass;
  nodePositions: Float32Array;
  nodeMesh: THREE.InstancedMesh;
  flatNodes: boolean;
  edgeLineSegments: THREE.LineSegments;
  /** Number of line segment pairs per edge curve */
  edgeSegsPerEdge: number;
  fireFlash: Float64Array;
  edgeSources: Uint32Array;
  edgeWeights: Float32Array;
  edgeFlash: Float64Array;
  /** Call each frame to billboard circle nodes. No-op when using spheres. */
  updateNodeBillboard(): void;
  /** Toggle between flat circles and 3D spheres. */
  setFlatNodes(flat: boolean): void;
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
// Node mesh helpers
// ---------------------------------------------------------------------------

const _dummy = new THREE.Object3D();

function createNodeMesh(
  N: number,
  positions: Float32Array,
  flat: boolean,
): { mesh: THREE.InstancedMesh; geo: THREE.BufferGeometry; mat: THREE.Material } {
  const geo = flat
    ? new THREE.CircleGeometry(1.0, 24)
    : new THREE.SphereGeometry(1.0, 12, 8);
  const mat = flat
    ? new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide })
    : new THREE.MeshPhongMaterial({ color: 0xffffff, flatShading: false });
  const mesh = new THREE.InstancedMesh(geo, mat, N);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  for (let i = 0; i < N; i++) {
    _dummy.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    _dummy.quaternion.identity();
    _dummy.updateMatrix();
    mesh.setMatrixAt(i, _dummy.matrix);
    mesh.setColorAt(i, new THREE.Color(1, 1, 1));
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  return { mesh, geo, mat };
}

function billboardInstances(
  mesh: THREE.InstancedMesh,
  positions: Float32Array,
  camera: THREE.Camera,
): void {
  const N = mesh.count;
  _dummy.quaternion.copy(camera.quaternion);
  for (let i = 0; i < N; i++) {
    _dummy.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    _dummy.updateMatrix();
    mesh.setMatrixAt(i, _dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
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
  renderer.toneMapping = THREE.ReinhardToneMapping;
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
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.4;

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(100, 200, 150);
  scene.add(dirLight);

  // --- Postprocessing ---
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.2, // strength
    0.2, // radius
    0.2, // threshold
  );
  bloomPass.enabled = false;
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  // --- Layout ---
  const { positions, edges, edgeWeights: edgeWeightsArr } = computeLayout(net, edgeWeightThreshold);

  // --- Nodes ---
  let flat = true;
  let nodeState = createNodeMesh(N, positions, flat);
  let nodeMesh = nodeState.mesh;
  scene.add(nodeMesh);

  // --- Edges (curved) ---
  const S = CURVE_SEGMENTS;
  const numEdges = edges.length;
  const vertsPerEdge = S * 2; // LineSegments: each segment = 2 verts
  const edgePositionArr = new Float32Array(numEdges * vertsPerEdge * 3);
  const edgeColorArr = new Float32Array(numEdges * vertsPerEdge * 3);
  const edgeSources = new Uint32Array(numEdges);

  const _a = new THREE.Vector3();
  const _b = new THREE.Vector3();
  const _mid = new THREE.Vector3();
  const _dir = new THREE.Vector3();
  const _perp = new THREE.Vector3();
  const _rand = new THREE.Vector3();
  const _ctrl = new THREE.Vector3();
  const _curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3());

  for (let e = 0; e < numEdges; e++) {
    const [ai, bi] = edges[e];
    edgeSources[e] = ai;

    _a.set(positions[ai * 3], positions[ai * 3 + 1], positions[ai * 3 + 2]);
    _b.set(positions[bi * 3], positions[bi * 3 + 1], positions[bi * 3 + 2]);

    _mid.addVectors(_a, _b).multiplyScalar(0.5);
    _dir.subVectors(_b, _a);
    const len = _dir.length();
    _dir.normalize();

    _rand.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize();
    _perp.crossVectors(_dir, _rand).normalize();
    if (_perp.lengthSq() < 0.001) {
      _rand.set(0, 1, 0);
      _perp.crossVectors(_dir, _rand).normalize();
    }
    _ctrl.copy(_mid).addScaledVector(_perp, len * CURVE_AMOUNT * (Math.random() * 0.5 + 0.75));

    _curve.v0.copy(_a);
    _curve.v1.copy(_ctrl);
    _curve.v2.copy(_b);
    const pts = _curve.getPoints(S);

    const baseVert = e * vertsPerEdge * 3;
    for (let s = 0; s < S; s++) {
      const p0 = pts[s];
      const p1 = pts[s + 1];
      const vi = baseVert + s * 6;
      edgePositionArr[vi + 0] = p0.x; edgePositionArr[vi + 1] = p0.y; edgePositionArr[vi + 2] = p0.z;
      edgePositionArr[vi + 3] = p1.x; edgePositionArr[vi + 4] = p1.y; edgePositionArr[vi + 5] = p1.z;
    }

    const y = edgeWeightsArr[e];
    const r = 1, g = 1, bl = 1 - y;
    for (let s = 0; s < S; s++) {
      const ci = baseVert + s * 6;
      edgeColorArr[ci + 0] = r; edgeColorArr[ci + 1] = g; edgeColorArr[ci + 2] = bl;
      edgeColorArr[ci + 3] = r; edgeColorArr[ci + 4] = g; edgeColorArr[ci + 5] = bl;
    }
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
  const fireFlash = new Float64Array(N);

  // --- Resize handler ---
  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    composer.setSize(w, h);
    bloomPass.resolution.set(w, h);
  }
  window.addEventListener("resize", resize);

  const viz: NetworkVisualization = {
    renderer,
    scene,
    camera,
    controls,
    composer,
    bloomPass,
    nodePositions: positions,
    nodeMesh,
    flatNodes: flat,
    edgeLineSegments,
    edgeSegsPerEdge: S,
    fireFlash,
    edgeSources,
    edgeWeights: edgeWeightsArr,
    edgeFlash,

    updateNodeBillboard() {
      if (flat) billboardInstances(nodeMesh, positions, camera);
    },

    setFlatNodes(newFlat: boolean) {
      if (newFlat === flat) return;
      flat = newFlat;
      viz.flatNodes = flat;

      // Copy instance colors from old mesh before disposing
      const oldColors = nodeMesh.instanceColor
        ? (nodeMesh.instanceColor.array as Float32Array).slice()
        : null;

      scene.remove(nodeMesh);
      nodeState.geo.dispose();
      nodeState.mat.dispose();

      nodeState = createNodeMesh(N, positions, flat);
      nodeMesh = nodeState.mesh;
      viz.nodeMesh = nodeMesh;

      if (oldColors && nodeMesh.instanceColor) {
        (nodeMesh.instanceColor.array as Float32Array).set(oldColors);
        nodeMesh.instanceColor.needsUpdate = true;
      }

      scene.add(nodeMesh);
    },

    resize,
    dispose() {
      window.removeEventListener("resize", resize);
      renderer.domElement.remove();
      renderer.dispose();
      composer.dispose();
      nodeState.geo.dispose();
      nodeState.mat.dispose();
      edgeGeo.dispose();
      edgeMat.dispose();
    },
  };

  return viz;
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

  for (let i = 0; i < N; i++) {
    if (firing[i]) flash[i] = 1.0;

    const c = activations[i];
    const m = flash[i];

    _tmpColor.setRGB(1 - c, 1 - m, 1);
    viz.nodeMesh.setColorAt(i, _tmpColor);

    flash[i] *= flashDecay;
    if (flash[i] < 0.01) flash[i] = 0;
  }

  if (viz.nodeMesh.instanceColor) {
    viz.nodeMesh.instanceColor.needsUpdate = true;
  }

  const colorAttr = viz.edgeLineSegments.geometry.getAttribute("color") as THREE.BufferAttribute;
  const colors = colorAttr.array as Float32Array;
  const eSources = viz.edgeSources;
  const eWeights = viz.edgeWeights;
  const eFlash = viz.edgeFlash;
  const numEdges = eSources.length;
  const S = viz.edgeSegsPerEdge;
  const stride = S * 2 * 3; // floats per edge in the color buffer

  for (let e = 0; e < numEdges; e++) {
    const src = eSources[e];
    if (firing[src]) eFlash[e] = 1.0;

    const c = eFlash[e];
    const y = eWeights[e];

    const r = 1 - c;
    const g = 1;
    const b = 1 - y;

    const base = e * stride;
    for (let s = 0; s < S; s++) {
      const ci = base + s * 6;
      colors[ci + 0] = r; colors[ci + 1] = g; colors[ci + 2] = b;
      colors[ci + 3] = r; colors[ci + 4] = g; colors[ci + 5] = b;
    }

    eFlash[e] *= flashDecay;
    if (eFlash[e] < 0.01) eFlash[e] = 0;
  }

  colorAttr.needsUpdate = true;
}
