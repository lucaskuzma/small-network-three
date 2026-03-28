import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import createGraph from "ngraph.graph";
import createLayout from "ngraph.forcelayout";
import { type NeuralNetwork, MAX_NEURONS, mulberry32 } from "./network.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CURVE_SEGMENTS = 8;
const CURVE_AMOUNT = 0.3;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
  edgeSegsPerEdge: number;
  edgeSources: Uint32Array;
  edgeWeights: Float32Array;
  updateNodeBillboard(): void;
  setFlatNodes(flat: boolean): void;
  setNodeCount(n: number): void;
  rebuildEdges(net: NeuralNetwork, edgeWeightThreshold: number): void;
  /** Re-run force layout, update node positions + edges. */
  reLayout(net: NeuralNetwork, edgeWeightThreshold: number): void;
  resize(): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Force-directed layout for all MAX_NEURONS nodes
// ---------------------------------------------------------------------------

function computeLayout(net: NeuralNetwork): Float32Array {
  const M = MAX_NEURONS;
  const { sparsity, numModules, interModuleFactor } = net.params;
  const interSparsity = sparsity * interModuleFactor;
  const modAssign = net.moduleAssignments;

  const graph = createGraph();
  for (let i = 0; i < M; i++) graph.addNode(i);

  for (let i = 0; i < M; i++) {
    for (let j = 0; j < M; j++) {
      if (i === j) continue;
      const same = numModules <= 1 || modAssign[i] === modAssign[j];
      const thr = same ? sparsity : interSparsity;
      if (net.sparsityRank[i * M + j] < thr) {
        graph.addLink(i, j);
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

  const positions = new Float32Array(M * 3);
  for (let i = 0; i < M; i++) {
    const p = layout.getNodePosition(i);
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z ?? 0;
  }

  layout.dispose();
  return positions;
}

// ---------------------------------------------------------------------------
// Node mesh helpers
// ---------------------------------------------------------------------------

const _dummy = new THREE.Object3D();

function createNodeMesh(
  positions: Float32Array,
  numVisible: number,
  flat: boolean,
): { mesh: THREE.InstancedMesh; geo: THREE.BufferGeometry; mat: THREE.Material } {
  const geo = flat
    ? new THREE.CircleGeometry(1.0, 24)
    : new THREE.SphereGeometry(1.0, 12, 8);
  const mat = flat
    ? new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide })
    : new THREE.MeshPhongMaterial({ color: 0xffffff, flatShading: false });
  const mesh = new THREE.InstancedMesh(geo, mat, MAX_NEURONS);
  mesh.count = numVisible;
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  for (let i = 0; i < MAX_NEURONS; i++) {
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
// Edge geometry builder (curved, seeded)
// ---------------------------------------------------------------------------

interface EdgeArrays {
  positionArr: Float32Array;
  colorArr: Float32Array;
  sources: Uint32Array;
  weights: Float32Array;
  numEdges: number;
}

const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _perp = new THREE.Vector3();
const _rv = new THREE.Vector3();
const _ctrl = new THREE.Vector3();
const _curve = new THREE.QuadraticBezierCurve3(
  new THREE.Vector3(),
  new THREE.Vector3(),
  new THREE.Vector3(),
);

function buildEdgeArrays(
  net: NeuralNetwork,
  positions: Float32Array,
  edgeWeightThreshold: number,
): EdgeArrays {
  const N = net.numNeurons;
  const M = MAX_NEURONS;
  const S = CURVE_SEGMENTS;
  const ew = net.effectiveWeights;

  // First pass: count edges
  let numEdges = 0;
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++)
      if (i !== j && Math.abs(ew[i * M + j]) > edgeWeightThreshold) numEdges++;

  const vertsPerEdge = S * 2;
  const positionArr = new Float32Array(numEdges * vertsPerEdge * 3);
  const colorArr = new Float32Array(numEdges * vertsPerEdge * 3);
  const sources = new Uint32Array(numEdges);
  const weights = new Float32Array(numEdges);

  // Second pass: fill geometry
  let e = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const w = ew[i * M + j];
      if (Math.abs(w) <= edgeWeightThreshold) continue;

      sources[e] = i;
      weights[e] = Math.abs(w);

      _a.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      _b.set(positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]);

      // Per-edge seeded PRNG for deterministic curve shape
      const erng = mulberry32(net.seed * 65537 + i * 997 + j);

      _mid.addVectors(_a, _b).multiplyScalar(0.5);
      _dir.subVectors(_b, _a);
      const len = _dir.length();
      _dir.normalize();

      _rv.set(erng() - 0.5, erng() - 0.5, erng() - 0.5).normalize();
      _perp.crossVectors(_dir, _rv).normalize();
      if (_perp.lengthSq() < 0.001) {
        _rv.set(0, 1, 0);
        _perp.crossVectors(_dir, _rv).normalize();
      }
      _ctrl
        .copy(_mid)
        .addScaledVector(_perp, len * CURVE_AMOUNT * (erng() * 0.5 + 0.75));

      _curve.v0.copy(_a);
      _curve.v1.copy(_ctrl);
      _curve.v2.copy(_b);
      const pts = _curve.getPoints(S);

      const baseVert = e * vertsPerEdge * 3;
      for (let s = 0; s < S; s++) {
        const p0 = pts[s];
        const p1 = pts[s + 1];
        const vi = baseVert + s * 6;
        positionArr[vi] = p0.x;
        positionArr[vi + 1] = p0.y;
        positionArr[vi + 2] = p0.z;
        positionArr[vi + 3] = p1.x;
        positionArr[vi + 4] = p1.y;
        positionArr[vi + 5] = p1.z;
      }

      const y = weights[e];
      const r = 1,
        g = 1,
        bl = 1 - y;
      for (let s = 0; s < S; s++) {
        const ci = baseVert + s * 6;
        colorArr[ci] = r;
        colorArr[ci + 1] = g;
        colorArr[ci + 2] = bl;
        colorArr[ci + 3] = r;
        colorArr[ci + 4] = g;
        colorArr[ci + 5] = bl;
      }

      e++;
    }
  }

  return { positionArr, colorArr, sources, weights, numEdges };
}

// ---------------------------------------------------------------------------
// Create visualization
// ---------------------------------------------------------------------------

export function createVisualization(
  net: NeuralNetwork,
  container: HTMLElement = document.body,
  edgeWeightThreshold = 0.05,
): NetworkVisualization {
  const N = net.numNeurons;

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
  camera.position.set(0, 0, 2048);

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
    0.2,
    0.2,
    0.2,
  );
  bloomPass.enabled = false;
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  // --- Layout (computed once for all MAX_NEURONS) ---
  const positions = computeLayout(net);

  // --- Nodes ---
  let flat = true;
  let nodeState = createNodeMesh(positions, N, flat);
  let nodeMesh = nodeState.mesh;
  scene.add(nodeMesh);

  // --- Edges ---
  const S = CURVE_SEGMENTS;
  let edgeData = buildEdgeArrays(net, positions, edgeWeightThreshold);

  let edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute(
    "position",
    new THREE.BufferAttribute(edgeData.positionArr, 3),
  );
  const edgeColorAttr = new THREE.BufferAttribute(edgeData.colorArr, 3);
  edgeColorAttr.setUsage(THREE.DynamicDrawUsage);
  edgeGeo.setAttribute("color", edgeColorAttr);

  const edgeMat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.15,
  });
  let edgeLineSegments = new THREE.LineSegments(edgeGeo, edgeMat);
  scene.add(edgeLineSegments);

  // --- Resize ---
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

  // --- Viz object ---
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
    edgeSources: edgeData.sources,
    edgeWeights: edgeData.weights,

    updateNodeBillboard() {
      if (flat) billboardInstances(nodeMesh, positions, camera);
    },

    setFlatNodes(newFlat: boolean) {
      if (newFlat === flat) return;
      flat = newFlat;
      viz.flatNodes = flat;

      const oldColors = nodeMesh.instanceColor
        ? (nodeMesh.instanceColor.array as Float32Array).slice()
        : null;
      const count = nodeMesh.count;

      scene.remove(nodeMesh);
      nodeState.geo.dispose();
      nodeState.mat.dispose();

      nodeState = createNodeMesh(positions, count, flat);
      nodeMesh = nodeState.mesh;
      viz.nodeMesh = nodeMesh;

      if (oldColors && nodeMesh.instanceColor) {
        (nodeMesh.instanceColor.array as Float32Array).set(oldColors);
        nodeMesh.instanceColor.needsUpdate = true;
      }
      scene.add(nodeMesh);
    },

    setNodeCount(n: number) {
      nodeMesh.count = n;
      nodeMesh.instanceMatrix.needsUpdate = true;
    },

    reLayout(netRef: NeuralNetwork, threshold: number) {
      const newPos = computeLayout(netRef);
      positions.set(newPos);

      // Update all node instance matrices with new positions
      for (let i = 0; i < MAX_NEURONS; i++) {
        _dummy.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        _dummy.quaternion.identity();
        _dummy.updateMatrix();
        nodeMesh.setMatrixAt(i, _dummy.matrix);
      }
      nodeMesh.instanceMatrix.needsUpdate = true;

      this.rebuildEdges(netRef, threshold);
    },

    rebuildEdges(netRef: NeuralNetwork, threshold: number) {
      scene.remove(edgeLineSegments);
      edgeGeo.dispose();

      edgeData = buildEdgeArrays(netRef, positions, threshold);

      edgeGeo = new THREE.BufferGeometry();
      edgeGeo.setAttribute(
        "position",
        new THREE.BufferAttribute(edgeData.positionArr, 3),
      );
      const colAttr = new THREE.BufferAttribute(edgeData.colorArr, 3);
      colAttr.setUsage(THREE.DynamicDrawUsage);
      edgeGeo.setAttribute("color", colAttr);

      edgeLineSegments = new THREE.LineSegments(edgeGeo, edgeMat);
      scene.add(edgeLineSegments);

      viz.edgeLineSegments = edgeLineSegments;
      viz.edgeSources = edgeData.sources;
      viz.edgeWeights = edgeData.weights;
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
): void {
  const N = net.numNeurons;
  const activations = net.activations;
  const refCounters = net.refractoryCounters;
  const refPeriods = net.refractionPeriods;

  // Nodes: RGB = (1-C, 1-M, 1)  C=activation, M=refractory flash
  for (let i = 0; i < N; i++) {
    const c = activations[i];
    const period = refPeriods[i];
    const m = period > 0 ? refCounters[i] / period : 0;

    _tmpColor.setRGB(1 - c, 1 - m, 1);
    viz.nodeMesh.setColorAt(i, _tmpColor);
  }

  if (viz.nodeMesh.instanceColor) {
    viz.nodeMesh.instanceColor.needsUpdate = true;
  }

  // Edges: RGB = (1-C, 1-M, 1-Y)  C=source activation, M=source refractory flash, Y=weight
  const colorAttr = viz.edgeLineSegments.geometry.getAttribute(
    "color",
  ) as THREE.BufferAttribute;
  const colors = colorAttr.array as Float32Array;
  const eSources = viz.edgeSources;
  const eWeights = viz.edgeWeights;
  const numEdges = eSources.length;
  const S = viz.edgeSegsPerEdge;
  const stride = S * 2 * 3;

  for (let e = 0; e < numEdges; e++) {
    const src = eSources[e];
    const period = refPeriods[src];
    const m = period > 0 ? refCounters[src] / period : 0;
    const y = eWeights[e];

    const r = 1 - activations[src];
    const g = 1 - m;
    const b = 1 - y;

    const base = e * stride;
    for (let s = 0; s < S; s++) {
      const ci = base + s * 6;
      colors[ci] = r;
      colors[ci + 1] = g;
      colors[ci + 2] = b;
      colors[ci + 3] = r;
      colors[ci + 4] = g;
      colors[ci + 5] = b;
    }
  }

  colorAttr.needsUpdate = true;
}
