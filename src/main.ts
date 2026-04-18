import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "./styles.css";

import * as THREE from "three";
import GUI from "lil-gui";
import { NeuralNetwork, DEFAULT_PARAMS, MAX_NEURONS, type NetworkMode } from "./network.ts";
import { createVisualization, updateColors } from "./visualization.ts";
import {
  createReadoutCharts,
  updateReadoutCharts,
  type ReadoutCharts,
} from "./charts.ts";
import { createAudioEngine, type SynthMode } from "./audio.ts";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let net: NeuralNetwork;
let viz: ReturnType<typeof createVisualization>;
let charts: ReadoutCharts;
let gui: GUI;
let step = 0;
let lastPulseTime = 0;
const audio = createAudioEngine();

const params = {
  // Simulation
  playing: true,
  pulse: false,
  bpm: 120,
  ticksPerFrame: 1,

  // Network (seed + decomposed params)
  seed: Math.floor(Math.random() * 0x7fffffff),
  ...DEFAULT_PARAMS,

  // Audio
  audioOn: false,
  synthMode: "oscillator" as SynthMode,
  masterVolume: 0.8,
  actTimbre: 0,
  grainSize: 0.1,
  grainSpread: 0,
  grainRamp: 0.5,
  pitchBias: 0,
  activationOffset: 0.5,

  // Visualization
  darkMode: false,
  edgeWeightThreshold: 0.05,
  flatNodes: true,
  bloom: false,
  bloomStrength: 0.2,
  bloomRadius: 0.2,
  bloomThreshold: 0.2,
};

// ---------------------------------------------------------------------------
// Build / rebuild (only on reseed)
// ---------------------------------------------------------------------------

function rebuild() {
  if (charts) charts.dispose();
  if (viz) viz.dispose();

  net = new NeuralNetwork(params.seed, params);
  step = 0;

  viz = createVisualization(net, document.body, params.edgeWeightThreshold);
  charts = createReadoutCharts(net);

  audio.configure(net);
  syncVisualParams();
}

const PALETTE = {
  paper: 0xffffff,
  paperDark: 0x00002b,
  graphite: 0x222222,
  graphiteDark: 0xc5ced2,
} as const;

function applyTheme() {
  const dark = params.darkMode;
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  viz.scene.background = new THREE.Color(dark ? PALETTE.paperDark : PALETTE.paper);
  for (const mat of charts.borderMaterials) {
    mat.color.set(dark ? PALETTE.graphiteDark : PALETTE.graphite);
  }
}

function syncVisualParams() {
  viz.setFlatNodes(params.flatNodes);
  viz.bloomPass.enabled = params.bloom;
  viz.bloomPass.strength = params.bloomStrength;
  viz.bloomPass.radius = params.bloomRadius;
  viz.bloomPass.threshold = params.bloomThreshold;
  applyTheme();
}

// ---------------------------------------------------------------------------
// Parameter change helpers
// ---------------------------------------------------------------------------

function onWeightParamChange() {
  net.updateParams({
    numModules: params.numModules,
    interModuleFactor: params.interModuleFactor,
    sparsity: params.sparsity,
    weightScale: params.weightScale,
  });
  viz.rebuildEdges(net, params.edgeWeightThreshold);
}

function onNumNeuronsChange() {
  net.updateParams({ numNeurons: params.numNeurons });
  viz.setNodeCount(params.numNeurons);
  viz.rebuildEdges(net, params.edgeWeightThreshold);
  charts.dispose();
  charts = createReadoutCharts(net);
  audio.configure(net);
}

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------

function buildGUI() {
  gui = new GUI({ title: "Small Network" });

  const sim = gui.addFolder("Simulation");
  sim.add(params, "playing").name("Play");
  sim.add(params, "pulse").name("Pulse");
  sim.add(params, "bpm", 1, 300, 1).name("BPM");
  sim.add(params, "ticksPerFrame", 1, 20, 1).name("Ticks / frame");

  const network = gui.addFolder("Network");

  // Mode-conditional control visibility
  type ModeController = ReturnType<typeof network.add>;
  const spikingOnly: ModeController[] = [];
  const ctrnnOnly: ModeController[] = [];

  function syncModeVisibility() {
    const isCtrnn = params.mode === 'ctrnn';
    for (const c of spikingOnly) c.show(!isCtrnn);
    for (const c of ctrnnOnly) c.show(isCtrnn);
  }

  network.add(params, "seed").name("Seed").disable();
  network
    .add(params, "numNeurons", 16, MAX_NEURONS, 1)
    .name("Neurons")
    .onChange(onNumNeuronsChange);
  network
    .add(params, "numModules", 1, 8, 1)
    .name("Modules")
    .onFinishChange(() => {
      onWeightParamChange();
      charts.dispose();
      charts = createReadoutCharts(net);
    });
  network
    .add(params, "interModuleFactor", 0, 1, 0.01)
    .name("Inter-module")
    .onChange(onWeightParamChange);
  network
    .add(params, "sparsity", 0.01, 1, 0.01)
    .name("Sparsity")
    .onChange(onWeightParamChange);
  network
    .add(params, "weightScale", 0.05, 1, 0.05)
    .name("Weight scale")
    .onChange(onWeightParamChange);
  network
    .add(params, "edgeWeightThreshold", 0, 0.3, 0.01)
    .name("Edge threshold")
    .onChange(() => viz.rebuildEdges(net, params.edgeWeightThreshold));

  network
    .add(params, "mode", ['spiking', 'ctrnn'] as NetworkMode[])
    .name("Mode")
    .onChange((mode: NetworkMode) => {
      net.updateParams({ mode });
      net.resetState();
      net.kickstart();
      syncModeVisibility();
    });

  // Spiking-only controls
  spikingOnly.push(
    network
      .add(params, "activationLeak", 0.8, 1, 0.01)
      .name("Act. leak")
      .onChange(() => net.updateParams({ activationLeak: params.activationLeak })),
    network
      .add(params, "refractionLeak", 0.3, 1, 0.01)
      .name("Refrac. leak")
      .onChange(() => net.updateParams({ refractionLeak: params.refractionLeak })),
    network
      .add(params, "outputDecay", 0.3, 1, 0.01)
      .name("Output decay")
      .onChange(() => net.updateParams({ outputDecay: params.outputDecay })),
    network
      .add(params, "refractionPeriod", 1, 96, 1)
      .name("Refrac. period")
      .onChange(() =>
        net.updateParams({ refractionPeriod: params.refractionPeriod }),
      ),
    network
      .add(params, "refractionVariation", 0, 100, 1)
      .name("Refrac. var.")
      .onChange(() =>
        net.updateParams({ refractionVariation: params.refractionVariation }),
      ),
  );

  // CTRNN-only controls
  ctrnnOnly.push(
    network
      .add(params, "dt", 0.005, 0.3, 0.005)
      .name("dt")
      .onChange(() => net.updateParams({ dt: params.dt })),
    network
      .add(params, "tauMin", 0.1, 4, 0.01)
      .name("Tau min")
      .onChange(() => {
        if (params.tauMin > params.tauMax) {
          params.tauMax = params.tauMin;
          gui.controllersRecursive().forEach((c) => c.updateDisplay());
        }
        net.updateParams({ tauMin: params.tauMin, tauMax: params.tauMax });
      }),
    network
      .add(params, "tauMax", 0.1, 6, 0.01)
      .name("Tau max")
      .onChange(() => {
        if (params.tauMax < params.tauMin) {
          params.tauMin = params.tauMax;
          gui.controllersRecursive().forEach((c) => c.updateDisplay());
        }
        net.updateParams({ tauMin: params.tauMin, tauMax: params.tauMax });
      }),
    network
      .add(params, "biasScale", 0, 2, 0.05)
      .name("Bias scale")
      .onChange(() => net.updateParams({ biasScale: params.biasScale })),
  );

  network
    .add(
      {
        randomize: () => {
          params.seed = Math.floor(Math.random() * 0x7fffffff);
          gui.controllers.forEach((c) => c.updateDisplay());
          rebuild();
        },
      },
      "randomize",
    )
    .name("Randomize weights");
  network
    .add(
      {
        layout: () => {
          viz.reLayout(net, params.edgeWeightThreshold);
          audio.configure(net);
        },
      },
      "layout",
    )
    .name("Layout");

  syncModeVisibility();

  const visual = gui.addFolder("Visual");
  visual
    .add(params, "darkMode")
    .name("Dark mode")
    .onChange(applyTheme);
  visual
    .add(params, "flatNodes")
    .name("Flat nodes")
    .onChange(() => viz.setFlatNodes(params.flatNodes));

  const bloomFolder = visual.addFolder("Bloom");
  bloomFolder
    .add(params, "bloom")
    .name("Enable")
    .onChange(() => (viz.bloomPass.enabled = params.bloom));
  bloomFolder
    .add(params, "bloomStrength", 0, 3, 0.05)
    .name("Strength")
    .onChange(() => (viz.bloomPass.strength = params.bloomStrength));
  bloomFolder
    .add(params, "bloomRadius", 0, 2, 0.05)
    .name("Radius")
    .onChange(() => (viz.bloomPass.radius = params.bloomRadius));
  bloomFolder
    .add(params, "bloomThreshold", 0, 1, 0.05)
    .name("Threshold")
    .onChange(() => (viz.bloomPass.threshold = params.bloomThreshold));

  // --- Audio ---
  const audioFolder = gui.addFolder("Audio");

  type Ctrl = ReturnType<typeof audioFolder.add>;
  const oscOnly: Ctrl[] = [];
  const granularOnly: Ctrl[] = [];

  function syncSynthVisibility() {
    const isOsc = params.synthMode === "oscillator";
    for (const c of oscOnly) c.show(isOsc);
    for (const c of granularOnly) c.show(!isOsc);
  }

  function sendActiveParams() {
    audio.setParam("masterVolume", params.masterVolume);
    if (params.synthMode === "oscillator") {
      audio.setParam("actTimbre", params.actTimbre);
    } else {
      audio.setParam("size", params.grainSize);
      audio.setParam("spread", params.grainSpread);
      audio.setParam("ramp", params.grainRamp);
      audio.setParam("pitchBias", params.pitchBias);
      audio.setParam("activationOffset", params.activationOffset);
    }
  }

  audioFolder
    .add(params, "audioOn")
    .name("Audio on")
    .onChange(async (on: boolean) => {
      if (on) {
        await audio.start();
        audio.setSynthMode(params.synthMode, net);
        sendActiveParams();
      } else {
        await audio.stop();
      }
    });
  audioFolder
    .add(params, "synthMode", ["oscillator", "granular"] as SynthMode[])
    .name("Synth mode")
    .onChange((mode: SynthMode) => {
      if (audio.running) {
        audio.setSynthMode(mode, net);
        sendActiveParams();
      }
      syncSynthVisibility();
    });
  audioFolder
    .add(params, "masterVolume", 0, 1, 0.05)
    .name("Volume")
    .onChange(() => audio.setParam("masterVolume", params.masterVolume));

  // Oscillator-only
  oscOnly.push(
    audioFolder
      .add(params, "actTimbre", 0, 1, 0.05)
      .name("Act. timbre")
      .onChange(() => audio.setParam("actTimbre", params.actTimbre)),
  );

  // Granular-only
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "audio/*";
  fileInput.style.display = "none";
  document.body.appendChild(fileInput);

  const audioStatus = { file: "No file loaded" };
  const fileLabel = audioFolder.add(audioStatus, "file").name("File").disable();
  granularOnly.push(fileLabel);

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    audio.loadFile(file).then(() => {
      audioStatus.file = file.name;
      fileLabel.updateDisplay();
    });
  });

  granularOnly.push(
    audioFolder
      .add({ load: () => fileInput.click() }, "load")
      .name("Load audio"),
    audioFolder
      .add(params, "grainSize", 0, 1, 0.05)
      .name("Grain size")
      .onChange(() => audio.setParam("size", params.grainSize)),
    audioFolder
      .add(params, "grainRamp", 0, 1, 0.05)
      .name("Grain ramp")
      .onChange(() => audio.setParam("ramp", params.grainRamp)),
    audioFolder
      .add(params, "grainSpread", 0, 1, 0.05)
      .name("Grain spread")
      .onChange(() => audio.setParam("spread", params.grainSpread)),
    audioFolder
      .add(params, "pitchBias", 0, 2, 0.05)
      .name("Pitch bias")
      .onChange(() => audio.setParam("pitchBias", params.pitchBias)),
    audioFolder
      .add(params, "activationOffset", 0, 1, 0.05)
      .name("Act. offset")
      .onChange(() => audio.setParam("activationOffset", params.activationOffset)),
  );

  syncSynthVisibility();
}

// ---------------------------------------------------------------------------
// Screen-X projection for audio stereo panning
// ---------------------------------------------------------------------------

const _vpMatrix = new THREE.Matrix4();
const _v4 = new THREE.Vector4();
const _panPositions = new Float32Array(MAX_NEURONS);

function updateAudioPan() {
  const N = net.numNeurons;
  const rp = viz.renderedPositions;

  _vpMatrix.multiplyMatrices(viz.camera.projectionMatrix, viz.camera.matrixWorldInverse);

  let minX = Infinity;
  let maxX = -Infinity;

  for (let i = 0; i < N; i++) {
    _v4.set(rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2], 1);
    _v4.applyMatrix4(_vpMatrix);
    const ndcX = _v4.x / _v4.w;
    _panPositions[i] = ndcX;
    if (ndcX < minX) minX = ndcX;
    if (ndcX > maxX) maxX = ndcX;
  }

  const range = maxX - minX;
  if (range > 0.0001) {
    for (let i = 0; i < N; i++) {
      _panPositions[i] = (_panPositions[i] - minX) / range;
    }
  } else {
    for (let i = 0; i < N; i++) _panPositions[i] = 0.5;
  }

  audio.updatePan(_panPositions);
}

// ---------------------------------------------------------------------------
// Animation loop
// ---------------------------------------------------------------------------

function animate() {
  requestAnimationFrame(animate);

  if (params.playing) {
    for (let t = 0; t < params.ticksPerFrame; t++) {
      net.tick(step++);
    }

    if (params.pulse) {
      const now = performance.now();
      const interval = 60_000 / params.bpm;
      if (now - lastPulseTime >= interval) {
        lastPulseTime = now;
        const strength = params.mode === 'ctrnn' ? 0.3 : 1.0;
        net.manualActivateMostWeighted(strength);
      }
    }
  }

  audio.updateVolumes(net);
  updateColors(viz, net, params.darkMode);
  viz.updateNodePositions(net);
  updateAudioPan();
  updateReadoutCharts(
    charts,
    net,
    viz.camera,
    viz.renderedPositions,
    params.playing,
    params.darkMode,
  );
  viz.controls.update();

  if (params.bloom) {
    viz.composer.render();
  } else {
    viz.renderer.render(viz.scene, viz.camera);
  }

  viz.renderer.autoClear = false;
  viz.renderer.clearDepth();
  viz.renderer.render(charts.scene, charts.camera);
  viz.renderer.autoClear = true;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

rebuild();
buildGUI();

window.addEventListener("resize", () => {
  viz.resize();
  charts.resize(window.innerWidth, window.innerHeight);
});

animate();
