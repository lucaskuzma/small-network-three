import GUI from "lil-gui";
import { NeuralNetwork, DEFAULT_PARAMS } from "./network.ts";
import { createVisualization, updateColors } from "./visualization.ts";
import {
  createReadoutCharts,
  updateReadoutCharts,
  type ReadoutCharts,
} from "./charts.ts";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let net: NeuralNetwork;
let viz: ReturnType<typeof createVisualization>;
let charts: ReadoutCharts;
let step = 0;
let lastPulseTime = 0;

const params = {
  // Simulation
  playing: true,
  pulse: false,
  bpm: 120,
  ticksPerFrame: 1,
  flashDecay: 0.85,

  // Network (seed + decomposed params)
  seed: Math.floor(Math.random() * 0x7fffffff),
  ...DEFAULT_PARAMS,

  // Visualization
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

  syncVisualParams();
}

function syncVisualParams() {
  viz.setFlatNodes(params.flatNodes);
  viz.bloomPass.enabled = params.bloom;
  viz.bloomPass.strength = params.bloomStrength;
  viz.bloomPass.radius = params.bloomRadius;
  viz.bloomPass.threshold = params.bloomThreshold;
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
}

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------

function buildGUI() {
  const gui = new GUI({ title: "Small Network" });

  const sim = gui.addFolder("Simulation");
  sim.add(params, "playing").name("Play");
  sim.add(params, "pulse").name("Pulse");
  sim.add(params, "bpm", 30, 300, 1).name("BPM");
  sim.add(params, "ticksPerFrame", 1, 20, 1).name("Ticks / frame");
  sim.add(params, "flashDecay", 0.5, 0.99, 0.01).name("Flash decay");

  const network = gui.addFolder("Network");
  network.add(params, "seed").name("Seed").disable();
  network
    .add(params, "numNeurons", 16, 512, 1)
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
    .add(params, "interModuleFactor", 0, 1, 0.05)
    .name("Inter-module")
    .onChange(onWeightParamChange);
  network
    .add(params, "sparsity", 0.01, 0.5, 0.01)
    .name("Sparsity")
    .onChange(onWeightParamChange);
  network
    .add(params, "weightScale", 0.05, 1, 0.05)
    .name("Weight scale")
    .onChange(onWeightParamChange);
  network
    .add(params, "activationLeak", 0.8, 1, 0.01)
    .name("Act. leak")
    .onChange(() => net.updateParams({ activationLeak: params.activationLeak }));
  network
    .add(params, "refractionLeak", 0.3, 1, 0.01)
    .name("Refrac. leak")
    .onChange(() => net.updateParams({ refractionLeak: params.refractionLeak }));
  network
    .add(params, "refractionPeriod", 1, 20, 1)
    .name("Refrac. period")
    .onChange(() =>
      net.updateParams({ refractionPeriod: params.refractionPeriod }),
    );
  network
    .add(params, "refractionVariation", 0, 100, 1)
    .name("Refrac. var.")
    .onChange(() =>
      net.updateParams({ refractionVariation: params.refractionVariation }),
    );
  network
    .add(params, "edgeWeightThreshold", 0, 0.3, 0.01)
    .name("Edge threshold")
    .onChange(() => viz.rebuildEdges(net, params.edgeWeightThreshold));
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
      { layout: () => viz.reLayout(net, params.edgeWeightThreshold) },
      "layout",
    )
    .name("Layout");

  const visual = gui.addFolder("Visual");
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
        if (params.numModules > 1) {
          net.manualActivateMostWeightedPerModule(1.0);
        } else {
          net.manualActivateMostWeighted(1.0);
        }
      }
    }
  }

  updateColors(viz, net, params.flashDecay);
  viz.updateNodeBillboard();
  updateReadoutCharts(
    charts,
    net,
    viz.camera,
    viz.nodePositions,
    params.playing,
    params.flashDecay,
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
