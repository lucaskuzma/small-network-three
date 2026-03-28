import GUI from "lil-gui";
import {
  NeuralNetwork,
  DEFAULT_NUM_NEURONS,
  DEFAULT_NUM_READOUTS,
  DEFAULT_N_OUTPUTS_PER_READOUT,
  DEFAULT_NUM_MODULES,
  DEFAULT_INTER_MODULE_FACTOR,
  DEFAULT_NETWORK_SPARSITY,
  DEFAULT_NETWORK_WEIGHT_SCALE,
  DEFAULT_ACTIVATION_LEAK,
  DEFAULT_REFRACTION_LEAK,
  DEFAULT_REFRACTION_PERIOD,
  DEFAULT_REFRACTION_VARIATION,
} from "./network.ts";
import { createVisualization, updateColors } from "./visualization.ts";
import { createReadoutCharts, updateReadoutCharts, type ReadoutCharts } from "./charts.ts";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let net: NeuralNetwork;
let viz: ReturnType<typeof createVisualization>;
let charts: ReadoutCharts;
let step = 0;

const params = {
  playing: true,
  ticksPerFrame: 1,
  flashDecay: 0.85,
  numNeurons: DEFAULT_NUM_NEURONS,
  numModules: DEFAULT_NUM_MODULES,
  interModuleFactor: DEFAULT_INTER_MODULE_FACTOR,
  sparsity: DEFAULT_NETWORK_SPARSITY,
  weightScale: DEFAULT_NETWORK_WEIGHT_SCALE,
  activationLeak: DEFAULT_ACTIVATION_LEAK,
  refractionLeak: DEFAULT_REFRACTION_LEAK,
  refractionPeriod: DEFAULT_REFRACTION_PERIOD,
  refractionVariation: DEFAULT_REFRACTION_VARIATION,
  edgeWeightThreshold: 0.05,
  reset: () => rebuild(),
};

// ---------------------------------------------------------------------------
// Build / rebuild
// ---------------------------------------------------------------------------

function rebuild() {
  if (charts) charts.dispose();
  if (viz) viz.dispose();

  net = new NeuralNetwork({
    numNeurons: params.numNeurons,
    numReadouts: DEFAULT_NUM_READOUTS,
    nOutputsPerReadout: DEFAULT_N_OUTPUTS_PER_READOUT,
  });

  net.randomizeWeights(
    params.sparsity,
    params.weightScale,
    params.numModules,
    params.numModules > 1 ? params.interModuleFactor : undefined,
  );
  net.randomizeOutputWeights();
  net.randomizeThresholds();
  net.enableRefractionDecay(
    params.refractionPeriod,
    params.refractionLeak,
    params.refractionVariation,
  );
  net.enableActivationLeak(params.activationLeak);

  if (params.numModules > 1 && net.moduleAssignments) {
    net.manualActivateMostWeightedPerModule(1.0);
  } else {
    net.manualActivateMostWeighted(1.0);
  }

  step = 0;

  viz = createVisualization(net, document.body, params.edgeWeightThreshold);
  charts = createReadoutCharts(net);
}

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------

function buildGUI() {
  const gui = new GUI({ title: "Small Network" });

  const sim = gui.addFolder("Simulation");
  sim.add(params, "playing").name("Play");
  sim.add(params, "ticksPerFrame", 1, 20, 1).name("Ticks / frame");
  sim.add(params, "flashDecay", 0.5, 0.99, 0.01).name("Flash decay");

  const network = gui.addFolder("Network");
  network.add(params, "numNeurons", 16, 512, 1).name("Neurons");
  network.add(params, "numModules", 1, 8, 1).name("Modules");
  network.add(params, "interModuleFactor", 0, 1, 0.05).name("Inter-module");
  network.add(params, "sparsity", 0.01, 0.5, 0.01).name("Sparsity");
  network.add(params, "weightScale", 0.05, 1, 0.05).name("Weight scale");
  network.add(params, "activationLeak", 0.8, 1, 0.01).name("Act. leak");
  network.add(params, "refractionLeak", 0.3, 1, 0.01).name("Refrac. leak");
  network.add(params, "refractionPeriod", 1, 20, 1).name("Refrac. period");
  network.add(params, "refractionVariation", 0, 100, 1).name("Refrac. var.");
  network.add(params, "edgeWeightThreshold", 0, 0.3, 0.01).name("Edge threshold");

  gui.add(params, "reset").name("Rebuild network");
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
  }

  updateColors(viz, net, params.flashDecay);
  updateReadoutCharts(charts, net, viz.camera, viz.nodePositions, params.playing);
  viz.controls.update();

  // Pass 1: main 3D scene
  viz.renderer.render(viz.scene, viz.camera);
  // Pass 2: HUD charts overlay (preserve color buffer, clear depth)
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
