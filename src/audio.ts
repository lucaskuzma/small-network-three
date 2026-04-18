import { type NeuralNetwork, MAX_NEURONS, mulberry32 } from "./network.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SynthMode = "oscillator" | "granular";

type AudioParamName =
  | "masterVolume"
  | "actTimbre"
  | "size"
  | "ramp"
  | "pitchBias"
  | "activationOffset";

export interface AudioEngine {
  start(): Promise<void>;
  stop(): Promise<void>;
  loadFile(file: File): Promise<void>;
  setSynthMode(mode: SynthMode, net: NeuralNetwork): void;
  configure(net: NeuralNetwork): void;
  updatePan(panPositions: Float32Array): void;
  updateVolumes(net: NeuralNetwork): void;
  setParam(name: AudioParamName, value: number): void;
  readonly running: boolean;
  readonly bufferLoaded: boolean;
  readonly synthMode: SynthMode;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Fiedler vector: spectral 1D ordering from network topology
// ---------------------------------------------------------------------------

export function computeFiedlerPositions(net: NeuralNetwork): Float32Array {
  const N = net.numNeurons;
  const M = MAX_NEURONS;
  const ew = net.effectiveWeights;

  const degree = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let d = 0;
    for (let j = 0; j < N; j++) d += Math.abs(ew[i * M + j]);
    degree[i] = d;
  }

  // α > λ_max of L (Gershgorin: λ_max ≤ 2 * max degree)
  let alpha = 0;
  for (let i = 0; i < N; i++) {
    if (degree[i] > alpha) alpha = degree[i];
  }
  alpha *= 2.1;

  // Power iteration on (αI - L) orthogonalized against the trivial
  // eigenvector (all-ones). Converges to the Fiedler vector.
  const rng = mulberry32(net.seed + 12345);
  const v = new Float64Array(N);
  for (let i = 0; i < N; i++) v[i] = rng() - 0.5;

  const tmp = new Float64Array(N);
  for (let iter = 0; iter < 300; iter++) {
    // tmp = (αI - L)v = (α - degree[i]) * v[i] + Σ_j |w_ij| * v[j]
    for (let i = 0; i < N; i++) {
      let Av = 0;
      const base = i * M;
      for (let j = 0; j < N; j++) Av += Math.abs(ew[base + j]) * v[j];
      tmp[i] = (alpha - degree[i]) * v[i] + Av;
    }

    let mean = 0;
    for (let i = 0; i < N; i++) mean += tmp[i];
    mean /= N;
    for (let i = 0; i < N; i++) tmp[i] -= mean;

    let norm = 0;
    for (let i = 0; i < N; i++) norm += tmp[i] * tmp[i];
    norm = Math.sqrt(norm);
    if (norm > 1e-12) {
      for (let i = 0; i < N; i++) v[i] = tmp[i] / norm;
    } else {
      break;
    }
  }

  let minV = Infinity, maxV = -Infinity;
  for (let i = 0; i < N; i++) {
    if (v[i] < minV) minV = v[i];
    if (v[i] > maxV) maxV = v[i];
  }

  const out = new Float32Array(N);
  const range = maxV - minV;
  if (range > 1e-12) {
    for (let i = 0; i < N; i++) out[i] = (v[i] - minV) / range;
  } else {
    for (let i = 0; i < N; i++) out[i] = i / (N - 1 || 1);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createAudioEngine(): AudioEngine {
  let ctx: AudioContext | null = null;
  let node: AudioWorkletNode | null = null;
  let workletReady = false;
  let running = false;
  let bufferLoaded = false;
  let decodedBuffer: AudioBuffer | null = null;
  let neuronCount = 0;
  let mode: SynthMode = "oscillator";

  async function ensureContext(): Promise<AudioContext> {
    if (!ctx) {
      ctx = new AudioContext();
    }
    if (!workletReady) {
      await ctx.audioWorklet.addModule("/network-oscillator-processor.js");
      await ctx.audioWorklet.addModule("/network-granular-processor.js");
      workletReady = true;
    }
    return ctx;
  }

  function createNode(synthMode: SynthMode) {
    if (!ctx) return;
    if (node) {
      node.disconnect();
      node = null;
    }
    const name = synthMode === "oscillator"
      ? "network-oscillator-processor"
      : "network-granular-processor";
    node = new AudioWorkletNode(ctx, name, {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });
    node.connect(ctx.destination);
  }

  function sendBuffer() {
    if (!node || !decodedBuffer) return;
    const channels: Float32Array[] = [];
    for (let ch = 0; ch < decodedBuffer.numberOfChannels; ch++) {
      channels.push(new Float32Array(decodedBuffer.getChannelData(ch)));
    }
    node.port.postMessage(
      { type: "loadBuffer", channelData: channels },
      channels.map((c) => c.buffer),
    );
  }

  const engine: AudioEngine = {
    get running() {
      return running;
    },
    get bufferLoaded() {
      return bufferLoaded;
    },
    get synthMode() {
      return mode;
    },

    async start() {
      const audioCtx = await ensureContext();
      if (!node) {
        createNode(mode);
      }
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
      running = true;
    },

    async stop() {
      if (ctx && ctx.state === "running") {
        await ctx.suspend();
      }
      running = false;
    },

    async loadFile(file: File) {
      const audioCtx = await ensureContext();
      const arrayBuffer = await file.arrayBuffer();
      const buffer = await audioCtx.decodeAudioData(arrayBuffer);
      decodedBuffer = buffer;
      bufferLoaded = true;
      if (mode === "granular") {
        sendBuffer();
      }
    },

    setSynthMode(newMode: SynthMode, net: NeuralNetwork) {
      mode = newMode;
      if (!ctx || !workletReady) return;
      createNode(mode);
      if (mode === "granular" && decodedBuffer) {
        sendBuffer();
      }
      engine.configure(net);
    },

    configure(net: NeuralNetwork) {
      neuronCount = net.numNeurons;
      const positions = computeFiedlerPositions(net);
      if (!node) return;
      if (mode === "oscillator") {
        const noteNumbers = new Float32Array(neuronCount);
        for (let i = 0; i < neuronCount; i++) {
          noteNumbers[i] = 32 + Math.round(positions[i] * 63);
        }
        node.port.postMessage({
          type: "configure",
          oscCount: neuronCount,
          noteNumbers,
        });
      } else {
        node.port.postMessage({
          type: "configure",
          grainCount: neuronCount,
          positions,
        });
      }
    },

    updatePan(panPositions: Float32Array) {
      if (!node || !running) return;
      if (mode === "granular" && !bufferLoaded) return;
      node.port.postMessage({ type: "updatePan", panPositions });
    },

    updateVolumes(net: NeuralNetwork) {
      if (!node || !running) return;
      if (mode === "granular" && !bufferLoaded) return;
      const N = net.numNeurons;
      const isCtrnn = net.params.mode === "ctrnn";
      const activations = net.activations;
      const vols = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        vols[i] = isCtrnn
          ? Math.abs(Math.tanh(activations[i]))
          : activations[i];
      }
      node.port.postMessage({ type: "updateVolumes", volumes: vols });
    },

    setParam(name: AudioParamName, value: number) {
      if (!node) return;
      const p = node.parameters.get(name);
      if (p) p.setValueAtTime(value, ctx?.currentTime ?? 0);
    },

    dispose() {
      if (node) {
        node.disconnect();
        node = null;
      }
      if (ctx) {
        void ctx.close();
        ctx = null;
      }
      workletReady = false;
      running = false;
      bufferLoaded = false;
      decodedBuffer = null;
    },
  };

  return engine;
}
