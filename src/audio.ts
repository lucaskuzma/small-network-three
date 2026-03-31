import { type NeuralNetwork } from "./network.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type AudioParamName = "size" | "spread" | "ramp" | "masterVolume";

export interface AudioEngine {
  /** Resume AudioContext and register worklet (first call only). */
  start(): Promise<void>;
  /** Suspend AudioContext. */
  stop(): Promise<void>;
  /** Decode an audio file and send its buffer to the worklet. */
  loadFile(file: File): Promise<void>;
  /** Send grain count and per-grain buffer positions to the worklet. */
  configure(numGrains: number, nodePositions: Float32Array): void;
  /** Read activations from the network and send per-grain volumes to the worklet. */
  updateVolumes(net: NeuralNetwork): void;
  /** Set a shared AudioParam on the worklet node. */
  setParam(name: AudioParamName, value: number): void;
  readonly running: boolean;
  readonly bufferLoaded: boolean;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Position helper: normalized distances from origin
// ---------------------------------------------------------------------------

function computeGrainPositions(
  numGrains: number,
  nodePositions: Float32Array,
): Float32Array {
  const out = new Float32Array(numGrains);
  let maxDist = 0;
  for (let i = 0; i < numGrains; i++) {
    const px = nodePositions[i * 3];
    const py = nodePositions[i * 3 + 1];
    const pz = nodePositions[i * 3 + 2];
    const d = Math.sqrt(px * px + py * py + pz * pz);
    out[i] = d;
    if (d > maxDist) maxDist = d;
  }
  if (maxDist > 0) {
    for (let i = 0; i < numGrains; i++) out[i] /= maxDist;
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
  let grainCount = 0;

  async function ensureContext(): Promise<AudioContext> {
    if (!ctx) {
      ctx = new AudioContext();
    }
    if (!workletReady) {
      await ctx.audioWorklet.addModule("/network-granular-processor.js");
      node = new AudioWorkletNode(ctx, "network-granular-processor", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
      });
      node.connect(ctx.destination);
      workletReady = true;
    }
    return ctx;
  }

  function sendBuffer(audioCtx: AudioContext, buffer: AudioBuffer) {
    if (!node) return;
    const channels: Float32Array[] = [];
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
      channels.push(new Float32Array(buffer.getChannelData(ch)));
    }
    node.port.postMessage(
      { type: "loadBuffer", channelData: channels },
      channels.map((c) => c.buffer),
    );
    decodedBuffer = buffer;
    bufferLoaded = true;
    void audioCtx;
  }

  const engine: AudioEngine = {
    get running() {
      return running;
    },
    get bufferLoaded() {
      return bufferLoaded;
    },

    async start() {
      const audioCtx = await ensureContext();
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
      sendBuffer(audioCtx, buffer);
    },

    configure(numGrains: number, nodePositions: Float32Array) {
      grainCount = numGrains;
      const positions = computeGrainPositions(numGrains, nodePositions);
      if (node) {
        node.port.postMessage({ type: "configure", grainCount: numGrains, positions });
      }
    },

    updateVolumes(net: NeuralNetwork) {
      if (!node || !running || !bufferLoaded) return;
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
