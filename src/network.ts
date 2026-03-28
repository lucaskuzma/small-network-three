// Default hyperparameters (from Python small-network)
export const DEFAULT_N_OUTPUTS_PER_READOUT = 5;
export const DEFAULT_NUM_READOUTS = 3;
export const DEFAULT_NUM_MODULES = 3;
export const DEFAULT_INTER_MODULE_FACTOR = 0.5;
export const NEURONS_PER_OUTPUT = 17;
export const DEFAULT_NUM_NEURONS =
  DEFAULT_NUM_READOUTS * DEFAULT_N_OUTPUTS_PER_READOUT * NEURONS_PER_OUTPUT; // 255

export const DEFAULT_ACTIVATION_LEAK = 0.98;
export const DEFAULT_REFRACTION_LEAK = 0.75;

export const DEFAULT_NETWORK_SPARSITY = 0.2;
export const DEFAULT_NETWORK_WEIGHT_SCALE = 0.3;
export const DEFAULT_WEIGHT_THRESHOLD = 0;

export const DEFAULT_OUTPUT_SPARSITY = 0.05;
export const DEFAULT_OUTPUT_WEIGHT_SCALE = 0.3;

export const DEFAULT_REFRACTION_PERIOD = 2;
export const DEFAULT_REFRACTION_VARIATION = 62;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function randn(): number {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 || 1e-30)) * Math.cos(2 * Math.PI * u2);
}

function clip(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Matrix-vector product: result[i] = sum_j mat[i * cols + j] * vec[j] */
function matVecTranspose(
  mat: Float64Array,
  rows: number,
  cols: number,
  vec: Float64Array | Uint8Array,
): Float64Array {
  // Computes mat^T @ vec  =>  out[j] = sum_i mat[i*cols + j] * vec[i]
  const out = new Float64Array(cols);
  for (let i = 0; i < rows; i++) {
    const vi = vec[i];
    if (vi === 0) continue;
    const base = i * cols;
    for (let j = 0; j < cols; j++) {
      out[j] += mat[base + j] * vi;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// NeuralNetworkState
// ---------------------------------------------------------------------------

export interface NeuralNetworkStateConfig {
  numNeurons?: number;
  numReadouts?: number;
  nOutputsPerReadout?: number;
}

export class NeuralNetworkState {
  numNeurons: number;
  numReadouts: number;
  nOutputsPerReadout: number;
  numOutputs: number;

  networkWeights: Float64Array; // N x N row-major
  thresholds: Float64Array;
  thresholdsCurrent: Float64Array;
  thresholdVariationRanges: Float64Array;
  thresholdVariationPeriods: Float64Array;
  outputWeights: Float64Array; // N x numOutputs row-major
  activations: Float64Array;
  firing: Uint8Array;
  outputs: Float64Array;
  refractoryCounters: Int32Array;
  refractionPeriod: Int32Array;

  useActivationLeak = false;
  activationLeak = DEFAULT_ACTIVATION_LEAK;
  refractionLeak = DEFAULT_REFRACTION_LEAK;
  useRefractionDecay = false;
  useTanhActivation = false;
  weightThreshold = 0;

  constructor(config: NeuralNetworkStateConfig = {}) {
    const N = (this.numNeurons = config.numNeurons ?? DEFAULT_NUM_NEURONS);
    this.numReadouts = config.numReadouts ?? DEFAULT_NUM_READOUTS;
    this.nOutputsPerReadout =
      config.nOutputsPerReadout ?? DEFAULT_N_OUTPUTS_PER_READOUT;
    this.numOutputs = this.numReadouts * this.nOutputsPerReadout;

    this.networkWeights = new Float64Array(N * N);
    this.thresholds = new Float64Array(N).fill(0.5);
    this.thresholdsCurrent = new Float64Array(N).fill(0.5);
    this.thresholdVariationRanges = new Float64Array(N);
    this.thresholdVariationPeriods = new Float64Array(N);
    this.outputWeights = new Float64Array(N * this.numOutputs);
    for (let i = 0; i < Math.min(N, this.numOutputs); i++) {
      this.outputWeights[i * this.numOutputs + i] = 1.0;
    }
    this.activations = new Float64Array(N);
    this.firing = new Uint8Array(N);
    this.outputs = new Float64Array(this.numOutputs);
    this.refractoryCounters = new Int32Array(N);
    this.refractionPeriod = new Int32Array(N);
  }

  getReadoutOutputs(): Float64Array[] {
    const result: Float64Array[] = [];
    for (let r = 0; r < this.numReadouts; r++) {
      const start = r * this.nOutputsPerReadout;
      result.push(
        this.outputs.slice(start, start + this.nOutputsPerReadout) as Float64Array,
      );
    }
    return result;
  }
}

// ---------------------------------------------------------------------------
// NeuralNetwork
// ---------------------------------------------------------------------------

export class NeuralNetwork {
  state: NeuralNetworkState;
  moduleAssignments: Int32Array | null = null;

  constructor(config: NeuralNetworkStateConfig = {}) {
    this.state = new NeuralNetworkState(config);
  }

  // -- Weight threshold helper ------------------------------------------------

  private applyWeightThreshold(
    src: Float64Array,
    dst: Float64Array,
    len: number,
  ): void {
    const t = this.state.weightThreshold;
    if (t <= 0) {
      dst.set(src);
      return;
    }
    for (let i = 0; i < len; i++) {
      dst[i] = Math.abs(src[i]) >= t ? src[i] : 0;
    }
  }

  // -- Tick -------------------------------------------------------------------

  tick(step: number): void {
    const s = this.state;
    const N = s.numNeurons;

    // 1. Effective weights (apply threshold)
    const effW = new Float64Array(N * N);
    this.applyWeightThreshold(s.networkWeights, effW, N * N);

    // incoming[i] = effW^T @ firing  (effW is NxN row-major)
    const incoming = matVecTranspose(effW, N, N, s.firing);

    // 2. Refractory gating
    const canReceive = new Uint8Array(N);
    if (s.useRefractionDecay) {
      for (let i = 0; i < N; i++) canReceive[i] = s.refractoryCounters[i] === 0 ? 1 : 0;
    } else {
      canReceive.fill(1);
    }

    // 3. Update activations
    const newAct = Float64Array.from(s.activations);
    if (s.useTanhActivation) {
      for (let i = 0; i < N; i++) {
        if (canReceive[i]) {
          newAct[i] = (Math.tanh(s.activations[i] + incoming[i]) + 1) / 2;
        }
      }
    } else {
      for (let i = 0; i < N; i++) {
        if (canReceive[i]) {
          newAct[i] = clip(s.activations[i] + incoming[i], 0, 1);
        }
      }
    }

    // 4. Firing
    const newFiring = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      newFiring[i] = canReceive[i] && newAct[i] >= s.thresholdsCurrent[i] ? 1 : 0;
    }

    // 5. Post-fire state
    const newRefrac = Int32Array.from(s.refractoryCounters);
    if (s.useRefractionDecay) {
      for (let i = 0; i < N; i++) {
        if (newFiring[i]) newRefrac[i] = s.refractionPeriod[i];
      }
    } else {
      for (let i = 0; i < N; i++) {
        if (newFiring[i]) newAct[i] = 0;
      }
    }

    // 6. Refraction leak on neurons that WERE refractory
    if (s.useRefractionDecay) {
      for (let i = 0; i < N; i++) {
        if (s.refractoryCounters[i] > 0) {
          newAct[i] *= s.refractionLeak;
        }
      }
    }

    // 7. Threshold variation
    let anyVariation = false;
    for (let i = 0; i < N; i++) {
      if (s.thresholdVariationPeriods[i] > 0) {
        anyVariation = true;
        break;
      }
    }
    if (anyVariation) {
      for (let i = 0; i < N; i++) {
        const p = s.thresholdVariationPeriods[i];
        if (p > 0) {
          const phase = (step * 2 * Math.PI) / p;
          const variation = Math.sin(phase) * s.thresholdVariationRanges[i];
          s.thresholdsCurrent[i] = clip(s.thresholds[i] + variation, 0, 1);
        }
      }
    }

    // 8. Outputs
    const newOut = new Float64Array(s.numOutputs);
    for (let i = 0; i < s.numOutputs; i++) {
      newOut[i] = s.outputs[i] * s.refractionLeak;
    }
    const effOW = new Float64Array(N * s.numOutputs);
    this.applyWeightThreshold(s.outputWeights, effOW, N * s.numOutputs);
    // effOW^T @ newFiring
    const outAdd = matVecTranspose(effOW, N, s.numOutputs, newFiring);
    for (let i = 0; i < s.numOutputs; i++) {
      newOut[i] = clip(newOut[i] + outAdd[i], 0, 1);
    }

    // 9. Activation leak
    if (s.useActivationLeak) {
      for (let i = 0; i < N; i++) newAct[i] *= s.activationLeak;
    }

    // 10. Decrement refractory
    if (s.useRefractionDecay) {
      for (let i = 0; i < N; i++) {
        newRefrac[i] = Math.max(0, newRefrac[i] - 1);
      }
    }

    // Commit
    s.activations = newAct;
    s.firing = newFiring;
    s.outputs = newOut;
    if (s.useRefractionDecay) s.refractoryCounters = newRefrac;
  }

  // -- Manual activation ------------------------------------------------------

  manualTrigger(idx: number): void {
    this.state.activations[idx] = 1;
  }

  manualActivate(idx: number, value: number): void {
    this.state.activations[idx] = clip(this.state.activations[idx] + value, 0, 1);
  }

  manualActivateMostWeighted(value: number): number {
    const N = this.state.numNeurons;
    const w = this.state.networkWeights;
    let bestIdx = 0;
    let bestSum = -Infinity;
    for (let i = 0; i < N; i++) {
      let sum = 0;
      const base = i * N;
      for (let j = 0; j < N; j++) sum += w[base + j];
      if (sum > bestSum) {
        bestSum = sum;
        bestIdx = i;
      }
    }
    this.manualActivate(bestIdx, value);
    return bestIdx;
  }

  manualActivateMostWeightedPerModule(value: number): number[] {
    if (!this.moduleAssignments) throw new Error("No module assignments");
    const N = this.state.numNeurons;
    const w = this.state.networkWeights;

    const totalWeights = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let sum = 0;
      const base = i * N;
      for (let j = 0; j < N; j++) sum += w[base + j];
      totalWeights[i] = sum;
    }

    const numModules = this.moduleAssignments.reduce((a, b) => Math.max(a, b), 0) + 1;
    const activated: number[] = [];
    for (let m = 0; m < numModules; m++) {
      let bestIdx = -1;
      let bestW = -Infinity;
      for (let i = 0; i < N; i++) {
        if (this.moduleAssignments[i] === m && totalWeights[i] > bestW) {
          bestW = totalWeights[i];
          bestIdx = i;
        }
      }
      if (bestIdx >= 0) {
        this.manualActivate(bestIdx, value);
        activated.push(bestIdx);
      }
    }
    return activated;
  }

  // -- Configuration ----------------------------------------------------------

  clearFiring(): void {
    this.state.firing.fill(0);
    this.state.outputs.fill(0);
  }

  enableActivationLeak(leak: number = DEFAULT_ACTIVATION_LEAK): void {
    this.state.useActivationLeak = true;
    this.state.activationLeak = clip(leak, 0, 1);
  }

  disableActivationLeak(): void {
    this.state.useActivationLeak = false;
  }

  setWeightThreshold(threshold: number = 0.05): void {
    this.state.weightThreshold = Math.max(0, threshold);
  }

  enableRefractionDecay(
    refractionPeriod: number = DEFAULT_REFRACTION_PERIOD,
    refractionLeak: number = DEFAULT_REFRACTION_LEAK,
    refractionVariation: number = DEFAULT_REFRACTION_VARIATION,
  ): void {
    const s = this.state;
    s.refractionLeak = clip(refractionLeak, 0, 1);
    s.useRefractionDecay = true;
    s.refractionPeriod = new Int32Array(s.numNeurons).fill(refractionPeriod);
    if (refractionVariation > 0) {
      for (let i = 0; i < s.numNeurons; i++) {
        s.refractionPeriod[i] += Math.floor(Math.random() * refractionVariation);
      }
    }
  }

  disableRefractionDecay(): void {
    this.state.useRefractionDecay = false;
  }

  // -- Weight randomization ---------------------------------------------------

  randomizeWeights(
    sparsity = DEFAULT_NETWORK_SPARSITY,
    scale = DEFAULT_NETWORK_WEIGHT_SCALE,
    numModules = 1,
    interModuleFactor?: number,
  ): void {
    const N = this.state.numNeurons;

    // Gaussian weights clipped to [-1, 1]
    for (let i = 0; i < N * N; i++) {
      this.state.networkWeights[i] = clip(randn() * scale, -1, 1);
    }

    // Sparsity mask
    if (numModules <= 1) {
      for (let i = 0; i < N * N; i++) {
        if (Math.random() >= sparsity) this.state.networkWeights[i] = 0;
      }
    } else {
      if (interModuleFactor == null)
        throw new Error("interModuleFactor required when numModules > 1");
      const moduleSize = Math.floor(N / numModules);
      const interSparsity = sparsity * interModuleFactor;
      const mask = new Uint8Array(N * N);

      for (let m = 0; m < numModules; m++) {
        const start = m * moduleSize;
        const end = m < numModules - 1 ? (m + 1) * moduleSize : N;
        // Intra-module
        for (let i = start; i < end; i++)
          for (let j = start; j < end; j++)
            if (Math.random() < sparsity) mask[i * N + j] = 1;
        // Inter-module
        for (let m2 = 0; m2 < numModules; m2++) {
          if (m2 === m) continue;
          const s2 = m2 * moduleSize;
          const e2 = m2 < numModules - 1 ? (m2 + 1) * moduleSize : N;
          for (let i = start; i < end; i++)
            for (let j = s2; j < e2; j++)
              if (Math.random() < interSparsity) mask[i * N + j] = 1;
        }
      }
      for (let i = 0; i < N * N; i++) {
        if (!mask[i]) this.state.networkWeights[i] = 0;
      }
    }

    // No self-connections
    for (let i = 0; i < N; i++) this.state.networkWeights[i * N + i] = 0;

    // Store module assignments
    if (numModules > 1) {
      const moduleSize = Math.floor(N / numModules);
      this.moduleAssignments = new Int32Array(N);
      for (let m = 0; m < numModules; m++) {
        const start = m * moduleSize;
        const end = m < numModules - 1 ? (m + 1) * moduleSize : N;
        for (let i = start; i < end; i++) this.moduleAssignments[i] = m;
      }
    }
  }

  randomizeOutputWeights(
    sparsity = DEFAULT_OUTPUT_SPARSITY,
    scale = DEFAULT_OUTPUT_WEIGHT_SCALE,
  ): void {
    const N = this.state.numNeurons;
    const K = this.state.numOutputs;
    for (let i = 0; i < N * K; i++) {
      this.state.outputWeights[i] =
        Math.random() < sparsity ? Math.random() * scale : 0;
    }
  }

  randomizeThresholds(): void {
    const N = this.state.numNeurons;
    for (let i = 0; i < N; i++) {
      const v = Math.random();
      this.state.thresholds[i] = v;
      this.state.thresholdsCurrent[i] = v;
    }
  }

  randomizeThresholdVariations(range = 0.1, period = 8): void {
    const N = this.state.numNeurons;
    for (let i = 0; i < N; i++) {
      this.state.thresholdVariationRanges[i] = Math.random() * range;
      this.state.thresholdVariationPeriods[i] =
        period > 0 ? Math.floor(Math.random() * period) : 0;
    }
  }

  getSpectralRadius(): number {
    // Simplified: compute power iteration estimate for largest eigenvalue magnitude
    const N = this.state.numNeurons;
    const w = this.state.networkWeights;
    let vec = new Float64Array(N);
    for (let i = 0; i < N; i++) vec[i] = randn();

    // Normalize
    let norm = 0;
    for (let i = 0; i < N; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    for (let i = 0; i < N; i++) vec[i] /= norm;

    let eigenvalue = 0;
    for (let iter = 0; iter < 100; iter++) {
      const next = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        let sum = 0;
        const base = i * N;
        for (let j = 0; j < N; j++) sum += w[base + j] * vec[j];
        next[i] = sum;
      }
      norm = 0;
      for (let i = 0; i < N; i++) norm += next[i] * next[i];
      norm = Math.sqrt(norm);
      eigenvalue = norm;
      if (norm > 0) for (let i = 0; i < N; i++) next[i] /= norm;
      vec = next;
    }
    return eigenvalue;
  }

  clear(): void {
    this.state = new NeuralNetworkState({
      numNeurons: this.state.numNeurons,
      numReadouts: this.state.numReadouts,
      nOutputsPerReadout: this.state.nOutputsPerReadout,
    });
  }

  /** Count non-zero entries in the network weight matrix. */
  connectionCount(): number {
    let count = 0;
    for (let i = 0; i < this.state.networkWeights.length; i++) {
      if (this.state.networkWeights[i] !== 0) count++;
    }
    return count;
  }
}
