export const MAX_NEURONS = 512;
export const DEFAULT_NUM_READOUTS = 3;
export const DEFAULT_N_OUTPUTS_PER_READOUT = 5;
const OUTPUT_SPARSITY = 0.05;
const OUTPUT_SCALE = 0.3;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

export type NetworkMode = 'spiking' | 'ctrnn';

export interface NetworkParams {
  mode: NetworkMode;
  numNeurons: number;
  numModules: number;
  interModuleFactor: number;
  sparsity: number;
  weightScale: number;
  // Spiking-specific
  activationLeak: number;
  refractionLeak: number;
  outputDecay: number;
  refractionPeriod: number;
  refractionVariation: number;
  // CTRNN-specific
  dt: number;
  tauMin: number;
  tauMax: number;
  biasScale: number;
}

export const DEFAULT_PARAMS: NetworkParams = {
  mode: 'spiking',
  numNeurons: DEFAULT_NUM_READOUTS * DEFAULT_N_OUTPUTS_PER_READOUT * 17, // 255
  numModules: 3,
  interModuleFactor: 0.5,
  sparsity: 0.2,
  weightScale: 0.3,
  activationLeak: 0.98,
  refractionLeak: 0.75,
  outputDecay: 0.75,
  refractionPeriod: 2,
  refractionVariation: 62,
  dt: 0.05,
  tauMin: 1.0,
  tauMax: 5.0,
  biasScale: 0.5,
};

// ---------------------------------------------------------------------------
// Seeded PRNG (mulberry32)
// ---------------------------------------------------------------------------

export function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function clip(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// ---------------------------------------------------------------------------
// NeuralNetwork
// ---------------------------------------------------------------------------

export class NeuralNetwork {
  readonly numReadouts = DEFAULT_NUM_READOUTS;
  readonly nOutputsPerReadout = DEFAULT_N_OUTPUTS_PER_READOUT;
  readonly numOutputs = DEFAULT_NUM_READOUTS * DEFAULT_N_OUTPUTS_PER_READOUT;

  seed: number;
  params: NetworkParams;

  // Base random arrays — regenerated only on reseed
  readonly baseWeights: Float64Array;
  readonly sparsityRank: Float64Array;
  readonly baseModuleRank: Float64Array;
  readonly baseThresholds: Float64Array;
  readonly baseRefractionVar: Float64Array;
  readonly baseOutputMagnitude: Float64Array;
  readonly outputSparsityRank: Float64Array;
  readonly baseTauRank: Float64Array;
  readonly baseBias: Float64Array;

  // Derived — recomputed when params change
  readonly effectiveWeights: Float64Array;
  readonly moduleAssignments: Int32Array;
  readonly outputWeights: Float64Array;
  readonly thresholds: Float64Array;
  readonly thresholdsCurrent: Float64Array;
  readonly refractionPeriods: Int32Array;
  readonly taus: Float64Array;
  readonly biases: Float64Array;

  // Simulation state
  readonly activations: Float64Array;
  readonly firing: Uint8Array;
  readonly outputs: Float64Array;
  readonly refractoryCounters: Int32Array;

  get numNeurons(): number {
    return this.params.numNeurons;
  }

  constructor(seed: number, params?: Partial<NetworkParams>) {
    this.seed = seed;
    this.params = { ...DEFAULT_PARAMS, ...params };

    const M = MAX_NEURONS;
    const K = this.numOutputs;

    this.baseWeights = new Float64Array(M * M);
    this.sparsityRank = new Float64Array(M * M);
    this.baseModuleRank = new Float64Array(M);
    this.baseThresholds = new Float64Array(M);
    this.baseRefractionVar = new Float64Array(M);
    this.baseOutputMagnitude = new Float64Array(M * K);
    this.outputSparsityRank = new Float64Array(M * K);
    this.baseTauRank = new Float64Array(M);
    this.baseBias = new Float64Array(M);

    this.effectiveWeights = new Float64Array(M * M);
    this.moduleAssignments = new Int32Array(M);
    this.outputWeights = new Float64Array(M * K);
    this.thresholds = new Float64Array(M);
    this.thresholdsCurrent = new Float64Array(M);
    this.refractionPeriods = new Int32Array(M);
    this.taus = new Float64Array(M);
    this.biases = new Float64Array(M);

    this.activations = new Float64Array(M);
    this.firing = new Uint8Array(M);
    this.outputs = new Float64Array(K);
    this.refractoryCounters = new Int32Array(M);

    this.randomizeWeights();
    this.recomputeDerived();
    this.kickstart();
  }

  // -- Seed management -------------------------------------------------------

  reseed(newSeed: number): void {
    this.seed = newSeed;
    this.randomizeWeights();
    this.recomputeDerived();
    this.resetState();
    this.kickstart();
  }

  randomizeWeights(): void {
    const rng = mulberry32(this.seed);
    const M = MAX_NEURONS;
    const K = this.numOutputs;
    const nPer = this.nOutputsPerReadout;
    const nRead = this.numReadouts;

    for (let i = 0; i < M * M; i++) {
      const u1 = rng();
      const u2 = rng();
      this.baseWeights[i] = clip(
        Math.sqrt(-2 * Math.log(u1 || 1e-30)) * Math.cos(2 * Math.PI * u2),
        -1,
        1,
      );
    }
    for (let i = 0; i < M * M; i++) this.sparsityRank[i] = rng();
    for (let i = 0; i < M; i++) this.baseModuleRank[i] = rng();
    for (let i = 0; i < M; i++) this.baseThresholds[i] = rng();
    for (let i = 0; i < M; i++) this.baseRefractionVar[i] = rng();
    for (let i = 0; i < M * K; i++) this.baseOutputMagnitude[i] = rng();
    for (let i = 0; i < M * K; i++) this.outputSparsityRank[i] = rng();
    for (let i = 0; i < M; i++) this.baseTauRank[i] = rng();
    for (let i = 0; i < M; i++) {
      const u1 = rng();
      const u2 = rng();
      this.baseBias[i] = clip(
        Math.sqrt(-2 * Math.log(u1 || 1e-30)) * Math.cos(2 * Math.PI * u2),
        -1,
        1,
      );
    }
  }

  // -- Derived recomputation -------------------------------------------------

  recomputeDerived(): void {
    this.recomputeModules();
    this.recomputeEffectiveWeights();
    this.recomputeOutputWeights();
    this.recomputeThresholds();
    this.recomputeRefractionPeriods();
    this.recomputeTaus();
    this.recomputeBiases();
  }

  private recomputeModules(): void {
    const nm = this.params.numModules;
    for (let i = 0; i < MAX_NEURONS; i++) {
      this.moduleAssignments[i] =
        nm <= 1 ? 0 : Math.min(Math.floor(this.baseModuleRank[i] * nm), nm - 1);
    }
  }

  private recomputeEffectiveWeights(): void {
    const N = this.numNeurons;
    const M = MAX_NEURONS;
    const { sparsity, weightScale, numModules, interModuleFactor } = this.params;
    const interSparsity = sparsity * interModuleFactor;

    this.effectiveWeights.fill(0);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const idx = i * M + j;
        const same =
          numModules <= 1 ||
          this.moduleAssignments[i] === this.moduleAssignments[j];
        if (this.sparsityRank[idx] < (same ? sparsity : interSparsity)) {
          this.effectiveWeights[idx] = this.baseWeights[idx] * weightScale;
        }
      }
    }
  }

  private recomputeOutputWeights(): void {
    const M = MAX_NEURONS;
    const K = this.numOutputs;
    const nPer = this.nOutputsPerReadout;
    const nRead = this.numReadouts;

    this.outputWeights.fill(0);
    for (let i = 0; i < M; i++) {
      const readout = this.moduleAssignments[i] % nRead;
      const colStart = readout * nPer;
      const colEnd = colStart + nPer;
      for (let k = colStart; k < colEnd; k++) {
        const idx = i * K + k;
        if (this.outputSparsityRank[idx] < OUTPUT_SPARSITY) {
          this.outputWeights[idx] = this.baseOutputMagnitude[idx] * OUTPUT_SCALE;
        }
      }
    }
  }

  private recomputeThresholds(): void {
    for (let i = 0; i < MAX_NEURONS; i++) {
      this.thresholds[i] = this.baseThresholds[i];
      this.thresholdsCurrent[i] = this.baseThresholds[i];
    }
  }

  private recomputeRefractionPeriods(): void {
    const { refractionPeriod, refractionVariation } = this.params;
    for (let i = 0; i < MAX_NEURONS; i++) {
      this.refractionPeriods[i] =
        refractionPeriod + Math.floor(this.baseRefractionVar[i] * refractionVariation);
    }
  }

  private recomputeTaus(): void {
    const { tauMin, tauMax } = this.params;
    for (let i = 0; i < MAX_NEURONS; i++) {
      this.taus[i] = tauMin + this.baseTauRank[i] * (tauMax - tauMin);
    }
  }

  private recomputeBiases(): void {
    const { biasScale } = this.params;
    for (let i = 0; i < MAX_NEURONS; i++) {
      this.biases[i] = this.baseBias[i] * biasScale;
    }
  }

  // -- State management ------------------------------------------------------

  resetState(): void {
    this.activations.fill(0);
    this.firing.fill(0);
    this.outputs.fill(0);
    this.refractoryCounters.fill(0);
  }

  kickstart(): void {
    if (this.params.mode === 'ctrnn') {
      this.kickstartCTRNN();
    } else if (this.params.numModules > 1) {
      this.manualActivateMostWeightedPerModule(1.0);
    } else {
      this.manualActivateMostWeighted(1.0);
    }
  }

  private kickstartCTRNN(): void {
    const N = this.numNeurons;
    const rng = mulberry32(this.seed + 777);
    for (let i = 0; i < N; i++) {
      this.activations[i] = (rng() - 0.5) * 0.2;
    }
  }

  // -- Live parameter updates ------------------------------------------------

  updateParams(
    changes: Partial<NetworkParams>,
  ): { weightsChanged: boolean; numNeuronsChanged: boolean } {
    const oldN = this.numNeurons;
    Object.assign(this.params, changes);

    const weightsChanged =
      "numNeurons" in changes ||
      "numModules" in changes ||
      "interModuleFactor" in changes ||
      "sparsity" in changes ||
      "weightScale" in changes;

    if ("numModules" in changes || "numNeurons" in changes) {
      this.recomputeModules();
      this.recomputeOutputWeights();
    }
    if (weightsChanged) {
      this.recomputeEffectiveWeights();
    }
    if ("refractionPeriod" in changes || "refractionVariation" in changes) {
      this.recomputeRefractionPeriods();
    }
    if ("tauMin" in changes || "tauMax" in changes) {
      this.recomputeTaus();
    }
    if ("biasScale" in changes) {
      this.recomputeBiases();
    }

    return { weightsChanged, numNeuronsChanged: this.numNeurons !== oldN };
  }

  // -- Tick ------------------------------------------------------------------

  tick(step: number): void {
    if (this.params.mode === 'ctrnn') {
      this.tickCTRNN();
    } else {
      this.tickSpiking(step);
    }
  }

  private tickSpiking(_step: number): void {
    const N = this.numNeurons;
    const M = MAX_NEURONS;

    // 1. Incoming activation: incoming[j] = Σ_i ew[i,j] * firing[i]
    const incoming = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      if (this.firing[i] === 0) continue;
      const base = i * M;
      for (let j = 0; j < N; j++) {
        incoming[j] += this.effectiveWeights[base + j];
      }
    }

    // 2. Update activations (gated by refractory)
    const newAct = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      newAct[i] =
        this.refractoryCounters[i] === 0
          ? clip(this.activations[i] + incoming[i], 0, 1)
          : this.activations[i];
    }

    // 3. Firing
    const newFiring = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      newFiring[i] =
        this.refractoryCounters[i] === 0 &&
        newAct[i] >= this.thresholdsCurrent[i]
          ? 1
          : 0;
    }

    // 4. New refractory counters
    const newRefrac = new Int32Array(N);
    for (let i = 0; i < N; i++) {
      newRefrac[i] = newFiring[i]
        ? this.refractionPeriods[i]
        : this.refractoryCounters[i];
    }

    // 5. Refractory leak (applied based on OLD refractory state)
    const refLeak = this.params.refractionLeak;
    for (let i = 0; i < N; i++) {
      if (this.refractoryCounters[i] > 0) newAct[i] *= refLeak;
    }

    // 6. Outputs (decay + new firing contributions)
    const K = this.numOutputs;
    const outDecay = this.params.outputDecay;
    for (let i = 0; i < K; i++) this.outputs[i] *= outDecay;
    for (let i = 0; i < N; i++) {
      if (newFiring[i] === 0) continue;
      const base = i * K;
      for (let k = 0; k < K; k++) this.outputs[k] += this.outputWeights[base + k];
    }
    for (let i = 0; i < K; i++) this.outputs[i] = clip(this.outputs[i], 0, 1);

    // 7. Activation leak
    const aLeak = this.params.activationLeak;
    for (let i = 0; i < N; i++) newAct[i] *= aLeak;

    // 8. Decrement refractory
    for (let i = 0; i < N; i++) newRefrac[i] = Math.max(0, newRefrac[i] - 1);

    // 9. Commit
    for (let i = 0; i < N; i++) {
      this.activations[i] = newAct[i];
      this.firing[i] = newFiring[i];
      this.refractoryCounters[i] = newRefrac[i];
    }
  }

  private tickCTRNN(): void {
    const N = this.numNeurons;
    const M = MAX_NEURONS;
    const dt = this.params.dt;

    for (let i = 0; i < N; i++) {
      let input = 0;
      const base = i * M;
      for (let j = 0; j < N; j++) {
        input += this.effectiveWeights[base + j]
               * Math.tanh(this.activations[j] + this.biases[j]);
      }
      this.activations[i] += (dt / this.taus[i])
                            * (-this.activations[i] + input);
    }

    const K = this.numOutputs;
    for (let k = 0; k < K; k++) this.outputs[k] = 0;
    for (let i = 0; i < N; i++) {
      const a = Math.tanh(this.activations[i]);
      const base = i * K;
      for (let k = 0; k < K; k++) {
        this.outputs[k] += this.outputWeights[base + k] * a;
      }
    }
    for (let k = 0; k < K; k++) {
      this.outputs[k] = (Math.tanh(this.outputs[k]) + 1) / 2;
    }
  }

  // -- Manual activation -----------------------------------------------------

  manualActivate(idx: number, value: number): void {
    if (this.params.mode === 'ctrnn') {
      this.activations[idx] += value;
    } else {
      this.activations[idx] = clip(this.activations[idx] + value, 0, 1);
    }
  }

  manualActivateMostWeighted(value: number): number {
    const N = this.numNeurons;
    const M = MAX_NEURONS;
    let bestIdx = 0;
    let bestSum = -Infinity;
    for (let i = 0; i < N; i++) {
      let sum = 0;
      const base = i * M;
      for (let j = 0; j < N; j++) sum += this.effectiveWeights[base + j];
      if (sum > bestSum) {
        bestSum = sum;
        bestIdx = i;
      }
    }
    this.manualActivate(bestIdx, value);
    return bestIdx;
  }

  manualActivateMostWeightedPerModule(value: number): number[] {
    const N = this.numNeurons;
    const M = MAX_NEURONS;

    const totals = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let sum = 0;
      const base = i * M;
      for (let j = 0; j < N; j++) sum += this.effectiveWeights[base + j];
      totals[i] = sum;
    }

    const activated: number[] = [];
    for (let m = 0; m < this.params.numModules; m++) {
      let bestIdx = -1;
      let bestW = -Infinity;
      for (let i = 0; i < N; i++) {
        if (this.moduleAssignments[i] === m && totals[i] > bestW) {
          bestW = totals[i];
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

  connectionCount(): number {
    const N = this.numNeurons;
    const M = MAX_NEURONS;
    let count = 0;
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        if (this.effectiveWeights[i * M + j] !== 0) count++;
    return count;
  }
}
