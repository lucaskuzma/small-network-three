// Per-neuron oscillator AudioWorkletProcessor.
// Each neuron is a phase-accumulator oscillator:
//   pitch = chromatic note from Fiedler topology
//   amplitude = activation
//   timbre = sine↔saw crossfade driven by smoothed activation
//   pan = screen-X position

const MAX_OSC = 512;

// Precomputed sine wavetable with guard point for linear interpolation
const SINE_N = 4096;
const SINE_TABLE = new Float32Array(SINE_N + 1);
for (let i = 0; i <= SINE_N; i++) {
  SINE_TABLE[i] = Math.sin((2 * Math.PI * i) / SINE_N);
}

function fastSin(phase) {
  const idx = phase * SINE_N;
  const i0 = idx | 0;
  const frac = idx - i0;
  return SINE_TABLE[i0] + frac * (SINE_TABLE[i0 + 1] - SINE_TABLE[i0]);
}

class NetworkOscillatorProcessor extends AudioWorkletProcessor {

  static get parameterDescriptors() {
    return [
      { name: "masterVolume", defaultValue: 0.8, minValue: 0, maxValue: 1, automationRate: "k-rate" },
      { name: "actTimbre",    defaultValue: 0,   minValue: 0, maxValue: 1, automationRate: "k-rate" },
    ];
  }

  constructor() {
    super();
    this.oscCount = 0;
    this.phases = new Float64Array(MAX_OSC);
    this.phaseInc = new Float32Array(MAX_OSC);
    this.amplitudes = new Float32Array(MAX_OSC);
    this.panL = new Float32Array(MAX_OSC);
    this.panR = new Float32Array(MAX_OSC);
    this.recentActivation = new Float32Array(MAX_OSC);

    // ~3s time constant applied once per quantum
    this.smoothCoeff = 1 - Math.exp(-128 / (sampleRate * 3));

    this.panL.fill(Math.SQRT1_2);
    this.panR.fill(Math.SQRT1_2);

    this.port.onmessage = (e) => {
      switch (e.data.type) {
        case "configure": {
          this.oscCount = Math.min(e.data.oscCount, MAX_OSC);
          const notes = e.data.noteNumbers;
          for (let i = 0; i < this.oscCount; i++) {
            this.phaseInc[i] = 440 * Math.pow(2, (notes[i] - 69) / 12) / sampleRate;
          }
          for (let i = 0; i < this.oscCount; i++) {
            this.phases[i] = Math.random();
          }
          this.recentActivation.fill(0);
          break;
        }
        case "updateVolumes": {
          const vols = e.data.volumes;
          for (let i = 0; i < vols.length && i < MAX_OSC; i++) {
            this.amplitudes[i] = vols[i];
          }
          break;
        }
        case "updatePan": {
          const pan = e.data.panPositions;
          const halfPi = Math.PI * 0.5;
          for (let i = 0; i < pan.length && i < MAX_OSC; i++) {
            const p = pan[i] * halfPi;
            this.panL[i] = Math.cos(p);
            this.panR[i] = Math.sin(p);
          }
          break;
        }
      }
    };
  }

  _p(params, name) {
    const arr = params[name];
    return arr ? arr[0] : 0;
  }

  process(_inputs, outputs, parameters) {
    const output = outputs[0];
    if (!output) return true;
    const outL = output[0];
    const outR = output[1] || output[0];
    if (!outL || !outR) return true;

    const n = this.oscCount;
    if (n <= 0) {
      for (let s = 0; s < outL.length; s++) {
        outL[s] = 0;
        outR[s] = 0;
      }
      return true;
    }

    const masterVol = this._p(parameters, "masterVolume");
    const actTimbre = this._p(parameters, "actTimbre");
    const sc = this.smoothCoeff;
    const blockLen = outL.length;
    const amps = this.amplitudes;
    const pL = this.panL;
    const pR = this.panR;

    // Per-quantum: EMA smoothing, timbre mix, count active oscillators
    const tmix = new Float32Array(n);
    let activeCount = 0;
    for (let i = 0; i < n; i++) {
      if (amps[i] < 0.001) continue;
      activeCount++;
      this.recentActivation[i] += sc * (amps[i] - this.recentActivation[i]);
      const m = this.recentActivation[i] * actTimbre;
      tmix[i] = m < 1 ? m : 1;
    }

    const norm = activeCount > 0 ? masterVol / Math.sqrt(activeCount) : 0;

    const phases = this.phases;
    const inc = this.phaseInc;

    for (let s = 0; s < blockLen; s++) {
      let sumL = 0;
      let sumR = 0;

      for (let i = 0; i < n; i++) {
        const amp = amps[i];
        if (amp < 0.001) continue;

        let phase = phases[i];
        const sine = fastSin(phase);
        const saw = 2 * phase - 1;
        const t = tmix[i];
        const sample = sine + t * (saw - sine);

        const out = sample * amp;
        sumL += out * pL[i];
        sumR += out * pR[i];

        phase += inc[i];
        if (phase >= 1) phase -= 1;
        phases[i] = phase;
      }

      outL[s] = sumL * norm;
      outR[s] = sumR * norm;
    }

    return true;
  }
}

registerProcessor("network-oscillator-processor", NetworkOscillatorProcessor);
