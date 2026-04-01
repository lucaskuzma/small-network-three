// Network granular synthesis AudioWorkletProcessor.
// Adapted from open-granular-js. Each grain maps to a neuron:
// per-grain buffer position (from spatial distance) and volume (from activation).

const MAX_GRAINS = 512;
const MAX_SIZE = 44100;

// ─── Grain ──────────────────────────────────────────────────────────────────

class Grain {
  constructor() {
    this.smoothOffset = 0;
    this.length = 4410;
    this.index = 0;
    this.delay = 0;
    this.ramp = 735;
    this._sampleL = 0;
    this._sampleR = 0;
  }

  sample(dataL, dataR, bufferLength) {
    const offset = Math.floor(this.smoothOffset);

    if (offset < this.delay) {
      this._advance();
      return;
    }

    const attackIndex = offset - this.delay;
    const audibleLength = this.length - this.delay;

    let envelope = 1;
    if (this.ramp > 0) {
      const attackFrac = attackIndex / this.ramp;
      const decayFrac = (audibleLength - attackIndex) / this.ramp;
      if (attackFrac < 1) {
        envelope = attackFrac;
      } else if (decayFrac < 1) {
        envelope = decayFrac > 0 ? decayFrac : 0;
      }
    }

    const readIndex = (this.index + offset) % bufferLength;
    this._sampleL = dataL[readIndex] * envelope;
    this._sampleR = dataR[readIndex] * envelope;

    this._advance();
  }

  _advance() {
    const cycleLen = this.length + this.delay;
    if (cycleLen <= 0) {
      this.smoothOffset = 0;
      return;
    }
    this.smoothOffset = (this.smoothOffset + 1) % cycleLen;
  }

  get atCycleBoundary() {
    return Math.floor(this.smoothOffset) === 0;
  }

  resample(bufferLength, bufferIndex, grainLength, grainDelay, grainRamp) {
    this.index = Math.floor(bufferIndex) % bufferLength;
    if (this.index < 0) this.index += bufferLength;

    this.length = Math.max(1, Math.floor(grainLength));
    this.delay = Math.max(0, Math.floor(grainDelay));

    const maxRamp = Math.floor(this.length / 2);
    this.ramp = Math.min(Math.floor(grainRamp), maxRamp);
  }
}

// ─── Processor ──────────────────────────────────────────────────────────────

class NetworkGranularProcessor extends AudioWorkletProcessor {

  static get parameterDescriptors() {
    return [
      { name: "size",         defaultValue: 0.1, minValue: 0, maxValue: 1, automationRate: "k-rate" },
      { name: "spread",       defaultValue: 0,   minValue: 0, maxValue: 1, automationRate: "k-rate" },
      { name: "ramp",         defaultValue: 0.5, minValue: 0, maxValue: 1, automationRate: "k-rate" },
      { name: "masterVolume", defaultValue: 0.8, minValue: 0, maxValue: 1, automationRate: "k-rate" },
    ];
  }

  constructor() {
    super();
    this.grains = [];
    for (let i = 0; i < MAX_GRAINS; i++) {
      this.grains.push(new Grain());
    }
    this.grainCount = 0;
    this.dataL = new Float32Array(0);
    this.dataR = new Float32Array(0);
    this.bufferLength = 0;

    // Per-grain arrays
    this.positions = new Float32Array(MAX_GRAINS);
    this.volumes = new Float32Array(MAX_GRAINS);
    this.smoothVolumes = new Float32Array(MAX_GRAINS);

    // One-pole low-pass coefficient (~5ms time constant)
    this.smoothCoeff = 1 - Math.exp(-1 / (sampleRate * 0.005));

    this.port.onmessage = (e) => {
      switch (e.data.type) {
        case "loadBuffer": {
          const { channelData } = e.data;
          this.dataL = channelData[0] || new Float32Array(0);
          this.dataR = channelData[1] || channelData[0] || new Float32Array(0);
          this.bufferLength = this.dataL.length;
          break;
        }
        case "configure": {
          this.grainCount = Math.min(e.data.grainCount, MAX_GRAINS);
          const pos = e.data.positions;
          for (let i = 0; i < pos.length && i < MAX_GRAINS; i++) {
            this.positions[i] = pos[i];
          }
          // Randomize initial phase so grains don't all start aligned
          for (let i = 0; i < this.grainCount; i++) {
            this.grains[i].smoothOffset = Math.random() * this.grains[i].length;
          }
          break;
        }
        case "updateVolumes": {
          const vols = e.data.volumes;
          for (let i = 0; i < vols.length && i < MAX_GRAINS; i++) {
            this.volumes[i] = vols[i];
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
    if (this.bufferLength === 0) return true;

    const size = this._p(parameters, "size");
    const spread = this._p(parameters, "spread");
    const ramp = this._p(parameters, "ramp");
    const masterVol = this._p(parameters, "masterVolume");

    const bufLen = this.bufferLength;
    const maxGrainLength = Math.min(MAX_SIZE, bufLen);
    const baseLength = Math.max(441, Math.floor(size * maxGrainLength));
    const baseDelay = Math.floor(spread * MAX_SIZE);

    const n = this.grainCount;
    if (n <= 0) {
      for (let s = 0; s < outL.length; s++) {
        outL[s] = 0;
        outR[s] = 0;
      }
      return true;
    }

    // Count active grains for normalization
    let activeCount = 0;
    for (let g = 0; g < n; g++) {
      if (this.volumes[g] > 0.001) activeCount++;
    }
    const amp = activeCount > 0 ? 1 / Math.sqrt(activeCount) : 0;

    const coeff = this.smoothCoeff;

    for (let s = 0; s < outL.length; s++) {
      let sumL = 0;
      let sumR = 0;

      for (let g = 0; g < n; g++) {
        const grain = this.grains[g];

        this.smoothVolumes[g] += coeff * (this.volumes[g] - this.smoothVolumes[g]);
        const vol = this.smoothVolumes[g];

        if (grain.atCycleBoundary) {
          const bufferIndex = this.positions[g] * (bufLen - 1);
          const grainLength = baseLength + g;
          const maxRamp = Math.floor(grainLength / 2);
          const grainRamp = Math.floor(ramp * maxRamp);
          grain.resample(bufLen, bufferIndex, grainLength, baseDelay, grainRamp);
        }

        grain._sampleL = 0;
        grain._sampleR = 0;
        grain.sample(this.dataL, this.dataR, bufLen);
        sumL += grain._sampleL * vol;
        sumR += grain._sampleR * vol;
      }

      outL[s] = sumL * amp * masterVol;
      outR[s] = sumR * amp * masterVol;
    }

    return true;
  }
}

registerProcessor("network-granular-processor", NetworkGranularProcessor);
