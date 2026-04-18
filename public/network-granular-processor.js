// Network granular synthesis AudioWorkletProcessor.
// Adapted from open-granular-js. Each grain maps to a neuron:
// per-grain buffer position (Fiedler topology), pan (screen X), volume (activation).

const MAX_GRAINS = 512;
const MAX_SIZE = 44100;

// ─── Grain ──────────────────────────────────────────────────────────────────

class Grain {
  constructor() {
    this.smoothOffset = 0;
    this.rate = 1.0;
    this.vol = 0;
    this.panL = 1;
    this.panR = 1;
    this.length = 4410;
    this.index = 0;
    this.ramp = 735;
    this._sampleL = 0;
    this._sampleR = 0;
  }

  sample(dataL, dataR, bufferLength) {
    const offset = Math.floor(this.smoothOffset);

    let envelope = 1;
    if (this.ramp > 0) {
      const attackFrac = offset / this.ramp;
      const decayFrac = (this.length - offset) / this.ramp;
      if (attackFrac < 1) {
        envelope = attackFrac;
      } else if (decayFrac < 1) {
        envelope = decayFrac > 0 ? decayFrac : 0;
      }
    }

    const readIndex = (this.index + offset) % bufferLength;
    const ev = envelope * this.vol;
    this._sampleL = dataL[readIndex] * ev * this.panL;
    this._sampleR = dataR[readIndex] * ev * this.panR;

    this._advance();
  }

  _advance() {
    if (this.length <= 0) {
      this.smoothOffset = 0;
      return;
    }
    this.smoothOffset = (this.smoothOffset + this.rate) % this.length;
  }

  get atCycleBoundary() {
    return Math.floor(this.smoothOffset) === 0;
  }

  updateParams(bufferLength, bufferIndex, grainLength, grainRamp, rate, vol, pan) {
    this.index = Math.floor(bufferIndex) % bufferLength;
    if (this.index < 0) this.index += bufferLength;

    this.length = Math.max(1, Math.floor(grainLength));

    const maxRamp = Math.floor(this.length / 2);
    this.ramp = Math.min(Math.floor(grainRamp), maxRamp);

    this.rate = rate;
    this.vol = vol;
    this.panL = Math.cos(pan * Math.PI * 0.5);
    this.panR = Math.sin(pan * Math.PI * 0.5);
  }
}

// ─── Processor ──────────────────────────────────────────────────────────────

class NetworkGranularProcessor extends AudioWorkletProcessor {

  static get parameterDescriptors() {
    return [
      { name: "size",         defaultValue: 0.1, minValue: 0, maxValue: 1, automationRate: "k-rate" },
      { name: "ramp",         defaultValue: 0.5, minValue: 0, maxValue: 1, automationRate: "k-rate" },
      { name: "masterVolume", defaultValue: 0.8, minValue: 0, maxValue: 1, automationRate: "k-rate" },
      { name: "pitchBias",       defaultValue: 0,   minValue: 0, maxValue: 2, automationRate: "k-rate" },
      { name: "activationOffset", defaultValue: 0.5, minValue: 0, maxValue: 1, automationRate: "k-rate" },
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
    this.positions = new Float32Array(MAX_GRAINS);     // Fiedler-based buffer positions [0..1]
    this.panPositions = new Float32Array(MAX_GRAINS);  // Screen-X stereo pan [0..1]
    this.volumes = new Float32Array(MAX_GRAINS);
    this.panPositions.fill(0.5);

    // Slow-smoothed activation for pitch mapping (~3s time constant)
    this.recentActivation = new Float32Array(MAX_GRAINS);
    this.pitchCoeff = 1 - Math.exp(-1 / (sampleRate * 3));

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
        case "updatePan": {
          const pan = e.data.panPositions;
          for (let i = 0; i < pan.length && i < MAX_GRAINS; i++) {
            this.panPositions[i] = pan[i];
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
    const ramp = this._p(parameters, "ramp");
    const masterVol = this._p(parameters, "masterVolume");
    const pitchBias = this._p(parameters, "pitchBias");
    const actOffset = this._p(parameters, "activationOffset");

    const bufLen = this.bufferLength;
    const maxGrainLength = Math.min(MAX_SIZE, bufLen);
    const baseLength = Math.max(441, Math.floor(size * maxGrainLength));

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

    const pCoeff = this.pitchCoeff;

    for (let s = 0; s < outL.length; s++) {
      let sumL = 0;
      let sumR = 0;

      for (let g = 0; g < n; g++) {
        const grain = this.grains[g];

        this.recentActivation[g] += pCoeff * (this.volumes[g] - this.recentActivation[g]);

        if (grain.atCycleBoundary) {
          const bufferIndex = (this.positions[g] + actOffset * this.volumes[g]) * (bufLen - 1);
          const grainLength = baseLength + g;
          const maxRamp = Math.floor(grainLength / 2);
          const grainRamp = Math.floor(ramp * maxRamp);
          const rate = 1 + pitchBias * this.recentActivation[g];
          grain.updateParams(bufLen, bufferIndex, grainLength, grainRamp, rate, this.volumes[g], this.panPositions[g]);
        }

        grain._sampleL = 0;
        grain._sampleR = 0;
        grain.sample(this.dataL, this.dataR, bufLen);
        sumL += grain._sampleL;
        sumR += grain._sampleR;
      }

      outL[s] = sumL * amp * masterVol;
      outR[s] = sumR * amp * masterVol;
    }

    return true;
  }
}

registerProcessor("network-granular-processor", NetworkGranularProcessor);
