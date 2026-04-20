import type { NeuralNetwork } from "./network.ts";

export interface PlaneReadoutParams {
  z: number;
  threshold: number;
  shape: number;
}

export interface Centroid {
  x: number;
  y: number;
  mag: number;
  valid: boolean;
}

export interface PlaneReadoutResult {
  pos: Centroid;
  neg: Centroid;
  isDipole: boolean;
}

export interface LayoutExtent {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

const EPS = 1e-6;

const _result: PlaneReadoutResult = {
  pos: { x: 0, y: 0, mag: 0, valid: false },
  neg: { x: 0, y: 0, mag: 0, valid: false },
  isDipole: false,
};

export function computePlaneReadout(
  net: NeuralNetwork,
  nodePositions: Float32Array,
  params: PlaneReadoutParams,
): PlaneReadoutResult {
  const N = net.numNeurons;
  const { z, threshold, shape } = params;
  const isCtrnn = net.params.mode === "ctrnn";
  const activations = net.activations;
  const usePow = shape !== 1;
  const invThr = 1 / threshold;

  let posSumW = 0, posSumX = 0, posSumY = 0;
  let negSumW = 0, negSumX = 0, negSumY = 0;

  for (let i = 0; i < N; i++) {
    const pz = nodePositions[i * 3 + 2];
    const dz = pz - z;
    const adz = dz < 0 ? -dz : dz;
    if (adz >= threshold) continue;

    const t = 1 - adz * invThr;
    const k = usePow ? Math.pow(t, shape) : t;

    const px = nodePositions[i * 3];
    const py = nodePositions[i * 3 + 1];

    if (isCtrnn) {
      const v = Math.tanh(activations[i]);
      if (v > 0) {
        const w = k * v;
        posSumW += w;
        posSumX += w * px;
        posSumY += w * py;
      } else if (v < 0) {
        const w = k * -v;
        negSumW += w;
        negSumX += w * px;
        negSumY += w * py;
      }
    } else {
      const a = activations[i];
      if (a > 0) {
        const w = k * a;
        posSumW += w;
        posSumX += w * px;
        posSumY += w * py;
      }
    }
  }

  _result.isDipole = isCtrnn;

  if (posSumW > EPS) {
    _result.pos.x = posSumX / posSumW;
    _result.pos.y = posSumY / posSumW;
    _result.pos.mag = posSumW;
    _result.pos.valid = true;
  } else {
    _result.pos.mag = 0;
    _result.pos.valid = false;
  }

  if (isCtrnn && negSumW > EPS) {
    _result.neg.x = negSumX / negSumW;
    _result.neg.y = negSumY / negSumW;
    _result.neg.mag = negSumW;
    _result.neg.valid = true;
  } else {
    _result.neg.mag = 0;
    _result.neg.valid = false;
  }

  return _result;
}

export function computeExtent(
  nodePositions: Float32Array,
  n: number,
): LayoutExtent {
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = nodePositions[i * 3];
    const y = nodePositions[i * 3 + 1];
    const z = nodePositions[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  if (!isFinite(minX)) {
    return { minX: -1, maxX: 1, minY: -1, maxY: 1, minZ: -1, maxZ: 1 };
  }
  return { minX, maxX, minY, maxY, minZ, maxZ };
}
