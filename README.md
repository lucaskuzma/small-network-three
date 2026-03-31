# small-network-three

3D neural network visualization built with Three.js. Supports two network modes:

- **Spiking** — threshold-and-fire neurons with refractory periods and activation leak
- **CTRNN** — continuous-time recurrent neural network with tau-based dynamics and bipolar activations

## Running

```bash
npm install
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`).

## Building

```bash
npm run build
npm run preview
```

## Controls

The GUI panel exposes simulation, network, and visual parameters. The **Mode** dropdown in the Network folder switches between spiking and CTRNN — each mode shows only its relevant controls.

Orbit the camera by dragging. Scroll to zoom.
