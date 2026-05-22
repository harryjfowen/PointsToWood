import * as THREE from "https://unpkg.com/three@0.163.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.163.0/examples/jsm/controls/OrbitControls.js";

const MAX_RENDER_POINTS = 240_000;
const DEFAULT_COMPACT_META_CANDIDATES = ["./data/pointcloud.meta.json"];
const DEFAULT_PLY_SOURCE_CANDIDATES = [
  "./data/uk01_lw_pl_3-p2w.ply",
  "./data/uk01_lw_pl_3_p2w.ply",
  "./data/pointcloud.ply",
  "../uk01_lw_pl_3-p2w.ply",
  "../uk01_lw_pl_3_p2w.ply",
];

const PLY_TYPE_BYTES = {
  char: 1,
  int8: 1,
  uchar: 1,
  uint8: 1,
  short: 2,
  int16: 2,
  ushort: 2,
  uint16: 2,
  int: 4,
  int32: 4,
  uint: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8,
};

const statusEl = document.getElementById("status");
const sourceEl = document.getElementById("source-file");
const sampleCountEl = document.getElementById("sample-count");
const splitCountEl = document.getElementById("split-count");

const thresholdInput = document.getElementById("threshold");
const thresholdValueEl = document.getElementById("threshold-value");
const softnessInput = document.getElementById("softness");
const softnessValueEl = document.getElementById("softness-value");
const pointSizeInput = document.getElementById("point-size");
const pointSizeValueEl = document.getElementById("point-size-value");

const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: true,
});
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(56, 1, 0.01, 100);
camera.position.set(1.5, 1.3, 1.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(0, 0, 0);

const material = new THREE.ShaderMaterial({
  uniforms: {
    uLeafColor: { value: new THREE.Color("#68d676") },
    uWoodColor: { value: new THREE.Color("#d48944") },
    uThreshold: { value: Number(thresholdInput.value) },
    uSoftness: { value: Number(softnessInput.value) },
    uPointSize: { value: Number(pointSizeInput.value) },
  },
  vertexShader: `
    attribute float aWood;
    varying float vWood;
    uniform float uPointSize;
    void main() {
      vWood = aWood;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      float depthScale = clamp(220.0 / -mvPosition.z, 0.55, 6.0);
      gl_PointSize = uPointSize * depthScale;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    precision highp float;
    varying float vWood;
    uniform vec3 uLeafColor;
    uniform vec3 uWoodColor;
    uniform float uThreshold;
    uniform float uSoftness;
    void main() {
      vec2 centered = gl_PointCoord - vec2(0.5);
      if (dot(centered, centered) > 0.25) discard;
      float mixValue = smoothstep(uThreshold - uSoftness, uThreshold + uSoftness, vWood);
      vec3 color = mix(uLeafColor, uWoodColor, mixValue);
      gl_FragColor = vec4(color, 0.96);
    }
  `,
  transparent: true,
  depthWrite: false,
});

let thresholdWoodCumulative = null;
let loadedCount = 0;

function onResize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

window.addEventListener("resize", onResize);
onResize();

thresholdInput.addEventListener("input", () => {
  const value = Number(thresholdInput.value);
  thresholdValueEl.textContent = value.toFixed(2);
  material.uniforms.uThreshold.value = value;
  updateSplitLabel();
});

softnessInput.addEventListener("input", () => {
  const value = Number(softnessInput.value);
  softnessValueEl.textContent = value.toFixed(3);
  material.uniforms.uSoftness.value = value;
});

pointSizeInput.addEventListener("input", () => {
  const value = Number(pointSizeInput.value);
  pointSizeValueEl.textContent = value.toFixed(1);
  material.uniforms.uPointSize.value = value;
});

function buildWoodCumulative(bins) {
  const cumulative = new Uint32Array(bins.length + 1);
  for (let i = bins.length - 1; i >= 0; i -= 1) {
    cumulative[i] = cumulative[i + 1] + bins[i];
  }
  return cumulative;
}

function updateSplitLabel() {
  if (!thresholdWoodCumulative || loadedCount === 0) {
    splitCountEl.textContent = "-";
    return;
  }
  const threshold = Number(thresholdInput.value);
  const index = Math.max(0, Math.min(100, Math.round(threshold * 100)));
  const wood = thresholdWoodCumulative[index];
  const leaf = loadedCount - wood;
  splitCountEl.textContent = `${leaf.toLocaleString()} leaf / ${wood.toLocaleString()} wood`;
}

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

async function fetchFirstAvailableCompact() {
  const params = new URLSearchParams(window.location.search);
  const metaParam = params.get("meta");
  const candidates = [];
  if (metaParam) {
    candidates.push(metaParam);
    candidates.push(`./data/${metaParam}`);
  }
  candidates.push(...DEFAULT_COMPACT_META_CANDIDATES);

  let lastError = null;
  for (const metaSource of [...new Set(candidates)]) {
    try {
      statusEl.textContent = `Trying ${metaSource} ...`;
      const metaResponse = await fetch(metaSource);
      if (!metaResponse.ok) {
        throw new Error(`${metaResponse.status} ${metaResponse.statusText}`);
      }
      const meta = await metaResponse.json();
      const binaryName = meta.binary || "pointcloud.bin";
      const binarySource = new URL(binaryName, new URL(metaSource, window.location.href)).toString();
      const binResponse = await fetch(binarySource);
      if (!binResponse.ok) {
        throw new Error(`${binResponse.status} ${binResponse.statusText}`);
      }
      const arrayBuffer = await binResponse.arrayBuffer();
      return { arrayBuffer, meta, source: metaSource };
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`Unable to load compact data (${lastError?.message || "unknown error"})`);
}

async function fetchFirstAvailablePly() {
  const params = new URLSearchParams(window.location.search);
  const fileParam = params.get("file");
  const candidates = [];
  if (fileParam) {
    candidates.push(`./data/${fileParam}`);
    candidates.push(fileParam);
  }
  candidates.push(...DEFAULT_PLY_SOURCE_CANDIDATES);

  let lastError = null;
  for (const source of [...new Set(candidates)]) {
    try {
      statusEl.textContent = `Trying ${source} ...`;
      const response = await fetch(source);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      const arrayBuffer = await response.arrayBuffer();
      return { arrayBuffer, source };
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`Unable to load PLY file (${lastError?.message || "unknown error"})`);
}

function parseHeader(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const headerScanLength = Math.min(bytes.length, 1024 * 1024);
  const headerText = new TextDecoder("ascii").decode(bytes.subarray(0, headerScanLength));
  const endMatch = headerText.match(/end_header\r?\n/);

  if (!endMatch) {
    throw new Error("PLY header not found");
  }

  const headerBytes = endMatch.index + endMatch[0].length;
  const lines = headerText.slice(0, headerBytes).split(/\r?\n/);
  let format = "";
  let vertexCount = 0;
  let inVertexElement = false;
  const properties = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    if (line.startsWith("format ")) {
      [, format] = line.split(/\s+/);
      continue;
    }

    if (line.startsWith("element ")) {
      const [, elementName, countText] = line.split(/\s+/);
      inVertexElement = elementName === "vertex";
      if (inVertexElement) {
        vertexCount = Number(countText);
      }
      continue;
    }

    if (inVertexElement && line.startsWith("property ")) {
      const tokens = line.split(/\s+/);
      if (tokens[1] === "list") {
        throw new Error("List vertex properties are not supported");
      }
      const type = tokens[1];
      const name = tokens[2];
      const byteSize = PLY_TYPE_BYTES[type];
      if (!byteSize) {
        throw new Error(`Unsupported PLY property type: ${type}`);
      }
      properties.push({ name, type, byteSize });
    }
  }

  if (!format || !vertexCount || properties.length === 0) {
    throw new Error("Invalid PLY metadata");
  }

  return { format, vertexCount, properties, headerBytes };
}

function readScalar(view, offset, type) {
  switch (type) {
    case "char":
    case "int8":
      return view.getInt8(offset);
    case "uchar":
    case "uint8":
      return view.getUint8(offset);
    case "short":
    case "int16":
      return view.getInt16(offset, true);
    case "ushort":
    case "uint16":
      return view.getUint16(offset, true);
    case "int":
    case "int32":
      return view.getInt32(offset, true);
    case "uint":
    case "uint32":
      return view.getUint32(offset, true);
    case "float":
    case "float32":
      return view.getFloat32(offset, true);
    case "double":
    case "float64":
      return view.getFloat64(offset, true);
    default:
      throw new Error(`Cannot read PLY type: ${type}`);
  }
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function parseCompactBinary(arrayBuffer, meta = {}) {
  if (meta.format && meta.format !== "xyzp_f32_le_v1") {
    throw new Error(`Unsupported compact format: ${meta.format}`);
  }

  const floats = new Float32Array(arrayBuffer);
  if (floats.length % 4 !== 0) {
    throw new Error("Compact binary payload must be interleaved x,y,z,prediction float32");
  }

  const availableCount = floats.length / 4;
  const expectedCount = Number(meta.pointCount || availableCount);
  const pointCount = Math.min(availableCount, expectedCount);

  const positions = new Float32Array(pointCount * 3);
  const wood = new Float32Array(pointCount);
  const bins = new Uint32Array(101);

  for (let i = 0; i < pointCount; i += 1) {
    const src = i * 4;
    const dst = i * 3;
    const pred = clamp01(floats[src + 3]);
    positions[dst] = floats[src];
    positions[dst + 1] = floats[src + 1];
    positions[dst + 2] = floats[src + 2];
    wood[i] = pred;
    const binIndex = Math.max(0, Math.min(100, Math.round(pred * 100)));
    bins[binIndex] += 1;
  }

  return {
    positions,
    wood,
    bins,
    loadedCount: pointCount,
    originalCount: Number(meta.sourceVertexCount || pointCount),
    step: Number(meta.sampleStep || 1),
    sourceDetail: meta.selection
      ? `wood ${Number(meta.selection.keptWood || 0).toLocaleString()} / leaf ${Number(
          meta.selection.keptLeaf || 0
        ).toLocaleString()}`
      : "preprocessed",
  };
}

function parseBinaryPly(arrayBuffer) {
  const { format, vertexCount, properties, headerBytes } = parseHeader(arrayBuffer);
  if (format !== "binary_little_endian") {
    throw new Error(`Expected binary_little_endian PLY, got ${format}`);
  }

  const propertyOffsets = [];
  let stride = 0;
  for (const property of properties) {
    propertyOffsets.push(stride);
    stride += property.byteSize;
  }

  const getIndex = (name) => properties.findIndex((prop) => prop.name === name);
  const xIndex = getIndex("x");
  const yIndex = getIndex("y");
  const zIndex = getIndex("z");
  const predictionIndex = getIndex("prediction");

  if (xIndex < 0 || yIndex < 0 || zIndex < 0 || predictionIndex < 0) {
    throw new Error("Expected x,y,z,prediction properties in PLY");
  }

  const requiredBytes = headerBytes + stride * vertexCount;
  if (requiredBytes > arrayBuffer.byteLength) {
    throw new Error("PLY payload appears truncated");
  }

  const step = Math.max(1, Math.ceil(vertexCount / MAX_RENDER_POINTS));
  const sampleCount = Math.ceil(vertexCount / step);
  const positions = new Float32Array(sampleCount * 3);
  const wood = new Float32Array(sampleCount);
  const bins = new Uint32Array(101);

  const view = new DataView(arrayBuffer, headerBytes);

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let out = 0;

  for (let i = 0; i < vertexCount; i += step) {
    const rowOffset = i * stride;
    const x = readScalar(view, rowOffset + propertyOffsets[xIndex], properties[xIndex].type);
    const y = readScalar(view, rowOffset + propertyOffsets[yIndex], properties[yIndex].type);
    const z = readScalar(view, rowOffset + propertyOffsets[zIndex], properties[zIndex].type);
    const pred = clamp01(
      readScalar(
        view,
        rowOffset + propertyOffsets[predictionIndex],
        properties[predictionIndex].type
      )
    );

    const pos = out * 3;
    positions[pos] = x;
    positions[pos + 1] = y;
    positions[pos + 2] = z;
    wood[out] = pred;

    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;

    const binIndex = Math.max(0, Math.min(100, Math.round(pred * 100)));
    bins[binIndex] += 1;
    out += 1;
  }

  const centerX = (minX + maxX) * 0.5;
  const centerY = (minY + maxY) * 0.5;
  const centerZ = (minZ + maxZ) * 0.5;
  const scale = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;

  for (let i = 0; i < out; i += 1) {
    const pos = i * 3;
    positions[pos] = (positions[pos] - centerX) / scale;
    positions[pos + 1] = (positions[pos + 1] - centerY) / scale;
    positions[pos + 2] = (positions[pos + 2] - centerZ) / scale;
  }

  return {
    positions: positions.subarray(0, out * 3),
    wood: wood.subarray(0, out),
    bins,
    loadedCount: out,
    originalCount: vertexCount,
    stride,
    step,
  };
}

async function init() {
  try {
    let parsed;
    let sourceLabel = "";

    try {
      const compact = await fetchFirstAvailableCompact();
      statusEl.textContent = "Parsing compact point cloud...";
      parsed = parseCompactBinary(compact.arrayBuffer, compact.meta);
      sourceLabel = `${compact.source} | ${parsed.sourceDetail}`;
    } catch (compactError) {
      console.warn("Compact load failed, falling back to PLY:", compactError);
      const { arrayBuffer, source } = await fetchFirstAvailablePly();
      statusEl.textContent = "Parsing binary PLY...";
      parsed = parseBinaryPly(arrayBuffer);
      sourceLabel = `${source} | sampling 1/${parsed.step}`;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(parsed.positions, 3));
    geometry.setAttribute("aWood", new THREE.BufferAttribute(parsed.wood, 1));
    geometry.computeBoundingSphere();

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    loadedCount = parsed.loadedCount;
    thresholdWoodCumulative = buildWoodCumulative(parsed.bins);
    updateSplitLabel();

    sampleCountEl.textContent = parsed.loadedCount.toLocaleString();
    sourceEl.textContent = sourceLabel;
    statusEl.textContent = `Loaded ${parsed.loadedCount.toLocaleString()} points`;
    setTimeout(() => {
      statusEl.style.opacity = "0.2";
    }, 1800);
  } catch (error) {
    console.error(error);
    statusEl.textContent = `Load failed: ${error.message}`;
    sourceEl.textContent = "No PLY loaded";
  }
}

thresholdValueEl.textContent = Number(thresholdInput.value).toFixed(2);
softnessValueEl.textContent = Number(softnessInput.value).toFixed(3);
pointSizeValueEl.textContent = Number(pointSizeInput.value).toFixed(1);

animate();
init();
