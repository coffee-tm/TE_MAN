import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXTENSION_NAME = "TE.ManConcurrentRunningHighlight";
const STROKE_STYLE_KEY = "teManConcurrentRunning";
const OUTLINE_CLASS = "te-man-concurrent-running-outline";
const STYLE_ELEMENT_ID = "te-man-concurrent-running-highlight-style";

const runningDisplayNodeIdsByPrompt = new Map();
let globalRunningDisplayNodeIds = new Set();

function escapeNodeIdForSelector(nodeId) {
  if (globalThis.CSS?.escape) {
    return globalThis.CSS.escape(nodeId);
  }
  return String(nodeId).replace(/["\\]/g, "\\$&");
}

function ensureStyleElement() {
  if (document.getElementById(STYLE_ELEMENT_ID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = STYLE_ELEMENT_ID;
  style.textContent = `
    .${OUTLINE_CLASS} {
      outline: 2px solid #00ff00 !important;
      outline-offset: 0 !important;
    }
  `;
  document.head.appendChild(style);
}

function flattenRunningNodeIds() {
  const merged = new Set();
  for (const nodeIds of runningDisplayNodeIdsByPrompt.values()) {
    for (const nodeId of nodeIds) {
      merged.add(String(nodeId));
    }
  }
  globalRunningDisplayNodeIds = merged;
}

function getRunningStrokeStyle() {
  const nodeId = String(this?.id ?? "");
  if (!nodeId || !globalRunningDisplayNodeIds.has(nodeId)) {
    return;
  }
  return { color: "#0f0" };
}

function ensureNodeStrokeStyle(node) {
  if (!node) {
    return;
  }
  node.strokeStyles ??= {};
  if (!node.strokeStyles[STROKE_STYLE_KEY]) {
    node.strokeStyles[STROKE_STYLE_KEY] = getRunningStrokeStyle;
  }
}

function patchVisibleGraphNodes() {
  const nodes = app?.canvas?.graph?.nodes;
  if (!Array.isArray(nodes)) {
    return;
  }
  for (const node of nodes) {
    ensureNodeStrokeStyle(node);
  }
}

function syncVueNodeOutlineClasses() {
  ensureStyleElement();

  const outlined = document.querySelectorAll(`.${OUTLINE_CLASS}[data-node-id]`);
  for (const element of outlined) {
    const nodeId = String(element.dataset.nodeId ?? "");
    if (!globalRunningDisplayNodeIds.has(nodeId)) {
      element.classList.remove(OUTLINE_CLASS);
    }
  }

  for (const nodeId of globalRunningDisplayNodeIds) {
    const nodeElements = document.querySelectorAll(
      `[data-node-id="${escapeNodeIdForSelector(nodeId)}"]`
    );
    for (const element of nodeElements) {
      element.classList.add(OUTLINE_CLASS);
    }
  }
}

function refreshRunningHighlights() {
  flattenRunningNodeIds();
  patchVisibleGraphNodes();
  syncVueNodeOutlineClasses();
  app?.canvas?.setDirty?.(true, true);
}

function updatePromptRunningNodes(promptId, nodes) {
  const runningNodeIds = new Set();
  for (const [nodeId, state] of Object.entries(nodes || {})) {
    if (state?.state !== "running") {
      continue;
    }
    const displayNodeId = state.display_node_id ?? state.node_id ?? nodeId;
    if (displayNodeId != null) {
      runningNodeIds.add(String(displayNodeId));
    }
  }

  if (runningNodeIds.size > 0) {
    runningDisplayNodeIdsByPrompt.set(String(promptId), runningNodeIds);
  } else {
    runningDisplayNodeIdsByPrompt.delete(String(promptId));
  }

  refreshRunningHighlights();
}

function clearPromptRunningNodes(promptId) {
  if (promptId == null) {
    return;
  }
  runningDisplayNodeIdsByPrompt.delete(String(promptId));
  refreshRunningHighlights();
}

function handleProgressState(event) {
  const detail = event?.detail;
  if (!detail?.prompt_id) {
    return;
  }
  updatePromptRunningNodes(detail.prompt_id, detail.nodes);
}

function handleExecutionFinished(event) {
  clearPromptRunningNodes(event?.detail?.prompt_id);
}

function handleGraphChanged() {
  refreshRunningHighlights();
}

app.registerExtension({
  name: EXTENSION_NAME,

  setup() {
    ensureStyleElement();

    api.addEventListener("progress_state", handleProgressState);
    api.addEventListener("execution_success", handleExecutionFinished);
    api.addEventListener("execution_error", handleExecutionFinished);
    api.addEventListener("execution_interrupted", handleExecutionFinished);
    api.addEventListener("graphChanged", handleGraphChanged);

    refreshRunningHighlights();
  },

  nodeCreated(node) {
    ensureNodeStrokeStyle(node);
  }
});
