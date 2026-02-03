const byId = (id) => document.getElementById(id);
const statusBadge = byId("statusBadge");
const latencyLabel = byId("latency");
const baseUrlInput = byId("baseUrl");
const saveBaseUrlBtn = byId("saveBaseUrl");
const refreshVoicesBtn = byId("refreshVoices");
const checkHealthBtn = byId("checkHealth");
const customVoiceSelect = byId("customVoice");

const baseKey = "qwen3-tts-base-url";

function normalizeBaseUrl(value) {
  if (!value) return window.location.origin;
  return value.replace(/\/$/, "");
}

function getBaseUrl() {
  return normalizeBaseUrl(baseUrlInput.value.trim());
}

function setStatus(text, tone = "idle") {
  statusBadge.textContent = text;
  statusBadge.className = "stat-value";
  statusBadge.classList.add(`status-${tone}`);
}

function setLatency(ms) {
  latencyLabel.textContent = ms;
}

function showMessage(targetId, text, tone = "idle") {
  const el = byId(targetId);
  el.textContent = text;
  el.className = "message";
  el.classList.add(`status-${tone}`);
}

function setDownload(anchorId, url, format) {
  const link = byId(anchorId);
  link.href = url;
  link.download = `tts_${Date.now()}.${format}`;
}

function setAudio(audioId, url) {
  const audio = byId(audioId);
  audio.src = url;
}

function setBusy(form, busy) {
  const button = form.querySelector("button[type='submit']");
  if (button) {
    if (!button.dataset.label) {
      button.dataset.label = button.textContent;
    }
    button.disabled = busy;
    button.textContent = busy ? "处理中..." : button.dataset.label;
  }
}

function readFormValue(form, name) {
  const field = form.querySelector(`[name='${name}']`);
  if (!field) return "";
  return field.value || "";
}

function isStreamEnabled(form) {
  const toggle = form.querySelector("[name='stream_segments']");
  return Boolean(toggle && toggle.checked);
}

async function readErrorMessage(res) {
  try {
    const text = await res.text();
    if (!text) return `HTTP ${res.status}`;
    try {
      const data = JSON.parse(text);
      return data.detail || data.message || JSON.stringify(data);
    } catch (err) {
      return text;
    }
  } catch (err) {
    return `HTTP ${res.status}`;
  }
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const message = await readErrorMessage(res);
    throw new Error(message || `HTTP ${res.status}`);
  }
  return res;
}

function b64ToArrayBuffer(b64) {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

class SegmentPlayer {
  constructor(audioEl) {
    this.audioEl = audioEl;
    this.ctx = null;
    this.nextTime = 0;
  }

  async reset() {
    if (this.ctx) {
      await this.ctx.close();
    }
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AudioCtx();
    await this.ctx.resume();
    this.nextTime = this.ctx.currentTime;
  }

  async enqueue(buffer) {
    if (!this.ctx) {
      await this.reset();
    }
    const audioBuffer = await this.ctx.decodeAudioData(buffer.slice(0));
    const source = this.ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.ctx.destination);
    const startAt = Math.max(this.nextTime, this.ctx.currentTime + 0.05);
    source.start(startAt);
    this.nextTime = startAt + audioBuffer.duration;
  }
}

async function streamSSE(url, options, onEvent) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const message = await readErrorMessage(res);
    throw new Error(message || `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\n\n/);
    buffer = parts.pop() || "";
    for (const chunk of parts) {
      const lines = chunk.split(/\n/);
      let event = "message";
      const dataLines = [];
      lines.forEach((line) => {
        if (line.startsWith("event:")) {
          event = line.replace("event:", "").trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.replace("data:", "").trim());
        }
      });
      if (!dataLines.length) continue;
      const dataText = dataLines.join("\n");
      let payload = dataText;
      try {
        payload = JSON.parse(dataText);
      } catch (err) {
        payload = { raw: dataText };
      }
      await onEvent(event, payload);
    }
  }
}

const streamControllers = {
  custom: null,
  design: null,
  clone: null,
};

const players = {
  custom: new SegmentPlayer(byId("customAudio")),
  design: new SegmentPlayer(byId("designAudio")),
  clone: new SegmentPlayer(byId("cloneAudio")),
};

async function runStreamSegments({
  mode,
  form,
  url,
  options,
  messageId,
  downloadId,
  outputFormat,
}) {
  if (streamControllers[mode]) {
    streamControllers[mode].abort();
  }
  const controller = new AbortController();
  streamControllers[mode] = controller;
  options.signal = controller.signal;

  const player = players[mode];
  await player.reset();
  setBusy(form, true);
  showMessage(messageId, "分段生成中...", "warn");
  const start = performance.now();
  let total = null;
  let received = 0;

  try {
    await streamSSE(url, options, async (event, payload) => {
      if (event === "meta") {
        total = payload.segments || null;
        showMessage(messageId, `开始生成，共 ${total || "?"} 段。`, "warn");
        return;
      }
      if (event === "chunk") {
        received += 1;
        if (payload.audio_b64) {
          const buffer = b64ToArrayBuffer(payload.audio_b64);
          await player.enqueue(buffer);
        }
        const progress = total ? `${received}/${total}` : `${received}`;
        showMessage(messageId, `已生成 ${progress} 段`, "warn");
        return;
      }
      if (event === "final" && payload.audio_b64) {
        const buffer = b64ToArrayBuffer(payload.audio_b64);
        const blob = new Blob([buffer], {
          type: outputFormat === "mp3" ? "audio/mpeg" : "audio/wav",
        });
        const fileUrl = URL.createObjectURL(blob);
        setAudio(`${mode}Audio`, fileUrl);
        setDownload(downloadId, fileUrl, outputFormat);
        return;
      }
      if (event === "error") {
        throw new Error(payload.detail || "流式请求失败");
      }
    });

    showMessage(messageId, "分段完成，可回放或下载。", "ok");
    setLatency(`${Math.round(performance.now() - start)} ms`);
    setStatus("完成", "ok");
  } catch (err) {
    if (err.name === "AbortError") {
      showMessage(messageId, "已中止请求。", "warn");
      return;
    }
    showMessage(messageId, err.message || "请求失败", "bad");
    setStatus("失败", "bad");
  } finally {
    streamControllers[mode] = null;
    setBusy(form, false);
  }
}

async function checkHealth() {
  const baseUrl = getBaseUrl();
  setStatus("检查中...", "warn");
  try {
    const res = await fetchJson(`${baseUrl}/healthz`);
    const data = await res.json();
    setStatus(data.status === "ok" ? "在线" : "异常", data.status === "ok" ? "ok" : "warn");
  } catch (err) {
    setStatus("不可用", "bad");
  }
}

async function loadVoices() {
  const baseUrl = getBaseUrl();
  customVoiceSelect.innerHTML = "";
  try {
    const res = await fetchJson(`${baseUrl}/v1/voices`);
    const data = await res.json();
    const voices = data.data || [];
    if (!voices.length) throw new Error("无内置声音");
    voices.forEach((voice) => {
      const option = document.createElement("option");
      option.value = voice;
      option.textContent = voice;
      customVoiceSelect.appendChild(option);
    });
    customVoiceSelect.value = voices[0];
  } catch (err) {
    const option = document.createElement("option");
    option.value = "vivian";
    option.textContent = "vivian";
    customVoiceSelect.appendChild(option);
    showMessage("customMessage", "未能加载内置声音，已使用默认值。", "warn");
  }
}

function setupTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((btn) => btn.classList.remove("active"));
      tab.classList.add("active");
      const target = tab.dataset.tab;
      document.querySelectorAll(".pane").forEach((pane) => {
        pane.classList.toggle("active", pane.dataset.pane === target);
      });
    });
  });
}

async function handleCustomSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const payload = {
    text: readFormValue(form, "text"),
    voice: readFormValue(form, "voice") || "vivian",
    speed: Number(readFormValue(form, "speed")) || 1,
    output_format: readFormValue(form, "output_format") || "wav",
  };
  const language = readFormValue(form, "language");
  if (language) payload.language = language;

  if (isStreamEnabled(form)) {
    await runStreamSegments({
      mode: "custom",
      form,
      url: `${getBaseUrl()}/v1/tts/custom/stream_segments`,
      options: {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
      messageId: "customMessage",
      downloadId: "customDownload",
      outputFormat: payload.output_format,
    });
    return;
  }

  setBusy(form, true);
  showMessage("customMessage", "生成中...", "warn");
  const start = performance.now();
  try {
    const res = await fetchJson(`${getBaseUrl()}/v1/tts/custom`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    setAudio("customAudio", url);
    setDownload("customDownload", url, payload.output_format);
    showMessage("customMessage", `完成 (${(blob.size / 1024).toFixed(1)} KB)`, "ok");
    setLatency(`${Math.round(performance.now() - start)} ms`);
    setStatus("完成", "ok");
  } catch (err) {
    showMessage("customMessage", err.message || "请求失败", "bad");
    setStatus("失败", "bad");
  } finally {
    setBusy(form, false);
  }
}

async function handleDesignSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const payload = {
    text: readFormValue(form, "text"),
    prompt_text: readFormValue(form, "prompt_text"),
    speed: Number(readFormValue(form, "speed")) || 1,
    output_format: readFormValue(form, "output_format") || "wav",
  };

  if (isStreamEnabled(form)) {
    await runStreamSegments({
      mode: "design",
      form,
      url: `${getBaseUrl()}/v1/tts/design/stream_segments`,
      options: {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
      messageId: "designMessage",
      downloadId: "designDownload",
      outputFormat: payload.output_format,
    });
    return;
  }

  setBusy(form, true);
  showMessage("designMessage", "生成中...", "warn");
  const start = performance.now();
  try {
    const res = await fetchJson(`${getBaseUrl()}/v1/tts/design`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    setAudio("designAudio", url);
    setDownload("designDownload", url, payload.output_format);
    showMessage("designMessage", `完成 (${(blob.size / 1024).toFixed(1)} KB)`, "ok");
    setLatency(`${Math.round(performance.now() - start)} ms`);
    setStatus("完成", "ok");
  } catch (err) {
    showMessage("designMessage", err.message || "请求失败", "bad");
    setStatus("失败", "bad");
  } finally {
    setBusy(form, false);
  }
}

async function handleCloneSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const formData = new FormData(form);
  const outputFormat = formData.get("output_format") || "wav";

  if (isStreamEnabled(form)) {
    await runStreamSegments({
      mode: "clone",
      form,
      url: `${getBaseUrl()}/v1/tts/clone/stream_segments`,
      options: {
        method: "POST",
        body: formData,
      },
      messageId: "cloneMessage",
      downloadId: "cloneDownload",
      outputFormat,
    });
    return;
  }

  setBusy(form, true);
  showMessage("cloneMessage", "生成中...", "warn");
  const start = performance.now();
  try {
    const res = await fetchJson(`${getBaseUrl()}/v1/tts/clone`, {
      method: "POST",
      body: formData,
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    setAudio("cloneAudio", url);
    setDownload("cloneDownload", url, outputFormat);
    showMessage("cloneMessage", `完成 (${(blob.size / 1024).toFixed(1)} KB)`, "ok");
    setLatency(`${Math.round(performance.now() - start)} ms`);
    setStatus("完成", "ok");
  } catch (err) {
    showMessage("cloneMessage", err.message || "请求失败", "bad");
    setStatus("失败", "bad");
  } finally {
    setBusy(form, false);
  }
}

function initBaseUrl() {
  const saved = localStorage.getItem(baseKey);
  baseUrlInput.value = saved || window.location.origin;
}

saveBaseUrlBtn.addEventListener("click", () => {
  const value = getBaseUrl();
  baseUrlInput.value = value;
  localStorage.setItem(baseKey, value);
  loadVoices();
  checkHealth();
});

refreshVoicesBtn.addEventListener("click", loadVoices);
checkHealthBtn.addEventListener("click", checkHealth);

byId("form-custom").addEventListener("submit", handleCustomSubmit);
byId("form-design").addEventListener("submit", handleDesignSubmit);
byId("form-clone").addEventListener("submit", handleCloneSubmit);

initBaseUrl();
setupTabs();
loadVoices();
checkHealth();
