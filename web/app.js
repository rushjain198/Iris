const $ = (id) => document.getElementById(id);
const ZONES = ["left", "center", "right"];
const COLORS = { clear: "#2fd16a", caution: "#ffb020", stop: "#ff4757" };
const caregiver = document.body.dataset.caregiver === "1";
const hhmmss = (t) => new Date(t * 1000).toTimeString().slice(0, 8);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---- radar: top-down fan, wearer at the bottom centre; each wedge = one zone, the arc = how near the obstacle is ----
const CX = 200, CY = 220, R0 = 26, R1 = 205;
const ANG = { left: [-62, -21], center: [-20, 20], right: [21, 62] };   // degrees from straight ahead
const pt = (r, deg) => { const a = (deg - 90) * Math.PI / 180; return [CX + r * Math.cos(a), CY + r * Math.sin(a)]; };
function wedgePath(r0, r1, a0, a1) {
  const [x0, y0] = pt(r1, a0), [x1, y1] = pt(r1, a1), [x2, y2] = pt(r0, a1), [x3, y3] = pt(r0, a0);
  return `M${x0},${y0} A${r1},${r1} 0 0 1 ${x1},${y1} L${x2},${y2} A${r0},${r0} 0 0 0 ${x3},${y3} Z`;
}
function arcPath(r, a0, a1) { const [x0, y0] = pt(r, a0 + 2), [x1, y1] = pt(r, a1 - 2); return `M${x0},${y0} A${r},${r} 0 0 1 ${x1},${y1}`; }
function buildRadar() {
  const svg = $("radar"); if (!svg) return;
  let h = "";
  for (const r of [70, 120, 170]) h += `<path d="${arcPath(r, -62, 62)}" stroke="#ffffff14" stroke-width="1" fill="none"/>`;
  for (const z of ZONES) {
    const [a0, a1] = ANG[z];
    h += `<path class="wedge" id="w-${z}" d="${wedgePath(R0, R1, a0, a1)}" fill="#2fd16a" opacity=".16"/>`;
    h += `<path class="arc" id="a-${z}" d="${arcPath(R1, a0, a1)}" stroke="#2fd16a" stroke-width="7" stroke-linecap="round" fill="none"/>`;
    const [lx, ly] = pt(R1 - 12, (a0 + a1) / 2);
    h += `<text x="${lx}" y="${ly}" fill="#8b98a8" font-size="12" font-weight="700" text-anchor="middle" letter-spacing="1">${z.toUpperCase()}</text>`;
  }
  h += `<g id="arrow" opacity="0"><path d="M0,-58 L14,-30 L5,-30 L5,0 L-5,0 L-5,-30 L-14,-30 Z" fill="#fff"/></g>`;
  h += `<circle cx="${CX}" cy="${CY}" r="11" fill="#8f7cff"/><circle cx="${CX}" cy="${CY}" r="18" fill="none" stroke="#8f7cff66" stroke-width="2"/>`;
  svg.innerHTML = h;
}
function updateRadar(s) {
  if (!$("radar")) return;
  for (const z of ZONES) {
    const o = s.zones[z] || { near: 0, level: "clear" }, c = COLORS[o.level] || COLORS.clear;
    const w = $("w-" + z), a = $("a-" + z);
    w.setAttribute("fill", c); w.setAttribute("opacity", o.level === "clear" ? ".12" : o.level === "caution" ? ".28" : ".42");
    const RMAX = R1 - 34;   // stay inside the zone labels
    const r = RMAX - Math.max(0, Math.min(1, o.near)) * (RMAX - R0 - 14);   // nearer -> arc closer to the wearer
    const [a0, a1] = ANG[z];
    a.setAttribute("d", arcPath(r, a0, a1)); a.setAttribute("stroke", c);
  }
  const g = $("arrow");
  if (s.clear) { const deg = (ANG[s.clear][0] + ANG[s.clear][1]) / 2; g.setAttribute("transform", `translate(${CX},${CY - 24}) rotate(${deg})`); g.setAttribute("opacity", "1"); }
  else g.setAttribute("opacity", "0");
}

// ---- direction word ----
function direction(s) {
  const c = s.zones.center || {}, worst = ZONES.reduce((m, z) => Math.max(m, ["clear", "caution", "stop"].indexOf((s.zones[z] || {}).level)), 0);
  if (c.level === "stop" && !s.clear) return ["STOP", "stop", "Nothing clear ahead. Stand still."];
  if (c.level === "stop") return [`GO ${s.clear.toUpperCase()} ${s.clear === "left" ? "←" : "→"}`, "stop", "Obstacle straight ahead."];
  if (worst === 0) return ["CLEAR AHEAD ↑", "clear", "Path is open."];
  if (s.clear === "center") return ["AHEAD ↑", "caution", "Something to the side. Keep straight."];
  if (s.clear) return [`BEAR ${s.clear.toUpperCase()} ${s.clear === "left" ? "←" : "→"}`, "caution", "Obstacle getting close."];
  return ["SLOW", "caution", "Obstacles around you."];
}

// ---- render ----
let lastEventT = 0, subTimer = null, firstStops = null;
let prevDesc = "", prevRead = "", prevColor = "", lastState = {};
function render(s) {
  $("src").textContent = s.source === "glasses" ? "Glasses live" : `${s.source}${s.stub ? " (stub)" : ""}`;
  $("livedot").className = "dot live";
  $("fps").textContent = `${s.fps} fps`;
  if ($("mutechip")) $("mutechip").hidden = !s.muted;

  const [word, lvl, sub] = direction(s);
  $("dir").textContent = word; $("dir").className = "dir " + lvl; $("dirsub").textContent = sub;
  $("viewwrap").className = "viewwrap " + (lvl === "clear" ? "" : lvl);
  updateRadar(s);
  $("zones").innerHTML = ZONES.map((z) => {
    const o = s.zones[z] || { near: 0, level: "clear" };
    return `<div class="z ${o.level}${s.clear === z ? " go" : ""}"><b>${z.toUpperCase()}</b><span>${o.level}</span>` +
      `<div class="bar"><i style="width:${Math.round(o.near * 100)}%;background:${COLORS[o.level]}"></i></div></div>`;
  }).join("");

  if ($("objs")) {
    const d = s.detections || [];
    $("objs").innerHTML = d.length ? d.map((x) => `<span class="obj">${esc(x.label)} <small>${x.zone} · ${Math.round(x.score * 100)}%</small></span>`).join("")
      : `<span class="empty">None detected</span>`;
  }

  const ev = s.events || [];
  $("log").innerHTML = ev.slice().reverse().map((e) => `<li class="${e.level}"><time>${hhmmss(e.t)}</time><span>${esc(e.text)}</span></li>`).join("")
    || `<li class="info"><time>—</time><span>Waiting for the first cue…</span></li>`;

  const last = ev[ev.length - 1];
  if (last && last.t > lastEventT && Date.now() / 1000 - last.t <= 5) {
    lastEventT = last.t;
    const st = $("subtitle"), ico = last.level === "stop" ? "⛔" : last.level === "caution" ? "⚠️" : "✅";
    st.innerHTML = `<span class="ico">${ico}</span>${esc(last.text)}`; st.classList.add("show");
    clearTimeout(subTimer); subTimer = setTimeout(() => st.classList.remove("show"), 3500);
  }
  $("emergency").classList.toggle("on", !!s.emergency);
  // SOS / fall check
  const sb = $("sosbanner");
  if (s.sos) { sb.hidden = false; $("sostext").textContent = `SOS — ${s.sos.reason} · ${Math.max(0, Math.round(Date.now() / 1000 - s.sos.t))} s ago`; }
  else if (s.checking) { sb.hidden = false; $("sostext").textContent = "Possible fall — asking: are you okay?"; }
  else sb.hidden = true;
  if ($("bw")) { $("bw").firstChild.textContent = `Fall check: ${s.watch ? "on" : "off"} `; $("bw").classList.toggle("on", !!s.watch); }
  if (s.color && s.color !== prevColor) { prevColor = s.color; if (!describing && $("desc")) $("desc").textContent = s.color; }

  if (caregiver) {
    $("nowlvl").textContent = lvl === "clear" ? "Clear" : lvl === "stop" ? "Stop" : "Caution";
    $("nowlvl").style.color = COLORS[lvl];
    const stops = ev.filter((e) => e.level === "stop");
    $("nstops").textContent = String(stops.length);
    const ls = stops[stops.length - 1];
    $("laststop").textContent = ls ? `${Math.max(0, Math.round(Date.now() / 1000 - ls.t))} s ago` : "none";
  } else {
    $("bm").classList.toggle("muted", !!s.muted);
    $("bm").firstChild.textContent = s.muted ? "Unmute voice " : "Mute voice ";
    // show whichever of describe / read changed last (re-showing describe every tick wiped read results)
    if (s.describe && s.describe !== prevDesc) { prevDesc = s.describe; if (!describing) $("desc").textContent = s.describe; }
    if (s.read && s.read !== prevRead) { prevRead = s.read; if (!describing) $("desc").textContent = s.read; }
  }
}

async function tick() {
  try { lastState = await (await fetch("/api/state", { cache: "no-store" })).json(); render(lastState); }
  catch (e) { $("src").textContent = "Disconnected"; $("livedot").className = "dot"; }
}

let describing = false;
async function describe() {
  describing = true; const box = $("desc"); box.classList.add("busy"); box.textContent = "Looking…";
  try { const r = await (await fetch("/api/describe", { method: "POST" })).json(); box.textContent = r.text || "No description."; }
  catch (e) { box.textContent = "Describe failed."; }
  box.classList.remove("busy"); describing = false;
}
async function readText() {
  describing = true; const box = $("desc"); box.classList.add("busy"); box.textContent = "Reading…";
  try { const r = await (await fetch("/api/read", { method: "POST" })).json(); box.textContent = r.text || "No text."; }
  catch (e) { box.textContent = "Read failed."; }
  box.classList.remove("busy"); describing = false;
}
async function post(path) { try { return await (await fetch(path, { method: "POST" })).json(); } catch (e) { return {}; } }
async function colorOf() { const r = await post("/api/color"); if ($("desc")) $("desc").textContent = r.text || "No color."; }
const sos = () => post("/api/sos"), okay = () => post("/api/okay"), watch = () => post("/api/watch");

// ---- voice commands: Chrome speech recognition in this tab (audio goes to Google); "Iris" prefix optional ----
let rec = null, listening = false;
function command(t) {
  t = t.toLowerCase();
  const say = (x) => { if ($("hint")) $("hint").textContent = `Heard: "${t}" → ${x}`; };
  if (/\b(i'?m|i am) (ok|okay|fine|alright)\b|\bcancel\b|\bfalse alarm\b/.test(t)) { say("I'm okay"); return okay(); }
  if (/\b(help|emergency|sos|call for help)\b/.test(t)) { say("SOS"); return sos(); }
  if (/\b(read|text|sign|label)\b/.test(t)) { say("read text"); return readText(); }
  if (/\b(colou?r)\b/.test(t)) { say("color"); return colorOf(); }
  if (/\b(describe|ahead|in front|what do you see|around)\b/.test(t)) { say("describe"); return describe(); }
  if (/\b(unmute|voice on|speak)\b/.test(t)) { say("unmute"); return lastState.muted && mute(); }
  if (/\b(mute|quiet|stop talking)\b/.test(t)) { say("mute"); return !lastState.muted && mute(); }
  if (/\b(watch me|fall check|fall detection)\b/.test(t)) { say("fall check"); return watch(); }
}
function toggleListen() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { $("bl").textContent = "Voice commands need Chrome"; $("bl").disabled = true; return; }
  listening = !listening;
  $("bl").firstChild.textContent = `Voice commands: ${listening ? "on" : "off"} `; $("bl").classList.toggle("on", listening);
  if (!listening) { try { rec && rec.stop(); } catch (e) {} return; }
  rec = new SR(); rec.lang = "en-US"; rec.continuous = true; rec.interimResults = false;
  rec.onresult = (e) => { for (let i = e.resultIndex; i < e.results.length; i++) if (e.results[i].isFinal) command(e.results[i][0].transcript); };
  rec.onend = () => { if (listening) try { rec.start(); } catch (e) {} };
  rec.onerror = (e) => { if (e.error === "not-allowed") { listening = false; $("bl").firstChild.textContent = "Mic blocked "; } };
  rec.start();
}
async function mute() { await fetch("/api/mute", { method: "POST" }); tick(); }

buildRadar();
if (!caregiver) {
  $("bd").onclick = describe; $("br").onclick = readText; $("bm").onclick = mute;
  $("bc").onclick = colorOf; $("bs").onclick = sos; $("bl").onclick = toggleListen; $("bw").onclick = watch; $("okbtn").onclick = okay;
  addEventListener("keydown", (e) => { if (e.target.tagName === "INPUT") return; const k = e.key.toLowerCase(); if (k === "d") describe(); if (k === "r") readText(); if (k === "m") mute(); if (k === "c") colorOf(); if (k === "h") sos(); if (k === "o") okay(); if (k === "v") toggleListen(); if (k === "f") watch(); });
}
setInterval(tick, 250); tick();
