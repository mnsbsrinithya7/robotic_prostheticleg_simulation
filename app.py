"""
Streamlit dashboard: MPC for a two-link robotic prosthetic leg.
Run:  streamlit run app.py          (keep this file next to prosthesis_mpc.py)
"""
import inspect
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import prosthesis_mpc as pm

st.set_page_config(page_title="Prosthesis MPC", layout="wide")
COL = {"PD + gravity comp.": "#2ca02c", "Computed-torque PD": "#1f77b4",
       "MPC": "#d62728", "MPC (no estimator)": "#ff7f0e"}
DEG = 180 / np.pi
ACT = pm.ACT          # activity presets live in prosthesis_mpc.py (shared with the CLI)


def stretch(fn, *args, **kw):
    """Full-width element on every Streamlit version: new releases use width="stretch",
    older ones use_container_width=True (removed in recent releases)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "width" in params and isinstance(params["width"].default, str):
        kw["width"] = "stretch"
    elif "use_container_width" in params:
        kw["use_container_width"] = True
    return fn(*args, **kw)


# ----------------------------------------------------------------------------- simulation
@st.cache_data(show_spinner=False)
def run_sim(duration, horizon, delay_ratio, activity, mismatch, friction, noise, terrain,
            contact, hip_lim, knee_lim, seed, with_noest, obstacles=True, density=4.0, placement="path"):
    ctrls = ("pd", "ctc", "mpc") + (("mpc_noest",) if with_noest else ())
    logs, rows, extra, healthy, info = pm.run_scenario(
        activity, duration, horizon, delay_ratio, mismatch, friction, noise, terrain, contact, hip_lim, knee_lim,
        seed, ctrls, obstacles, density, placement=placement)
    return logs, pd.DataFrame(rows).T.round(3), healthy, info, extra


# ----------------------------------------------------------------------------- plots
def tracking_fig(logs, healthy_delayed):
    t = logs["MPC"]["t"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Hip angle q1 [deg]", "Knee angle q2 [deg]",
                                        "Coordination error ‖q_p(t) − q_L(t−Γ)‖ [deg]", "Knee torque [Nm]"))
    for j in range(2):
        fig.add_trace(go.Scatter(x=t, y=healthy_delayed[:, j] * DEG, name="Reference q_L(t−Γ)",
                                 line=dict(color="black", dash="dash", width=2),
                                 showlegend=(j == 0), legendgroup="ref"), row=j + 1, col=1)
    for n, L in logs.items():
        kw = dict(line=dict(color=COL[n], width=1.8), legendgroup=n)
        fig.add_trace(go.Scatter(x=t, y=L["q"][:, 0] * DEG, name=n, **kw), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=L["q"][:, 1] * DEG, showlegend=False, **kw), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=np.linalg.norm(L["q"] - L["qref"], axis=1) * DEG,
                                 showlegend=False, **kw), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=L["u"][:, 1], showlegend=False, **kw), row=4, col=1)
    fig.update_xaxes(title_text="time [s]", row=4, col=1)
    fig.update_layout(height=900, hovermode="x unified", legend=dict(orientation="h", y=1.06),
                      margin=dict(t=70, b=30))
    return fig


def cyclogram_fig(logs, ref):
    fig = go.Figure(go.Scatter(x=ref[:, 0] * DEG, y=ref[:, 1] * DEG, name="Reference",
                               line=dict(color="black", dash="dash", width=3)))
    for n, L in logs.items():
        fig.add_trace(go.Scatter(x=L["q"][:, 0] * DEG, y=L["q"][:, 1] * DEG, name=n,
                                 line=dict(color=COL[n], width=1.5)))
    fig.update_layout(height=500, xaxis_title="hip q1 [deg]", yaxis_title="knee q2 [deg]",
                      title="Hip-knee cyclogram")
    return fig


def ankle_path(L, l1, l2):
    q, hip = L["q"], L["hip"]
    return (hip[:, 0] + l1 * np.sin(q[:, 0]) + l2 * np.sin(q[:, 0] + q[:, 1]),
            hip[:, 1] - l1 * np.cos(q[:, 0]) - l2 * np.cos(q[:, 0] + q[:, 1]))


def terrain_fig(logs, info):
    """world view of the terrain (stairs or road with obstacles) and every controller's ankle path."""
    tr = info["terrain"]
    P = np.array(tr["poly"])
    paths = {n: ankle_path(L, info["l1"], info["l2"]) for n, L in logs.items()}
    x0 = min(p[0].min() for p in paths.values()) - 0.3
    x1 = max(p[0].max() for p in paths.values()) + 0.3
    y0 = min(min(p[1].min() for p in paths.values()), P[:, 1].min()) - 0.1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=P[:, 0], y=np.full(len(P), y0), line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], fill="tonexty", fillcolor="#d9c7a8",
                             line=dict(color="#8b6b3e", width=2), name="contact surface (ankle level)"))
    for n, (fx, fy) in paths.items():
        fig.add_trace(go.Scatter(x=fx, y=fy, name=n + " ankle", line=dict(color=COL[n], width=1.3)))
    hip = next(iter(logs.values()))["hip"]
    fig.add_trace(go.Scatter(x=hip[:, 0], y=hip[:, 1], name="hip (pelvis)",
                             line=dict(color="black", dash="dot", width=1)))
    if tr["kind"] == "road":
        for o in tr.get("obstacles", []):
            fig.add_annotation(x=o["x"], y=info["ground"] + max(o["h"], 0) + 0.03, text=o["k"], showarrow=False,
                               font=dict(size=10, color="#555"))
        title = f"Road: {info['n_stones']} stones, {info['n_cracks']} cracks · ground speed {info['speed']:.2f} m/s"
    else:
        title = (f"Stairs {'up' if tr['dir'] > 0 else 'down'}: riser {tr['riser']*100:.0f} cm, tread "
                 f"{tr['tread']*100:.0f} cm · pelvis speed {info['speed']:.2f} m/s")
    fig.update_layout(height=430, title=title, xaxis_title="x [m]", yaxis_title="y [m]",
                      xaxis=dict(range=[x0, x1]), yaxis=dict(range=[y0, hip[:, 1].max() + 0.1]),
                      legend=dict(orientation="h", y=-0.2))
    if tr["kind"] == "stairs":
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# ----------------------------------------------------------------------------- MPC estimator study plots
def mpc_single_fig(L, name, Ts):
    t = L["t"]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        subplot_titles=("Hip angle q1 [deg]", "Knee angle q2 [deg]",
                                        "Coordination error ‖e(t)‖ [deg]", "Joint torques [Nm]",
                                        "Disturbance: estimate d̂ vs true (low-pass) [Nm]"))
    for j in range(2):
        fig.add_trace(go.Scatter(x=t, y=L["qref"][:, j] * DEG, name="Reference q_L(t−Γ)",
                                 line=dict(color="black", dash="dash", width=2), showlegend=(j == 0)),
                      row=j + 1, col=1)
        fig.add_trace(go.Scatter(x=t, y=L["q"][:, j] * DEG, name=name, line=dict(color=COL[name], width=1.8),
                                 showlegend=(j == 0)), row=j + 1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.linalg.norm(L["q"] - L["qref"], axis=1) * DEG, showlegend=False,
                             line=dict(color=COL[name])), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=L["u"][:, 0], name="hip torque", line=dict(color="#1f77b4")), row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=L["u"][:, 1], name="knee torque", line=dict(color="#9467bd")), row=4, col=1)
    if np.all(np.isfinite(L["dtrue"])):
        dl = pm.lowpass(L["dtrue"], Ts)
        fig.add_trace(go.Scatter(x=t, y=dl[:, 0], name="true hip (low-pass)",
                                 line=dict(color="#1f77b4", width=1), opacity=.45), row=5, col=1)
        fig.add_trace(go.Scatter(x=t, y=dl[:, 1], name="true knee (low-pass)",
                                 line=dict(color="#9467bd", width=1), opacity=.45), row=5, col=1)
    fig.add_trace(go.Scatter(x=t, y=L["dhat"][:, 0], name="d̂ hip", line=dict(color="#1f77b4", width=2.2)),
                  row=5, col=1)
    fig.add_trace(go.Scatter(x=t, y=L["dhat"][:, 1], name="d̂ knee", line=dict(color="#9467bd", width=2.2)),
                  row=5, col=1)
    fig.update_xaxes(title_text="time [s]", row=5, col=1)
    fig.update_layout(height=1050, hovermode="x unified", legend=dict(orientation="h", y=1.05),
                      margin=dict(t=70, b=30))
    return fig


def est_compare_figs(lw, lo, Tg, cmp_):
    n1, n2 = "MPC", "MPC (no estimator)"
    t = lw["t"]
    m = t > Tg
    f1 = go.Figure()
    for n, L in ((n1, lw), (n2, lo)):
        f1.add_trace(go.Scatter(x=t, y=np.linalg.norm(L["q"] - L["qref"], axis=1) * DEG, name=n,
                                line=dict(color=COL[n], width=1.8)))
    f1.update_layout(height=360, title="Coordination error ‖e(t)‖ over time", xaxis_title="time [s]",
                     yaxis_title="[deg]", hovermode="x unified")

    cw, co = pm.per_cycle_rmse(lw, Tg), pm.per_cycle_rmse(lo, Tg)
    k = np.arange(1, min(len(cw), len(co)) + 1)
    f2 = go.Figure([go.Bar(x=k, y=cw[:len(k)], name=n1, marker_color=COL[n1]),
                    go.Bar(x=k, y=co[:len(k)], name=n2, marker_color=COL[n2])])
    f2.update_layout(barmode="group", height=360, title="Per-cycle coordination RMSE",
                     xaxis_title="gait cycle", yaxis_title="RMSE [deg]")

    f3 = go.Figure([go.Box(y=np.linalg.norm(L["q"][m] - L["qref"][m], axis=1) * DEG, name=n,
                           marker_color=COL[n], boxmean=True) for n, L in ((n1, lw), (n2, lo))])
    f3.update_layout(height=360, title="Error distribution after the first cycle", yaxis_title="‖e‖ [deg]")

    f4 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                       subplot_titles=("Signed hip error q1 − q1,ref [deg]", "Signed knee error q2 − q2,ref [deg]"))
    for j in range(2):
        for n, L in ((n1, lw), (n2, lo)):
            f4.add_trace(go.Scatter(x=t, y=(L["q"][:, j] - L["qref"][:, j]) * DEG, name=n, showlegend=(j == 0),
                                    line=dict(color=COL[n], width=1.4)), row=j + 1, col=1)
        f4.add_hline(y=0, line=dict(color="black", width=1), row=j + 1, col=1)
    f4.update_layout(height=520, hovermode="x unified", title="Offset (bias) — what the estimator removes")

    keys = [k_ for k_ in ["RMSE hip [deg]", "RMSE knee [deg]", "Coord. error mean [deg]", "Coord. error max [deg]",
                          "Coord. error P95 [deg]", "Last-cycle RMSE [deg]", "Torque RMS hip [Nm]",
                          "Torque RMS knee [Nm]", "Mean positive power [W]", "Torque rate RMS [Nm/s]"]
            if k_ in cmp_.index]
    f5 = go.Figure([go.Bar(name=n, x=keys, y=[cmp_.loc[k_, col] for k_ in keys], marker_color=COL[n])
                    for n, col in ((n1, "MPC + estimator"), (n2, "MPC (no estimator)"))])
    f5.update_layout(barmode="group", height=430, title="Key metrics side by side (lower is better)")
    f6 = go.Figure(go.Bar(x=keys, y=[cmp_.loc[k_, "Change [%]"] for k_ in keys],
                          marker_color=["#2a7a2a" if cmp_.loc[k_, "Change [%]"] < 0 else "#d62728" for k_ in keys]))
    f6.update_layout(height=380, title="Change with the estimator relative to without [%] (negative = lower)",
                     yaxis_title="%")
    return f1, f2, f3, f4, f5, f6


# ----------------------------------------------------------------------------- animation (HTML5 canvas + JS)
ANIM_HTML = """
<div style="font-family:Arial,sans-serif;color:#222">
 <div style="display:flex;gap:14px;align-items:center;margin-bottom:8px">
  <button id="pp" style="padding:6px 16px;border-radius:6px;border:1px solid #888;cursor:pointer">Pause</button>
  <label>Speed <select id="sp"><option value="0.25">0.25x</option><option value="0.5">0.5x</option>
   <option value="1" selected>1x</option><option value="2">2x</option></select></label>
  <input type="range" id="sl" min="0" max="0" value="0" style="flex:1"><span id="tl" style="width:70px"></span>
 </div>
 <canvas id="cv" style="width:100%;height:440px;border:1px solid #ddd;border-radius:10px;background:#fff"></canvas>
</div>
<script>
const D = __DATA__, N = D.t.length, H = 440, R = 180/Math.PI;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const sl = document.getElementById('sl'), pp = document.getElementById('pp'), sp = document.getElementById('sp');
sl.max = N-1;
let idx = 0, last = null, playing = true;
function size(){ const d = window.devicePixelRatio||1; cv.width = cv.clientWidth*d; cv.height = H*d; }
window.addEventListener('resize', size); size();
function lerpQ(a, i){ const i0 = Math.floor(i)%N, i1 = (i0+1)%N, f = i-Math.floor(i);
  return [a[i0][0]+(a[i1][0]-a[i0][0])*f, a[i0][1]+(a[i1][1]-a[i0][1])*f]; }
function lerpH(i){ const i0 = Math.floor(i)%N, i1 = Math.min(i0+1, N-1), f = i-Math.floor(i), a = D.hip;
  return [a[i0][0]+(a[i1][0]-a[i0][0])*f, a[i0][1]+(a[i1][1]-a[i0][1])*f]; }
function pts(q, sc, hx, hy){
  const kx = D.l1*Math.sin(q[0]), ky = -D.l1*Math.cos(q[0]);
  const fx = kx + D.l2*Math.sin(q[0]+q[1]), fy = ky - D.l2*Math.cos(q[0]+q[1]);
  return [[hx,hy],[hx+kx*sc,hy-ky*sc],[hx+fx*sc,hy-fy*sc]]; }
function leg(p, color, w, alpha){
  ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w;
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.moveTo(p[0][0],p[0][1]); ctx.lineTo(p[1][0],p[1][1]); ctx.lineTo(p[2][0],p[2][1]); ctx.stroke();
  for (const c of p){ ctx.beginPath(); ctx.arc(c[0],c[1],w*0.65,0,6.3); ctx.fill(); }
  ctx.globalAlpha = 1; }
function ground(x0, pw, hx, hy, sc, i){                 // terrain in the hip frame (stairs / stones / cracks)
  if (!D.poly){ const gy = hy - D.ground*sc;
    ctx.fillStyle='#d9c7a8'; ctx.fillRect(x0,gy,pw,H-gy); ctx.strokeStyle='#8b6b3e'; ctx.lineWidth=3;
    ctx.beginPath(); ctx.moveTo(x0,gy); ctx.lineTo(x0+pw,gy); ctx.stroke(); return; }
  const hp = lerpH(i), P = D.poly.map(p=>[hx+(p[0]-hp[0])*sc, hy-(p[1]-hp[1])*sc]);
  ctx.fillStyle='#d9c7a8'; ctx.beginPath(); ctx.moveTo(P[0][0],H+5);
  P.forEach(p=>ctx.lineTo(p[0],p[1])); ctx.lineTo(P[P.length-1][0],H+5); ctx.closePath(); ctx.fill();
  ctx.strokeStyle='#8b6b3e'; ctx.lineWidth=3; ctx.beginPath(); P.forEach((p,j)=>j?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])); ctx.stroke(); }
function draw(i){
  const d = window.devicePixelRatio||1; ctx.setTransform(d,0,0,d,0,0);
  const W = cv.clientWidth, n = D.names.length, pw = W/n, sc = (H-120)/1.0, hy = 70;
  ctx.clearRect(0,0,W,H);
  const qh = lerpQ(D.healthy, i), qr = lerpQ(D.qref, i);
  D.names.forEach((name,k)=>{
    const x0 = k*pw, hx = x0+pw/2;
    ctx.save(); ctx.beginPath(); ctx.rect(x0,0,pw,H); ctx.clip();
    if (k>0){ ctx.strokeStyle='#ddd'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x0,0); ctx.lineTo(x0,H); ctx.stroke(); }
    ground(x0, pw, hx, hy, sc, i);                                    // ground
    ctx.fillStyle='#8b6b3e'; ctx.fillRect(hx-26,hy-9,52,12);          // pelvis
    ctx.beginPath(); ctx.strokeStyle = D.colors[k]; ctx.globalAlpha = .35; ctx.lineWidth = 2;   // foot trail
    for (let j=0;j<45;j++){ const ii=(Math.floor(i)-45+j+N*2)%N; const f=pts(D.q[name][ii],sc,hx,hy)[2];
      if(j==0) ctx.moveTo(f[0],f[1]); else ctx.lineTo(f[0],f[1]); }
    ctx.stroke(); ctx.globalAlpha = 1;
    leg(pts(qh,sc,hx,hy), '#1f77b4', 16, .28);                        // healthy limb (ghost)
    const q = lerpQ(D.q[name], i);
    leg(pts(q,sc,hx,hy), D.colors[k], 8, 1);                          // prosthesis
    const err = Math.hypot(q[0]-qr[0], q[1]-qr[1])*R;
    ctx.fillStyle='#222'; ctx.font='bold 15px Arial'; ctx.textAlign='center'; ctx.fillText(name, hx, 24);
    ctx.font='13px Arial'; ctx.fillStyle = err>8?'#d62728':'#2a7a2a';
    ctx.fillText('coordination error: '+err.toFixed(1)+'°', hx, 44);
    ctx.restore(); });
  ctx.font='12px Arial'; ctx.textAlign='left'; ctx.fillStyle='#1f77b4'; ctx.fillText('■ healthy limb q_L(t)', 10, H-10);
  ctx.fillStyle='#555'; ctx.fillText('■ prosthesis (follows with delay Γ = '+D.shift.toFixed(2)+' s)', 160, H-10);
  document.getElementById('tl').textContent = 't = '+(i*D.Ts).toFixed(2)+' s'; }
function frame(ts){
  if (playing && last!==null){ idx += (ts-last)/1000*parseFloat(sp.value)/D.Ts; if (idx>=N) idx -= N; }
  last = ts; draw(idx); sl.value = Math.floor(idx); requestAnimationFrame(frame); }
pp.onclick = ()=>{ playing=!playing; pp.textContent = playing?'Pause':'Play'; };
sl.oninput = ()=>{ idx = +sl.value; draw(idx); };
requestAnimationFrame(frame);
</script>
"""


def animation_html(logs, healthy, info):
    names = list(logs)
    data = dict(names=names, colors=[COL[n] for n in names], Ts=info["Ts"], ground=info["ground"],
                l1=info["l1"], l2=info["l2"], shift=info["shift"],
                t=np.round(logs[names[0]]["t"], 3).tolist(),
                healthy=np.round(healthy, 4).tolist(),
                qref=np.round(logs[names[0]]["qref"], 4).tolist(),
                q={n: np.round(logs[n]["q"], 4).tolist() for n in names},
                hip=np.round(logs[names[0]]["hip"], 4).tolist(), poly=info["terrain"]["poly"])
    return ANIM_HTML.replace("__DATA__", json.dumps(data))


# ----------------------------------------------------------------------------- UI
st.title("🦿 MPC for a Two-Link Robotic Prosthetic Leg")
st.caption("Lagrangian dynamics · gait coordination q_p(t) → q_L(t − Γ) · Ma, Zhang & Xu, IEEE TNSRE 2024")

with st.sidebar.form("params"):
    st.header("Simulation settings")
    duration = st.slider("Duration [s]", 3.0, 12.0, 6.0, 0.5)
    activity = st.selectbox("Activity", list(ACT), format_func=lambda k: ACT[k]["n"])
    delay_ratio = st.slider("Delay ratio Γ / Tg", 0.0, 1.0, 0.5, 0.05)
    horizon = st.slider("MPC horizon [steps × 20 ms]", 5, 30, 15)
    st.subheader("Uncertainty")
    mismatch = st.slider("Mass mismatch", 0.0, 0.6, 0.25, 0.05)
    friction = st.slider("Joint friction [Nms/rad]", 0.0, 4.0, 1.5, 0.25)
    noise = st.slider("Random torque noise", 0.0, 1.0, 0.3, 0.05)
    terrain = st.slider("Terrain roughness [m]", 0.0, 0.006, 0.002, 0.0005, format="%.4f")
    contact = st.checkbox("Foot-ground contact", True)
    st.subheader("Terrain & obstacles")
    obstacles = st.checkbox("Road obstacles: stones & cracks", True,
                            help="Used for walking, jogging and running. Stairs Up/Down always use a "
                                 "17 cm riser / 28 cm tread staircase with the moving-hip Lagrangian model.")
    density = st.slider("Obstacles per 10 m of road", 1.0, 10.0, 4.0, 0.5)
    placement = st.selectbox("Obstacle placement", ["path", "random"],
                             format_func=lambda k: {"path": "In the foot path (perturbation course)",
                                                    "random": "Random along the road"}[k])
    st.subheader("Limits & options")
    hip_lim = st.slider("Hip torque limit [Nm]", 30, 300, 150, 5)
    knee_lim = st.slider("Knee torque limit [Nm]", 30, 250, 120, 5)
    with_noest = st.checkbox("Also run MPC without disturbance estimator", True)
    seed = st.number_input("Random seed", 0, 999, 1)
    stretch(st.form_submit_button, "▶ Run simulation")

with st.spinner("Simulating controllers (about 10–40 s)…"):
    logs, table, healthy, info, extra = run_sim(duration, horizon, delay_ratio, activity, mismatch, friction, noise,
                                                terrain, contact, hip_lim, knee_lim, int(seed), with_noest,
                                                obstacles, float(density), placement)
ref_delayed = logs["MPC"]["qref"]

c1, c2, c3, c4 = st.columns(4)
m, ct, pd_ = table.loc["MPC"], table.loc["Computed-torque PD"], table.loc["PD + gravity comp."]
c1.metric("MPC knee RMSE", f"{m['RMSE knee [deg]']:.2f}°", f"{m['RMSE knee [deg]'] - ct['RMSE knee [deg]']:+.2f}° vs CTC", delta_color="inverse")
c2.metric("MPC hip RMSE", f"{m['RMSE hip [deg]']:.2f}°", f"{m['RMSE hip [deg]'] - ct['RMSE hip [deg]']:+.2f}° vs CTC", delta_color="inverse")
c3.metric("MPC max coord. error", f"{m['Coord. error max [deg]']:.1f}°", f"{m['Coord. error max [deg]'] - pd_['Coord. error max [deg]']:+.1f}° vs PD", delta_color="inverse")
c4.metric("MPC solve time", f"{info['solve_ms']:.2f} ms", "limit: 20 ms")

tab_3d, tab_anim, tab_plots, tab_metrics, tab_est, tab_model = st.tabs(
    ["🧍 3D Simulator", "🎬 Quick animation", "📈 Plots", "📊 Metrics", "🆚 MPC estimator study", "📐 Model"])

with tab_anim:
    st.markdown("Blue ghost = **healthy limb** q_L(t). Coloured leg = **prosthesis** following with a half-cycle lag. "
                "The text shows the live coordination error.")
    components.html(animation_html(logs, healthy, info), height=540)

with tab_plots:
    stretch(st.plotly_chart, tracking_fig(logs, ref_delayed))
    stretch(st.plotly_chart, cyclogram_fig(logs, ref_delayed))
    st.markdown("**Terrain and foot path** — contact surface for the ankle point and the ankle trajectory "
                "of each controller in world coordinates.")
    stretch(st.plotly_chart, terrain_fig(logs, info))

with tab_metrics:
    stretch(st.dataframe, table)
    keys = ["RMSE hip [deg]", "RMSE knee [deg]", "Coord. error max [deg]", "Mean positive power [W]"]
    fig = go.Figure([go.Bar(name=n, x=keys, y=[table.loc[n, k] for k in keys], marker_color=COL[n])
                     for n in table.index])
    fig.update_layout(barmode="group", height=420)
    stretch(st.plotly_chart, fig)
    st.caption("Peak contact force = largest foot–ground force after the first cycle. Obstacle/riser contact = share "
               "of time the foot touches a stone, crack edge or stair riser.")
    stretch(st.download_button, "Download metrics CSV", table.to_csv().encode(), "metrics.csv", "text/csv")

with tab_est:
    if "MPC (no estimator)" not in logs:
        st.info("This comparison needs both MPC variants. Tick **Also run MPC without disturbance estimator** "
                "in the sidebar and press **Run simulation**.")
    else:
        lw, lo = logs["MPC"], logs["MPC (no estimator)"]
        rw = {**table.loc["MPC"].to_dict(), **extra["MPC"]}
        ro = {**table.loc["MPC (no estimator)"].to_dict(), **extra["MPC (no estimator)"]}
        cmp_ = pm.estimator_comparison_table(rw, ro)
        st.markdown(
            "Both controllers use the same LTV-MPC, horizon, weights, nominal model, noise realisation and terrain. "
            "The only difference is the offset-free **disturbance estimator**: every 20 ms it compares the measured "
            "joint velocities with the one-step prediction of the nominal model and updates "
            "d̂ ← d̂ + α·M(q)·(q̇ − q̇_pred)/Ts (α = 0.4, |d̂| ≤ 40 Nm). The MPC then plans with u + d̂ and removes d̂ "
            "from the feed-forward. Without it, d̂ = 0 and mass error, friction, contact and terrain act as "
            "unmodelled torque.")

        def pct(k):
            return cmp_.loc[k, "Change [%]"] if k in cmp_.index else float("nan")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Knee RMSE", f"{rw['RMSE knee [deg]']:.2f}° vs {ro['RMSE knee [deg]']:.2f}°",
                  f"{pct('RMSE knee [deg]'):+.1f}% with estimator", delta_color="inverse")
        k2.metric("Mean coord. error", f"{rw['Coord. error mean [deg]']:.2f}° vs {ro['Coord. error mean [deg]']:.2f}°",
                  f"{pct('Coord. error mean [deg]'):+.1f}%", delta_color="inverse")
        k3.metric("Last-cycle RMSE", f"{rw['Last-cycle RMSE [deg]']:.2f}° vs {ro['Last-cycle RMSE [deg]']:.2f}°",
                  f"{pct('Last-cycle RMSE [deg]'):+.1f}%", delta_color="inverse")
        if "Unexplained disturbance RMS [Nm]" in rw:
            k4.metric("Unexplained disturbance", f"{rw['Unexplained disturbance RMS [Nm]']:.1f} vs "
                      f"{ro['Unexplained disturbance RMS [Nm]']:.1f} Nm",
                      f"{pct('Unexplained disturbance RMS [Nm]'):+.1f}%", delta_color="inverse")
        better = [k for k in cmp_.index if "[deg]" in k and k.startswith(("RMSE", "Coord", "Last", "Abs"))
                  and cmp_.loc[k, "Difference"] < 0]
        worse = [k for k in cmp_.index if "[deg]" in k and k.startswith(("RMSE", "Coord", "Last", "Abs"))
                 and cmp_.loc[k, "Difference"] > 0]
        st.markdown(f"**This run** ({ACT[info['act']]['n']}, seed {int(seed)}): the estimator lowered "
                    f"{len(better)} of {len(better) + len(worse)} tracking-error metrics"
                    + (f"; higher with the estimator: {', '.join(worse)}." if worse else ".")
                    + f" Torque RMS knee {rw['Torque RMS knee [Nm]']:.1f} vs {ro['Torque RMS knee [Nm]']:.1f} Nm, "
                    f"solve time {rw.get('Solve time mean [ms]', float('nan')):.2f} vs "
                    f"{ro.get('Solve time mean [ms]', float('nan')):.2f} ms.")

        st.subheader("MPC with disturbance estimator")
        stretch(st.plotly_chart, mpc_single_fig(lw, "MPC", info["Ts"]))
        stretch(st.dataframe, pd.DataFrame({"MPC + estimator": rw}).round(3))
        st.subheader("MPC without disturbance estimator")
        stretch(st.plotly_chart, mpc_single_fig(lo, "MPC (no estimator)", info["Ts"]))
        stretch(st.dataframe, pd.DataFrame({"MPC (no estimator)": ro}).round(3))

        st.subheader("Comparison")
        f1, f2, f3, f4, f5, f6 = est_compare_figs(lw, lo, info["Tg"], cmp_)
        stretch(st.plotly_chart, f1)
        a, b = st.columns(2)
        with a:
            stretch(st.plotly_chart, f2)
        with b:
            stretch(st.plotly_chart, f3)
        stretch(st.plotly_chart, f4)
        stretch(st.plotly_chart, cyclogram_fig({"MPC": lw, "MPC (no estimator)": lo}, ref_delayed))
        stretch(st.plotly_chart, f5)
        stretch(st.plotly_chart, f6)
        stretch(st.dataframe, cmp_.round(3))
        st.caption("Difference = with − without. Change [%] = difference / |without|. All metrics are computed "
                   "after the first gait cycle, except First-cycle RMSE. Unexplained disturbance = RMS of "
                   "(true lumped disturbance, low-passed 0.1 s) − d̂.")
        stretch(st.download_button, "Download estimator comparison CSV", cmp_.round(4).to_csv().encode(),
                "mpc_estimator_comparison.csv", "text/csv")

with tab_model:
    st.subheader("Lagrangian model (paper Eq. 8)")
    st.latex(r"M(q)\ddot q + C(q,\dot q)\dot q + N(q) = u + J^T F_c")
    st.subheader("Ground contact (Eq. 7)")
    st.latex(r"F_{c} = k\,\delta + c\,\dot\delta")
    st.subheader("Gait coordination objective (Eq. 13)")
    st.latex(r"e(t) = q_p(t) - q_L(t-\Gamma),\quad \Gamma = T_g/2,\quad \lim_{t\to\infty}\|e(t)\| = 0")
    st.subheader("MPC problem (each 20 ms)")
    st.latex(r"\min_{\delta u}\ \sum_{k=1}^{N} e_k^T Q e_k + \delta u_k^T R\,\delta u_k"
             r"\quad \text{s.t. } e_{k+1}=A_k e_k + B_k\delta u_k + c_k,\ |u|\le u_{max},\ q_{inf}\le q\le q_{sup}")
    st.markdown("Input: `u = u_ff + δu`, where `u_ff` is the inverse dynamics of the reference and a disturbance "
                "estimator removes unmodelled torque (friction, contact, mass error).")
    st.subheader("Stairs: Lagrangian model with a moving hip")
    st.markdown("On stairs the hip (pelvis) follows a planned path p_h(t) that rises or falls one step every half "
                "gait cycle, with a small 2×-gait-frequency bounce. Writing the link centres of mass as "
                "p_h + r_i(q) and applying Lagrange's equations gives one extra term; the base-velocity terms cancel:")
    st.latex(r"M(q)\ddot q + C(q,\dot q)\dot q + N(q) + B(q)\,\ddot p_h(t) = u + J^T F_c,\qquad "
             r"B(q) = m_1 J_{c1}^T(q) + m_2 J_{c2}^T(q)")
    st.markdown("The vertical column of B equals N(q)/g, so climbing acceleration acts like extra gravity. "
                "All model-based controllers use the planned p̈_h in their feed-forward. The stance part of the stair "
                "gait comes from inverse kinematics with the foot fixed on its tread; the swing part is a template "
                "shaped to typical stair-gait ranges and tuned for at least 7 cm ankle clearance over each nosing "
                "(synthetic reference, not measured data). Staircase: riser 17 cm, tread 28 cm; treads and risers are "
                "both contact surfaces, so a low swing foot hits the riser.")
    st.subheader("Road obstacles")
    st.markdown("Stones are raised-cosine bumps (2–4.5 cm high, 8–18 cm long); cracks are narrow depressions "
                "(1.2–2.5 cm deep, 3–8 cm wide). The contact force acts along the local surface normal. The foot's "
                "world position is x = v·t + x_foot, with v the kinematically consistent ground speed "
                "(stance foot does not slide over the road). Obstacles use their own random stream, so roughness "
                "and torque noise are identical with obstacles on or off.")


# ----------------------------------------------------------------------------- 3D simulator fed by the Python results
def simulator_html(logs, healthy, info):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prosthetic_leg_simulator.html")
    with open(path, encoding="utf-8") as f:
        html = f.read()
    names = list(logs)
    r4 = lambda x: np.round(x, 4).tolist()
    data = dict(act=info["act"], Tg=info["Tg"], Ts=info["Ts"], n=len(logs[names[0]]["t"]), names=names,
                qmin=info["qmin"], qmax=info["qmax"], healthy=r4(healthy), qref=r4(logs[names[0]]["qref"]),
                hip=r4(logs[names[0]]["hip"]), terrain=info["terrain"],
                d={n: dict(q=r4(logs[n]["q"]), qd=r4(logs[n]["qd"]), u=r4(logs[n]["u"]),
                           fc=np.round(np.linalg.norm(logs[n]["fc"], axis=1), 1).tolist()) for n in names})
    tag = '<script src="https://cdnjs'
    return html.replace(tag, "<script>window.SIM_DATA=" + json.dumps(data) + ";</script>" + tag, 1)


with tab_3d:
    st.caption("The 3D figure plays back the exact joint angles, torques and constraint results from the Python "
               "simulation above. Change the activity or any setting in the sidebar and press Run simulation.")
    components.html(simulator_html(logs, healthy, info), height=1450, scrolling=True)
