//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// Live viewer: robot pose, ego/depth images, chat, plan overlay.
const $ = (id) => document.getElementById(id);
const S = {
  follow: false, ghost: true, overlay: false, scene: true, ghostOp: 0.5, ovOp: 0.55,
  THREE: null,
  jointNames: null,   // deploy-order joint names for LIVE (from /api/live/start)
};

// ── three.js
async function initThree() {
  const vp = $("viewport");
  try {
    const THREE = await import("three");
    const { STLLoader } = await import("three/addons/loaders/STLLoader.js");
    const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
    const URDFLoader = (await import("urdf-loader")).default;
    S.THREE = THREE;
    const w = vp.clientWidth, h = vp.clientHeight;
    // ACES tone mapping, warm key + cool fill + hemisphere, dark floor with grid, distance fog.
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(2, devicePixelRatio)); renderer.setSize(w, h);
    renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.08;
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    vp.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    {
      const c = document.createElement("canvas"); c.width = 2; c.height = 256;
      const g2 = c.getContext("2d"), gr = g2.createLinearGradient(0, 0, 0, 256);
      gr.addColorStop(0, "#050506"); gr.addColorStop(0.62, "#0c0d0f"); gr.addColorStop(1, "#16171a");
      g2.fillStyle = gr; g2.fillRect(0, 0, 2, 256);
      const sky = new THREE.CanvasTexture(c); sky.colorSpace = THREE.SRGBColorSpace;
      scene.background = sky;
    }
    scene.fog = new THREE.Fog(0x0e0f11, 8, 30);   // near/far re-fit to the camera distance each frame
    const camera = new THREE.PerspectiveCamera(42, w / h, 0.01, 100);
    camera.position.set(2.4, 2.4, 1.7);
    scene.add(new THREE.HemisphereLight(0x32353c, 0x040405, 1.0));
    const key = new THREE.DirectionalLight(0xfff6e8, 2.6);
    key.position.set(2.5, 6, 3.5); key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048); key.shadow.radius = 5; key.shadow.bias = -0.0004; key.shadow.normalBias = 0.02;
    Object.assign(key.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 20 });
    scene.add(key); scene.add(key.target);
    const fill = new THREE.DirectionalLight(0x7589ad, 0.55); fill.position.set(-3, 2.2, -2.5); scene.add(fill);
    // 40 m ground fits a whole room scan; 12 mm low so it does not z-fight the scan floor.
    const GSIZE = 40;
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x0f1012, roughness: 0.92, metalness: 0.0 });
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(GSIZE, GSIZE), groundMat);
    ground.rotation.x = -Math.PI / 2; ground.position.y = -0.012; ground.receiveShadow = true;
    scene.add(ground); S.ground = ground;
    const mkGrid = (div, op) => {
      const gh = new THREE.GridHelper(GSIZE, div, 0xe0e8ff, 0xe0e8ff);
      gh.material.transparent = true; gh.material.opacity = op; gh.material.depthWrite = false;
      gh.position.y = -0.010; return gh;
    };
    const grid = new THREE.Group(); grid.add(mkGrid(GSIZE, 0.085)); grid.add(mkGrid(GSIZE * 4, 0.03));
    scene.add(grid); S.grid = grid;
    {
      // soft pool of light under the robot (the website floor's center glow)
      const c = document.createElement("canvas"); c.width = c.height = 128;
      const g2 = c.getContext("2d"), gr = g2.createRadialGradient(64, 64, 0, 64, 64, 64);
      gr.addColorStop(0, "rgba(150,160,185,0.30)"); gr.addColorStop(1, "rgba(150,160,185,0)");
      g2.fillStyle = gr; g2.fillRect(0, 0, 128, 128);
      const glow = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 4.2), new THREE.MeshBasicMaterial({
        map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
      glow.rotation.x = -Math.PI / 2; glow.position.y = -0.008; scene.add(glow); S.glow = glow;
    }
    const root = new THREE.Group(); root.rotation.x = -Math.PI / 2; scene.add(root); // Z-up
    const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.target.set(0, 0, 0.8);
    // world axes (X red / Y green / Z blue), hidden until toggled
    const worldAxes = new THREE.AxesHelper(0.6); worldAxes.visible = false; root.add(worldAxes); S.worldAxes = worldAxes;
    const robotAxes = new THREE.AxesHelper(0.35); robotAxes.visible = false; root.add(robotAxes); S.robotAxes = robotAxes;
    Object.assign(S, { renderer, camera, controls, scene, root });
    CUT.uCutEye.value = new THREE.Vector3(); CUT.uCutFocus.value = new THREE.Vector3();
    controls.addEventListener("start", () => { S._camDrag = true; });
    controls.addEventListener("end", () => { S._camDrag = false; S._fRig = null; });
    const _rp = new THREE.Vector3();
    let _lastT = performance.now();
    (function animate() {
      requestAnimationFrame(animate); controls.update();
      const _now = performance.now(), _dt = Math.min(0.1, (_now - _lastT) / 1000); _lastT = _now;
      // key light + shadow box + floor glow ride along with the robot; fog fits the zoom level
      if (S.robot) {
        S.robot.getWorldPosition(_rp);
        key.target.position.set(_rp.x, 0, _rp.z); key.position.set(_rp.x + 2.5, 6, _rp.z + 3.5);
        if (S.glow) S.glow.position.set(_rp.x, -0.008, _rp.z);
        followTick(_dt, _rp);
        CUT.uCutEye.value.copy(camera.position); CUT.uCutFocus.value.set(_rp.x, _rp.y, _rp.z);
        CUT.uCutOn.value = 1;
      } else CUT.uCutOn.value = 0;
      const dcam = camera.position.distanceTo(controls.target);
      scene.fog.near = dcam + 5; scene.fog.far = dcam + 24;
      renderer.render(scene, camera);
    })();
    new ResizeObserver(() => { const w = vp.clientWidth, h = vp.clientHeight; if (w && h) { camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h); } }).observe(vp);

    const mkLoader = () => {
      const mgr = new THREE.LoadingManager();
      const ld = new URDFLoader(mgr); ld.workingPath = "/robot/";
      ld.loadMeshCb = (path, m, done) => new STLLoader(mgr).load(path, (g) => {
        g.computeVertexNormals();
        done(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0x9aa2ab, metalness: 0.25, roughness: 0.55 })));
      }, undefined, (e) => done(null, e));
      return { ld, mgr };
    };
    // Tint by mutating mesh materials: the URDF loader sets colours after loadMeshCb and STL
    // meshes load async, so retry until meshes exist and re-apply against late overrides.
    const applyTint = (robot, rgb, opacity) => {
      const mats = [];
      robot.traverse((o) => {
        if (o.isMesh && o.material) {
          const m = o.material;
          if (m.color) m.color.setRGB(rgb[0], rgb[1], rgb[2]);
          if (m.emissive) m.emissive.setRGB(rgb[0] * 0.12, rgb[1] * 0.12, rgb[2] * 0.12);
          m.opacity = opacity; m.transparent = true; m.depthWrite = opacity >= 0.99;
          m.needsUpdate = true; mats.push(m);
        }
      });
      robot.userData.tintMats = mats;
      return mats.length;
    };
    const tintWhenReady = (getRobot, rgb, opacity, tries = 40) => {
      const r = getRobot();
      const n = r ? applyTint(r, rgb, opacity) : 0;
      if ((!r || n === 0) && tries > 0) { requestAnimationFrame(() => tintWhenReady(getRobot, rgb, opacity, tries - 1)); return; }
      // meshes found + tinted; re-apply twice more to beat any late loader override
      setTimeout(() => { const rr = getRobot(); if (rr) applyTint(rr, rgb, opacity); }, 300);
      setTimeout(() => { const rr = getRobot(); if (rr) applyTint(rr, rgb, opacity); }, 900);
    };
    // White shell + near-black joints for the actual robot (ghost/plan keep their tints).
    const styleRobot = (robot) => {
      if (!robot) return;
      robot.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        o.castShadow = true;
        const m = o.material, l = m.color ? (m.color.r + m.color.g + m.color.b) / 3 : 0.7;
        if (m.color) l > 0.45 ? m.color.setRGB(0.93, 0.93, 0.95) : m.color.setRGB(0.07, 0.07, 0.08);
        if ("roughness" in m) { m.roughness = l > 0.45 ? 0.38 : 0.5; m.metalness = l > 0.45 ? 0.05 : 0.25; }
        m.needsUpdate = true;
      });
    };
    const A = mkLoader();
    A.ld.load("/robot/urdf/g1_29dof_rev_1_0.urdf", (r) => { S.robot = r; root.add(r); });
    const G = mkLoader();
    G.ld.load("/robot/urdf/g1_29dof_rev_1_0.urdf", (r) => { S.ghostRobot = r; r.visible = S.ghost; root.add(r); });
    G.mgr.onLoad = () => tintWhenReady(() => S.ghostRobot, [0.40, 0.55, 1.00], S.ghostOp);
    const O = mkLoader();
    O.ld.load("/robot/urdf/g1_29dof_rev_1_0.urdf", (r) => { S.overlayRobot = r; r.visible = false; root.add(r); });
    O.mgr.onLoad = () => { tintWhenReady(() => S.overlayRobot, [1.00, 0.55, 0.20], S.ovOp); setTimeout(buildStrobes, 1500); };
    A.mgr.onLoad = () => { styleRobot(S.robot); $("loading").style.display = "none"; loadScene(); };
    setTimeout(() => { $("loading").style.display = "none"; }, 12000);
  } catch (e) { console.error(e); $("loading").innerHTML = "3D unavailable — " + (e.message || e); }
}
async function loadScene() {
  if (!S.THREE || S.sceneGrp) return;
  try {
    const j = await (await fetch("/api/scene")).json();
    if (!j.meshes || !j.meshes.length) return;
    addSceneMeshes(j);
  } catch (e) { console.warn("scene load failed", e); }
}
// Scan textures: max anisotropy (grazing-angle floors/walls), trilinear mipmaps, and a cache
// keyed on the data URI so per-material meshes share one GPU upload.
const _texCache = new Map();
function texFor(uri) {
  let tex = _texCache.get(uri);
  if (tex) return tex;
  const T = S.THREE;
  tex = new T.TextureLoader().load(uri);
  tex.colorSpace = T.SRGBColorSpace;
  tex.anisotropy = S.renderer.capabilities.getMaxAnisotropy();
  tex.generateMipmaps = true;
  tex.minFilter = T.LinearMipmapLinearFilter;
  tex.magFilter = T.LinearFilter;
  _texCache.set(uri, tex);
  return tex;
}

// Cutaway: scene geometry in a cone from camera to robot is discarded (dithered rim) so walls
// never hide the robot; the floor (world y < 0.12 m) is kept. Uniforms are shared by all scene materials.
const CUT = { uCutEye: { value: null }, uCutFocus: { value: null }, uCutOn: { value: 0 } };
function withCutaway(mat) {
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, CUT);
    sh.vertexShader = "varying vec3 vCutW;\n" + sh.vertexShader.replace("#include <project_vertex>",
      "#include <project_vertex>\n  vCutW = (modelMatrix * vec4(transformed, 1.0)).xyz;");
    sh.fragmentShader = "varying vec3 vCutW;\nuniform vec3 uCutEye; uniform vec3 uCutFocus; uniform float uCutOn;\n" +
      sh.fragmentShader.replace("void main() {", `void main() {
  if (uCutOn > 0.5) {
    vec3 sg = uCutFocus - uCutEye; float L = length(sg); vec3 dr = sg / L, w = vCutW - uCutEye; float sp = dot(w, dr);
    if (vCutW.y > 0.12 && sp > 0.0 && sp < L - 0.42) {
      float r = length(w - dr * sp), R = 0.16 + 0.52 * sp / L, k = (r - R * 0.72) / (R * 0.28);
      float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
      if (r < R * 0.72 || (r < R && n > k)) discard;
    }
  }`);
  };
  return mat;
}
// Follow camera: rides the robot's smoothed heading at a fixed offset angle; dragging resets the offset.
const _wrapA = (a) => Math.atan2(Math.sin(a), Math.cos(a));
function followTick(dt, rp) {
  if (!S.follow || !S.robot || !S.controls || S._camDrag) return;
  const T = S.THREE, cam = S.camera, tgt = S.controls.target;
  if (!S._fq) { S._fq = new T.Quaternion(); S._fx = new T.Vector3(); }
  S.robot.getWorldQuaternion(S._fq); S._fx.set(1, 0, 0).applyQuaternion(S._fq);
  const head = Math.atan2(S._fx.z, S._fx.x);
  if (S._fHead == null) S._fHead = head;
  S._fHead += _wrapA(head - S._fHead) * (1 - Math.exp(-dt * 0.9));
  const off = cam.position.clone().sub(tgt), dh = Math.max(0.5, Math.hypot(off.x, off.z)), dz = off.y;
  if (S._fRig == null) S._fRig = _wrapA(Math.atan2(off.z, off.x) - S._fHead);
  const kf = 1 - Math.exp(-dt * 3);
  tgt.x += (rp.x - tgt.x) * kf; tgt.z += (rp.z - tgt.z) * kf; tgt.y += (0.18 + 0.5 * rp.y - tgt.y) * kf;
  const a = S._fHead + S._fRig;
  cam.position.set(tgt.x + dh * Math.cos(a), tgt.y + dz, tgt.z + dh * Math.sin(a));
}
function addSceneMeshes(j) {
  const T = S.THREE, grp = new T.Group();
  (j.meshes || []).forEach((m) => {
    const g = new T.BufferGeometry();
    g.setAttribute("position", new T.BufferAttribute(new Float32Array(m.v), 3));
    g.setIndex(m.f);
    // Normals only for lit meshes (MeshBasicMaterial ignores them).
    const lit = !(m.tex && m.uv);
    if (lit) g.computeVertexNormals();
    let mat;
    if (!lit) {
      // Textured scan: unlit so its noisy normals don't shatter into facets.
      g.setAttribute("uv", new T.BufferAttribute(new Float32Array(m.uv), 2));
      const tex = texFor(m.tex);
      // FrontSide: single-sided roof/walls vanish from outside and stay solid inside;
      // S.twoSided (the "2-sided" toggle) restores DoubleSide. toneMapped false: texture is a photo.
      mat = withCutaway(new T.MeshBasicMaterial({ map: tex, side: S.twoSided ? T.DoubleSide : T.FrontSide, toneMapped: false }));
      S.scanMats = (S.scanMats || []); S.scanMats.push(mat);
    } else {
      mat = withCutaway(new T.MeshStandardMaterial({ color: new T.Color(m.color[0], m.color[1], m.color[2]), metalness: 0.1, roughness: 0.85, transparent: m.opacity < 1, opacity: m.opacity, side: T.DoubleSide }));
    }
    const mesh = new T.Mesh(g, mat);
    if (lit) { mesh.castShadow = true; mesh.receiveShadow = true; }
    if (m.body) {
      // Free body (e.g. the box): body-local verts; liveObjects() moves it to the sim's pose.
      mesh.position.set(m.pos[0], m.pos[1], m.pos[2]);
      mesh.quaternion.set(m.quat[0], m.quat[1], m.quat[2], m.quat[3]);
      (S.objMeshes[m.body] = S.objMeshes[m.body] || []).push(mesh);
    }
    grp.add(mesh);
  });
  S.sceneGrp = grp; S.scene = true; grp.visible = true; S.root.add(grp);
  // Scene extent so "Top" can fit it.
  try {
    const bb = new T.Box3().setFromObject(grp);
    if (isFinite(bb.min.x) && isFinite(bb.max.x)) S.sceneBox = bb;
  } catch (e) { /* empty group */ }
  $("tg-scene").classList.add("on");
}
function poseOne(robot, q) {
  if (!robot || !q) return;
  robot.position.set(q[0], q[1], q[2]); robot.quaternion.set(q[4], q[5], q[6], q[3]);
  const jn = S.jointNames;   // from /api/live/start; skip until known
  if (!jn) return;
  for (let i = 0; i < jn.length; i++) robot.setJointValue(jn[i], q[7 + i]);
}
function yawFromQ(q) {   // yaw (rad) from a qpos row [x,y,z, qw,qx,qy,qz, ...]
  const w = q[3], x = q[4], y = q[5], z = q[6];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}
// Default view: 3/4 from above-behind the robot's START, looking the way it heads.
// The scene root maps qpos (Z-up) -> world (Y-up): (x,y,z) -> (x, z, -y). So height is
// world-Y and the ground is the world X–Z plane; compute the camera there (not in qpos-xy).
function viewBehind(q) {
  if (!q || !S.controls) return;
  const yaw = yawFromQ(q);
  const wx = q[0], wz = -q[1];                    // robot ground position in world
  const fx = Math.cos(yaw), fz = -Math.sin(yaw);  // heading in world (X,Z)
  const rx = -fz, rz = fx;                         // right = forward × up(0,1,0)
  // Wide high start view from behind-above, centred ahead of the robot.
  const AHEAD = 1.5, BACK = 4.5, SIDE = 1.2, UP = 6.5;
  S.controls.target.set(wx + fx * AHEAD, 0.3, wz + fz * AHEAD);
  S.camera.position.set(wx - fx * BACK + rx * SIDE, UP, wz - fz * BACK + rz * SIDE);
  S.controls.update();
  S._fRig = null; S._fHead = null;   // follow camera re-latches its angle from this behind-the-robot view
}
// viser-style keyboard camera: WASD = fly (translate cam+target on the view plane;
// W/S forward-back, A/D strafe), arrows = orbit (yaw/pitch). Runs while keys held.
function camMove() {
  if (!S.controls || !S.THREE) { S._camLoop = false; return; }
  if (!S.keys.size) { S._camLoop = false; return; }
  const T = S.THREE, cam = S.camera, tgt = S.controls.target, K = S.keys;
  const dist = cam.position.distanceTo(tgt);
  const mv = Math.max(0.02, dist * 0.02);   // move step scales with zoom
  const rot = 0.03;                          // orbit step (rad)
  // WASD translate along camera forward (flattened to ground) + right
  const fwd = new T.Vector3(); cam.getWorldDirection(fwd); fwd.z = 0; if (fwd.lengthSq() > 1e-6) fwd.normalize();
  const right = new T.Vector3().crossVectors(fwd, new T.Vector3(0, 0, 1)).normalize();
  const d = new T.Vector3();
  if (K.has("w")) d.addScaledVector(fwd, mv);
  if (K.has("s")) d.addScaledVector(fwd, -mv);
  if (K.has("d")) d.addScaledVector(right, mv);
  if (K.has("a")) d.addScaledVector(right, -mv);
  if (d.lengthSq() > 0) { cam.position.add(d); tgt.add(d); if (S.follow) { S.follow = false; $("tg-follow").classList.remove("on"); } }
  // arrows orbit around the target (turntable yaw + pitch)
  let yaw = 0, pitch = 0;
  if (K.has("arrowleft")) yaw += rot;
  if (K.has("arrowright")) yaw -= rot;
  if (K.has("arrowup")) pitch += rot;
  if (K.has("arrowdown")) pitch -= rot;
  if (yaw || pitch) {
    const off = new T.Vector3().subVectors(cam.position, tgt);
    if (yaw) off.applyAxisAngle(new T.Vector3(0, 0, 1), yaw);
    if (pitch) { const rx = new T.Vector3().crossVectors(off, new T.Vector3(0, 0, 1)).normalize(); off.applyAxisAngle(rx, pitch); }
    cam.position.copy(tgt).add(off);
  }
  cam.lookAt(tgt);
  requestAnimationFrame(camMove);
}

// Plan strobe: STROBE_N amber poses sampled across the active plan (clones of the plan robot).
const STROBE_N = 10;
function buildStrobes() {
  const src = S.overlayRobot;
  if (!src || S.strobes || !S.root) return;
  S.strobes = [];
  for (let i = 0; i < STROBE_N; i++) {
    const c = src.clone(true);
    const a = 0.10 + 0.32 * i / (STROBE_N - 1);
    c.traverse((o) => {
      if (!o.isMesh || !o.material) return;
      o.material = o.material.clone();
      if (o.material.color) o.material.color.setRGB(1.0, 0.70, 0.25);
      if (o.material.emissive) o.material.emissive.setRGB(0.10, 0.06, 0.02);
      o.material.opacity = a; o.material.transparent = true; o.material.depthWrite = false;
      o.renderOrder = 2;
    });
    c.visible = false; S.root.add(c); S.strobes.push(c);
  }
}

// ── wiring
function toast(m) { const t = $("toast"); t.textContent = m; t.style.opacity = 1; clearTimeout(t._h); t._h = setTimeout(() => t.style.opacity = 0, 2200); }
function wire() {
  // robots / camera
  const setActual = (v) => { S.actual = v; if (S.robot) S.robot.visible = v; $("actualToggle").classList.toggle("on", v); $("tg-actual").classList.toggle("on", v); };
  const setGhost = (v) => { S.ghost = v; if (S.ghostRobot) S.ghostRobot.visible = v; $("ghostToggle").classList.toggle("on", v); $("tg-ghost").classList.toggle("on", v); };
  const setOverlay = (v) => { S.overlay = v; $("overlayToggle").classList.toggle("on", v); $("tg-overlay").classList.toggle("on", v); };
  // Turning follow off does not reframe; "Reset view" does.
  const setFollow = (v) => { S.follow = v; S._fRig = null; S._fHead = null; $("tg-follow").classList.toggle("on", v); const fb = $("followToggle"); if (fb) fb.classList.toggle("on", v); };
  $("actualToggle").onclick = () => setActual(!S.actual); $("tg-actual").onclick = () => setActual(!S.actual);
  $("ghostToggle").onclick = () => setGhost(!S.ghost); $("tg-ghost").onclick = () => setGhost(!S.ghost);
  $("overlayToggle").onclick = () => setOverlay(!S.overlay); $("tg-overlay").onclick = () => setOverlay(!S.overlay);
  $("tg-follow").onclick = () => setFollow(!S.follow);
  if ($("followToggle")) $("followToggle").onclick = () => setFollow(!S.follow);
  $("ghostOp").oninput = (e) => { S.ghostOp = +e.target.value; $("ghostOpV").textContent = S.ghostOp.toFixed(2); ((S.ghostRobot && S.ghostRobot.userData.tintMats) || []).forEach((m) => { m.opacity = S.ghostOp; m.transparent = true; }); };
  $("ovOp").oninput = (e) => { S.ovOp = +e.target.value; $("ovOpV").textContent = S.ovOp.toFixed(2); ((S.overlayRobot && S.overlayRobot.userData.tintMats) || []).forEach((m) => { m.opacity = S.ovOp; m.transparent = true; }); };
  setFollow(false); setGhost(true); setActual(true);
  // visualize: scene / grid / axes / camera
  $("tg-scene").onclick = () => { S.scene = !S.scene; if (S.sceneGrp) S.sceneGrp.visible = S.scene; else if (S.scene) loadScene(); $("tg-scene").classList.toggle("on", S.scene); };
  $("tg-grid").onclick = () => { const v = !S.grid.visible; S.grid.visible = v; if (S.ground) S.ground.visible = v;   // also hides the solid floor
    $("tg-grid").classList.toggle("on", v); };
  $("tg-axes").onclick = () => { S.worldAxes.visible = !S.worldAxes.visible; $("tg-axes").classList.toggle("on", S.worldAxes.visible); };
  $("tg-robotaxes").onclick = () => { S.robotAxes.visible = !S.robotAxes.visible; $("tg-robotaxes").classList.toggle("on", S.robotAxes.visible); };
  // Operator message box: Enter sends.
  $("humanMsg").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); sendHumanMsg(); }
  });
  const camPreset = (dx, dy, dz) => { if (!S.controls) return; const t = S.controls.target; S.camera.position.set(t.x + dx, t.y + dy, t.z + dz); };
  $("camReset").onclick = () => { if (S._liveQ) viewBehind(S._liveQ); else camPreset(2.2, 1.5, 2.2); };
  const sideBtn = $("btnTwoSided");
  if (sideBtn) sideBtn.onclick = () => {
    S.twoSided = !S.twoSided;
    (S.scanMats || []).forEach((mm) => {
      mm.side = S.twoSided ? S.THREE.DoubleSide : S.THREE.FrontSide; mm.needsUpdate = true;
    });
    sideBtn.classList.toggle("on", S.twoSided);
    sideBtn.textContent = S.twoSided ? "2-sided" : "1-sided";
  };
  $("camTop").onclick = () => {
    // Straight down, high enough that the scene's wider horizontal extent fits the frustum.
    const bb = S.sceneBox;
    if (!bb || !S.controls) { camPreset(0.01, 4, 0.01); return; }
    const cx = (bb.min.x + bb.max.x) / 2, cz = (bb.min.z + bb.max.z) / 2;
    const ex = bb.max.x - bb.min.x, ez = bb.max.z - bb.min.z;
    const fov = (S.camera.fov * Math.PI) / 180;
    // fit the WIDER of the two extents, allowing for the viewport aspect, + 15% margin
    const need = Math.max(ez, ex / Math.max(S.camera.aspect, 1e-3));
    const h = bb.max.y + (need / 2) / Math.tan(fov / 2) * 1.15;
    S.controls.target.set(cx, bb.min.y, cz);
    S.camera.position.set(cx + 0.001, h, cz);   // tiny offset: exactly vertical degenerates OrbitControls
    S.camera.far = Math.max(S.camera.far, h * 3);
    S.camera.updateProjectionMatrix();
    S.controls.update();
  };
  $("camFront").onclick = () => camPreset(0, 0.8, 3.5);    // in front (+Z)
  $("camSide").onclick = () => camPreset(3.5, 0.8, 0);     // to the side (+X)
  // Viewport view pill drives the same presets; 3/4 = the start view.
  const vseg = $("viewSeg");
  if (vseg) {
    const pick = (v) => vseg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === v)));
    vseg.querySelectorAll("button").forEach((b) => b.onclick = () => {
      const v = b.dataset.v;
      if (v === "34") { if (S._liveQ) viewBehind(S._liveQ); else camPreset(2.2, 1.5, 2.2); }
      else if (v === "front") $("camFront").click();
      else if (v === "side") $("camSide").click();
      else if (v === "top") $("camTop").click();
      pick(v); S._fRig = null;
    });
  }
  // resizers
  wireFolds();
  setFold(true);   // the side panel starts folded
  // Immersive layout (the viewer is live-only).
  document.body.classList.add("imm");
  wireWinFolds();
  dragH($("rz-l"), "--lw", 1, 200, 520); dragH($("rz-r"), "--rw", -1, 240, 600);
  makeEgoDraggable("egofloat");     // the floating ego PiP can be dragged anywhere
  makeEgoDraggable("depthfloat");   // the depth window drags on its own
  // keyboard: WASD fly the camera · arrows orbit
  const KEYMOVE = new Set(["w", "a", "s", "d", "arrowup", "arrowdown", "arrowleft", "arrowright"]);
  S.keys = new Set();
  const typing = (e) => ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName);
  addEventListener("keydown", (e) => {
    if (typing(e)) return;
    const k = e.key.toLowerCase();
    if (KEYMOVE.has(k)) { e.preventDefault(); if (!S.keys.has(k)) { S.keys.add(k); if (!S._camLoop) { S._camLoop = true; requestAnimationFrame(camMove); } } }
  });
  addEventListener("keyup", (e) => S.keys.delete(e.key.toLowerCase()));
  addEventListener("blur", () => S.keys.clear());
}
// ── foldable panels
// The right panel folds to a strip (never zero, so the chevron stays clickable). Folding writes the
// same custom property as the drag handle (--rw) and restores the pre-fold size.
const FOLD_STRIP_PX = 28;   // > the 22px chevron + 3px inset, else the affordance clips
function setFold(on) {
  const st = document.documentElement;   // --rw lives on :root
  const col = document.querySelector(".col.right");
  if (!col || col.classList.contains("folded") === !!on) return;
  if (on) {
    S._rwPrev = getComputedStyle(st).getPropertyValue("--rw").trim() || "340px";
    st.style.setProperty("--rw", FOLD_STRIP_PX + "px");
  } else {
    st.style.setProperty("--rw", S._rwPrev || "340px");
  }
  col.classList.toggle("folded", !!on);
  const b = $("foldR"); if (b) { b.textContent = on ? "‹" : "›"; b.title = on ? "expand the panel" : "collapse the panel"; }
}
function isFolded() { return !!document.querySelector(".col.right.folded"); }
function wireFolds() {
  const r = $("foldR");
  if (r) r.onclick = () => setFold(!isFolded());
}
// Pointer events (mouse + touch + pen); pointer capture keeps the drag if the finger leaves the bar.
function dragPointer(handle, onMove, onEnd) {
  if (!handle) return;
  handle.style.touchAction = "none";
  handle.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    e.preventDefault();
    const pid = e.pointerId, start = onMove(null, e);
    try { handle.setPointerCapture(pid); } catch (_) {}
    const mv = (ev) => { if (ev.pointerId === pid) { ev.preventDefault(); onMove(start, ev); } };
    const up = (ev) => {
      if (ev.pointerId !== pid) return;
      handle.removeEventListener("pointermove", mv); handle.removeEventListener("pointerup", up);
      handle.removeEventListener("pointercancel", up);
      if (onEnd) onEnd();
    };
    handle.addEventListener("pointermove", mv); handle.addEventListener("pointerup", up);
    handle.addEventListener("pointercancel", up);
  });
}
function dragH(handle, v, sign, min, max) {
  const st = document.documentElement;
  dragPointer(handle, (s, e) => {
    if (!s) return { x: e.clientX, cur: parseFloat(getComputedStyle(st).getPropertyValue(v)) || (sign > 0 ? 520 : 340) };
    let x = s.cur + sign * (e.clientX - s.x); x = Math.max(min, Math.min(max, x));
    st.style.setProperty(v, x + "px");
  });
}
// ── ego / depth window folding
// The ego window folds to its title strip; while folded no image is downloaded.
function winFolded(id) { const e = $(id); return !!(e && e.classList.contains("wfolded")); }
function setWinFold(id, on) {
  const e = $(id), b = $(id === "egofloat" ? "egoFold" : "depthFold");
  if (!e) return;
  e.classList.toggle("wfolded", !!on);
  if (b) { b.textContent = on ? "▾" : "▴"; b.title = (on ? "unfold" : "fold") + (id === "egofloat" ? " the ego camera" : " the depth camera"); }
  try { localStorage.setItem("halo.fold." + id, on ? "1" : "0"); } catch (_) {}
  if (!on) { if (id === "egofloat") S._liveImgT = 0; else S._liveDepthT = 0; }   // refresh at once
  placeDepth();
}
function wireWinFolds() {
  for (const [id, btn] of [["egofloat", "egoFold"], ["depthfloat", "depthFold"]]) {
    const b = $(btn); if (!b) continue;
    b.onclick = (e) => { e.stopPropagation(); setWinFold(id, !winFolded(id)); };
    let v = null; try { v = localStorage.getItem("halo.fold." + id); } catch (_) {}
    if (v === "1") setWinFold(id, true);
  }
}
// The floating ego PiP drags from anywhere except its controls; dragging re-anchors it top/left
// and clamps it on-screen.
function makeEgoDraggable(id) {
  const el = document.getElementById(id);
  if (!el) return;
  // touch-action:none so a finger drag is not a page scroll / pinch.
  el.style.touchAction = "none";
  el.addEventListener("pointerdown", (e) => {
    if (e.target.closest("select, option, input, button, a")) return;   // let controls work
    if (e.pointerType === "mouse" && e.button !== 0) return;
    const sx = e.clientX, sy = e.clientY;
    const r = el.getBoundingClientRect();
    const ox = sx - r.left, oy = sy - r.top;
    let dragging = false;
    const pid = e.pointerId;
    const mv = (ev) => {
      if (ev.pointerId !== pid) return;
      if (!dragging) {
        // Start only after a >4 px move so a plain click does not re-anchor (jump) the PiP.
        if (Math.abs(ev.clientX - sx) < 4 && Math.abs(ev.clientY - sy) < 4) return;
        dragging = true; el._dragged = true;
        el.style.right = "auto"; el.style.bottom = "auto";
      }
      ev.preventDefault();
      const w = el.offsetWidth, h = el.offsetHeight;
      // keep clear of the top bar and grabbable
      const x = Math.max(0, Math.min(innerWidth - w, ev.clientX - ox));
      const y = Math.max(54, Math.min(innerHeight - Math.min(h, 80), ev.clientY - oy));
      el.style.left = x + "px"; el.style.top = y + "px";
      if (el.id === "egofloat") placeDepth();   // an undragged depth window follows the ego window
    };
    const up = (ev) => {
      if (ev.pointerId !== pid) return;
      removeEventListener("pointermove", mv); removeEventListener("pointerup", up);
      removeEventListener("pointercancel", up);
    };
    addEventListener("pointermove", mv, { passive: false });
    addEventListener("pointerup", up); addEventListener("pointercancel", up);
  });
}

// ── LIVE mode
// Shows CONNECTING until the first g1/state frame.
async function startLive() {
  // enter LIVE mode immediately (visible reset), regardless of connection outcome.
  S._liveDepthT = 0; S._liveDepthGot = false;
  S._liveFramed = false;   // reframe the camera on the first live pose
  if (S.overlayRobot) S.overlayRobot.visible = false;   // hidden until the first live plan arrives
  // obs image panel → live ego (pulled from get_state; placeholder until it arrives)
  $("obsimg").style.display = "none"; $("obsempty").style.display = "";
  $("obsempty").textContent = "Connecting…";
  S._liveImgT = 0; S._liveImgOkT = 0;
  $("srclabel").textContent = "LIVE · connecting…";
  $("modeBadgeT").textContent = "LIVE";
  missionReset();                            // reset the orchestrator stream to "connecting"
  toast("LIVE: connecting to tracker…");

  try {
    const r = await (await fetch("/api/live/start")).json();
    // Live joint names before poll() so poseOne can pose the robot (even on error).
    if (r.joint_names) S.jointNames = r.joint_names;
    // Single-task (non-agentic) run: nobody is driving the view, so follow the robot by default.
    if (r.follow_default && !S._followAuto) {
      S._followAuto = true; S.follow = true;
      const tg = $("tg-follow"); if (tg) tg.classList.add("on");
    }
    if (!r.ok) { $("srclabel").textContent = "LIVE · error"; toast("live: " + (r.error || "can't open subscriber")); }
    else { toast(`LIVE ← ${r.channel} @ ${r.host}:${r.port}`); }
  } catch (e) { toast("live start failed: " + e); }
  poll();
  pollMission();                             // start the VLM mission-orchestrator stream

  function poll() {
    fetch("/api/live").then((x) => x.json()).then((d) => {
      if (d.have && d.qpos) {
        if (S.robot) S.robot.visible = S.actual;
        poseOne(S.robot, d.qpos);
        S._liveQ = d.qpos;
        if (S.robotAxes && S.robotAxes.visible) { S.robotAxes.position.set(d.qpos[0], d.qpos[1], d.qpos[2]); S.robotAxes.quaternion.set(d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[3]); }
        // blue reference ghost (streamed in RobotState.ref_motion when the tracker ships it)
        if (S.ghostRobot) {
          if (d.ref_qpos && S.ghost) { poseOne(S.ghostRobot, d.ref_qpos); S.ghostRobot.visible = true; }
          else S.ghostRobot.visible = false;
        }
        if (!S._liveFramed) { viewBehind(d.qpos); S._liveFramed = true; }   // 3/4-behind the live robot, once
        // Integer Hz: the fractional digit is noise.
        const _hz = Math.round(d.hz || 0);
        $("srclabel").textContent = `LIVE · ${_hz} Hz${d.tracker_ready ? "" : " · warming"}`;
        $("modeBadgeT").textContent = `LIVE ${_hz}Hz`;
        liveEgo();                                          // refresh the ego image (throttled)
        liveDepth();
        livePlan();
        liveObjects();
      } else {
        // No tracker state: only the 3D pose needs it; the ego image comes from the generator.
        $("srclabel").textContent = "LIVE · waiting for tracker…";
        $("modeBadgeT").textContent = "LIVE · waiting";
        liveEgo();
        liveDepth();
        livePlan();
      }
      setTimeout(poll, 33);                   // ~30 Hz browser refresh
    }).catch(() => { $("srclabel").textContent = "LIVE · reconnecting…"; setTimeout(poll, 400); });
  }
}
// Free bodies (the box) follow the sim: /api/objects = the tracker's live poses (sim only; 204 on
// the real robot). Polled at <= 10 Hz.
S.objMeshes = {};                 // body name -> [mesh, ...]
function liveObjects() {
  const now = performance.now();
  if (S._objBusy || now - (S._objT || 0) < 100 || !Object.keys(S.objMeshes).length) return;
  S._objBusy = true; S._objT = now;
  fetch("/api/objects").then((r) => (r.status === 200 ? r.json() : null)).then((d) => {
    for (const [name, o] of Object.entries((d && d.objects) || {})) {
      for (const mesh of S.objMeshes[name] || []) {
        mesh.position.set(o.pos[0], o.pos[1], o.pos[2]);
        mesh.quaternion.set(o.quat[0], o.quat[1], o.quat[2], o.quat[3]);
        mesh.visible = !o.hidden;
      }
    }
  }).catch(() => {}).finally(() => { S._objBusy = false; });
}
// Depth window (/api/live_depth = tracker g1/depth, depth policies only). Sits directly under the
// ego window, same width, until the operator drags it.
function placeDepth() {
  const d = $("depthfloat"), e = $("egofloat");
  if (!d || !e || d._dragged || d.style.display === "none") return;
  if (getComputedStyle(d).position !== "fixed") return;
  const r = e.getBoundingClientRect();
  if (!r.width) return;
  d.style.right = "auto"; d.style.bottom = "auto";
  d.style.left = r.left + "px"; d.style.width = r.width + "px";
  d.style.top = Math.min(innerHeight - 60, r.bottom + 10) + "px";
}
addEventListener("resize", placeDepth);
(() => {
  const e = document.getElementById("egofloat");
  if (!e) return;
  if (window.ResizeObserver) new ResizeObserver(() => placeDepth()).observe(e);
  const i = document.getElementById("obsimg");
  if (i) i.addEventListener("load", placeDepth);
})();
function depthShow(src, msg) {
  const d = $("depthfloat"), img = $("depthimg"), em = $("depthempty");
  if (!d) return;
  if (src) { img.src = src; img.style.display = ""; em.style.display = "none"; }
  else { img.style.display = "none"; em.textContent = msg || "No depth"; em.style.display = ""; }
  d.style.display = "";
  placeDepth();
}
function liveDepth() {
  const now = performance.now();
  // 150 ms once depth is flowing; 1 s while waiting (runs without a depth policy never send any).
  const gap = S._liveDepthGot ? 150 : 1000;
  if (S._liveDepthBusy || now - (S._liveDepthT || 0) < gap) return;
  if (winFolded("depthfloat") && S._liveDepthGot) return;   // folded: no downloads
  S._liveDepthT = now; S._liveDepthBusy = true;
  fetch("/api/live_depth?t=" + now).then((r) => (r.status === 200 ? r.blob() : null)).then((b) => {
    S._liveDepthBusy = false;
    const img = $("depthimg");
    if (b && b.size) {
      const url = URL.createObjectURL(b);
      if (img._url) URL.revokeObjectURL(img._url);
      img._url = url;
      S._liveDepthGot = true;
      depthShow(url);
    } else if (S._liveDepthGot) {
      depthShow(null, "No depth");                  // had depth, the stream stopped
    } else {
      $("depthfloat").style.display = "none";       // no depth yet: stay hidden
    }
  }).catch(() => { S._liveDepthBusy = false; });
}
// Throttled ego-image refresh (~8 Hz) via /api/live_image. Server prefers the
// generator's g1/obs topic (sim + real), falls back to get_state (sim render).
function liveEgo() {
  const now = performance.now();
  if (S._liveImgBusy || now - (S._liveImgT || 0) < 120) return;
  if (winFolded("egofloat")) return;                          // folded: no downloads
  S._liveImgT = now; S._liveImgBusy = true;
  // Request the panel's pixel width (server downscales; ~20 KB vs ~120 KB per frame).
  const pw = (($("obswrap") || {}).clientWidth || 360) * (devicePixelRatio || 1);
  fetch("/api/live_image?t=" + now + "&w=" + Math.round(pw)).then((r) => r.ok ? r.blob() : null).then((b) => {
    S._liveImgBusy = false;
    if (b && b.size) {
      const url = URL.createObjectURL(b);
      const img = $("obsimg");
      if (img._url) URL.revokeObjectURL(img._url); img._url = url;
      img.src = url; img.style.display = ""; $("obsempty").style.display = "none";
      S._liveImgOkT = performance.now();
    } else if (S._liveImgOkT && performance.now() - S._liveImgOkT < 8000) {
      // No new frame (204): keep the last image rather than flashing the panel.
    } else { $("obsempty").textContent = "No image"; $("obsempty").style.display = ""; $("obsimg").style.display = "none"; }
  }).catch(() => { S._liveImgBusy = false; });
}
// Live plan overlay ("plan" toggle): newest generator plan (world frame) at ~4 Hz, drawn as
// strobes plus the moving robot at the frame matching the time since the plan was made.
function livePlan() {
  const now = performance.now();
  const hide = () => { if (S.overlayRobot) S.overlayRobot.visible = false; (S.strobes || []).forEach((r) => { r.visible = false; }); S._livePlanIdx = null; };
  if (!S.overlay) { hide(); return; }
  if (S._livePlan && S.overlayRobot) {                // animate between fetches
    const pl = S._livePlan, n = pl.qpos.length;
    const k = Math.round(Math.max(0, (Date.now() / 1000 + (S._livePlanSkew || 0)) - pl.stamp) * (pl.fps || 30));
    poseOne(S.overlayRobot, pl.qpos[Math.max(0, Math.min(n - 1, k))]); S.overlayRobot.visible = true;
  }
  if (S._livePlanBusy || (S._livePlanT != null && now - S._livePlanT < 250)) return;
  S._livePlanT = now; S._livePlanBusy = true;
  fetch("/api/live_plan").then((r) => r.json()).then((j) => {
    S._livePlanBusy = false;
    if (!S.overlay) return;
    if (!j || !j.plan || !j.plan.qpos || j.plan.qpos.length < 2) { S._livePlan = null; hide(); return; }
    S._livePlanSkew = (j.now || Date.now() / 1000) - Date.now() / 1000;   // server clock = plan stamps
    S._livePlan = j.plan;
    if (S._livePlanIdx === j.plan.idx) return;
    S._livePlanIdx = j.plan.idx;
    const st = S.strobes || [], n = j.plan.qpos.length;
    st.forEach((r, i) => { poseOne(r, j.plan.qpos[Math.round(i * (n - 1) / Math.max(1, st.length - 1))]); r.visible = true; });
  }).catch(() => { S._livePlanBusy = false; });
}

// ── mission orchestrator stream
// LIVE polls /api/live_mission (~2 Hz) for the orchestrator's g1/mission decisions.
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function missionReset() {
  S._misSeq = -1; S._planEl = null; S._lastPlanReply = null; S._lastPlanVer = null; S._lastDecSay = null;
  $("msgs").innerHTML = `<div class="empty" id="msgsEmpty">Connecting…</div>`;
}
// operator message — a boxed turn with a left lavender bar (terminal transcript)
function addHumanBubble(text) {
  const empty = $("msgsEmpty"); if (empty) empty.remove();
  const host = $("msgs"), el = document.createElement("div");
  el.className = "turn user";
  el.innerHTML = `<div class="ubox">${escapeHtml(text)}</div>`;
  finalizePrevPlanWidget();   // a new message interrupts the current mission -> freeze its running step as cancelled
  host.appendChild(el);
  newPlanWidget();           // this request gets its OWN checklist, right below the message
  // Anti-flash: baseline = the plan live before this message, so the previous plan isn't painted here.
  if (S._planEl) S._planEl._baseVer = (typeof S._lastPlanVer === "number") ? S._lastPlanVer : null;
  if (S._planEl) S._planEl._baseSig = S._lastPlanSig || null;
  host.scrollTop = host.scrollHeight;
}
// A new operator message interrupts the running plan: mark its running steps cancelled.
function finalizePrevPlanWidget() {
  const el = S._planEl;
  if (!el) return;
  const running = el.querySelectorAll(".pstep.in_progress");
  if (!running.length) return;
  running.forEach((p) => {
    p.classList.remove("in_progress"); p.classList.add("cancelled");
    const pm = p.querySelector(".pmark"); if (pm) pm.textContent = "✕";
  });
  el.classList.add("stopped"); el.classList.remove("waiting", "finished");
  const head = el.querySelector(".plan-prog"); if (head) head.textContent = "cancelled";
  el._sig = "final";   // it is no longer S._planEl after newPlanWidget(); keep it frozen
}
// A step-decision line (the orchestrator's `say` for one decision): small, with a dot, not a bubble.
function _sayTurnEl(m, line) {
  const el = document.createElement("div"); el.className = "turn agent say";
  el.style.setProperty("--st", "var(--blue)");
  const tag = [m.action, m.subtask].filter(Boolean).join(" · ");
  el.innerHTML = `<div class="tline"><span class="tl-task">[${escapeHtml(tag || "step")}]</span>` +
    `<span class="tl-say">${escapeHtml(line)}</span></div>`;
  return el;
}
function applyMissionSnap(snap) {
  if (snap.error) return;
  const hist = snap.history || [];
  // Track the newest decision for the Thinking box (steps are shown by the plan checklist).
  let latest = null;
  hist.forEach((d) => {
    if (d.seq > S._misSeq) { S._misSeq = d.seq; latest = d; }
  });
  // Live plan version: a new message's widget uses it as its anti-flash baseline.
  const _wpv = snap.plan && snap.plan.plan && snap.plan.plan.version;
  if (typeof _wpv === "number") S._lastPlanVer = _wpv;
  if (snap.plan && snap.plan.plan) S._lastPlanSig = planAuthorSig(snap.plan.plan);
  applyDecisions(snap.decisions); // the orchestrator's per-step lines (`say`), one at a time
  maybeAddReplyBubble(snap);     // the robot's reply (T1) -> one chat message
  renderPlan(snap.plan, snap);   // checkpoint checklist (live; hidden when g1/plan is stale)
  setEmptyState(snap);   // "connecting…" until the orchestrator is up + waiting, then "say hi"
  updateLiveThink(latest || (hist.length ? hist[hist.length - 1] : null), snap);
}
S._pageOpenedMs = Date.now();
// Live step decisions: repeats dropped, a run shows only its newest line.
function applyDecisions(list) {
  if (!Array.isArray(list)) return;
  const first = !S._decSeen; S._decSeen = S._decSeen || new Set();
  const host = $("msgs");
  for (const m of list) {
    const key = (m.stamp || "") + ":" + m.say;
    if (S._decSeen.has(key)) continue;
    S._decSeen.add(key);
    const fresh = !first || (m.stamp && m.stamp * 1000 >= S._pageOpenedMs - 2000);
    const line = String(m.say || "").trim();
    if (!fresh || !line || line === S._lastDecSay) continue;
    S._lastDecSay = line;
    const empty = $("msgsEmpty"); if (empty) empty.remove();
    const nearBottom = host.scrollHeight - host.scrollTop - host.clientHeight < 120;
    const lastEl = host.lastElementChild;
    if (lastEl && lastEl.classList.contains("say")) lastEl.remove();
    host.appendChild(_sayTurnEl(m, line));
    if (nearBottom) host.scrollTop = host.scrollHeight;
  }
}
// T1's reply as one chat message, once per authored answer (shown above the plan checklist).
function maybeAddReplyBubble(snap) {
  if (!snap || !snap.connected) return;            // don't resurrect a dead orchestrator's last reply
  const del = snap.deliberation;
  if (!del || !del.authored) return;
  const reply = String(del.reply || "").trim();
  if (!reply) return;                                // no reply -> nothing to show
  // Show the PLAN's stated reply once per authored answer (dedup on _lastPlanReply).
  if (S._lastPlanReply === reply) return;
  S._lastPlanReply = reply;
  addReplyBubble(reply);
}
// Agent reply turn, inserted above this message's plan widget (message -> reply -> checklist).
function addReplyBubble(text) {
  const empty = $("msgsEmpty"); if (empty) empty.remove();
  const host = $("msgs"), el = document.createElement("div");
  el.className = "turn agent reply";
  el.style.setProperty("--st", "var(--blue)");
  el.innerHTML = `<div class="tline"><span class="tl-task">[reply]</span>` +
    `<span class="tl-say">${escapeHtml(text)}</span></div>`;
  const nearBottom = host.scrollHeight - host.scrollTop - host.clientHeight < 120;
  // before this turn's plan widget, else append
  if (S._planEl && S._planEl.parentNode === host) host.insertBefore(el, S._planEl);
  else host.appendChild(el);
  if (nearBottom) host.scrollTop = host.scrollHeight;
}
// One plan checklist per operator message. planAuthorSig: identity of an authored plan
// (reply + mission + ordered skills), unchanged by step-status updates.
function planAuthorSig(plan) {
  if (!plan) return null;
  const sk = (plan.steps || []).map((s) => s.subtask || "").join(",");
  return [plan.run_id || "", plan.mission || "", plan.reply || "", sk].join("\u241e");
}
function newPlanWidget() {
  const el = document.createElement("div");
  el.className = "planbox"; el.style.display = "none";
  // Anti-flash baseline: paint only versions > _baseVer. Null for lazily created widgets (else
  // the incoming plan's own version would hide it forever); the typed path re-stamps it.
  el._baseVer = null;
  el.innerHTML = `<div class="plan-head"><span class="plan-title">Mission plan</span>` +
                 `<span class="plan-prog mono"></span></div>` +
                 `<div class="plan-bar"><i></i></div>` +
                 `<div class="plan-steps"></div>`;
  $("msgs").appendChild(el);
  S._planEl = el;
  return el;
}
// Max age (s since receipt of g1/plan, heartbeated ~1 Hz) for a live plan to count as current.
const PLAN_STALE_S = 30;
function renderPlan(env, snap) {
  const plan = env && env.plan;
  const steps = (plan && plan.steps) || [];
  // Zero steps is normal before the first plan: keep the widget and show a waiting state.
  const box = S._planEl || newPlanWidget();
  // Hide until this message's own plan (version > _baseVer) arrives.
  const _pver = plan && typeof plan.version === "number" ? plan.version : null;
  if (box._baseVer != null && _pver != null && _pver <= box._baseVer) {
    box.style.display = "none"; box._sig = null;
    return;
  }
  // Same authored plan as when the message was sent (versions bump on status changes): hide.
  if (box._baseSig != null && plan && planAuthorSig(plan) === box._baseSig) {
    box.style.display = "none"; box._sig = null;
    return;
  }
  // No mission yet (bootstrap plan) -> no panel; a mission with no steps means T1 is authoring.
  const hasMission = !!(plan && String(plan.mission || "").trim());
  if (!hasMission && !steps.length) {
    box.style.display = "none"; box._sig = null;
    return;
  }
  // Hide a plan whose g1/plan feed has gone quiet (age_s >= PLAN_STALE_S).
  const age = env && env.age_s;
  if (!(snap && snap.connected && age != null && age < PLAN_STALE_S)) {
    box.style.display = "none"; box._sig = null;
    return;
  }
  const total = steps.length;
  const done = steps.filter((s) => s.status === "completed").length;
  const cur = steps.find((s) => s.status === "in_progress");
  // Conversational answer (authored, zero steps) -> no panel. Must precede the sig dedup below (the
  // plan is republished ~1 Hz). Unauthored zero steps = T1 still authoring -> "planning…" below.
  if (!total && plan && plan.authored) {
    box.style.display = "none"; box._sig = null;
    return;
  }
  box.style.display = "";
  // Operator stop: /stop publishes a hold without mutating the plan; show the running step cancelled.
  const stopped = !!(snap && snap.current && snap.current.subtask === "hold"
                     && /stopped/i.test(String(snap.current.reason || "")));
  // Dedup key includes plan identity (run_id + revision) and the stop flag, not just the statuses.
  const sig = (plan && plan.run_id || "") + ":" + (plan && plan.revision != null ? plan.revision : "")
    + ":" + (plan && plan.authored ? "a" : "u")
    + ":" + steps.map((s) => s.status[0]).join("") + (plan && plan.done ? "1" : "0")
    + ":" + (stopped ? "s" : "");
  if (box._sig === sig) return;   // no change -> don't rebuild (keeps the spinner smooth)
  box._sig = sig;

  const head = box.querySelector(".plan-prog");
  const bar = box.querySelector(".plan-bar i");
  const body = box.querySelector(".plan-steps");
  if (!total) {
    // Unauthored zero-step state: T1 is authoring, so show a waiting frame.
    box.classList.add("waiting");
    box.classList.remove("finished");
    head.textContent = "planning…";
    bar.style.width = "0%";
    body.innerHTML =
      `<div class="pstep waiting"><span class="pmark"><i></i></span>` +
      `<span class="ptitle">Waiting for the plan…</span></div>`;
    return;
  }
  const finished = !!(plan && plan.done) || done === total;
  box.classList.toggle("finished", finished);
  box.classList.toggle("stopped", stopped && !finished);
  box.classList.remove("waiting");
  head.textContent = (stopped && !finished) ? "stopped"
    : `${done}/${total}` + (finished ? " · done" : "");
  // The bar gives the in-progress step half credit; the count is completed steps only.
  const prog = total ? (done + (cur ? 0.5 : 0)) / total : 0;
  bar.style.width = Math.min(100, prog * 100) + "%";
  const mark = (st) => st === "completed" ? "✓" : st === "failed" ? "✗" : st === "skipped" ? "↷"
    : st === "cancelled" ? "✕" : st === "in_progress" ? '<i></i>' : "○";
  body.innerHTML = steps.map((s, i) => {
    // Under an operator stop, the running step reads as CANCELLED (a ✕), not a live spinner.
    const st = (stopped && !finished && s.status === "in_progress") ? "cancelled" : s.status;
    return `<div class="pstep ${st}"><span class="pmark">${mark(st)}</span>` +
      `<span class="pnum mono">${i + 1}</span>` +
      `<span class="ptitle">${escapeHtml(s.title || s.subtask)}</span>` +
      (s.subtask && s.title ? `<span class="psk mono">${escapeHtml(s.subtask)}</span>` : "") +
      `</div>`;
  }).join("");
}
// "connecting…" until the orchestrator heartbeat is fresh (keyed on heartbeat_age_s, not the sticky
// `snap.have`), then invite the operator. Only touches the empty placeholder.
function setEmptyState(snap) {
  const e = $("msgsEmpty");
  if (!e) return;
  const live = !!(snap && snap.connected && snap.heartbeat_age_s != null);
  const ready = live && !!(snap.have || (snap.current && snap.current.subtask === "hold"));
  e.innerHTML = ready
    ? `Tell the robot what to do.`
    : `Connecting…`;
}
function pollMission() {
  fetch("/api/live_mission").then((x) => x.json()).then((snap) => {
    applyMissionSnap(snap); setTimeout(pollMission, 500);
  }).catch(() => { setTimeout(pollMission, 900); });
}

// Fade the thinking text on change.
function setThinkText(txt) {
  const el = $("liveThinkText");
  if (!el || el.textContent === txt) return;
  el.style.opacity = "0.25";
  el.textContent = txt;
  requestAnimationFrame(() => requestAnimationFrame(() => { el.style.opacity = "1"; }));
}
// Fade the old line out, then type the new one in.
let _twTimer = null;
function streamThinkText(txt) {
  const el = $("liveThinkText");
  if (!el) return;
  if (_twTimer) { clearInterval(_twTimer); _twTimer = null; }
  el.style.opacity = "0";                       // fade the current thought away
  setTimeout(() => {
    el.textContent = ""; el.style.opacity = "1";
    const full = txt || "";
    // ~22 chunks at 12 ms -> ~260 ms, inside one 1 Hz tick.
    const step = Math.max(2, Math.ceil(full.length / 22));
    let i = 0;
    _twTimer = setInterval(() => {
      i = Math.min(full.length, i + step);
      el.textContent = full.slice(0, i);
      if (i >= full.length) { clearInterval(_twTimer); _twTimer = null; }
    }, 12);
  }, 110);
}
// Re-trigger the tick pulse on each new decision, even if the text is unchanged.
function pulseThink() {
  const box = $("liveThink");
  if (!box || box.style.display === "none") return;
  box.classList.remove("tick"); void box.offsetWidth; box.classList.add("tick");
}
function del_ver(snap) { const d = snap && snap.deliberation; return d && d.plan_version != null ? d.plan_version : null; }
function updateLiveThink(d, snap) {
  const box = $("liveThink");
  if (!box) return;
  if (!d) { box.style.display = "none"; S._thinkSeq = -1; return; }
  const cur = snap && snap.current;
  const holding = !!(cur && cur.subtask === "hold");
  // Deliberation and /stop holds both publish `hold`; the orchestrator prefixes the gate's reason with
  // "Deliberating" (a contract with the orchestrator node).
  const gating = holding && /^Deliberating/.test(String(cur.reason || ""));
  const ver = del_ver(snap);
  if (S._optThink) {
    const age = Date.now() - S._optThink.t;
    const planLanded = ver != null && ver !== S._optThink.ver;
    const answered = !!cur && String(cur.reason || "") !== S._optThink.reason && !gating && age > 1500;
    if (age > 30000 || planLanded || answered) S._optThink = null;
  }
  S._lastDelVer = ver; S._lastCurReason = cur ? String(cur.reason || "") : null;
  // Only while a request is being planned (sent -> plan landed); otherwise no line at all.
  const _pending = !!(snap && snap.deliberation && snap.deliberation.authored === false);
  if (!(gating || S._optThink || _pending)) { box.style.display = "none"; S._thinkSeq = -1; return; }
  box.style.display = "flex";
  box.classList.toggle("hold", holding);
  const del = snap && snap.deliberation;
  // planning -> plan authored.
  let txt, seq;
  if (gating || S._optThink) {
    txt = "The robot is thinking and planning…"; seq = "think:v" + ((del && del.plan_version) ?? "?");
  } else if (holding) {
    setThinkText("holding — waiting for you…"); S._thinkSeq = "hold"; return;
  } else if (del && del.authored === false) {
    txt = "The robot is thinking and planning…"; seq = "think:pending";
  }
  if (seq !== S._thinkSeq) { S._thinkSeq = seq; pulseThink(); streamThinkText(txt); }
  else setThinkText(txt);
}

// Typed operator input: shown as an operator turn and appended to the orchestrator's inbox.
function sendText(text) {
  if (!text) return;
  addHumanBubble(text);
  postMissionInput(text);
}
// POST typed text to the orchestrator's inbox and show the thinking pill.
function postMissionInput(text) {
  if (!text) return;
  startOptimisticThink();
  fetch("/api/mission_input", { method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ text }) })
    .then((r) => r.json()).then((j) => { if (j.error) toast("send failed: " + j.error); })
    .catch((e) => toast("send failed: " + e));
}
// Optimistic thinking: show the pill on send; updateLiveThink ends it when the plan lands, the
// orchestrator answers some other way, or after 30 s.
function startOptimisticThink() {
  S._optThink = { t: Date.now(), ver: S._lastDelVer, reason: S._lastCurReason };
  const box = $("liveThink");
  if (box) { box.style.display = "flex"; box.classList.add("hold"); }
  S._thinkSeq = "think:optimistic"; pulseThink(); setThinkText("The robot is thinking and planning…");
}
function sendHumanMsg() {
  const inp = $("humanMsg");
  const text = (inp.value || "").trim();
  if (!text) return;
  inp.value = "";
  sendText(text);
}

wire(); initThree(); startLive();
