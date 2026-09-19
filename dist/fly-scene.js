import * as THREE from "./vendor/three.module.min.js";

// Anatomical meshes; only the button gesture is animated here.
export class FlyScene {
  constructor(canvas) {
    this.canvas = canvas;
    this.ready = false;
    this.parts = new Map();
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#111a20");
    this.camera = new THREE.PerspectiveCamera(36, 1, .1, 100);
    this.yaw = .45;
    this.pitch = .38;
    this.scene.add(new THREE.HemisphereLight(0xd9edff, 0x463021, 2.6));
    const light = new THREE.DirectionalLight(0xffe4b5, 4);
    light.position.set(2, 7, 5); light.castShadow = true;
    light.shadow.mapSize.set(1024, 1024);
    light.shadow.camera.left = -5; light.shadow.camera.right = 5;
    light.shadow.camera.top = 5; light.shadow.camera.bottom = -5;
    light.shadow.normalBias = .035;
    this.scene.add(light);
    const rim = new THREE.DirectionalLight(0x78d2e8, 2.3);
    rim.position.set(-3, 3, -4); this.scene.add(rim);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshStandardMaterial({ color: 0x19262d, roughness: .9 }));
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true;
    this.scene.add(ground);
    const grid = new THREE.GridHelper(16, 32, 0x2d424a, 0x22323a);
    grid.position.y = .001; grid.material.transparent = true; grid.material.opacity = .42;
    this.scene.add(grid);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    let drag = null;
    canvas.addEventListener("pointerdown", e => {
      drag = {x:e.clientX, y:e.clientY}; canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", e => {
      if (!drag) return;
      this.yaw -= (e.clientX - drag.x) * .009;
      this.pitch = THREE.MathUtils.clamp(this.pitch + (e.clientY - drag.y) * .006, .12, .9);
      drag = {x:e.clientX, y:e.clientY};
    });
    canvas.addEventListener("pointerup", () => { drag = null; });
    canvas.addEventListener("pointercancel", () => { drag = null; });
    canvas.addEventListener("keydown", e => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home"].includes(e.key)) return;
      e.preventDefault(); e.stopPropagation();
      if (e.key === "Home") { this.yaw = .45; this.pitch = .38; }
      if (e.key === "ArrowLeft") this.yaw -= .12;
      if (e.key === "ArrowRight") this.yaw += .12;
      if (e.key === "ArrowUp") this.pitch = Math.min(.9, this.pitch + .08);
      if (e.key === "ArrowDown") this.pitch = Math.max(.12, this.pitch - .08);
    });
    this.resize();
  }
  resize() {
    const {width, height} = this.canvas.getBoundingClientRect();
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }
  async load() {
    const [metaResponse, binaryResponse] = await Promise.all([fetch("assets/flybody/fly.json"), fetch("assets/flybody/fly.bin")]);
    if (!metaResponse.ok || !binaryResponse.ok) throw new Error("Fly model could not load");
    const meta = await metaResponse.json(), binary = await binaryResponse.arrayBuffer();
    const materials = Object.fromEntries(Object.entries(meta.materials).map(([name, rgba]) => [name, new THREE.MeshStandardMaterial({
      color: new THREE.Color(...rgba.slice(0, 3)), roughness: name === "red" ? .28 : .56,
      metalness: .06, transparent: rgba[3] < 1, opacity: rgba[3], side: THREE.DoubleSide,
      depthWrite: rgba[3] === 1,
    })]));
    const geometries = Object.fromEntries(Object.entries(meta.meshes).map(([name, data]) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(binary, data.positions.offset, data.positions.count), 3));
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(binary, data.indices.offset, data.indices.count), 1));
      geometry.computeVertexNormals();
      return [name, geometry];
    }));
    function transform(object, data) {
      object.position.fromArray(data.position);
      const [w, x, y, z] = data.quaternion;
      object.quaternion.set(x, y, z, w).normalize();
    }
    const build = data => {
      const group = new THREE.Group(); group.name = data.name;
      transform(group, data); this.parts.set(data.name, group);
      for (const part of data.meshes) {
        const mesh = new THREE.Mesh(geometries[part.mesh], materials[part.material]);
        transform(mesh, part); mesh.castShadow = part.material !== "membrane"; mesh.receiveShadow = true;
        group.add(mesh);
      }
      for (const child of data.children) group.add(build(child));
      return group;
    };
    this.fly = new THREE.Group();
    this.fly.rotation.x = -Math.PI / 2;
    this.fly.scale.setScalar(13);
    this.fly.add(build(meta.body)); this.scene.add(this.fly);
    // Place the original resting pose on the table, then lift one foreleg.
    this.fly.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(this.fly);
    this.fly.position.y = -bounds.min.y + .03;
    this.fly.updateMatrixWorld(true);
    this.tip = this.parts.get("claw_T1_right");
    this.joints = ["tibia_T1_right", "femur_T1_right", "coxa_T1_right"].map(name => this.parts.get(name));
    if (!this.tip || this.joints.some(j => !j)) throw new Error("Fly foreleg is missing");
    const foot = this.tip.getWorldPosition(new THREE.Vector3());
    this.buttonPosition = new THREE.Vector3(foot.x + .25, .43, foot.z);
    this.makeController();
    this.ready = true;
    this.resize();
  }
  makeController() {
    const material = new THREE.MeshStandardMaterial({color:0x34454e, roughness:.48, metalness:.35});
    const deck = new THREE.Mesh(new THREE.BoxGeometry(1.1, .26, 1.35), material);
    deck.position.copy(this.buttonPosition); deck.position.y = .16; deck.position.z -= .32;
    deck.castShadow = true; deck.receiveShadow = true; this.scene.add(deck);
    const rim = new THREE.Mesh(new THREE.CylinderGeometry(.29, .32, .10, 48), new THREE.MeshStandardMaterial({color:0x11191e, metalness:.65, roughness:.3}));
    rim.position.copy(this.buttonPosition); rim.position.y = .32; this.scene.add(rim);
    this.button = new THREE.Mesh(new THREE.CylinderGeometry(.24, .26, .14, 48), new THREE.MeshStandardMaterial({color:0xff704d, roughness:.3, emissive:0xff4526, emissiveIntensity:.08}));
    this.button.position.copy(this.buttonPosition); this.button.castShadow = true; this.scene.add(this.button);
    const stick = new THREE.Mesh(new THREE.CylinderGeometry(.035, .035, .35, 16), new THREE.MeshStandardMaterial({color:0x99a9ac, metalness:.7, roughness:.2}));
    stick.position.set(deck.position.x, .43, deck.position.z - .35); this.scene.add(stick);
    const ball = new THREE.Mesh(new THREE.SphereGeometry(.13, 24, 16), new THREE.MeshStandardMaterial({color:0x131e24, roughness:.32}));
    ball.position.copy(stick.position); ball.position.y = .64; this.scene.add(ball);
    const cable = new THREE.CatmullRomCurve3([new THREE.Vector3(deck.position.x + .55,.12,deck.position.z),new THREE.Vector3(deck.position.x+1,.05,deck.position.z),new THREE.Vector3(deck.position.x+1.8,.04,deck.position.z-.6),new THREE.Vector3(4,.04,-.8)]);
    this.scene.add(new THREE.Mesh(new THREE.TubeGeometry(cable, 32, .025, 8, false), new THREE.MeshStandardMaterial({color:0x090f13})));
  }
  reach(target) {
    // CCD IK moves the original articulated foreleg to the button.
    for (let iteration = 0; iteration < 7; iteration++) {
      for (const joint of this.joints) {
        const localTip = joint.worldToLocal(this.tip.getWorldPosition(new THREE.Vector3()));
        const localTarget = joint.worldToLocal(target.clone());
        const rotation = new THREE.Quaternion().setFromUnitVectors(localTip.normalize(), localTarget.normalize());
        joint.quaternion.multiply(rotation);
        joint.updateMatrixWorld(true);
      }
    }
  }
  update({pressAge = Infinity, paused = false} = {}) {
    if (!this.canvas.clientWidth || !this.canvas.clientHeight) return;
    this.camera.position.set(Math.sin(this.yaw) * 8, 1.0 + Math.sin(this.pitch) * 7, Math.cos(this.yaw) * 8);
    this.camera.lookAt(.0, .85, 0);
    if (this.ready) {
      // A contact begins in the same physics frame as the flap decision.
      const contact = Math.max(0, 1 - pressAge / .16);
      this.button.position.y = this.buttonPosition.y - contact * .065;
      this.button.material.emissiveIntensity = .08 + contact * .7;
      const target = this.buttonPosition.clone();
      target.y += .1 + (1 - contact) * .24 - contact * .065;
      this.reach(target);
      this.contact = contact;
      this.paused = paused;
    }
    this.renderer.render(this.scene, this.camera);
  }
}
