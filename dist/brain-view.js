const positive = [104, 230, 210], negative = [255, 153, 108];
const color = (value, alpha = 1) => `rgba(${(value >= 0 ? positive : negative).join(',')},${alpha})`;

export class BrainView {
  constructor(canvas, inspector) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.inspector = inspector;
    this.yaw = -.2; this.pitch = .12; this.selected = null; this.hovered = null;
    this.projected = []; this.sample = null; this.pass = 3;
    this.surfaceCanvas = document.createElement('canvas');
    this.surfaceCanvas.width = 640; this.surfaceCanvas.height = 410;
    this.surfaceReady = fetch('assets/brain-shell.json').then(r => {
      if (!r.ok) throw new Error('Brain surface could not load');
      return r.json();
    }).then(surface => { this.surface = surface; if (this.graph) this.setGraph(this.graph); })
      .catch(() => { this.surfaceError = true; });
    let drag = null, moved = false;
    canvas.addEventListener('pointerdown', e => { drag = {x:e.clientX, y:e.clientY}; moved = false; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => {
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width * 640, y = (e.clientY - rect.top) / rect.height * 410;
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
        this.yaw += dx * .008; this.pitch = Math.max(-1, Math.min(1, this.pitch + dy * .006));
        drag = {x:e.clientX, y:e.clientY};
      }
      this.hovered = this.projected.map((p,i) => ({i, d:Math.hypot(p.x-x,p.y-y)})).sort((a,b) => a.d-b.d)[0];
      if (this.hovered?.d > 16) this.hovered = null;
      canvas.style.cursor = drag ? 'grabbing' : this.hovered ? 'pointer' : 'grab';
    });
    canvas.addEventListener('pointerup', () => { if (!moved) this.selected = this.hovered?.i === this.selected ? null : this.hovered?.i ?? null; drag = null; });
    canvas.addEventListener('pointercancel', () => { drag = null; });
    canvas.addEventListener('pointerleave', () => { this.hovered = null; });
    canvas.addEventListener('keydown', e => {
      if (e.key === 'Escape') this.selected = null;
      if (e.key === 'Home') {this.yaw=-.2;this.pitch=.12;this.selected=null;}
      if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','Escape'].includes(e.key)) return;
      e.preventDefault(); e.stopPropagation();
      if(e.key==='ArrowLeft')this.yaw-=.12;
      if(e.key==='ArrowRight')this.yaw+=.12;
      if (!this.graph) return;
      if(e.key==='ArrowUp')this.selected=((this.selected??0)+1)%this.graph.nodes.length;
      if(e.key==='ArrowDown')this.selected=((this.selected??1)-1+this.graph.nodes.length)%this.graph.nodes.length;
    });
  }
  setGraph(graph) {
    this.graph = graph;
    const coords = this.surface?.vertices || graph.nodes.map(n => n.somaLocation);
    const min = [0,1,2].map(axis=>Math.min(...coords.map(p=>p[axis]))), max = [0,1,2].map(axis=>Math.max(...coords.map(p=>p[axis])));
    const scale = Math.max(...max.map((v,i)=>v-min[i]));
    const normalize = p => [(p[0]-(min[0]+max[0])/2)/scale, (p[2]-(min[2]+max[2])/2)/scale, -(p[1]-(min[1]+max[1])/2)/scale];
    this.points = graph.nodes.map(n=>normalize(n.somaLocation));
    this.surfacePoints = this.surface?.vertices.map(normalize);
    this.surfaceKey = null;
    this.inputs = new Set(graph.inputs); this.outputs = new Set(graph.outputs);
  }
  update(sample, controller) { this.sample=sample; this.controller=controller; }
  clear() { this.sample=null; this.selected=null; this.hovered=null; }
  draw(time, live) {
    const ctx = this.ctx;
    ctx.clearRect(0,0,640,410);
    ctx.fillStyle='#111a20'; ctx.fillRect(0,0,640,410);
    if(!this.graph)return;
    const mlp=this.controller?.kind==='mlp';
    const values=mlp ? this.sample?.activity || [] : this.sample?.passes?.[this.pass] || this.sample?.activity || [];
    const signals=this.sample?.edgePasses?.[this.pass] || this.sample?.edgeActivity || [];
    const points=mlp ? Array.from({length:this.controller.w['hidden.bias'].length},(_,i)=>[(i%7-3)/7,(Math.floor(i/7)-2.5)/7,0]) : this.points;
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    const project=([x,y,z])=>{
      const rx=x*cy+z*sy,rz=-x*sy+z*cy,ry=y*cp-rz*sp,depth=y*sp+rz*cp;
      const perspective=1/(1+depth*.4);
      return {x:320+rx*540*perspective,y:200+ry*540*perspective,z:depth,scale:perspective};
    };
    this.projected=points.map(project);
    if (!mlp && this.surfacePoints) {
      const key = `${this.yaw}:${this.pitch}`;
      if (key !== this.surfaceKey) {
        this.surfaceKey = key;
        const sc = this.surfaceCanvas.getContext('2d');
        sc.clearRect(0,0,640,410);
        const projected = this.surfacePoints.map(project);
        const faces = this.surface.faces.map(face => ({face,depth:face.reduce((sum,i)=>sum+projected[i].z,0)/3})).sort((a,b)=>b.depth-a.depth);
        for (const {face} of faces) {
          const [a,b,c] = face.map(i=>projected[i]);
          const normal = (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
          // Opaque muted surface underneath a transparent live circuit overlay.
          const light = Math.min(1,Math.abs(normal)/110);
          sc.fillStyle = `rgb(${28+Math.round(light*20)},${49+Math.round(light*23)},${61+Math.round(light*29)})`;
          sc.strokeStyle='rgba(127,179,199,.065)'; sc.lineWidth=.45;
          sc.beginPath();sc.moveTo(a.x,a.y);sc.lineTo(b.x,b.y);sc.lineTo(c.x,c.y);sc.closePath();sc.fill();sc.stroke();
        }
      }
      ctx.drawImage(this.surfaceCanvas,0,0);
    }
    const focus=this.selected??this.hovered?.i;
    const src=mlp?[]:this.controller?.w.src || this.graph.edges.map(e=>e.source_index);
    const dst=mlp?[]:this.controller?.w.dst || this.graph.edges.map(e=>e.target_index);
    const maxSignal=Math.max(.001,...signals.map(Math.abs));
    let strongest=-1;
    for(let e=0;e<src.length;e++){
      const a=this.projected[src[e]], b=this.projected[dst[e]];
      if(!a||!b)continue;
      const cut=this.controller?.intervention==='lesion'&&this.outputs.has(dst[e]);
      if(cut)continue;
      const connected=focus==null||src[e]===focus||dst[e]===focus;
      const signal=this.sample?.bypassed?0:(signals[e]||0), strength=Math.abs(signal)/maxSignal;
      if(strongest<0||Math.abs(signal)>Math.abs(signals[strongest]||0)) strongest=e;
      const alpha=connected ? .025+strength*.65 : .013;
      ctx.strokeStyle=values.length?color(signal,alpha):'rgba(106,139,153,.10)';
      ctx.lineWidth=connected? .5+strength*1.8 : .4;
      const bend=(src[e]<dst[e]?1:-1)*12;
      const mx=(a.x+b.x)/2+bend, my=(a.y+b.y)/2-bend;
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo(mx,my,b.x,b.y);ctx.stroke();
      if(live&&strength>.18&&connected){
        const t=(time*1.5+e*.137)%1,u=1-t;
        ctx.fillStyle=color(signal,.85);
        ctx.beginPath();ctx.arc(u*u*a.x+2*u*t*mx+t*t*b.x,u*u*a.y+2*u*t*my+t*t*b.y,1.2+strength*1.1,0,Math.PI*2);ctx.fill();
      }
    }
    this.projected.map((p,i)=>({...p,i})).sort((a,b)=>b.z-a.z).forEach(p=>{
      const v=values[p.i]||0;
      const connected=focus==null||p.i===focus||src.some((s,e)=>(s===focus&&dst[e]===p.i)||(s===p.i&&dst[e]===focus));
      const radius=(mlp?5:3.4)*p.scale;
      ctx.globalAlpha=connected?1:.18;
      if(values.length&&Math.abs(v)>.45){ctx.fillStyle=color(v,.10);ctx.beginPath();ctx.arc(p.x,p.y,radius*3,0,Math.PI*2);ctx.fill();}
      ctx.fillStyle=values.length?color(v,.3+Math.abs(v)*.7):'#4d646f';
      ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fill();
      if(!mlp&&(this.inputs.has(p.i)||this.outputs.has(p.i))){
        ctx.strokeStyle=this.inputs.has(p.i)?'#b8eaff':'#ffcd89';ctx.lineWidth=1;
        ctx.beginPath();ctx.arc(p.x,p.y,radius+2.8,0,Math.PI*2);ctx.stroke();
      }
      if(p.i===focus){ctx.strokeStyle='#fff';ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(p.x,p.y,radius+7,0,Math.PI*2);ctx.stroke();}
      ctx.globalAlpha=1;
    });
    ctx.fillStyle='#8fa5b1';ctx.font='14px system-ui';
    ctx.fillText(mlp?'Standard network units':this.surface ? 'MaleCNS brain surface' : 'Brain surface unavailable',20,388);
    ctx.textAlign='right';ctx.fillText('Drag to rotate',620,388);ctx.textAlign='left';
    const chosen=focus ?? (values.length ? values.reduce((best,v,i)=>Math.abs(v)>Math.abs(values[best])?i:best,0):null);
    if(chosen==null)this.inspector.textContent='Start a run to see activity. Select a neuron to follow its connections.';
    else if(mlp)this.inspector.textContent=`Unit ${chosen+1}: ${(values[chosen]||0).toFixed(3)}. These are standard network units.`;
    else {
      const node=this.graph.nodes[chosen];
      this.inspector.textContent=`${focus==null?'Strongest neuron: ':''}${node.type||node.superclass} · ID ${node.bodyId} · value ${(values[chosen]||0).toFixed(3)}`;
    }
    const path=document.getElementById('path-readout');
    if(mlp)path.textContent='This controller has no fly connections.';
    else if(this.sample?.bypassed)path.textContent='Wiring bypassed. Inputs feed the output directly.';
    else if(strongest>=0&&signals.length)path.textContent=`Strongest path: ${this.graph.nodes[src[strongest]].bodyId} → ${this.graph.nodes[dst[strongest]].bodyId} · signal ${signals[strongest].toFixed(3)}`;
    else path.textContent='Paths light up when the controller runs.';
  }
}
