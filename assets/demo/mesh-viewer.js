() => {
  if (customElements.get("direct3d-mesh-viewer")) return;

  // The generation pipeline exports one indexed triangle mesh per binary glTF.
  function readGeometry(bytes) {
    const header = new DataView(bytes);
    if (header.getUint32(0, true) !== 0x46546c67 || header.getUint32(4, true) !== 2) {
      throw new Error("Expected a binary glTF 2.0 mesh.");
    }
    const jsonLength = header.getUint32(12, true);
    const gltf = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 20, jsonLength)));
    const binaryOffset = 20 + jsonLength + 8;
    const primitive = gltf.meshes[0].primitives[0];
    function attribute(id, ArrayType, width) {
      const accessor = gltf.accessors[id], view = gltf.bufferViews[accessor.bufferView];
      const offset = binaryOffset + (view.byteOffset || 0) + (accessor.byteOffset || 0);
      return new ArrayType(bytes, offset, accessor.count * width);
    }
    const positions = attribute(primitive.attributes.POSITION, Float32Array, 3);
    const IndexArray = {5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array}[gltf.accessors[primitive.indices].componentType];
    const indices = attribute(primitive.indices, IndexArray, 1);
    let normals;
    if (primitive.attributes.NORMAL !== undefined) {
      normals = attribute(primitive.attributes.NORMAL, Float32Array, 3);
    } else {
      // Accumulate area-weighted vertex normals once for both materials.
      normals = new Float32Array(positions.length);
      for (let i = 0; i < indices.length; i += 3) {
        const a = 3 * indices[i], b = 3 * indices[i + 1], c = 3 * indices[i + 2];
        const bx = positions[b] - positions[a], by = positions[b + 1] - positions[a + 1], bz = positions[b + 2] - positions[a + 2];
        const cx = positions[c] - positions[a], cy = positions[c + 1] - positions[a + 1], cz = positions[c + 2] - positions[a + 2];
        const nx = by * cz - bz * cy, ny = bz * cx - bx * cz, nz = bx * cy - by * cx;
        normals[a] += nx; normals[b] += nx; normals[c] += nx;
        normals[a + 1] += ny; normals[b + 1] += ny; normals[c + 1] += ny;
        normals[a + 2] += nz; normals[b + 2] += nz; normals[c + 2] += nz;
      }
      for (let i = 0; i < normals.length; i += 3) {
        const length = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
        normals[i] /= length; normals[i + 1] /= length; normals[i + 2] /= length;
      }
    }
    const {min, max} = gltf.accessors[primitive.attributes.POSITION];
    const center = min.map((v, i) => (v + max[i]) / 2);
    const size = Math.max(...max.map((v, i) => v - min[i])) || 1;
    return {positions, normals, indices, center, size};
  }

  const meshVertex = `#version 300 es
  layout(location=0) in vec3 position;
  layout(location=1) in vec3 normal;
  uniform vec3 center;
  uniform float size;
  uniform vec2 angles, scale, pan;
  out vec3 vNormal;
  vec3 rotate(vec3 p) {
    float cy=cos(angles.x), sy=sin(angles.x), cx=cos(angles.y), sx=sin(angles.y);
    vec3 q=vec3(cy*p.x+sy*p.z, p.y, -sy*p.x+cy*p.z);
    return vec3(q.x, cx*q.y-sx*q.z, sx*q.y+cx*q.z);
  }
  void main() {
    vec3 p=rotate((position-center)*2.0/size);
    vNormal=rotate(normal);
    gl_Position=vec4(p.xy*scale+pan, -p.z*0.5, 1.0);
  }`;
  const meshFragment = `#version 300 es
  precision highp float;
  in vec3 vNormal;
  out vec4 color;
  void main() {
    vec3 n=normalize(vNormal);
    if (!gl_FrontFacing) n=-n;
    color=vec4(n*0.5+0.5, 1.0);
  }`;
  const screenVertex = `#version 300 es
  out vec2 uv;
  void main() {
    uv=vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position=vec4(uv*2.0-1.0, 0.0, 1.0);
  }`;
  const screenFragment = `#version 300 es
  precision highp float;
  in vec2 uv;
  uniform sampler2D surface, depths;
  uniform vec2 pixel, scale;
  uniform bool showNormals;
  out vec4 color;
  void main() {
    vec4 sampleNormal=texture(surface,uv);
    if(sampleNormal.a<0.5) {color=vec4(0.0); return;}
    vec3 n=normalize(sampleNormal.rgb*2.0-1.0);
    if(showNormals) {color=vec4(n*0.5+0.5,1.0); return;}
    float depth=texture(depths,uv).r;
    float occlusion=0.0;
    for(int ring=0;ring<2;ring++) {
      float radius=(ring==0?0.020:0.073);
      for(int i=0;i<8;i++) {
        float angle=(float(i)+0.5*float(ring))*0.78539816;
        vec2 offset=vec2(cos(angle),sin(angle))*max(radius*scale*0.5,pixel*1.5);
        float other=texture(depths,uv+offset).r;
        vec3 delta=vec3(2.0*offset/scale,4.0*(depth-other));
        float d=length(delta);
        occlusion+=max(dot(n,delta)/max(d,0.0001)-0.10,0.0)*(1.0-smoothstep(radius,4.0*radius,d));
      }
    }
    // A broad key and grazing rim separate bevels; nearby depth samples shade recesses.
    vec3 key=normalize(vec3(-0.65,0.76,0.62));
    float diffuse=max(dot(n,key),0.0);
    float sky=0.5+0.5*n.y;
    float rim=pow(1.0-clamp(n.z,0.0,1.0),3.0)*max(0.0,n.x+0.35);
    float gloss=pow(max(dot(n,normalize(key+vec3(0,0,1))),0.0),26.0);
    vec3 clay=vec3(0.40,0.43,0.47)+vec3(0.14)*sky+vec3(0.38,0.36,0.33)*diffuse;
    clay+=vec3(0.12)*gloss+vec3(0.11,0.12,0.13)*rim;
    color=vec4(clay*(1.0-clamp(occlusion/16.0*1.8,0.0,0.48)),1.0);
  }`;

  function createProgram(gl, vertex, fragment) {
    const program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, vertex], [gl.FRAGMENT_SHADER, fragment]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(program, shader);
      gl.deleteShader(shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return program;
  }

  function createRenderer(canvas, mesh) {
    const gl = canvas.getContext("webgl2", {alpha:true, antialias:false});
    if (!gl) throw new Error("WebGL 2 is required for interactive preview. You can download the GLB below.");
    const geometry = createProgram(gl, meshVertex, meshFragment);
    const shade = createProgram(gl, screenVertex, screenFragment);
    const locations = (p, names) => Object.fromEntries(names.map(k => [k, gl.getUniformLocation(p,k)]));
    const g = locations(geometry, ["center","size","angles","scale","pan"]);
    const s = locations(shade, ["pixel","scale","surface","depths","showNormals"]);
    const vao = gl.createVertexArray(), buffers = [];
    gl.bindVertexArray(vao);
    for (const [location, values] of [mesh.positions, mesh.normals].entries()) {
      const buffer = gl.createBuffer(); buffers.push(buffer);
      gl.bindBuffer(gl.ARRAY_BUFFER,buffer); gl.bufferData(gl.ARRAY_BUFFER,values,gl.STATIC_DRAW);
      gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location,3,gl.FLOAT,false,0,0);
    }
    const index = gl.createBuffer(); buffers.push(index);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,index); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,mesh.indices,gl.STATIC_DRAW);
    const indexType = {1:gl.UNSIGNED_BYTE,2:gl.UNSIGNED_SHORT,4:gl.UNSIGNED_INT}[mesh.indices.BYTES_PER_ELEMENT];
    const {center,size} = mesh, count = mesh.indices.length;
    gl.bindVertexArray(null);
    const frame = gl.createFramebuffer(), surface = gl.createTexture(), depth = gl.createTexture();
    let width = 0, height = 0;
    function texture(unit, tex, format, channels, type, attachment) {
      gl.activeTexture(unit); gl.bindTexture(gl.TEXTURE_2D,tex);
      gl.texImage2D(gl.TEXTURE_2D,0,format,width,height,0,channels,type,null);
      for (const param of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D,param,gl.NEAREST);
      for (const param of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D,param,gl.CLAMP_TO_EDGE);
      gl.framebufferTexture2D(gl.FRAMEBUFFER,attachment,gl.TEXTURE_2D,tex,0);
    }
    function draw(camera, normals, geometryChanged) {
      const dpr = Math.min(2, Math.max(1.25, window.devicePixelRatio || 1));
      const w = Math.max(1, Math.round(canvas.clientWidth*dpr)), h = Math.max(1, Math.round(canvas.clientHeight*dpr));
      gl.bindFramebuffer(gl.FRAMEBUFFER,frame);
      if (w !== width || h !== height) {
        width = canvas.width = w; height = canvas.height = h;
        texture(gl.TEXTURE0,surface,gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE,gl.COLOR_ATTACHMENT0);
        texture(gl.TEXTURE1,depth,gl.DEPTH_COMPONENT24,gl.DEPTH_COMPONENT,gl.UNSIGNED_INT,gl.DEPTH_ATTACHMENT);
        geometryChanged = true;
      }
      const sx = camera.zoom*Math.min(1,h/w), sy = camera.zoom*Math.min(1,w/h);
      gl.viewport(0,0,w,h);
      if (geometryChanged) {
        gl.enable(gl.DEPTH_TEST); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
        gl.useProgram(geometry); gl.bindVertexArray(vao);
        gl.uniform3fv(g.center,center); gl.uniform1f(g.size,size);
        gl.uniform2f(g.angles,camera.yaw,camera.pitch); gl.uniform2f(g.scale,sx,sy);
        gl.uniform2f(g.pan,camera.x,camera.y);
        gl.drawElements(gl.TRIANGLES,count,indexType,0); gl.bindVertexArray(null);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.disable(gl.DEPTH_TEST); gl.useProgram(shade);
      gl.uniform1i(s.surface,0); gl.uniform1i(s.depths,1); gl.uniform1i(s.showNormals,normals);
      gl.uniform2f(s.pixel,1/w,1/h); gl.uniform2f(s.scale,sx,sy);
      gl.drawArrays(gl.TRIANGLES,0,3);
    }
    function dispose() {
      buffers.forEach(buffer => gl.deleteBuffer(buffer));
      gl.deleteVertexArray(vao); gl.deleteFramebuffer(frame); gl.deleteTexture(surface); gl.deleteTexture(depth);
      gl.deleteProgram(geometry); gl.deleteProgram(shade);
      gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
    }
    return {draw, dispose};
  }

  class MeshViewer extends HTMLElement {
    static get observedAttributes() { return ["src"]; }
    connectedCallback() {
      this.innerHTML = `<div class="viewer-toolbar"><div role="group" aria-label="Surface shading">
        <button type="button" data-mode="clay" aria-pressed="true">White clay</button>
        <button type="button" data-mode="normal" aria-pressed="false">Normal</button></div>
        <button type="button" class="reset-view">Reset view</button></div>
        <div class="viewer-stage"><canvas tabindex="0" aria-label="Generated mesh. Drag to orbit, scroll to zoom."></canvas>
        <span class="viewer-status" role="status"></span></div>`;
      this.canvas = this.querySelector("canvas");
      this.status = this.querySelector(".viewer-status");
      this.controls = new AbortController();
      const events = {signal:this.controls.signal};
      this.mode = "clay";
      this.pendingFrame = 0;
      this.pointers = new Map();
      this.reset();
      this.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => {
        this.mode = button.dataset.mode;
        this.querySelectorAll("[data-mode]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
        this.redraw(false);
      }, events));
      this.querySelector(".reset-view").addEventListener("click", () => this.reset(), events);
      const canvas = this.canvas;
      canvas.addEventListener("contextmenu", e => e.preventDefault(), events);
      canvas.addEventListener("pointerdown", e => {
        canvas.setPointerCapture(e.pointerId);
        this.pointers.set(e.pointerId, [e.clientX,e.clientY]);
      }, events);
      const distance = () => {
        const [a,b] = this.pointers.values(); return Math.max(1,Math.hypot(a[0]-b[0],a[1]-b[1]));
      };
      canvas.addEventListener("pointermove", e => {
        const old = this.pointers.get(e.pointerId); if (!old) return;
        const before = this.pointers.size === 2 ? distance() : 0;
        this.pointers.set(e.pointerId,[e.clientX,e.clientY]);
        const dx=e.clientX-old[0], dy=e.clientY-old[1];
        if (before) this.zoom(distance()/before);
        else if (e.shiftKey || e.buttons === 2) {
          this.camera.x += 2*dx/canvas.clientWidth; this.camera.y -= 2*dy/canvas.clientHeight;
        } else {
          this.camera.yaw += dx*0.009; this.camera.pitch = Math.max(-1.57, Math.min(1.57,this.camera.pitch+dy*0.009));
        }
        this.redraw();
      }, events);
      for (const type of ["pointerup","pointercancel","lostpointercapture"]) {
        canvas.addEventListener(type, e => this.pointers.delete(e.pointerId), events);
      }
      canvas.addEventListener("wheel", e => {
        if (!this.renderer) return; e.preventDefault(); this.zoom(Math.exp(-Math.sign(e.deltaY)*0.1)); this.redraw();
      }, {...events,passive:false});
      canvas.addEventListener("keydown", e => {
        switch (e.key) {
          case "ArrowLeft": this.camera.yaw -= 0.1; break;
          case "ArrowRight": this.camera.yaw += 0.1; break;
          case "ArrowUp": this.camera.pitch = Math.max(-1.57,this.camera.pitch-0.1); break;
          case "ArrowDown": this.camera.pitch = Math.min(1.57,this.camera.pitch+0.1); break;
          case "+": case "=": this.zoom(1.2); break;
          case "-": this.zoom(1/1.2); break;
          case "Home": this.reset(); break;
          default: return;
        }
        e.preventDefault(); this.redraw();
      }, events);
      this.observer = new ResizeObserver(() => this.redraw()); this.observer.observe(canvas);
      this.loadMesh();
    }
    reset() { this.camera = {yaw:0,pitch:0,zoom:0.88,x:0,y:0}; this.redraw(); }
    zoom(factor) { this.camera.zoom = Math.max(0.2,Math.min(12,this.camera.zoom*factor)); }
    redraw(geometryChanged=true) {
      this.geometryChanged ||= geometryChanged;
      if (!this.renderer || this.pendingFrame) return;
      this.pendingFrame = requestAnimationFrame(() => {
        this.pendingFrame = 0;
        this.renderer?.draw(this.camera,this.mode==="normal",this.geometryChanged);
        this.geometryChanged = false;
      });
    }
    attributeChangedCallback(name, oldValue, value) { if (this.isConnected && this.canvas && oldValue !== value) this.loadMesh(); }
    async loadMesh() {
      this.request?.abort(); this.renderer?.dispose(); this.renderer = null;
      const url = this.getAttribute("src");
      this.status.textContent = url ? "Loading mesh…" : "Generate a mesh to preview it here.";
      if (!url) return;
      const request = new AbortController(); this.request = request;
      try {
        const response = await fetch(url,{signal:request.signal});
        if (!response.ok) throw new Error("Could not load the generated GLB.");
        const bytes = await response.arrayBuffer();
        if (request.signal.aborted) return;
        this.renderer = createRenderer(this.canvas,readGeometry(bytes));
        this.status.textContent = ""; this.reset();
      } catch (error) { if (!request.signal.aborted) this.status.textContent = error.message; }
    }
    disconnectedCallback() {
      this.request?.abort(); this.controls.abort(); this.observer.disconnect();
      cancelAnimationFrame(this.pendingFrame); this.renderer?.dispose(); this.renderer = null; this.canvas = null;
    }
  }
  customElements.define("direct3d-mesh-viewer",MeshViewer);
}
