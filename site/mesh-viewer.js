// A worker owns each in-flight decode. Changing case also cancels its network request.
export function readMesh(url, signal) {
  return new Promise((resolve,reject)=>{
    if(signal.aborted) {reject(new DOMException('Aborted','AbortError'));return;}
    const worker=new Worker(new URL('./mesh-reader.js',import.meta.url));
    function finish(callback,value) {
      signal.removeEventListener('abort',abort);
      worker.terminate();
      callback(value);
    }
    function abort() {finish(reject,new DOMException('Aborted','AbortError'));}
    signal.addEventListener('abort',abort,{once:true});
    worker.onmessage=({data})=>data.error ? finish(reject,new Error(data.error)) : finish(resolve,data);
    worker.onerror=()=>finish(reject,new Error('Unable to decode the full-resolution mesh.'));
    worker.postMessage(url);
  });
}

const vertexSource = `#version 300 es
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
uniform vec2 angles;
uniform vec2 scale;
out vec3 vNormal;
vec3 rotate(vec3 p) {
  float cy=cos(angles.x), sy=sin(angles.x), cx=cos(angles.y), sx=sin(angles.y);
  vec3 q=vec3(cy*p.x+sy*p.z,p.y,-sy*p.x+cy*p.z);
  return vec3(q.x,cx*q.y-sx*q.z,sx*q.y+cx*q.z);
}
void main(){
  vec3 p=rotate(position);
  gl_Position=vec4(p.xy*scale,-p.z*0.45,1.0);
  vNormal=rotate(normal);
}`;
const fragmentSource = `#version 300 es
precision highp float;
in vec3 vNormal;
out vec4 color;
void main(){
  vec3 n=normalize(vNormal);
  if(n.z<0.0)n=-n;
  color=vec4(0.5*n+0.5,1.0);
}`;

export function createViewer(canvas, mesh, camera, onChange) {
  const gl = canvas.getContext('webgl2', {alpha:true, antialias:true, premultipliedAlpha:false, preserveDrawingBuffer:false});
  if (!gl) throw new Error('3D requires WebGL 2. The front preview is still available.');
  const program = gl.createProgram();
  const shaders = [];
  const buffers = [];
  const controls = new AbortController();
  let observer;
  function dispose() {
    controls.abort();
    observer?.disconnect();
    for (const buffer of buffers) gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
    for (const shader of shaders) gl.deleteShader(shader);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
  try {
    for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource],[gl.FRAGMENT_SHADER, fragmentSource]]) {
      const shader=gl.createShader(type);
      shaders.push(shader);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) throw new Error('Unable to initialize the 3D shader.');
      gl.attachShader(program,shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program,gl.LINK_STATUS)) throw new Error('Unable to initialize the 3D view.');
    gl.useProgram(program);
    for (const [location, values] of [[0,mesh.vertices],[1,mesh.normals]]) {
      const buffer=gl.createBuffer(); buffers.push(buffer);
      gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
      gl.bufferData(gl.ARRAY_BUFFER,values,gl.STATIC_DRAW);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location,3,gl.FLOAT,false,0,0);
    }
    const indices=gl.createBuffer(); buffers.push(indices);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indices);
    const indexCount=mesh.indices.length;
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,mesh.indices,gl.STATIC_DRAW);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0,0,0,0);
    const angles=gl.getUniformLocation(program,'angles'), scale=gl.getUniformLocation(program,'scale');
    function draw() {
      const rect=canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const dpr=window.devicePixelRatio||1;
      const w=Math.max(1,Math.round(rect.width*dpr)), h=Math.max(1,Math.round(rect.height*dpr));
      if (canvas.width!==w || canvas.height!==h) {canvas.width=w;canvas.height=h;}
      gl.viewport(0,0,w,h);
      gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
      gl.uniform2f(angles,camera.yaw,camera.pitch);
      gl.uniform2f(scale,camera.zoom*Math.min(1,h/w),camera.zoom*Math.min(1,w/h));
      gl.drawElements(gl.TRIANGLES,indexCount,gl.UNSIGNED_INT,0);
    }
    function zoom(factor) {camera.zoom=Math.max(0.45,Math.min(5,camera.zoom*factor));}
    function orbit(x,y) {camera.yaw+=x;camera.pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,camera.pitch+y));}
    const events={signal:controls.signal}, active=new Map();
    function distance() {
      const [a,b]=active.values();
      return Math.max(1,Math.hypot(a.x-b.x,a.y-b.y));
    }
    canvas.addEventListener('pointerdown',e=>{
      if(e.button!==0) return;
      canvas.setPointerCapture(e.pointerId);active.set(e.pointerId,{x:e.clientX,y:e.clientY});
    },events);
    canvas.addEventListener('pointermove',e=>{
      const old=active.get(e.pointerId);
      if(!old) return;
      const previous=active.size===2?distance():0;
      active.set(e.pointerId,{x:e.clientX,y:e.clientY});
      if(previous) zoom(distance()/previous);
      else if(active.size===1) orbit((e.clientX-old.x)*.012,(e.clientY-old.y)*.012);
      onChange();
    },events);
    const end=e=>active.delete(e.pointerId);
    for(const name of ['pointerup','pointercancel','lostpointercapture']) canvas.addEventListener(name,end,events);
    canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-Math.max(-120,Math.min(120,e.deltaY))*.002));onChange();},{...events,passive:false});
    canvas.addEventListener('keydown',e=>{
      switch(e.key) {
        case 'ArrowLeft':orbit(-.12,0);break; case 'ArrowRight':orbit(.12,0);break;
        case 'ArrowUp':orbit(0,-.12);break; case 'ArrowDown':orbit(0,.12);break;
        case '+':case '=':zoom(1.15);break; case '-':zoom(1/1.15);break;
        case 'Home':Object.assign(camera,{yaw:0,pitch:0,zoom:1});break; default:return;
      }
      e.preventDefault();onChange();
    },events);
    observer=new ResizeObserver(draw);observer.observe(canvas);
    draw();
    return {draw,dispose};
  } catch(e) {dispose();throw e;}
}
