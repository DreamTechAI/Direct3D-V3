// Decode in a worker so that large original meshes never stall gallery navigation.
async function readPayload(url) {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error('The mesh could not be downloaded.');
  const reader = response.body.getReader();
  const prefix = [];
  let length = 0;
  while (length < 2) {
    const {done, value} = await reader.read();
    if (done) throw new Error('Incomplete mesh download.');
    prefix.push(value);
    length += value.length;
  }
  const compressed = prefix[0][0] === 31 && (prefix[0][1] ?? prefix[1]?.[0]) === 139;
  const stream = new ReadableStream({
    start(controller) { for (const part of prefix) controller.enqueue(part); },
    async pull(controller) {
      const {done, value} = await reader.read();
      if (done) controller.close(); else controller.enqueue(value);
    },
    cancel(reason) { return reader.cancel(reason); }
  });
  if (compressed && !self.DecompressionStream) throw new Error('This browser cannot decompress the mesh.');
  return new Response(compressed ? stream.pipeThrough(new DecompressionStream('gzip')) : stream).arrayBuffer();
}

function unpack(data) {
  const header = new DataView(data);
  if (data.byteLength < 64 || header.getUint32(0) !== 0x44334632) throw new Error('Invalid mesh data.');
  const count = header.getUint32(4,true), triangles = header.getUint32(8,true);
  const axes = [0,1,2].map(i => header.getInt32(12 + 4*i,true));
  const center = [0,1,2].map(i => header.getFloat64(24 + 8*i,true));
  const factor = header.getFloat64(48,true);
  let offset = 64;
  function components(format) {
    let array;
    if (format === 0) {
      offset = Math.ceil(offset/8)*8;
      array = new Float64Array(data,offset,count*3);
    } else if (format === 0xffffffff) {
      array = new Float32Array(data,offset,count*3);
    } else {
      array = new Int32Array(data,offset,count*3);
    }
    offset += array.byteLength;
    return [array, format === 0xffffffff || format === 0 ? 1 : format];
  }
  const [points, ps] = components(header.getUint32(56,true));
  const [shading, ns] = components(header.getUint32(60,true));
  if (!count || !triangles || offset + triangles*12 !== data.byteLength) throw new Error('Incomplete mesh data.');
  // Convert only the final camera-space attributes to the GPU's native floats.
  const vertices = new Float32Array(count*3), normals = new Float32Array(count*3);
  for (let dst=0; dst<3; dst++) {
    const src=Math.abs(axes[dst])-1, sign=Math.sign(axes[dst]);
    for (let i=0; i<count*3; i+=3) {
      vertices[i+dst]=sign*(points[i+src]/ps-center[src])*factor;
      normals[i+dst]=sign*shading[i+src]/ns;
    }
  }
  const indices = new Uint32Array(data,offset,triangles*3).slice();
  return {vertices,normals,indices};
}

self.onmessage = async ({data:url}) => {
  try {
    const mesh = unpack(await readPayload(url));
    self.postMessage(mesh,[mesh.vertices.buffer,mesh.normals.buffer,mesh.indices.buffer]);
  } catch(error) {
    self.postMessage({error:error.message});
  }
};
