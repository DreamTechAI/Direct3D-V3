import meshStats from './mesh-stats.js';
import {readMesh, createViewer} from './mesh-viewer.js';

const cases = [
  [6,'Serpent guardian'],[7,'Armored gnome'],[15,'Armored panda'],[21,'Siege tower'],
  [25,'Masked warrior'],[45,'Warhammer knight'],[46,'Mechanical sentinel'],[48,'Expedition suit'],
  [51,'Arcane engineer'],[60,'Siege mech'],[68,'Runic knight'],[71,'Dwarven smith'],
  [76,'Forest golem'],[87,'Golden warrior']
];
const methods = [
  ['ours','Direct3D-V3'],['trellis2','TRELLIS.2'],['hi3d','Hi3DV1.0'],
  ['meshy','Meshy 7.1 4K'],['tripo','Tripo H3.1'],['rodin','Gen-2.5 Extreme-High']
];
const $=id=>document.getElementById(id);
const media=new URL('./media/',import.meta.url);
const geometryRoot=new URL('https://raw.githubusercontent.com/DreamTechAI/Direct3D-V3/6579db016572a17e97d3e3dd5ede9a4a9d3825e5/');
const input=(id,suffix='')=>new URL(`inputs/${id}${suffix}.webp`,media).href;
const mesh=(id,method,extension)=>new URL(`meshes/${id}/${method}.${extension}`,media).href;
const icon=(name,cl='ic')=>`<svg class="${cl}" aria-hidden="true"><use href="#${name}"/></svg>`;
const camera={yaw:0,pitch:0,zoom:1};
let selected=-1, frame=0;
let batch=new AbortController();
const viewers=new Set();
let tiles=[], leaveDialog=null;
function scheduleDraw() {
  if(frame) return;
  frame=requestAnimationFrame(()=>{frame=0;for(const viewer of viewers)viewer.draw();});
}
function resetCamera() {Object.assign(camera,{yaw:0,pitch:0,zoom:1});scheduleDraw();}
const dialog=$('info-dialog'), body=$('dialog-body');
function showDialog(title) {
  if(dialog.open) dialog.close();
  $('dialog-title').textContent=title;
  body.replaceChildren();
  dialog.showModal();
  scheduleHero();
}
// Fit the complete figure first and support inspecting any region at source resolution.
function inspectImage(frame,image,button) {
  const events=new AbortController(), options={signal:events.signal};
  let zoomed=false, x=0, y=0, drag=null;
  function paint() {
    const w=frame.clientWidth, h=frame.clientHeight;
    const scale=zoomed ? Math.max(image.naturalWidth/w,image.naturalHeight/h,1) : 1;
    const limitX=Math.max(0,(image.naturalWidth-w)/2),limitY=Math.max(0,(image.naturalHeight-h)/2);
    x=zoomed?Math.max(-limitX,Math.min(limitX,x)):0;
    y=zoomed?Math.max(-limitY,Math.min(limitY,y)):0;
    image.style.transform=zoomed?`translate(${x}px,${y}px) scale(${scale})`:'';
    frame.classList.toggle('pixel-view',zoomed);
    button.textContent=zoomed?'Fit to view':'Actual pixels';
    button.setAttribute('aria-pressed',String(zoomed));
  }
  function toggle(event) {
    if(!image.naturalWidth)return;
    zoomed=!zoomed;
    if(zoomed && event) {
      const rect=frame.getBoundingClientRect();
      const scale=Math.max(image.naturalWidth/rect.width,image.naturalHeight/rect.height,1);
      x=(rect.width/2-event.clientX+rect.left)*(scale-1);
      y=(rect.height/2-event.clientY+rect.top)*(scale-1);
    }
    paint();
  }
  button.addEventListener('click',()=>toggle(),options);
  image.draggable=false;
  image.addEventListener('pointerdown',e=>{
    if(e.button!==0)return;
    drag={px:e.clientX,py:e.clientY,x,y,moved:false};
    image.setPointerCapture(e.pointerId);
  },options);
  image.addEventListener('pointermove',e=>{
    if(!drag)return;
    const dx=e.clientX-drag.px,dy=e.clientY-drag.py;
    if(Math.hypot(dx,dy)>4)drag.moved=true;
    if(zoomed && drag.moved){x=drag.x+dx;y=drag.y+dy;paint();}
  },options);
  image.addEventListener('pointerup',e=>{if(drag&&!drag.moved)toggle(e);drag=null;},options);
  image.addEventListener('pointercancel',()=>{drag=null;},options);
  window.addEventListener('resize',paint,options);paint();
  return ()=>{events.abort();image.style.transform='';frame.classList.remove('pixel-view');};
}
function detailButton() {
  const button=document.createElement('button');
  button.type='button';button.className='detail-btn';return button;
}
function showImage(title,url,alt) {
  showDialog(title);
  const frame=document.createElement('div');frame.className='image-inspection';
  const image=new Image();image.src=url;image.alt=alt;frame.append(image);
  const button=detailButton();body.append(frame,button);
  leaveDialog=inspectImage(frame,image,button);
}
$('dialog-close').onclick=()=>dialog.close();
dialog.addEventListener('click',e=>{
  if(e.target!==dialog)return;
  const r=dialog.getBoundingClientRect();
  if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();
});
dialog.addEventListener('close',()=>{
  leaveDialog?.();leaveDialog=null;body.replaceChildren();
});
for(const [id,title] of [['teaser-open','Generated meshes'],['framework-open','Generation framework'],['architecture-open','Dual-granularity DiT architecture']]) {
  const figure=$(id).querySelector('img');
  $(id).onclick=()=>showImage(title,figure.src,figure.alt);
}
$('abstract-open').onclick=()=>{
  showDialog('Abstract');
  const content=document.createElement('div');content.className='text-content';
  const paragraphs=ABSTRACT.split('\n\n');
  for(const text of paragraphs){const p=document.createElement('p');p.textContent=text;content.append(p);}
  body.append(content);
};
$('cite-open').onclick=()=>{
  showDialog('BibTeX');
  const pre=document.createElement('pre');pre.textContent=CITATION;
  const button=document.createElement('button');button.className='btn btn-accent small copy-btn';
  button.type='button';button.innerHTML=icon('copy')+'Copy BibTeX';
  button.onclick=async()=>{
    try {
      if(!navigator.clipboard)throw new Error();
      await navigator.clipboard.writeText(CITATION);
      button.textContent='Copied';
    }catch{
      const selection=window.getSelection(),range=document.createRange();
      range.selectNodeContents(pre);selection.removeAllRanges();selection.addRange(range);
      button.textContent='Citation selected';
    }
  };
  body.append(pre,button);
};

function refreshStatus() {
  const ready=tiles.filter(t=>t.state==='ready').length;
  const loading=tiles.some(t=>t.state==='loading'), failed=tiles.some(t=>t.state==='error');
  $('reset-view').hidden=!ready;
  const loadAll=$('load-all');
  loadAll.disabled=loading||ready===methods.length;
  loadAll.querySelector('span').textContent=loading?'Loading…':ready===methods.length?'All 3D loaded':ready?'Load remaining':'Load all 3D';
  $('compare-status').textContent=failed?'A 3D view could not load. Its preview remains available. Select Retry to try again.'
    :loading?'Loading mesh geometry…'
    :ready?`${ready} of ${methods.length} meshes loaded · Shared camera · Drag to orbit, scroll or pinch to zoom.`
    :'Static previews · Orthographic front view';
}
function inputTile(id,caseName,label,suffix) {
  const button=document.createElement('button');button.type='button';button.className='input-tile';
  button.setAttribute('aria-label',`Enlarge ${label} for ${caseName}`);
  const img=new Image();img.loading='lazy';img.src=input(id,suffix);img.alt=`${label} for ${caseName}`;img.width=650;img.height=650;
  const caption=document.createElement('span');caption.className='input-caption';caption.innerHTML=`<span>${label}</span>${icon('expand')}`;
  button.append(img,caption);button.onclick=()=>showImage(`${caseName} · ${label}`,img.src,img.alt);return button;
}
function makeTile(id,caseName,method,label) {
  const stats=meshStats[`${id}/${method}`];
  const card=document.createElement('article');card.className='method-card '+method;
  const stage=document.createElement('div');stage.className='tile-stage';
  const poster=new Image();poster.loading='lazy';poster.src=mesh(id,method,'thumb.webp');
  poster.className='tile-poster';poster.alt=`${caseName}, ${label}, front-view surface normals`;
  poster.width=768;poster.height=768;stage.append(poster);
  const foot=document.createElement('div');foot.className='tile-foot';
  const heading=document.createElement('h4');
  heading.textContent=label;
  const button=document.createElement('button');button.type='button';button.className='tile-load';
  button.innerHTML=icon('cube','ico-cube')+'Load 3D · '+Math.ceil(stats.bytes/1e6)+' MB';
  button.title=stats.triangles.toLocaleString('en-US')+' original triangles';
  button.setAttribute('aria-label',`Load ${label} mesh for ${caseName}`);
  const detail=detailButton();detail.hidden=true;
  foot.append(heading,detail,button);card.append(stage,foot);
  let disposed=false, viewer=null, canvas=null, expanded=false, stopInspect=null;
  const tile={element:card,state:'idle',load,dispose};
  const expand=document.createElement('button');expand.type='button';expand.className='tile-expand';
  expand.innerHTML=icon('expand');expand.setAttribute('aria-label',`Enlarge ${label} for ${caseName}`);
  function open() {
    showDialog(`${caseName} · ${label}`);
    // Moving the existing canvas preserves its full geometry and GPU buffers.
    const location=document.createComment('Expanded mesh');
    card.replaceWith(location);body.append(card);expand.hidden=true;expanded=true;
    if(!viewer){
      poster.loading='eager';poster.src=mesh(id,method,'webp');
      detail.hidden=false;stopInspect=inspectImage(stage,poster,detail);
    } else {button.disabled=false;button.textContent='Front view';}
    scheduleDraw();
    leaveDialog=()=>{
      stopInspect?.();stopInspect=null;detail.hidden=true;
      expanded=false;expand.hidden=false;location.replaceWith(card);
      if(viewer){button.textContent='3D active';button.disabled=true;}
      expand.focus({preventScroll:true});scheduleDraw();
    };
  }
  expand.onclick=open;stage.append(expand);
  poster.addEventListener('click',()=>{if(!expanded)open();});
  async function load() {
    if(disposed||tile.state==='loading')return;
    if(viewer){resetCamera();return;}
    tile.state='loading';button.disabled=true;button.textContent='Loading…';refreshStatus();
    try{
      const url=new URL(`${id}/${method}.mesh.gz`,geometryRoot).href;
      const data=await readMesh(url,batch.signal);
      if(disposed)return;
      canvas=document.createElement('canvas');canvas.className='tile-canvas';canvas.tabIndex=0;
      canvas.dataset.triangles=String(data.indices.length/3);
      canvas.setAttribute('aria-label',`Interactive ${label} mesh. Drag or use arrow keys to orbit; plus and minus to zoom; Home for front view.`);
      stage.insertBefore(canvas,expand);
      viewer=createViewer(canvas,data,camera,scheduleDraw);
      viewers.add(viewer);stopInspect?.();stopInspect=null;detail.hidden=true;
      poster.hidden=true;tile.state='ready';
      button.textContent=expanded?'Front view':'3D active';button.disabled=!expanded;
    }catch(e){
      if(disposed)return;
      tile.state='error';canvas?.remove();button.disabled=false;button.textContent='Retry 3D';button.title=e.message;
    }
    refreshStatus();
  }
  function dispose(){disposed=true;if(viewer){viewers.delete(viewer);viewer.dispose();}}
  button.onclick=load;
  return tile;
}
function selectCase(i,scrollStrip=false) {
  i=(i+cases.length)%cases.length;
  if(i===selected)return;
  for(const tile of tiles)tile.dispose();
  batch.abort();batch=new AbortController();tiles=[];
  resetCamera();selected=i;
  const [id,name]=cases[i];
  $('case-name').textContent=name;
  const strip=$('case-strip');
  Array.from(strip.children).forEach((b,j)=>b.setAttribute('aria-pressed',String(j===i)));
  if(scrollStrip){const b=strip.children[i];strip.scrollTo({left:b.offsetLeft-strip.offsetLeft-(strip.clientWidth-b.offsetWidth)/2,behavior:'auto'});}
  const modelGrid=$('models');modelGrid.replaceChildren();
  const inputs=document.createElement('div');inputs.className='inputs-panel';
  inputs.append(inputTile(id,name,'RGB input',''),inputTile(id,name,'Normal input','-normal'));
  modelGrid.append(inputs);
  tiles=methods.map(([key,label])=>makeTile(id,name,key,label));
  for(const tile of tiles)modelGrid.append(tile.element);
  refreshStatus();
}
cases.forEach(([id,name],i)=>{
  const button=document.createElement('button');button.type='button';button.className='case-thumb';
  button.title=name;button.setAttribute('aria-label',name);
  const image=new Image();image.src=input(id,'-thumb');image.alt='';image.width=90;image.height=90;image.loading='lazy';
  button.append(image);button.onclick=()=>selectCase(i,true);
  button.onkeydown=e=>{
    const offsets={ArrowRight:1,ArrowLeft:-1,Home:-i,End:cases.length-1-i};
    if(!(e.key in offsets))return;
    e.preventDefault();const next=(i+offsets[e.key]+cases.length)%cases.length;
    selectCase(next,true);$('case-strip').children[next].focus({preventScroll:true});
  };
  $('case-strip').append(button);
});
$('case-prev').onclick=()=>selectCase(selected-1,true);
$('case-next').onclick=()=>selectCase(selected+1,true);
$('load-all').onclick=async()=>{
  const group=tiles;
  for(const tile of group){if(group!==tiles)break;if(tile.state!=='ready')await tile.load();}
};
$('reset-view').onclick=resetCamera;
const slides=Array.from($('hero-photos').children);
const heroInspect=$('hero-inspect'), heroName=$('hero-name'), playback=$('hero-playback');
const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
let playing=!reducedMotion.matches, heroInView=false;
let heroIndex=0, requestedIndex=0, heroTimer;
function canRotate() {
  return playing && heroInView && !document.hidden && !dialog.open && !heroInspect.classList.contains('detail-view');
}
function scheduleHero() {
  clearTimeout(heroTimer);
  if(canRotate()) heroTimer=setTimeout(()=>showHero(requestedIndex+1,true),6000);
}
function setHeroDetail(enabled) {
  heroInspect.classList.toggle('detail-view',enabled);
  $('hero-detail-toggle').setAttribute('aria-pressed',String(enabled));
}
async function showHero(index,automatic=false) {
  clearTimeout(heroTimer);
  requestedIndex=(index+slides.length)%slides.length;
  const next=requestedIndex, slide=slides[next];
  try { await slide.decode(); } catch {
    if(next===requestedIndex) scheduleHero();
    return;
  }
  if(next!==requestedIndex) return;
  if(automatic && !canRotate()) {
    requestedIndex=heroIndex;
    return;
  }
  slides[heroIndex].classList.remove('is-active');
  slides[heroIndex].setAttribute('aria-hidden','true');
  slide.classList.add('is-active');
  slide.removeAttribute('aria-hidden');
  heroIndex=next;
  setHeroDetail(false);
  heroName.setAttribute('aria-live',automatic?'off':'polite');
  heroName.textContent=slide.dataset.name;
  heroInspect.setAttribute('aria-label',`Enlarge the clay render of ${slide.dataset.name}`);
  scheduleHero();
}
function setPlaying(enabled) {
  playing=enabled;
  playback.title=playing?'Pause slideshow':'Play slideshow';
  playback.setAttribute('aria-label',playback.title);
  playback.querySelector('use').setAttribute('href',playing?'#pause':'#play');
  scheduleHero();
}
heroInspect.onclick=()=>{
  const slide=slides[heroIndex];
  showImage(`${slide.dataset.name} · Clay render`,slide.src,slide.alt);
};
$('hero-prev').onclick=()=>showHero(requestedIndex-1);
$('hero-next').onclick=()=>showHero(requestedIndex+1);
$('hero-detail-toggle').onclick=()=>{
  setHeroDetail(!heroInspect.classList.contains('detail-view'));
  scheduleHero();
};
playback.onclick=()=>setPlaying(!playing);
reducedMotion.addEventListener('change',()=>setPlaying(!reducedMotion.matches));
document.addEventListener('visibilitychange',scheduleHero);
dialog.addEventListener('close',scheduleHero);
new IntersectionObserver(([entry])=>{
  heroInView=entry.isIntersecting;
  scheduleHero();
}).observe(heroInspect);
setPlaying(playing);
selectCase(0);

const CITATION=`@article{wu2026direct3dv3,
  author  = {Wu, Shuang and Qian, Jiachen and Lin, Youtian and Zhang, Feihu and Torr, Philip and Yao, Yao and Cao, Xun},
  title   = {{Direct3D-V3}: Latent-Free Diffusion Transformers in the Native Voxel Domain},
  journal = {ACM Transactions on Graphics},
  year    = {2026},
  volume  = {45},
  number  = {6},
  articleno = {244},
  doi     = {10.1145/3842524}
}`;

const ABSTRACT = "Native 3D generation has largely been built around latent diffusion, with recent advances concentrating on stronger variational autoencoders (VAE) and increasingly elaborate latent representations, from vecset and triplanes to structured sparse voxels. Despite their architectural differences, these methods all share the same premise: geometry is first compressed into a latent space and only then modeled by the generator. This paradigm carries two intrinsic limitations. First, VAE reconstruction removes fine geometric detail before the diffusion model ever sees the data. Second, the two-stage pipeline creates an objective mismatch, since the VAE is trained for reconstruction whereas the diffusion model is trained for distribution matching.\n\nWe take a different route and propose Direct3D-V3, a latent-free diffusion transformer (DiT) framework that discards the VAE altogether and operates in the native voxel domain, directly modeling raw SDF voxel grids during both training and sampling. To make this direct voxel formulation geometrically reliable, we introduce an implicit manifold regularization over neighboring voxels to suppress non-watertight surfaces, and convert predicted SDF grids into meshes via Marching Cubes so that rendered normal losses can supervise the model directly from its predicted SDF. To improve alignment between the conditioning image and generated geometry, Direct3D-V3 also predicts the camera parameters of the conditioning image as an auxiliary task, encouraging the DiT to learn the mapping from image observations to 3D shape. This latent-free design removes the reconstruction bottleneck, brings training under an objective in the native voxel domain, and yields meshes with substantially richer local detail.\n\nThe predicted camera and native SDF voxel output further make Direct3D-V3 extensible to generative reconstruction, where diffusion is interleaved with normal-guided refinement to improve alignment with the observed view while the generative prior completes unseen regions. Extensive experiments show that Direct3D-V3 achieves competitive results to state-of-the-art latent-based methods, while producing finer geometry.";
