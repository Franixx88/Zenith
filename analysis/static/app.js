"use strict";
const $ = (id) => document.getElementById(id);
let state = null, selected = null, busy = false, orientation = localStorage.getItem("zenith.orientation") || "w";
let showArrow = localStorage.getItem("zenith.arrows") !== "false", annotations = [], pendingPromotion = null;
let importKind = "fen", toastTimer, drag = null, source = null, connected = false;
const pieceNames = {p:"пешка",n:"конь",b:"слон",r:"ладья",q:"ферзь",k:"король"};
const pieceSrc = (piece) => `/pieces/${piece === piece.toUpperCase() ? "w" : "b"}${piece.toUpperCase()}.svg`;
const isWhite = (piece) => piece === piece.toUpperCase();
const number = new Intl.NumberFormat("ru-RU");
const compact = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)} млн` : n >= 1000 ? `${(n / 1000).toFixed(n >= 1e5 ? 0 : 1)} тыс.` : number.format(n || 0);

function toast(text, error = false) { clearTimeout(toastTimer); $("toast").textContent = text; $("toast").classList.toggle("error",error); $("toast").hidden = false; toastTimer = setTimeout(() => $("toast").hidden = true, 3800); }
function showError(id, message) { $(id).textContent = message || ""; $(id).hidden = !message; }
function updateState(next) {
  if(state&&state.token===next.token&&(next.revision<state.revision||(next.revision===state.revision&&next.sequence<state.sequence)))return;
  const moved = !state || state.revision !== next.revision;
  const previousNode = state?.node;
  state = next;
  if (moved) { selected = null; if (previousNode !== next.node) annotations = []; }
  render();
}
async function action(payload) {
  if (!state) throw new Error("Нет соединения с доской.");
  if (busy) throw new Error("Дождись завершения предыдущего действия.");
  busy = true;
  try {
    const response = await fetch("/api/action", {method:"POST",headers:{"Content-Type":"application/json","X-Zenith-Token":state.token},body:JSON.stringify({...payload,revision:state.revision})});
    const data = await response.json();
    if (!response.ok) { if (response.status === 409) await refresh(); throw new Error(data.error || "Ошибка запроса"); }
    updateState(data);
    return data;
  } finally { busy = false; }
}
function safely(promise) { Promise.resolve(promise).catch(error => toast(error.message,true)); }
async function refresh() { const response = await fetch("/api/state"); if (!response.ok) throw new Error("Сервер недоступен"); updateState(await response.json()); }

function squareAt(row,col) { return orientation === "w" ? "abcdefgh"[col] + (8-row) : "hgfedcba"[col] + (row+1); }
function squareCenter(square) { const file = square.charCodeAt(0)-97, rank = Number(square[1])-1; return orientation === "w" ? [(file+.5)*100,(7-rank+.5)*100] : [(7-file+.5)*100,(rank+.5)*100]; }
function renderBoard() {
  const fragment = document.createDocumentFragment();
  const legalTargets = new Set(selected ? state.legal.filter(move => move.startsWith(selected)).map(move => move.slice(2,4)) : []);
  for(let row=0;row<8;row++) for(let col=0;col<8;col++) {
    const square = squareAt(row,col), piece = state.pieces[square];
    const button = document.createElement("button");
    button.type = "button"; button.dataset.square = square; button.setAttribute("role","gridcell");
    button.className = `square ${(square.charCodeAt(0)-97+Number(square[1]))%2 ? "dark" : "light"}`;
    if(state.lastMove?.slice(0,2)===square || state.lastMove?.slice(2,4)===square) button.classList.add("last");
    if(square===selected) button.classList.add("selected");
    button.setAttribute("aria-selected",String(square===selected));
    if(square===state.king) button.classList.add("check");
    if(legalTargets.has(square)) { button.classList.add("legal"); if(piece) button.classList.add("capture"); }
    button.setAttribute("aria-label",`${square}${piece ? `, ${isWhite(piece)?"белая":"чёрная"} ${pieceNames[piece.toLowerCase()]}` : ", пусто"}`);
    if(piece) { const img = document.createElement("img"); img.src=pieceSrc(piece); img.className="piece"; img.alt=""; img.draggable=false; button.append(img); }
    if(col===0) { const rank=document.createElement("span");rank.className="coord rank";rank.textContent=square[1];button.append(rank); }
    if(row===7) { const file=document.createElement("span");file.className="coord file";file.textContent=square[0];button.append(file); }
    button.addEventListener("click", event => { if(event.detail===0) safely(chooseSquare(square)); });
    fragment.append(button);
  }
  $("board").replaceChildren(fragment);
  $("board").setAttribute("aria-label",`Шахматная доска, ${orientation==="w"?"белые":"чёрные"} снизу`);
  $("arrows").setAttribute("aria-label","Стрелки анализа");
  $("arrow-toggle").setAttribute("aria-pressed",String(showArrow));
  renderArrows();
}
function drawLine(from,to,color,user=false) {
  const [x1,y1]=squareCenter(from),[x2,y2]=squareCenter(to);
  const svg=document.createElementNS("http://www.w3.org/2000/svg",from===to?"circle":"line");
  if(from===to) { svg.setAttribute("cx",x1);svg.setAttribute("cy",y1);svg.setAttribute("r",43);svg.setAttribute("fill","none");svg.setAttribute("stroke-width",8); }
  else { const length=Math.hypot(x2-x1,y2-y1), offset=length>0?16/length:0; svg.setAttribute("x1",x1);svg.setAttribute("y1",y1);svg.setAttribute("x2",x2-(x2-x1)*offset);svg.setAttribute("y2",y2-(y2-y1)*offset);svg.setAttribute("stroke-width",user?14:12);svg.setAttribute("marker-end",`url(#arrow-${color})`);svg.setAttribute("stroke-linecap","round"); }
  svg.setAttribute("stroke",color==="mint"?"#247e61":"#dc8946");svg.setAttribute("opacity",user?.85:.72);$("arrow-lines").append(svg);
}
function renderArrows() { $("arrow-lines").replaceChildren(); const best=state?.analysis?.pv?.[0]?.uci;if(showArrow&&best)drawLine(best.slice(0,2),best.slice(2,4),"mint");for(const a of annotations)drawLine(a.from,a.to,"orange",true); }
async function chooseSquare(square) {
  if(busy||!state) return;
  if(selected) {
    const moves=state.legal.filter(move=>move.startsWith(selected+square));
    if(moves.length) {
      if(moves.length>1) { pendingPromotion={from:selected,to:square,revision:state.revision};openPromotion(moves);return; }
      await action({action:"move",move:moves[0]});return;
    }
  }
  const piece=state.pieces[square];
  selected=piece&&((state.turn==="w")===isWhite(piece))&&selected!==square?square:null;
  if(!selected)annotations=[];
  renderBoard();
}
function pointSquare(event) { const rect=$("board").getBoundingClientRect();const x=(event.clientX-rect.left)/rect.width,y=(event.clientY-rect.top)/rect.height; if(x<0||y<0||x>=1||y>=1)return null;return squareAt(Math.floor(y*8),Math.floor(x*8)); }
$("board").addEventListener("pointerdown", event=>{
  if(busy||!state||![0,2].includes(event.button))return;
  const square=pointSquare(event);if(!square)return;
  event.preventDefault();$("board").setPointerCapture(event.pointerId);
  drag={square,x:event.clientX,y:event.clientY,right:event.button===2,moved:false,pointer:event.pointerId};
});
$("board").addEventListener("pointermove", event=>{
  if(!drag)return;
  if(Math.hypot(event.clientX-drag.x,event.clientY-drag.y)>5)drag.moved=true;
  const piece=state.pieces[drag.square];
  if(drag.moved&&!drag.right&&piece&&((state.turn==="w")===isWhite(piece))) {
    if(!drag.ghost) { const img=document.createElement("img");img.src=pieceSrc(piece);img.className="drag-ghost";img.style.width=`${$("board").getBoundingClientRect().width/8}px`;document.body.append(img);drag.ghost=img;$("board").querySelector(`[data-square="${drag.square}"]`)?.classList.add("drag-source"); }
    drag.ghost.style.left=`${event.clientX-drag.ghost.width/2}px`;drag.ghost.style.top=`${event.clientY-drag.ghost.width/2}px`;
  }
});
$("board").addEventListener("pointerup", event=>{
  if(!drag)return;
  const original=drag,target=pointSquare(event);drag=null;original.ghost?.remove();
  $("board").querySelector(".drag-source")?.classList.remove("drag-source");
  if(original.right) { if(target) { const index=annotations.findIndex(a=>a.from===original.square&&a.to===target);if(index>=0)annotations.splice(index,1);else annotations.push({from:original.square,to:target});renderArrows(); } return; }
  if(!target)return;
  if(original.moved) { const piece=state.pieces[original.square];if(piece&&((state.turn==="w")===isWhite(piece))) { selected=original.square;safely(chooseSquare(target)); } }
  else safely(chooseSquare(target));
});
$("board").addEventListener("pointercancel",()=>{drag?.ghost?.remove();drag=null;renderBoard();});
$("board").addEventListener("contextmenu",event=>event.preventDefault());

function scoreText(info) {
  if(info?.mate!==undefined&&info.mate!==null) { if(info.mate===0&&info.winner) return info.winner==="w"?"1–0":"0–1";return `${info.mate<0?"−":""}M${Math.abs(info.mate)}`; }
  if(info?.cp===undefined||info.cp===null)return "—";
  if(Math.abs(info.cp)>=28872&&Math.abs(info.cp)<=29000)return info.cp>0?"TB+":"TB−";
  const value=info.cp/100;return `${value>0?"+":value<0?"−":""}${Math.abs(value).toFixed(2)}`;
}
function renderAnalysis() {
  const a=state.analysis||{},active=state.enabled;
  $("engine-name").textContent=state.engine;
  $("analysis-toggle").setAttribute("aria-checked",String(active));
  $("engine-dot").style.background=active&&connected?"":"#73818a";
  const statuses={starting:"НАЧИНАЕМ",running:"АНАЛИЗ",paused:"ПАУЗА",waiting:"ОЖИДАНИЕ",terminal:"РЕЗУЛЬТАТ",complete:"ГОТОВО",error:"ОШИБКА"};
  $("analysis-status").textContent=!connected?"НЕТ СВЯЗИ":!active?"ПАУЗА":statuses[a.status]||"АНАЛИЗ";
  $("analysis-status").classList.toggle("paused",!active||!connected);
  $("engine-subtitle").textContent=!connected?"Переподключение к серверу…":!active?"Анализ на паузе":a.status==="terminal"?"Позиция завершена":`${state.settings.Threads} потоков · без лимита времени`;
  $("score").textContent=scoreText(a);
  const whiteAdvantage=a.mate?Math.sign(a.mate)*10000:a.cp||0;
  $("assessment").textContent=a.mate!==null&&a.mate!==undefined?"Матовая позиция":a.cp===undefined||a.cp===null?"Ожидание оценки":Math.abs(a.cp)>=28872&&Math.abs(a.cp)<=29000?`Табличная победа ${a.cp>0?"белых":"чёрных"}`:Math.abs(whiteAdvantage)<35?"Примерное равенство":whiteAdvantage>0?"Перевес белых":"Перевес чёрных";
  $("depth").textContent=a.depth??"—";$("nps").textContent=a.nps?compact(a.nps):"—";$("nodes").textContent=compact(a.nodes||0);$("tbhits").textContent=number.format(a.tbhits||0);
  const fraction=a.winner?(a.winner==="w"?100:0):a.mate?(a.mate>0?100:0):100/(1+Math.exp(-(a.cp||0)/300));
  $("eval-fill").style.height=`${Math.max(0,Math.min(100,fraction))}%`;document.querySelector(".evaluation").classList.toggle("flipped",orientation==="b");
  document.querySelector(".evaluation").setAttribute("aria-label",`Оценка за белых: ${scoreText(a)}`);
  const pv=$("pv");pv.replaceChildren();
  const lines=a.variations?.length?a.variations:a.pv?.length?[a]:[];
  pv.classList.toggle("multiple",lines.length>1);
  const requested=state.settings.MultiPV||1;
  $("pv-count").hidden=!state.options.MultiPV;
  if(![...$("pv-count").options].some(option=>Number(option.value)===requested)){const option=document.createElement("option");option.value=requested;option.textContent=`${requested} линий`;$("pv-count").append(option);}
  $("pv-count").value=String(requested);
  $("pv-title").textContent=requested>1?"ЛУЧШИЕ ВАРИАНТЫ":"ЛУЧШИЙ ВАРИАНТ";
  if(lines.length) {
    lines.forEach((line,rank)=>{
      const row=document.createElement("div");row.className="pv-row";
      if(lines.length>1){const score=document.createElement("strong");score.className="pv-score";score.textContent=scoreText(line);score.title=`Вариант ${rank+1}, глубина ${line.depth??a.depth??0}, оценка за белых`;row.append(score);}
      const continuation=document.createElement("div");continuation.className="pv-continuation";
      line.pv.forEach((move,index)=>{if(move.turn==="w"||index===0) {const num=document.createElement("span");num.className="pv-number";num.textContent=`${move.number}${move.turn==="w"?".":"…"} `;continuation.append(num);}const button=document.createElement("button");button.className="pv-move";button.textContent=move.san;button.title="Открыть продолжение до этого хода";button.onclick=()=>safely(action({action:"pv",moves:line.pv.slice(0,index+1).map(m=>m.uci)}));continuation.append(button);});
      row.append(continuation);pv.append(row);
    });
  } else {const span=document.createElement("span");span.className="muted";span.textContent=state.result?"В этой позиции партия завершена.":active?"Движок ищет продолжение…":"Включи анализ, чтобы увидеть лучший ход.";pv.append(span);}
  $("play-best").disabled=!a.pv?.length;
  $("resource-label").textContent=`${state.settings.Threads} потоков · ${state.settings.Hash} МБ хеш`;
  showError("engine-error",state.error);
  if(a.node===state.node&&(a.cp!==undefined||a.mate!==undefined))state.evaluations[state.node]=a;
  renderArrows();renderChart();
}

function renderMoves() {
  const container=$("moves");
  const oldScroll=container.scrollTop;
  if(state.tree.length===1) {container.innerHTML='<div class="empty-state"><span class="empty-board">♘</span><strong>Исследуй позицию</strong><p>Сделай ход на доске или импортируй партию. Ответвления сохранятся как варианты.</p></div>';$("promote").disabled=true;return;}
  container.replaceChildren();
  const map=new Map(state.tree.map(n=>[n.id,n])), ancestors=new Set(state.path);
  function moveButton(node) { const button=document.createElement("button");button.className=`move-button${node.id===state.node?" current":ancestors.has(node.id)?" ancestor":""}`;button.textContent=node.san;button.dataset.node=node.id;button.onclick=()=>safely(action({action:"navigate",to:node.id}));if(node.id===state.node)button.setAttribute("aria-current","step");return button; }
  function variation(start) { const div=document.createElement("div");div.className="variation-line";const marker=document.createElement("span");marker.className="variation-start";marker.textContent="↳ ";div.append(marker);let node=start;while(node) {const n=document.createElement("span");n.className="move-number";n.textContent=`${Math.ceil(node.ply/2)}${node.ply%2?".":"…"}`;div.append(n,moveButton(node));const parent=map.get(node.parent);if(parent&&parent.children.length>1&&parent.children[0]===node.id)for(const id of parent.children.slice(1))div.append(variation(map.get(id)));node=node.children.length?map.get(node.children[0]):null;}return div; }
  let parent=map.get(0),node=parent.children.length?map.get(parent.children[0]):null,row=null;
  while(node) {
    if(node.ply%2===1||!row) {row=document.createElement("div");row.className="move-row";const n=document.createElement("span");n.className="move-number";n.textContent=`${Math.ceil(node.ply/2)}.`;row.append(n);if(node.ply%2===0){const blank=document.createElement("span");blank.className="move-button";blank.textContent="…";row.append(blank);}container.append(row);}
    row.append(moveButton(node));
    if(parent.children.length>1) {for(const id of parent.children.slice(1))container.append(variation(map.get(id)));row=null;}
    parent=node;node=node.children.length?map.get(node.children[0]):null;
  }
  container.scrollTop=oldScroll;
  const active=container.querySelector(".current");if(active){const rect=active.getBoundingClientRect(),outer=container.getBoundingClientRect();if(rect.top<outer.top||rect.bottom>outer.bottom)active.scrollIntoView({block:"nearest"});}
  $("promote").disabled=state.node===0;
}
function renderChart() {
  const ids=state.path,points=[];const length=Math.max(ids.length-1,1);
  ids.forEach((id,index)=>{const value=state.evaluations[id];if(value&&(value.cp!==null&&value.cp!==undefined||value.mate!==null&&value.mate!==undefined)) {const cp=value.mate?Math.sign(value.mate)*900:value.cp||0;points.push(`${index*640/length},${30-Math.max(-900,Math.min(900,cp))/900*25}`);}});
  $("chart-line").setAttribute("d",points.map((p,i)=>(i?"L":"M")+p).join(" "));
  $("chart-cursor").setAttribute("x1",ids.length===1?0:640);$("chart-cursor").setAttribute("x2",ids.length===1?0:640);
  $("chart-caption").textContent=points.length?`${points.length} поз. · за белых`:"Ходы появятся здесь";
}
function render() {
  renderBoard();renderMoves();renderAnalysis();$("fen").value=state.fen;
  const black=state.headers.Black==="?"?"Чёрные":state.headers.Black,white=state.headers.White==="?"?"Белые":state.headers.White;
  $("top-player").textContent=orientation==="w"?black:white;$("bottom-player").textContent=orientation==="w"?white:black;$("top-side").textContent=orientation==="w"?"ЧЁРНЫЕ":"БЕЛЫЕ";
  const topAvatar=document.querySelector(".player-strip .avatar"),bottomAvatar=document.querySelector(".bottom-player .avatar");topAvatar.className=`avatar ${orientation==="w"?"black":"white"}-avatar`;bottomAvatar.className=`avatar ${orientation==="w"?"white":"black"}-avatar`;
  topAvatar.textContent=orientation==="w"?"♟":"♙";bottomAvatar.textContent=orientation==="w"?"♙":"♟";
  $("turn-label").textContent=state.result?"Позиция завершена":`Ход ${state.turn==="w"?"белых":"чёрных"}${state.check?" · шах":""}`;
  const labels={CHECKMATE:"Мат",STALEMATE:"Пат",INSUFFICIENT_MATERIAL:"Недостаточно материала",FIFTY_MOVES:"Правило 50 ходов",THREEFOLD_REPETITION:"Троекратное повторение",SEVENTYFIVE_MOVES:"Правило 75 ходов",FIVEFOLD_REPETITION:"Пятикратное повторение"};
  $("game-result").hidden=!state.result;$("game-result").textContent=`${state.result||""} · ${labels[state.termination]||"Партия завершена"}`;
  const current=state.tree.find(n=>n.id===state.node);$("move-counter").textContent=state.node===0?"Начальная позиция":`Ход ${Math.ceil(current.ply/2)} · ${state.tree.length-1} в дереве`;
  $("previous").disabled=state.node===0;$("first").disabled=state.node===0;$("next").disabled=!current.children.length;$("last").disabled=!current.children.length;
  $("board-message").hidden=!state.result;$("board-message").textContent=`${labels[state.termination]||"Партия завершена"} · ${state.result}`;
  renderMaterial();
}
function renderMaterial() {const values={p:1,n:3,b:3,r:5,q:9,k:0};let white=0,black=0;for(const p of Object.values(state.pieces)){if(isWhite(p))white+=values[p.toLowerCase()];else black+=values[p];}const difference=white-black;$("top-material").textContent=(orientation==="w"?-difference:difference)>0?`+${Math.abs(difference)}`:"";$("bottom-material").textContent=(orientation==="w"?difference:-difference)>0?`+${Math.abs(difference)}`:"";}

function openPromotion(moves) {const container=$("promotion-choices");container.replaceChildren();for(const letter of ["q","r","b","n"]){const move=moves.find(m=>m.endsWith(letter));if(!move)continue;const button=document.createElement("button"),img=document.createElement("img");img.src=pieceSrc(state.turn==="w"?letter.toUpperCase():letter);img.alt=pieceNames[letter];button.title=pieceNames[letter];button.append(img);button.onclick=()=>{const revision=pendingPromotion?.revision;$("promotion-dialog").close();pendingPromotion=null;if(revision!==state.revision){toast("Позиция уже изменилась",true);return;}safely(action({action:"move",move}));};container.append(button);}$("promotion-dialog").showModal();}
function flip(){orientation=orientation==="w"?"b":"w";localStorage.setItem("zenith.orientation",orientation);if(state)render();}
function toggleAnalysis(){if(state)safely(action({action:"analysis",enabled:!state.enabled}));}
function openImport(kind="fen") {importKind=kind;$("import-dialog").showModal();setImportKind(kind);$("import-text").focus();}
function setImportKind(kind) {importKind=kind;$("tab-fen").classList.toggle("active",kind==="fen");$("tab-pgn").classList.toggle("active",kind==="pgn");$("import-label").textContent=kind==="fen"?"Описание позиции FEN":"Партия с вариантами PGN";$("import-text").placeholder=kind==="fen"?"Вставь FEN позиции…":"Вставь PGN партии…";$("pgn-file-label").hidden=kind!=="pgn";showError("import-error",null);}
const settingLabels={MultiPV:["Линий анализа","Больше вариантов — меньше глубина за то же время"],Threads:["Потоки","8 — все потоки твоего процессора"],Hash:["Хеш-таблица, МБ","Общая память для всех потоков"],SyzygyPath:["Папка Syzygy","Локальные таблицы эндшпиля"],SyzygyProbeLimit:["Syzygy: максимум фигур"],SyzygyProbeDepth:["Syzygy: глубина пробы"],"Null Move Pruning":["Null move pruning"],"Selective Pruning":["Селективные отсечения"],"Search Enhancements":["Улучшения поиска"],"Modern Search":["Современный поиск"],"Singular Extensions":["Singular extensions","Эксперимент"],"SMP Voting":["Голосование SMP","Эксперимент"],"RFP Margin":["Запас RFP"],"Futility Margin":["Запас futility"],"NMP Base Reduction":["Базовое сокращение NMP"],"LMR Divisor":["Делитель LMR, в сотых"]};
function openSettings() {if(!state)return;$("settings-fields").replaceChildren();$("advanced-fields").replaceChildren();for(const [name,schema] of Object.entries(state.options)){const row=document.createElement("div"),label=document.createElement("label"),input=document.createElement("input");row.className="setting-row";label.textContent=settingLabels[name]?.[0]||name;label.htmlFor=`setting-${name}`;if(settingLabels[name]?.[1]){const small=document.createElement("small");small.textContent=settingLabels[name][1];label.append(small);}input.id=label.htmlFor;input.dataset.option=name;input.type=schema.type==="check"?"checkbox":schema.type==="spin"?"number":"text";if(schema.type==="check")input.checked=state.settings[name];else input.value=state.settings[name]??schema.default??"";if(schema.type==="spin"){input.min=schema.min;input.max=schema.max;input.required=true;}row.append(label,input);$( ["Threads","Hash","MultiPV","SyzygyPath"].includes(name)?"settings-fields":"advanced-fields").append(row);}showError("settings-error",null);$("settings-dialog").showModal();}

$("flip").onclick=flip;$("analysis-toggle").onclick=toggleAnalysis;
for(const to of ["first","previous","next","last"])$(to).onclick=()=>safely(action({action:"navigate",to}));
$("arrow-toggle").onclick=()=>{showArrow=!showArrow;localStorage.setItem("zenith.arrows",String(showArrow));$("arrow-toggle").setAttribute("aria-pressed",String(showArrow));renderArrows();};
$("play-best").onclick=()=>{const best=state?.analysis?.pv?.[0];if(best)safely(action({action:"move",move:best.uci}));};
$("pv-count").onchange=()=>safely(action({action:"settings",settings:{MultiPV:Number($("pv-count").value)}}));
$("promote").onclick=()=>safely(action({action:"promote"}));
$("import-button").onclick=()=>openImport();$("import-top").onclick=()=>openImport("pgn");$("tab-fen").onclick=()=>setImportKind("fen");$("tab-pgn").onclick=()=>setImportKind("pgn");
$("settings-button").onclick=openSettings;$("settings-top").onclick=openSettings;$("help-button").onclick=()=>$("help-dialog").showModal();
$("new-button").onclick=()=>{if(state.tree.length>1&&!confirm("Открыть новую доску? Для отдельного сохранения текущей партии сначала скачай PGN."))return;safely(action({action:"new"}));};
for(const button of document.querySelectorAll(".close-dialog"))button.onclick=()=>button.closest("dialog").close();
for(const dialog of document.querySelectorAll("dialog"))dialog.addEventListener("click",event=>{if(event.target===dialog){const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)dialog.close();}});
$("import-form").onsubmit=async event=>{event.preventDefault();const button=event.submitter;button.disabled=true;try{await action({action:importKind,text:$("import-text").value});$("import-dialog").close();toast("Позиция загружена");}catch(error){showError("import-error",error.message);}finally{button.disabled=false;}};
$("pgn-file").onchange=async event=>{const file=event.target.files[0];if(!file)return;if(file.size>512*1024){showError("import-error","Файл больше 512 КБ");return;}$("import-text").value=await file.text();};
$("settings-form").onsubmit=async event=>{event.preventDefault();const settings={};for(const input of $("settings-form").querySelectorAll("[data-option]"))settings[input.dataset.option]=input.type==="checkbox"?input.checked:input.type==="number"?Number(input.value):input.value;try{await action({action:"settings",settings});$("settings-dialog").close();toast("Настройки применены");}catch(error){showError("settings-error",error.message);}};
$("clear-hash").onclick=async()=>{try{await action({action:"clear"});toast("Хеш и история поиска очищены");}catch(error){showError("settings-error",error.message);}};
$("copy-fen").onclick=async()=>{try{await navigator.clipboard.writeText(state.fen);toast("FEN скопирован");}catch{ $("fen").focus();$("fen").select();toast("FEN выделен — нажми Ctrl+C");}};
$("export-button").onclick=async()=>{try{const response=await fetch("/api/pgn");if(!response.ok)throw new Error("Не удалось скачать PGN");const blob=await response.blob(),url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download="zenith-analysis.pgn";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(error){toast(error.message,true);}};
$("eval-chart").onclick=event=>{if(!state)return;const rect=$("eval-chart").getBoundingClientRect(),index=Math.max(0,Math.min(state.path.length-1,Math.round((event.clientX-rect.left)/rect.width*(state.path.length-1))));safely(action({action:"navigate",to:state.path[index]}));};
document.addEventListener("keydown",event=>{if(event.target.closest("input,textarea,select,[contenteditable=true]")||document.querySelector("dialog[open]")||(event.target.closest("button")&&event.key===" "))return;const key=event.key.toLowerCase();if(["arrowleft","arrowright","home","end"," ","f","a","?"].includes(key))event.preventDefault();if(key==="f")flip();else if(key===" ")toggleAnalysis();else if(key==="a")$("arrow-toggle").click();else if(key==="?")$("help-dialog").showModal();else{const to={arrowleft:"previous",arrowright:"next",home:"first",end:"last"}[key];if(to)safely(action({action:"navigate",to}));}});

async function connect() {
  try {await refresh();source=new EventSource("/api/events");source.onopen=()=>{connected=true;if(state)renderAnalysis();};source.addEventListener("state",event=>{connected=true;updateState(JSON.parse(event.data));});source.addEventListener("analysis",event=>{const packet=JSON.parse(event.data);if(!state||packet.revision!==state.revision||packet.sequence<state.sequence)return;state.sequence=packet.sequence;state.analysis=packet.analysis;state.error=packet.error;state.options=packet.options;state.settings=packet.settings;state.engine=packet.engine;renderAnalysis();});source.onerror=()=>{connected=false;if(state)renderAnalysis();};}
  catch(error){showError("engine-error",error.message+". Запусти start-analysis.cmd.");setTimeout(connect,2500);}
}
connect();
