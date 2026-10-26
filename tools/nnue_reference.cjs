// Independent binary exporter and dense forward reference. Node built-ins only.
// Usage: node tools/nnue_reference.cjs fixture target/synthetic.znnue
//        node tools/nnue_reference.cjs verify target/synthetic.znnue target/walk.tsv
//        node tools/nnue_reference.cjs score target/synthetic.znnue FENS
// Synthetic weights check arithmetic; they are NOT a trained playing network.
'use strict';
const fs = require('node:fs');
const assert = require('node:assert/strict');
const INPUTS=768, HIDDEN=256, HEADER=64, SIZE=394820;
const fields=[1,INPUTS,HIDDEN,1,255,64,400,1];
function hash(bytes) {
  let h=0xcbf29ce484222325n;
  for(let i=0;i<bytes.length;i++) h=BigInt.asUintN(64,(h^BigInt(i>=48&&i<56?0:bytes[i]))*0x100000001b3n);
  return h;
}
function fixture(path) {
  const bytes=Buffer.alloc(SIZE);
  bytes.write('ZNNUE',0,'ascii');
  fields.forEach((v,i)=>bytes.writeUInt32LE(v,8+4*i));
  bytes.writeBigUInt64LE(BigInt(SIZE-HEADER),40);
  let state=0x42a1783d;
  function next(){state^=state<<13;state^=state>>>17;state^=state<<5;return state>>>0;}
  let offset=HEADER;
  for(const [count,range,shift] of [[HIDDEN,601,200],[INPUTS*HIDDEN,65,32],[2*HIDDEN,257,128]]) {
    for(let i=0;i<count;i++,offset+=2) bytes.writeInt16LE(next()%range-shift,offset);
  }
  bytes.writeInt32LE(-12345,offset);
  bytes.writeBigUInt64LE(hash(bytes),48);
  fs.writeFileSync(path,bytes,{flag:'wx'});
  console.log(`Synthetic test fixture only: ${path}, ${bytes.length} bytes`);
}
function load(path) {
  const b=fs.readFileSync(path);
  assert.equal(b.length,SIZE);
  assert.equal(b.subarray(0,8).toString('hex'),'5a4e4e5545000000');
  fields.forEach((v,i)=>assert.equal(b.readUInt32LE(8+4*i),v));
  assert.equal(b.readBigUInt64LE(40),BigInt(SIZE-HEADER));
  assert.equal(b.readBigUInt64LE(48),hash(b));
  assert.equal(b.readBigUInt64LE(56),0n);
  let off=HEADER;
  function values(n,bound){return Array.from({length:n},()=>{const v=b.readInt16LE(off);off+=2;assert.ok(Math.abs(v)<=bound);return v;});}
  const bias=values(HIDDEN,4096),ft=values(INPUTS*HIDDEN,512),out=values(2*HIDDEN,8192),ob=b.readInt32LE(off);
  assert.ok(Math.abs(ob)<=16777216);
  return {bias,ft,out,ob};
}
function forward(net,fen) {
  const [placement,stm]=fen.split(' '),pieces=[];
  let rank=7,file=0;
  for(const ch of placement) {
    if(ch==='/'){rank--;file=0;continue;}
    if(/[1-8]/.test(ch)){file+=Number(ch);continue;}
    const kind='pnbrqk'.indexOf(ch.toLowerCase());assert.ok(kind>=0);
    pieces.push({square:rank*8+file++,kind,color:ch===ch.toUpperCase()?0:1});
  }
  let raw=net.ob;
  const us=stm==='w'?0:1;
  for(let half=0;half<2;half++) {
    const p=us^half;
    for(let neuron=0;neuron<HIDDEN;neuron++) {
      let sum=net.bias[neuron];
      for(const piece of pieces) {
        const square=p===1?piece.square^56:piece.square;
        const feature=((piece.color===p?0:1)*6+piece.kind)*64+square;
        sum+=net.ft[feature*HIDDEN+neuron];
      }
      raw+=Math.max(0,Math.min(255,sum))*net.out[half*HIDDEN+neuron];
    }
  }
  return [raw,Math.max(-28000,Math.min(28000,Math.trunc(raw*400/(255*64))))];
}
const [command,path,trace]=process.argv.slice(2);
if(command==='fixture'&&path&&!trace) fixture(path);
else if(command==='score'&&path&&trace) {
  const net=load(path);
  for(const fen of fs.readFileSync(trace,'utf8').replace(/^\uFEFF/,'').split(/\r?\n/)) {
    if(!fen.trim()||fen.startsWith('#'))continue;
    console.log([fen,...forward(net,fen)].join('\t'));
  }
}
else if(command==='verify'&&path&&trace) {
  const net=load(path);
  let count=0;
  for(const line of fs.readFileSync(trace,'utf8').replace(/^\uFEFF/,'').split(/\r?\n/)) {
    if(!line.trim())continue;
    const [fen,raw,score]=line.split('\t');
    assert.deepEqual(forward(net,fen),[Number(raw),Number(score)],`row ${count+1}: ${fen}`);
    count++;
  }
  assert.ok(count>0,'empty trace');
  console.log(`Independent Node dense reference matches Rust: ${count} positions`);
} else {console.error('usage: fixture OUTPUT | score NETWORK FENS | verify NETWORK TRACE');process.exitCode=1;}
