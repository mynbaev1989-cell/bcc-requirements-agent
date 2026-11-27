"""Интерактивный HTML-отчёт: один самодостаточный файл (CSS и JS внутри, без сети).

Данные встраиваются JSON-ом (он же литерал JS) в обычный <script> и рисуются из JS через textContent,
поэтому текст модели и документа не может внедрить разметку. В JSON дополнительно экранируются < > &."""

from __future__ import annotations

import html
import json
from collections import Counter

from reqcheck import criteria
from reqcheck.clarify import PRIORITY_RU
from reqcheck.models import CRITICAL, MAJOR, MINOR, SEV_RANK, SEVERITY_RU, RunResult


def _data(result: RunResult) -> dict:
    """Собрать всё, что нужно странице, в простой словарь."""
    doc = result.document
    p = result.passport
    sev = Counter(f.severity for f in result.findings)
    open_q = [q for q in result.questions if not q.answer]
    findings = []
    # Сначала критичные: жюри и автор документа видят главное без прокрутки (порядок внутри уровня — как в отчёте)
    for f in sorted(result.findings, key=lambda x: -SEV_RANK.get(x.severity, 0)):
        findings.append({
            "fid": f.fid, "severity": f.severity, "criterion": criteria.name(f.criterion),
            "title": f.title, "explanation": f.explanation, "example": f.example, "suggestion": f.suggestion,
            "rids": f.rids, "critic": f.critic, "resolution": f.resolution,
            "quotes": [{"text": q.text, "line": q.line} for q in f.quotes],
        })
    questions = [{
        "qid": q.qid, "text": q.text, "priority": q.priority, "addressee": q.addressee, "why": q.why,
        "options": q.options, "assumption": q.assumption, "fids": q.fids, "lines": q.lines, "answer": q.answer,
    } for q in result.questions]
    reqs = []
    for r in result.requirements:
        asm = result.assessments.get(r.rid)
        reqs.append({"rid": r.rid, "source_id": r.source_id if not r.source_id_assigned else "", "summary": r.summary,
                     "quote": r.quote, "kind": r.kind, "line_start": r.line_start, "line_end": r.line_end,
                     "section": r.section, "score": asm.score if asm else None, "light": asm.light if asm else ""})
    return {
        "title": (doc.title if doc else "") or p.get("document_title", "") or "Документ",
        "lines": list(doc.lines) if doc else [],
        "verdict_dev": result.verdict_dev, "verdict_test": result.verdict_test, "score": result.document_score,
        "counts": {CRITICAL: sev[CRITICAL], MAJOR: sev[MAJOR], MINOR: sev[MINOR]},
        "n_requirements": len(result.requirements), "n_questions": len(result.questions),
        "n_open_questions": len(open_q), "n_blockers": sum(1 for q in open_q if q.priority == "blocker"),
        "findings": findings, "questions": questions, "requirements": reqs,
        "sev_ru": SEVERITY_RU, "priority_ru": PRIORITY_RU,
        "passport": {"input_path": p.get("input_path", ""), "sha256": p.get("sha256", ""),
                     "report_questions_file": p.get("report_questions_file", "")},
    }


def _json_safe(data: dict) -> str:
    """JSON для вставки в <script>: < > & заменены на \\u-коды, закрыть тег или начать комментарий нельзя."""
    raw = json.dumps(data, ensure_ascii=False, default=str)
    return (raw.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def render_html(result: RunResult) -> str:
    data = _data(result)
    title = html.escape(data["title"])
    return _TEMPLATE.replace("{{TITLE}}", title).replace("{{DATA}}", _json_safe(data))


_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Отчёт: {{TITLE}}</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--text:#1f2328;--muted:#6a6f76;--line:#e3e1db;--accent:#2c5fb3;
--crit:#d64545;--crit-bg:#fde8e8;--maj:#e8830c;--maj-bg:#fff0dc;--min:#c9a400;--min-bg:#fff8d0;--flash:#ffe08a}
@media (prefers-color-scheme:dark){:root{--bg:#16181b;--panel:#1f2226;--text:#e6e8ea;--muted:#9aa0a6;--line:#31353b;
--accent:#7aa7f0;--crit:#ff7b7b;--crit-bg:#4a2326;--maj:#ffa94d;--maj-bg:#4a3318;--min:#e6c84a;--min-bg:#443d14;--flash:#7a6420}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:16px 20px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{margin:0 0 8px;font-size:20px}
.verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px;margin:8px 0}
.verdict{border:1px solid var(--line);border-radius:8px;padding:8px 12px;background:var(--bg)}
.verdict b{display:block;font-size:12px;text-transform:uppercase;color:var(--muted);letter-spacing:.04em}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.chip{border-radius:999px;padding:2px 12px;font-size:13px;border:1px solid var(--line);background:var(--bg)}
.chip.critical{background:var(--crit-bg);color:var(--crit);border-color:var(--crit)}
.chip.major{background:var(--maj-bg);color:var(--maj);border-color:var(--maj)}
.chip.minor{background:var(--min-bg);color:var(--min);border-color:var(--min)}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;padding:16px 20px;align-items:start}
@media (max-width:860px){main{grid-template-columns:1fr;padding:12px}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.panel>h2{margin:0;padding:10px 14px;font-size:14px;border-bottom:1px solid var(--line);color:var(--muted)}
#doc{max-height:82vh;overflow:auto;font:13px/1.55 ui-monospace,Consolas,monospace;position:relative}
.ln{display:flex;border-left:4px solid transparent}
.ln .n{flex:0 0 3.5em;text-align:right;padding:0 8px;color:var(--muted);user-select:none}
.ln .t{flex:1;white-space:pre-wrap;word-break:break-word;padding-right:8px}
.ln.critical{background:var(--crit-bg);border-left-color:var(--crit);cursor:pointer}
.ln.major{background:var(--maj-bg);border-left-color:var(--maj);cursor:pointer}
.ln.minor{background:var(--min-bg);border-left-color:var(--min);cursor:pointer}
.ln.flash{animation:fl 1.4s}
@keyframes fl{0%,60%{background:var(--flash)}}
.tabs{display:flex;border-bottom:1px solid var(--line)}
.tab{flex:1;padding:10px;border:0;background:none;color:var(--muted);font:inherit;cursor:pointer;border-bottom:2px solid transparent}
.tab[aria-selected=true]{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
.pane{padding:12px;max-height:82vh;overflow:auto}
.pane[hidden],.card[hidden]{display:none}
.filters{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px;font-size:13px}
.card{border:1px solid var(--line);border-left:4px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:10px;background:var(--bg)}
.card.critical{border-left-color:var(--crit)}.card.major{border-left-color:var(--maj)}.card.minor{border-left-color:var(--min)}
.card.sel{outline:2px solid var(--accent)}
.card h3{margin:0 0 4px;font-size:15px}
.meta{font-size:12px;color:var(--muted);margin-bottom:4px}
.quote{border-left:3px solid var(--line);margin:6px 0;padding:2px 10px;color:var(--muted);font-size:13px;white-space:pre-wrap}
.sugg{background:var(--panel);border:1px dashed var(--line);border-radius:6px;padding:6px 10px;margin-top:6px;font-size:13px}
.lbl{font-weight:600}
a.lnk{color:var(--accent);cursor:pointer;text-decoration:underline}
textarea{width:100%;min-height:60px;font:inherit;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px}
label.opt{display:block;margin:2px 0}
button.btn{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;font:inherit;cursor:pointer}
.empty{color:var(--muted);padding:8px}
</style>
</head>
<body>
<header>
<h1 id="title"></h1>
<div class="verdicts">
<div class="verdict"><b>Разработка</b><span id="vdev"></span></div>
<div class="verdict"><b>Тестирование</b><span id="vtest"></span></div>
</div>
<div class="stats" id="stats"></div>
</header>
<main>
<section class="panel"><h2>Исходный документ</h2><div id="doc"></div></section>
<section class="panel">
<div class="tabs" role="tablist">
<button class="tab" role="tab" data-tab="findings" aria-selected="true">Находки</button>
<button class="tab" role="tab" data-tab="questions" aria-selected="false">Вопросы</button>
<button class="tab" role="tab" data-tab="reqs" aria-selected="false">Требования</button>
</div>
<div class="pane" id="pane-findings">
<div class="filters" id="filters"></div><div id="flist"></div></div>
<div class="pane" id="pane-questions" hidden><div id="qlist"></div>
<p><button class="btn" id="dl" type="button">Скачать ответы</button></p></div>
<div class="pane" id="pane-reqs" hidden><div id="rlist"></div></div>
</section>
</main>
<script>
(function(){
"use strict";
// Литерал вместо <script type="application/json">: так отчёт работает и в просмотрщиках, которые исполняют все <script>
var D={{DATA}};
var SEV=["critical","major","minor"],RANK={critical:0,major:1,minor:2};
// Создать элемент; текст всегда через textContent (никакого innerHTML)
function el(tag,cls,text,parent){var e=document.createElement(tag);if(cls)e.className=cls;
 if(text!==undefined&&text!==null)e.textContent=text;if(parent)parent.appendChild(e);return e;}
function $(id){return document.getElementById(id);}

// Шапка
$("title").textContent=D.title;
$("vdev").textContent=D.verdict_dev||"—";
$("vtest").textContent=D.verdict_test||"—";
var st=$("stats");
if(D.score!==null&&D.score!==undefined)el("span","chip","Балл: "+D.score+"/100",st);
el("span","chip","Требований: "+D.n_requirements,st);
SEV.forEach(function(s){el("span","chip "+s,D.sev_ru[s]+": "+D.counts[s],st);});
el("span","chip","Вопросов: "+D.n_questions+" (без ответа "+D.n_open_questions+")",st);
el("span","chip"+(D.n_blockers?" critical":""),"Блокеров: "+D.n_blockers,st);

// Строки документа: на каждую строку — худшая критичность и список находок
var lineSev={},lineF={};
D.findings.forEach(function(f){f.quotes.forEach(function(q){
 if(!q.line)return;
 var cur=lineSev[q.line];
 if(cur===undefined||RANK[f.severity]<RANK[cur])lineSev[q.line]=f.severity;
 (lineF[q.line]=lineF[q.line]||[]);if(lineF[q.line].indexOf(f.fid)<0)lineF[q.line].push(f.fid);});});
var docEl=$("doc"),lineEls={};
D.lines.forEach(function(t,i){var n=i+1;
 var row=el("div","ln"+(lineSev[n]?" "+lineSev[n]:""),null,docEl);row.id="L"+n;
 el("span","n",String(n),row);el("span","t",t,row);lineEls[n]=row;
 if(lineF[n])row.addEventListener("click",function(){openFindings(lineF[n]);});});
function flash(n){var r=lineEls[n];if(!r)return;
 docEl.scrollTo({top:r.offsetTop-docEl.clientHeight/3,behavior:"smooth"});
 r.classList.remove("flash");void r.offsetWidth;r.classList.add("flash");}

// Вкладки
function showTab(name){document.querySelectorAll(".tab").forEach(function(b){
 b.setAttribute("aria-selected",b.dataset.tab===name?"true":"false");});
 ["findings","questions","reqs"].forEach(function(t){$("pane-"+t).hidden=(t!==name);});}
document.querySelectorAll(".tab").forEach(function(b){b.addEventListener("click",function(){showTab(b.dataset.tab);});});

// Фильтр критичности
var on={critical:true,major:true,minor:true},fbox=$("filters");
SEV.forEach(function(s){var l=el("label",null,null,fbox),c=document.createElement("input");
 c.type="checkbox";c.checked=true;c.addEventListener("change",function(){on[s]=c.checked;applyFilter();});
 l.appendChild(c);l.appendChild(document.createTextNode(" "+D.sev_ru[s]));});
function applyFilter(){document.querySelectorAll("#flist .card").forEach(function(c){c.hidden=!on[c.dataset.sev];});}

// Карточки находок
var cards={};
function lineRef(n,parent){var a=el("a","lnk","стр. "+n,parent);a.addEventListener("click",function(e){e.stopPropagation();flash(n);});}
var fl=$("flist");
if(!D.findings.length)el("div","empty","Находок нет.",fl);
D.findings.forEach(function(f){
 var c=el("div","card "+f.severity,null,fl);c.dataset.sev=f.severity;cards[f.fid]=c;
 el("div","meta",f.fid+" · "+(D.sev_ru[f.severity]||f.severity)+" · "+f.criterion+(f.rids.length?" · "+f.rids.join(", "):""),c);
 el("h3",null,f.title,c);
 if(f.explanation)el("div",null,f.explanation,c);
 if(f.example)el("div","meta","Пример: "+f.example,c);
 f.quotes.forEach(function(q){var d=el("div","quote",q.text,c);if(q.line){d.appendChild(document.createTextNode(" "));lineRef(q.line,d);}});
 if(f.suggestion){var s=el("div","sugg",null,c);el("span","lbl","Предложение: ",s);s.appendChild(document.createTextNode(f.suggestion));}
 if(f.resolution)el("div","meta","Учёт ответа автора: "+f.resolution,c);
 c.addEventListener("click",function(){
  Object.keys(cards).forEach(function(k){cards[k].classList.remove("sel");});c.classList.add("sel");
  var first=f.quotes.filter(function(q){return q.line;})[0];if(first)flash(first.line);});});
function openFindings(fids){showTab("findings");
 SEV.forEach(function(s){on[s]=true;});fbox.querySelectorAll("input").forEach(function(i){i.checked=true;});applyFilter();
 Object.keys(cards).forEach(function(k){cards[k].classList.remove("sel");});
 fids.forEach(function(id){if(cards[id])cards[id].classList.add("sel");});
 if(cards[fids[0]])cards[fids[0]].scrollIntoView({behavior:"smooth",block:"center"});}

// Вопросы
var qs=[],ql=$("qlist");
if(!D.questions.length)el("div","empty","Вопросов нет.",ql);
D.questions.forEach(function(q){
 var c=el("div","card"+(q.priority==="blocker"?" critical":""),null,ql);
 el("div","meta",q.qid+" · "+(D.priority_ru[q.priority]||q.priority)+" · Кому: "+q.addressee,c);
 el("h3",null,q.text,c);
 if(q.why){var w=el("div",null,null,c);el("span","lbl","Почему важно: ",w);w.appendChild(document.createTextNode(q.why));}
 var radios=[];
 q.options.forEach(function(o,i){var l=el("label","opt",null,c),r=document.createElement("input");
  r.type="radio";r.name="opt-"+q.qid;l.appendChild(r);l.appendChild(document.createTextNode(" "+(i+1)+") "+o));
  l.addEventListener("dblclick",function(){r.checked=false;});radios.push(r);});
 if(q.assumption)el("div","meta","Если ответа не будет: "+q.assumption,c);
 if(q.fids.length){var m=el("div","meta","Находки: ",c);
  q.fids.forEach(function(id,i){if(i)m.appendChild(document.createTextNode(", "));
   var a2=el("a","lnk",id,m);a2.addEventListener("click",function(){openFindings([id]);});});}
 var ta=el("textarea",null,null,c);ta.placeholder="Свой ответ";ta.value=q.answer||"";
 qs.push({q:q,radios:radios,ta:ta});});

// Файл ответов: тот же формат, что у <отчёт>_questions.md (его читает clarify.parse_answers)
function answerOf(x){var parts=[];
 x.radios.forEach(function(r,i){if(r.checked)parts.push("Вариант "+(i+1)+": "+x.q.options[i]);});
 var t=x.ta.value.trim();if(t)parts.push(t);
 // строка вида «## Q-…» в ответе оборвала бы блок при разборе
 return parts.join("\n").replace(/^(\s*##\s+Q-)/gm," $1");}
function buildFile(){
 var p=D.passport,L=["# Вопросы агента автору требований: "+D.title,""];
 if(p.sha256)L.push("Документ: `"+p.input_path+"` (SHA-256 "+p.sha256.slice(0,16)+"…)","");
 L.push("Ответы собраны в HTML-отчёте. Сохраните файл и запустите повторно с параметром --answers.","");
 qs.forEach(function(x){var q=x.q;
  L.push("## "+q.qid+" · "+(D.priority_ru[q.priority]||q.priority)+" · Кому: "+q.addressee,"","Вопрос: "+q.text);
  if(q.why)L.push("Почему важно: "+q.why);
  if(q.options.length)L.push("Варианты: "+q.options.map(function(o,i){return "("+(i+1)+") "+o;}).join("; "));
  if(q.assumption)L.push("Если ответа не будет: "+q.assumption);
  var refs=q.fids.join(", ");if(q.lines.length)refs+=" (строки: "+q.lines.join(", ")+")";
  L.push("Находки: "+refs);
  L.push(("Ответ: "+answerOf(x)).replace(/\s+$/,""),"");});
 return L.join("\n")+"\n";}
$("dl").addEventListener("click",function(){
 var blob=new Blob([buildFile()],{type:"text/markdown;charset=utf-8"});
 var a=document.createElement("a");a.href=URL.createObjectURL(blob);
 a.download=D.passport.report_questions_file||"report_questions.md";
 document.body.appendChild(a);a.click();a.remove();setTimeout(function(){URL.revokeObjectURL(a.href);},1000);});

// Требования
var rl=$("rlist");
if(!D.requirements.length)el("div","empty","Требований нет.",rl);
D.requirements.forEach(function(r){var c=el("div","card",null,rl);
 el("div","meta",r.rid+(r.source_id?" · "+r.source_id:"")+" · "+r.kind+(r.score!==null?" · балл "+r.score+"/100":"")+" · "+r.section,c);
 el("div",null,r.summary,c);if(r.quote)el("div","quote",r.quote,c);
 var a=el("a","lnk","стр. "+r.line_start+(r.line_end!==r.line_start?"–"+r.line_end:""),c);
 a.addEventListener("click",function(){flash(r.line_start);});});
})();
</script>
</body>
</html>
"""
