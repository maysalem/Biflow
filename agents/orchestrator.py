import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# l'orchestrateur est dans agents/ : ce dossier est déjà sur le chemin de Python
from profiler import OUT_DIR, ROOT

load_dotenv(ROOT / ".env")

PIPELINE = ["profiler", "etl", "kpi", "analyst", "dashboard", "auditor"]
NEXT = {n: (PIPELINE[i + 1] if i + 1 < len(PIPELINE) else "finish")
        for i, n in enumerate(PIPELINE)}

# nom de l'étape -> fichier dans agents/ (qui doit contenir une fonction run())
MODULES = {
    "profiler":  "profiler_agent",
    "etl":       "etl_agent",
    "kpi":       "kpi_agent",
    "analyst":   "analyst_agent",
    "dashboard": "dashboard_agent",
    "auditor":   "auditor_agent",
}

MAX_RETRY = 1         # réessais par agent
MAX_AUDIT_LOOP = 1    # retours en arrière demandés après audit
USE_LLM = "--no-llm" not in sys.argv


class State(TypedDict, total=False):
    run_id: str
    steps: list        # ce que chaque agent a fait
    decisions: list    # décisions d'aiguillage (règle ou LLM)
    retries: dict
    audit_loops: int
    next: str          # prochaine destination choisie
    status: str
    summary: str


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ============================================================================
# LLM : choix parmi des options autorisées + résumé final
# ============================================================================
_LLM = None   # la "boîte" : vide au départ, remplie au premier besoin


def get_llm():
    global _LLM
    if not USE_LLM:
        return None
    model = os.environ.get("LLM_MODEL")
    if not model:
        return None
    if _LLM is None:
        _LLM = init_chat_model(model, temperature=0)
    return _LLM


def text_of(msg) -> str:
    c = msg.content
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def compact(step: dict) -> dict:
    """Version courte d'une étape, pour ne pas envoyer de longs textes au LLM."""
    return {"agent": step["agent"], "status": step["status"],
            "reason": step.get("reason"), "error": (step.get("error") or "")[:300],
            "result": {k: v for k, v in step.get("result", {}).items() if k != "outputs"}}


DECISION_PROMPT = """Tu es l'Orchestrator d'un pipeline BI multi-agents (projet BIFlow, données Olist).
Pipeline : profiler -> etl -> kpi -> analyst -> dashboard -> auditor.
On te donne le résultat de l'agent qui vient de s'exécuter et une liste d'OPTIONS AUTORISÉES.
Tu dois choisir UNE option de cette liste, jamais autre chose.
Principes : un échec temporaire (réseau, limite de débit) justifie un réessai ; une erreur de code ou de données
qui reviendra à l'identique justifie l'arrêt. Ne stoppe après le profiler que si les données sont inexploitables :
des problèmes à corriger par l'ETL ne sont pas une raison d'arrêter. Après un audit non conforme, recalcule l'étape
la plus probablement en cause (écarts de formule -> KPI ; données incohérentes -> ETL) ; n'accepte avec avertissement
que si l'écart paraît mineur.
Réponds UNIQUEMENT par un JSON : {"choice": "<option>", "reason": "une phrase en français"}"""


def llm_choose(agent: str, step: dict, options: dict, state: State):
    """Renvoie (option, raison) choisie par le LLM, ou None si indisponible ou réponse invalide."""
    llm = get_llm()
    if llm is None:
        return None
    payload = {"agent": agent, "step": compact(step), "options": list(options),
               "retries": state.get("retries", {}), "audit_loops": state.get("audit_loops", 0)}
    try:
        ai = llm.invoke([SystemMessage(DECISION_PROMPT),
                         HumanMessage(json.dumps(payload, ensure_ascii=False, default=str))])
        text = text_of(ai)
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
        if data.get("choice") in options:
            return data["choice"], str(data.get("reason", ""))
    except Exception as e:   # quota, réseau, JSON invalide : on retombe sur l'option par défaut
        print(f"  (Orchestrator : LLM indisponible, option par défaut : {str(e)[:120]})")
    return None


def decide(name: str, step: dict, state: State, retries: dict) -> dict:
    """Détermine la prochaine destination. Python fixe les options, le LLM choisit parmi elles."""
    nxt = NEXT[name]
    res = step.get("result", {})
    loops = state.get("audit_loops", 0)

    if step["status"] == "failed":
        if step.get("fatal"):
            return {"agent": name, "next": "finish", "by": "rule",
                    "reason": "erreur fatale (quota épuisé) : arrêt sans réessai"}
        options = {"stop": "finish"}
        if retries.get(name, 0) <= MAX_RETRY:
            options = {"retry": name, **options}
    elif step["status"] == "success" and res.get("stop"):
        return {"agent": name, "next": "finish", "by": "rule",
                "reason": "l'agent a demandé l'arrêt (données inexploitables)"}
    elif name == "auditor" and step["status"] == "success" \
            and res.get("passed", True) is False and loops < MAX_AUDIT_LOOP:
        options = {"recalculate_kpi": "retry_kpi", "recalculate_etl": "retry_etl",
                   "accept_with_warning": "finish"}
    elif name == "profiler" and step["status"] == "success" and res.get("high_issues", 0) > 0:
        options = {"continue": nxt, "stop": "finish"}
    else:
        return {"agent": name, "next": nxt, "by": "rule", "reason": "étape terminée normalement"}

    if len(options) == 1:
        label = next(iter(options))
        return {"agent": name, "next": options[label], "by": "rule",
                "reason": f"une seule option autorisée : {label}"}

    picked = llm_choose(name, step, options, state)
    if picked:
        label, reason = picked
        return {"agent": name, "next": options[label], "by": "llm", "choice": label, "reason": reason}
    label = next(iter(options))   # option par défaut = la première
    return {"agent": name, "next": options[label], "by": "rule", "choice": label,
            "reason": "LLM indisponible : option par défaut"}


# ============================================================================
# NŒUDS DU GRAPHE
# ============================================================================
def make_node(name: str):
    def node(state: State) -> State:
        steps = list(state.get("steps", []))
        decisions = list(state.get("decisions", []))
        retries = dict(state.get("retries", {}))
        start = now()

        # Agent pas encore écrit -> on saute (une vraie erreur d'import reste visible)
        if importlib.util.find_spec(MODULES[name]) is None:
            step = {"agent": name, "status": "skipped",
                    "reason": "agent pas encore implémenté", "at": start}
        else:
            t0 = time.time()
            step = {"agent": name, "at": start}
            try:
                module = importlib.import_module(MODULES[name])
                fn = getattr(module, "run", None)
                if fn is None:
                    step.update(status="skipped", reason="pas de fonction run() dans le module")
                else:
                    result = fn() or {}
                    step.update(status="success" if result.get("ok", True) else "failed",
                                result=result)
            except (Exception, SystemExit) as e:   # SystemExit : ex. LLM_MODEL absent du .env
                step.update(status="failed", error=str(e),
                            trace=traceback.format_exc(limit=4),
                            fatal="quota" in str(e).lower())
            step["seconds"] = round(time.time() - t0, 1)

        steps.append(step)
        if step["status"] == "failed":
            retries[name] = retries.get(name, 0) + 1

        d = decide(name, step, state, retries)
        decisions.append({**d, "at": now()})
        return {"steps": steps, "decisions": decisions, "retries": retries, "next": d["next"]}
    return node


def make_retry(target: str):
    """Retour en arrière demandé après un audit non conforme."""
    def node(state: State) -> State:
        return {"audit_loops": state.get("audit_loops", 0) + 1, "next": target}
    return node


SUMMARY_PROMPT = """Tu es l'Orchestrator d'un pipeline BI multi-agents (projet BIFlow).
Rédige en français un résumé de cette exécution, en 4 à 6 phrases, pour un responsable non technique :
ce qui a été exécuté, ce qui a réussi, échoué ou été ignoré (agents pas encore implémentés), les décisions
d'aiguillage prises et pourquoi, et ce qu'il faut faire ensuite. N'invente aucun chiffre : utilise uniquement
ceux fournis. Réponds par du texte simple, sans JSON."""


def llm_summary(state: State, status: str) -> str:
    llm = get_llm()
    if llm is None:
        return "Résumé LLM non disponible (LLM désactivé ou LLM_MODEL absent)."
    payload = {"status": status,
               "steps": [compact(s) for s in state["steps"]],
               "decisions": state.get("decisions", [])}
    try:
        ai = llm.invoke([SystemMessage(SUMMARY_PROMPT),
                         HumanMessage(json.dumps(payload, ensure_ascii=False, default=str))])
        return text_of(ai).strip()
    except Exception as e:
        return f"Résumé LLM non disponible : {str(e)[:200]}"


def finish(state: State) -> State:
    steps = state["steps"]
    last = steps[-1]
    if last["status"] == "failed":
        status = "failed"
    elif last.get("result", {}).get("stop"):
        status = "stopped"
    else:
        status = "completed"

    summary = llm_summary(state, status)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. journal technique (JSON)
    log = {"run_id": state["run_id"], "status": status, "summary": summary,
           "steps": steps, "decisions": state.get("decisions", [])}
    (OUT_DIR / "orchestrator_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 2. rapport lisible (Markdown)
    L = ["# Rapport de l'Orchestrator", "",
         f"- Exécution : `{state['run_id']}`",
         f"- Statut final : **{status}**", "",
         "## Résumé", "", summary, "",
         "## Étapes exécutées", ""]
    for s in steps:
        dur = f" ({s['seconds']} s)" if "seconds" in s else ""
        L.append(f"- **{s['agent']}** : {s['status']}{dur} (à {s['at']})")
        if s.get("reason"):
            L.append(f"  - Raison : {s['reason']}")
        if s.get("error"):
            L.append(f"  - Erreur : {s['error'][:300]}")
        for k in ("verdict", "issues", "high_issues", "data_gaps", "reused"):
            if k in s.get("result", {}):
                L.append(f"  - {k} : {s['result'][k]}")
    L += ["", "## Décisions d'aiguillage", ""]
    for d in state.get("decisions", []):
        who = "LLM" if d["by"] == "llm" else "règle"
        L.append(f"- après **{d['agent']}** -> `{d['next']}` ({who}) : {d['reason']}")
    (OUT_DIR / "orchestrator_report.md").write_text("\n".join(L), encoding="utf-8")

    return {"status": status, "summary": summary}


def build_graph():
    g = StateGraph(State)
    for name in PIPELINE:
        g.add_node(name, make_node(name))
    g.add_node("retry_kpi", make_retry("kpi"))
    g.add_node("retry_etl", make_retry("etl"))
    g.add_node("finish", finish)

    g.add_edge(START, PIPELINE[0])
    destinations = {n: n for n in PIPELINE}
    destinations.update({"finish": "finish", "retry_kpi": "retry_kpi", "retry_etl": "retry_etl"})
    for name in PIPELINE:
        g.add_conditional_edges(name, lambda s: s["next"], destinations)

    g.add_edge("retry_kpi", "kpi")
    g.add_edge("retry_etl", "etl")
    g.add_edge("finish", END)
    return g.compile()


if __name__ == "__main__":
    app = build_graph()
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    print(f"Lancement de {run_id} ({'avec' if USE_LLM else 'sans'} LLM)...\n")
    final = app.invoke({"run_id": run_id, "steps": [], "decisions": [],
                        "retries": {}, "audit_loops": 0})

    print(f"\nRun {final['run_id']} : {final['status']}")
    for s in final["steps"]:
        extra = f" ({s['seconds']} s)" if "seconds" in s else ""
        print(f"  [{s['status']:8}] {s['agent']}{extra}")
        if s["status"] == "failed":
            print(f"             erreur : {s.get('error', '')[:200]}")
    print("\n" + final.get("summary", ""))
    print("\nJournal : data/profile/orchestrator_log.json")
    print("Rapport : data/profile/orchestrator_report.md")