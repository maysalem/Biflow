import json
import os
import re
import statistics
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from profiler import OUT_DIR, RAW_DIR, ROOT, category_translation_status, load_profile

load_dotenv(ROOT / ".env")
# Le profil est recalculé automatiquement s'il est périmé (ou avec l'option --refresh)
PROFILE = load_profile(refresh="--refresh" in sys.argv)


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


@tool
def get_schema() -> str:
    """Schéma de toutes les tables : colonnes avec leur type, clés candidates, nombre de lignes. Sert à vérifier quelles données existent (ex. coût d'achat, stock, événements, sessions)."""
    return _j({t: {"rows": tab["rows"], "key_candidates": tab["key_candidates"],
                   "columns": {c: i["dtype"] for c, i in tab["columns"].items()}}
               for t, tab in PROFILE["tables"].items()})


@tool
def get_profile_overview() -> str:
    """Vue d'ensemble : score global et son détail, et par table : lignes, doublons, % moyen de manquants, % moyen d'atypiques, score."""
    keys = ("rows", "duplicates", "duplicates_pct", "avg_missing_pct", "avg_outlier_pct", "reliability_score")
    return _j({"global_score": PROFILE["global_score"], "score_details": PROFILE["score_details"],
               "tables": {n: {k: t[k] for k in keys} for n, t in PROFILE["tables"].items()}})


@tool
def list_flagged_columns() -> str:
    """Colonnes avec un point d'attention : manquants > 1 %, type suspect (dates ou nombres en texte), négatifs inattendus, atypiques > 5 %, et tables avec doublons > 1 %."""
    out = []
    for tname, t in PROFILE["tables"].items():
        for col, i in t["columns"].items():
            notes = list(i["hints"])
            if i["missing_pct"] > 1:
                notes.append(f"{i['missing_count']} manquants ({i['missing_pct']} %)")
            if i.get("negatives_unexpected"):
                notes.append(f"{i['negatives']} valeurs négatives inattendues")
            if i.get("outlier_pct", 0) > 5:
                notes.append(f"{i['outlier_pct']} % atypiques")
            if notes:
                out.append({"table": tname, "column": col, "notes": notes})
        if t["duplicates_pct"] > 1:
            out.append({"table": tname, "column": None,
                        "notes": [f"{t['duplicates']} lignes en double ({t['duplicates_pct']} %)"]})
    return _j(out)


@tool
def get_column_profile(table: str, column: str) -> str:
    """Profil détaillé d'une colonne : type, manquants, unicité, min, max, moyenne, valeurs fréquentes, dates."""
    try:
        return _j(PROFILE["tables"][table]["columns"][column])
    except KeyError:
        return "Table ou colonne inconnue."


@tool
def get_relations_report() -> str:
    """Relations entre tables : relations de référence vérifiées (lignes orphelines ET valeurs distinctes orphelines, avec exemples) et relations détectées automatiquement, avec l'évaluation de la détection."""
    return _j({"verified": PROFILE["relations"], "detected": PROFILE["detected_relations"],
               "detection_eval": PROFILE["relation_detection_eval"]})


@tool
def get_business_rule_results() -> str:
    """Résultats des règles de cohérence métier : commandes livrées sans date, dates manquantes par statut, commandes sans article ou sans paiement, dates incohérentes, prix <= 0, poids nuls, écart articles/paiements."""
    return _j(PROFILE["business_rules"])


@tool
def get_category_translation_status() -> str:
    """État des traductions de catégories de produits : produits sans catégorie, produits dont la catégorie n'a pas de traduction (LIGNES) et catégories distinctes non traduites (VALEURS), avec le nombre de produits par catégorie."""
    try:
        return _j(category_translation_status())
    except Exception as e:
        return f"Erreur : {e}"


@tool
def get_value_counts(table: str, column: str) -> str:
    """Répartition des 15 valeurs les plus fréquentes d'une colonne de la table brute (ex. order_status, payment_type, customer_state)."""
    f = RAW_DIR / f"{table}.csv"
    if not f.exists():
        return "Table inconnue."
    try:
        s = pd.read_csv(f, usecols=[column])[column]
    except ValueError:
        return "Colonne inconnue."
    return _j({"total_rows": int(len(s)), "missing": int(s.isna().sum()), "distinct": int(s.nunique()),
               "top": {str(k): int(v) for k, v in s.value_counts().head(15).items()}})


@tool
def get_date_coverage(table: str, column: str) -> str:
    """Période couverte par une colonne de dates : min, max, premiers et derniers mois, et mois suspects à volume très faible (période incomplète, risque de fausses alertes de baisse)."""
    try:
        col = PROFILE["tables"][table]["columns"][column]
    except KeyError:
        return "Table ou colonne inconnue."
    m = col.get("monthly_counts")
    if not m:
        return "Cette colonne n'est pas une colonne de dates détectée."
    median = statistics.median(m.values())
    ks = list(m)
    return _j({"min": col["date_range"]["min"], "max": col["date_range"]["max"], "months": len(m),
               "median_rows_per_month": median,
               "suspect_low_volume_months": {k: v for k, v in m.items() if v < 0.3 * median},
               "first_months": {k: m[k] for k in ks[:3]}, "last_months": {k: m[k] for k in ks[-3:]}})


@tool
def sample_rows(table: str, n: int = 5) -> str:
    """Retourne n lignes (maximum 10) d'une table brute pour voir de vraies valeurs."""
    f = RAW_DIR / f"{table}.csv"
    if not f.exists():
        return "Table inconnue."
    return pd.read_csv(f, nrows=max(1, min(n, 10))).to_string()


TOOLS = [get_schema, get_profile_overview, list_flagged_columns, get_column_profile,
         get_relations_report, get_business_rule_results, get_category_translation_status,
         get_value_counts, get_date_coverage, sample_rows]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}



SYSTEM_PROMPT = """Tu es le Data Profiler Agent d'un pipeline BI multi-agents (projet BIFlow).
Jeu de données : Olist, marketplace e-commerce brésilien en 9 tables (clients, commandes, lignes de
commande, paiements, avis, produits, vendeurs, géolocalisation, traduction des catégories).
Périmètre du projet : TOUS les produits et toutes les catégories du jeu de données, sans filtre de domaine.
Objectifs métier du projet COMPLET : chiffre d'affaires, nombre de commandes, panier moyen, best-sellers,
marge, rotation du stock, ventes par catégorie / vendeur / région, délais et retards de livraison,
satisfaction client (avis), détection des tendances et des baisses de ventes, détection d'anomalies,
explication des variations (événements, saisonnalité), et traçabilité de tous les calculs.
Tu agis AVANT l'orchestrateur. Ta décision sera lue par l'Orchestrator, l'agent ETL, l'agent KPI,
l'agent BI Analyst et l'Auditor.

DÉMARCHE OBLIGATOIRE (appelle les outils, au minimum dans cet ordre) :
1. get_schema, get_profile_overview, list_flagged_columns
2. get_business_rule_results, get_relations_report
3. get_category_translation_status et sample_rows sur product_category_name_translation (style des noms anglais)
4. get_value_counts(olist_orders_dataset, order_status)
5. get_date_coverage(olist_orders_dataset, order_purchase_timestamp)
6. get_value_counts(olist_order_payments_dataset, payment_type) et get_value_counts(olist_order_reviews_dataset, review_score)
7. En cas de doute : get_column_profile, get_value_counts ou sample_rows avant de conclure.

RÈGLES :
- N'invente AUCUN nombre. Chaque problème doit avoir un champ "evidence" qui cite l'outil et les chiffres exacts.
- Distingue toujours LIGNES et VALEURS DISTINCTES : une relation avec 13 lignes orphelines n'a pas forcément
  13 valeurs distinctes ; un produit sans catégorie n'est pas une catégorie sans traduction.
- N'affirme pas qu'un manque est "normal" sans preuve : appuie-toi sur les règles métier et sur la répartition
  par statut. Signale aussi les anomalies réelles (ex. commandes livrées sans date).
- Jugement métier : une latitude ou longitude négative au Brésil est normale ; les colonnes de texte libre des avis
  sont facultatives ; un chiffre d'affaires ne doit pas compter les commandes annulées ou indisponibles
  (indique l'action ETL correspondante). La géolocalisation sert aux analyses par région : juge si la table est
  exploitable (doublons par code postal) ou si les colonnes d'état et de ville des clients suffisent.
- Valeurs atypiques = à examiner, pas des erreurs. Dates ou nombres stockés en texte = problème pour l'ETL.
- Évalue chaque besoin du projet (liste des objectifs ci-dessus) : pour chacun, vérifie dans le schéma que les
  colonnes nécessaires existent. Un besoin sans donnée (coût d'achat, stock, événements, sessions de visite, etc.)
  va dans "data_gaps" avec les KPI touchés, les options possibles (donnée simulée et déclarée, donnée externe,
  KPI retiré) et ta recommandation. Ce sont des décisions à faire prendre par un humain, pas des détails.
- Si la couverture temporelle montre des mois incomplets, signale le risque de fausses alertes de baisse
  et dis comment l'agent BI Analyst doit les exclure.
- Gravité : high = fausserait un KPI du projet si non traité ; medium = dégrade un segment ou une analyse ;
  low = mineur.
- Traductions : pour chaque catégorie non traduite, propose une traduction anglaise dans le même style que la
  colonne anglaise de la table de traduction. Ce sont des PROPOSITIONS à faire valider par un humain.
- N'attribue jamais un problème à une colonne sans l'avoir vue dans les sorties des outils.

SORTIE : quand tu as fini, réponds UNIQUEMENT avec un JSON valide, sans texte autour, de la forme :
{"summary": "4 à 6 phrases en français",
 "fit_for_project": {"verdict": "yes|partially|no", "reasoning": "..."},
 "issues": [{"id": "I1", "table": "...", "column": "... ou null", "severity": "high|medium|low",
             "relevant_for_kpis": true, "diagnosis": "...", "evidence": "outil + chiffres exacts",
             "action_for_etl": "..."}],
 "etl_plan": [{"order": 1, "step": "...", "tables": ["..."]}],
 "proposed_translations": [{"original": "...", "english": "...", "confidence": "high|medium|low"}],
 "confidence": "high|medium|low"}],
 "data_gaps": [{"need": "...", "kpis_affected": ["..."], "options": ["..."], "recommendation": "..."}]}
Classe les issues par priorité décroissante."""


# ============================================================================
# OUTILS DE L'AGENT : appel du LLM, lecture de la réponse, garde-fous
# ============================================================================
def text_of(msg) -> str:
    c = msg.content
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c



def parse_json(text: str) -> dict:
    import json
    import re

    text = text.strip()

    # Retirer les balises Markdown ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Extraire l'objet JSON
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("Pas de JSON valide trouvé dans la réponse.")

    json_text = text[start:end + 1]

    # Première tentative : JSON normal
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    
    json_fixed = re.sub(
        r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
        r'\\\\',
        json_text
    )

    # Deuxième tentative
    try:
        return json.loads(json_fixed)
    except json.JSONDecodeError as e:
        extrait = json_text[max(0, e.pos - 80):e.pos + 80]
        raise ValueError(
            f"Réponse finale non JSON : {e}\n"
            f"Extrait problématique : {extrait}"
        ) from e



def _is_daily_quota_exhausted(message: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", message.lower())
    return "quota" in normalized and "perday" in normalized


def invoke_with_retry(llm, messages, attempts: int = 4):
    """Réessaie en cas de limite de débit ou d'erreur temporaire ; les autres erreurs remontent tout de suite."""
    for k in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:
            msg = str(e).lower()
            if _is_daily_quota_exhausted(msg):
                raise RuntimeError(
                    "Le quota journalier du modèle LLM est épuisé. "
                    "Aucune nouvelle tentative automatique ne sera faite : attendez son renouvellement, "
                    "vérifiez les quotas et la facturation, ou configurez un autre modèle dans LLM_MODEL. "
                    f"Détail du fournisseur : {e}"
                ) from e
            transient = any(t in msg for t in ("429", "resource_exhausted", "503", "unavailable", "timed out"))
            if k == attempts - 1 or not transient:
                raise
            wait = 20 * (k + 1)
            print(f"  (limite de débit ou erreur temporaire : nouvelle tentative dans {wait} s)")
            time.sleep(wait)


def _norm(text: str) -> str:
    text = re.sub(r"(?<=\d)[ \u202f\u00a0](?=\d{3}\b)", "", text)   # "261 831" -> "261831"
    return re.sub(r"(?<=\d),(?=\d)", ".", text)                    # "26,18" -> "26.18"


def validate(decision: dict, tool_outputs: str) -> dict:
    """Garde-fous : (1) les tables et colonnes citées existent, (2) une preuve est fournie,
    (3) les chiffres cités se retrouvent dans les sorties des outils, (4) les traductions concernent
    de vraies catégories non traduites."""
    outputs = _norm(tool_outputs)
    for issue in decision.get("issues", []):
        t = PROFILE["tables"].get(issue.get("table"))
        col = issue.get("column")
        issue["verified"] = bool(t is not None and (col in (None, "null", "") or col in t["columns"]))
        issue["has_evidence"] = bool(str(issue.get("evidence", "")).strip())
        claim = _norm(f"{issue.get('diagnosis', '')} {issue.get('evidence', '')}")
        numbers = re.findall(r"\d+(?:\.\d+)?", claim)
        issue["ungrounded_numbers"] = sorted({n for n in numbers if len(n) >= 3 and n not in outputs})
    try:
        real = category_translation_status()["untranslated_categories"]
    except Exception:
        real = {}
    for tr in decision.get("proposed_translations", []):
        tr["verified"] = tr.get("original") in real
        tr["status"] = "proposition à valider par un humain"
    return decision


def run_profiler_agent(max_steps: int = 18) -> dict:
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL est absent du fichier .env (ex. google_genai:gemini-2.5-flash).")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
    llm = init_chat_model(model, temperature=temperature).bind_tools(TOOLS)

    messages = [SystemMessage(SYSTEM_PROMPT),
                HumanMessage("Analyse la qualité des données Olist et donne ta décision finale.")]
    trace, outputs, repairs = [], [], 0

    for step in range(1, max_steps + 1):
        ai = invoke_with_retry(llm, messages)
        messages.append(ai)

        if not ai.tool_calls:
            try:
                decision = parse_json(text_of(ai))
            except (json.JSONDecodeError, ValueError):
                repairs += 1
                if repairs > 2:
                    raise RuntimeError("Réponse finale non JSON : " + text_of(ai)[:300])
                messages.append(HumanMessage(
                    "Ta réponse n'est pas un JSON valide. Renvoie UNIQUEMENT le JSON demandé, sans texte autour."))
                continue
            decision["trace"] = trace
            return validate(decision, "\n".join(outputs))

        for call in ai.tool_calls:
            name, args = call["name"], call["args"]
            print(f"  [étape {step}] outil appelé : {name} {args}")
            trace.append({"step": step, "tool": name, "args": args})
            fn = TOOLS_BY_NAME.get(name)
            try:
                result = str(fn.invoke(args)) if fn else f"Outil inconnu : {name}"
            except Exception as e:
                result = f"Erreur de l'outil : {e}"
            outputs.append(result)
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    raise RuntimeError("L'agent n'a pas conclu dans le nombre d'étapes autorisé.")


# ============================================================================
# RAPPORTS
# ============================================================================
def write_agent_report(decision: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "profiler_agent_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    fit = decision.get("fit_for_project", {})
    L = ["# Synthèse du Data Profiler Agent", "", decision.get("summary", ""), "",
         f"**Adéquation au projet : {fit.get('verdict', '?')}** - {fit.get('reasoning', '')}", "",
         "## Problèmes priorisés", ""]
    rank = {"high": 0, "medium": 1, "low": 2}
    for i in sorted(decision.get("issues", []), key=lambda x: rank.get(x.get("severity"), 3)):
        flags = []
        if not i.get("verified"):
            flags.append("table ou colonne introuvable")
        if not i.get("has_evidence"):
            flags.append("sans preuve")
        if i.get("ungrounded_numbers"):
            flags.append("chiffres non retrouvés dans les outils : " + ", ".join(i["ungrounded_numbers"]))
        warn = f" [À CONTRÔLER : {' ; '.join(flags)}]" if flags else ""
        kpi = "touche les KPI" if i.get("relevant_for_kpis") else "sans effet sur les KPI"
        L += [f"- **[{i.get('severity')}] {i.get('table')}.{i.get('column')}** ({kpi}){warn}",
              f"  - Diagnostic : {i.get('diagnosis')}",
              f"  - Preuve : {i.get('evidence')}",
              f"  - Action ETL : {i.get('action_for_etl')}"]

    L += ["", "## Plan pour l'agent ETL", ""]
    for s in sorted(decision.get("etl_plan", []), key=lambda x: x.get("order", 99)):
        L.append(f"{s.get('order')}. {s.get('step')} ({', '.join(s.get('tables', []))})")

    L += ["", "## Traductions proposées (à valider par un humain)", ""]
    for tr in decision.get("proposed_translations", []):
        flag = "" if tr.get("verified") else " (catégorie non reconnue : ignorer)"
        L.append(f"- {tr.get('original')} -> {tr.get('english')} (confiance : {tr.get('confidence')}){flag}")

    L += ["", "## Données manquantes pour le projet : décisions à prendre", ""]
    for g in decision.get("data_gaps", []):
        L += [f"- **{g.get('need')}** (KPI touchés : {', '.join(g.get('kpis_affected', []))})",
              f"  - Options : {' ; '.join(g.get('options', []))}",
              f"  - Recommandation : {g.get('recommendation')}"]

    L += ["", "## Traçabilité : outils appelés par l'agent", ""]
    L += [f"- étape {t['step']} : {t['tool']} {t['args']}" for t in decision.get("trace", [])]
    (OUT_DIR / "agent_report.md").write_text("\n".join(L), encoding="utf-8")

if __name__ == "__main__":
    print("Le Data Profiler Agent réfléchit...")
    decision = run_profiler_agent()
    write_agent_report(decision)
    print("\n" + decision.get("summary", ""))
    print("\nRapport : data/profile/agent_report.md")
    print("Décision pour l'ETL : data/profile/profiler_agent_decision.json")