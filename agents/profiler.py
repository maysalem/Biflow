import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "olist"
OUT_DIR = ROOT / "data" / "profile"
PROFILE_PATH = OUT_DIR / "profile.json"

# Colonnes de texte libre : vides par nature, non pénalisées dans le score
OPTIONAL_COLUMNS = {"review_comment_title", "review_comment_message"}
# Colonnes où une valeur négative serait anormale
NON_NEGATIVE_HINTS = ("price", "freight", "weight", "length", "height", "width",
                      "payment_value", "installments", "qty")
# Colonnes pour lesquelles les "valeurs atypiques" statistiques n'ont pas de sens
NO_OUTLIER_SUFFIXES = ("_id", "_lat", "_lng", "_prefix", "_sequential")

# Relations de référence : (table enfant, colonne, table parent, colonne)
RELATIONS = [
    ("olist_orders_dataset", "customer_id", "olist_customers_dataset", "customer_id"),
    ("olist_order_items_dataset", "order_id", "olist_orders_dataset", "order_id"),
    ("olist_order_items_dataset", "product_id", "olist_products_dataset", "product_id"),
    ("olist_order_items_dataset", "seller_id", "olist_sellers_dataset", "seller_id"),
    ("olist_order_payments_dataset", "order_id", "olist_orders_dataset", "order_id"),
    ("olist_order_reviews_dataset", "order_id", "olist_orders_dataset", "order_id"),
    ("olist_products_dataset", "product_category_name",
     "product_category_name_translation", "product_category_name"),
]


def is_text(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or s.dtype == object


def is_date_name(name: str) -> bool:
    return "date" in name or "timestamp" in name or name.endswith("_at")


# ----------------------------------------------------------------------------
# 1. Profil d'une colonne et d'une table
# ----------------------------------------------------------------------------
def profile_column(s: pd.Series) -> dict:
    name = str(s.name)
    info = {
        "dtype": str(s.dtype),
        "missing_count": int(s.isna().sum()),
        "missing_pct": round(float(s.isna().mean() * 100), 2),
        "unique": int(s.nunique()),
        "hints": [],
    }
    if pd.api.types.is_numeric_dtype(s):
        x = s.dropna()
        if len(x):
            negatives = int((x < 0).sum())
            info.update(
                min=float(x.min()), max=float(x.max()), mean=round(float(x.mean()), 2),
                negatives=negatives,
                negatives_unexpected=bool(negatives and any(k in name for k in NON_NEGATIVE_HINTS)),
            )
            if not name.endswith(NO_OUTLIER_SUFFIXES):
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                out = int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum())
                info.update(outliers=out, outlier_pct=round(out / len(x) * 100, 2))
            if info["unique"] <= 12:
                info["top_values"] = {str(k): int(v) for k, v in x.value_counts().head(10).items()}
    elif is_text(s):
        x = s.dropna().astype(str)
        if len(x):
            if info["unique"] <= 30:
                info["top_values"] = {str(k): int(v) for k, v in x.value_counts().head(10).items()}
            sample = x.sample(min(len(x), 2000), random_state=0)
            if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.95:
                info["hints"].append("nombres stockés comme texte")
            if is_date_name(name):
                parsed = pd.to_datetime(x, errors="coerce")
                if parsed.notna().mean() > 0.95:
                    info["hints"].append("dates stockées comme texte (à convertir)")
                    p = parsed.dropna()
                    info["date_range"] = {"min": str(p.min()), "max": str(p.max())}
                    m = p.dt.to_period("M").value_counts().sort_index()
                    info["monthly_counts"] = {str(k): int(v) for k, v in m.items()}
    return info


def profile_table(df: pd.DataFrame) -> dict:
    cols = {c: profile_column(df[c]) for c in df.columns}
    n_rows = len(df)
    dup = int(df.duplicated().sum())
    dup_pct = dup / n_rows * 100 if n_rows else 0.0

    used = [i["missing_pct"] for c, i in cols.items() if c not in OPTIONAL_COLUMNS]
    missing = float(np.mean(used)) if used else 0.0
    outs = [i["outlier_pct"] for i in cols.values() if "outlier_pct" in i]
    outlier_rate = float(np.mean(outs)) if outs else 0.0

    # Score de fiabilité d'une table : choix de conception (poids à justifier dans le rapport)
    score = max(0.0, min(100.0, 100 - missing - dup_pct - 0.5 * outlier_rate))
    return {
        "rows": n_rows, "columns_count": df.shape[1],
        "duplicates": dup, "duplicates_pct": round(dup_pct, 2),
        "avg_missing_pct": round(missing, 2),
        "avg_outlier_pct": round(outlier_rate, 2),
        "reliability_score": round(score, 1),
        "key_candidates": [c for c in df.columns if df[c].notna().all() and df[c].is_unique],
        "columns": cols,
    }


# ----------------------------------------------------------------------------
# 2. Relations entre tables
# ----------------------------------------------------------------------------
def check_relations(tables: dict) -> list:
    """Vérifie les relations de référence. Distingue LIGNES orphelines et VALEURS DISTINCTES orphelines."""
    results = []
    for child, ccol, parent, pcol in RELATIONS:
        if child not in tables or parent not in tables:
            continue
        if ccol not in tables[child].columns or pcol not in tables[parent].columns:
            continue
        values = tables[child][ccol].dropna()
        parent_set = set(tables[parent][pcol].dropna())
        bad = values[~values.isin(parent_set)]
        results.append({
            "child": f"{child}.{ccol}", "parent": f"{parent}.{pcol}",
            "checked_rows": int(len(values)),
            "orphan_rows": int(len(bad)),
            "orphan_distinct_values": int(bad.nunique()),
            "orphan_pct": round(len(bad) / len(values) * 100, 2) if len(values) else 0.0,
            "orphan_examples": {str(k): int(v) for k, v in bad.value_counts().head(20).items()},
        })
    return results


def detect_relations(tables: dict, profiles: dict, min_match: float = 0.95) -> list:
    """Découvre les relations probables : clé candidate + colonne de même nom dont les valeurs existent dans la clé."""
    found, seen = [], set()
    for parent, prof in profiles.items():
        for pcol in prof["key_candidates"]:
            parent_values = set(tables[parent][pcol])
            for child, cdf in tables.items():
                if child == parent or pcol not in cdf.columns:
                    continue
                values = cdf[pcol].dropna()
                if values.empty:
                    continue
                match = float(values.isin(parent_values).mean())
                key = frozenset([f"{child}.{pcol}", f"{parent}.{pcol}"])
                if match >= min_match and key not in seen:
                    seen.add(key)
                    found.append({
                        "child": f"{child}.{pcol}", "parent": f"{parent}.{pcol}",
                        "match_pct": round(match * 100, 2),
                        "type": "1-1" if cdf[pcol].is_unique else "N-1",
                    })
    return found


# ----------------------------------------------------------------------------
# 3. Règles métier (cohérence)
# ----------------------------------------------------------------------------
def business_rules(tables: dict) -> list:
    orders = tables.get("olist_orders_dataset")
    items = tables.get("olist_order_items_dataset")
    pays = tables.get("olist_order_payments_dataset")
    prods = tables.get("olist_products_dataset")
    results = []

    def has(df, *cols):
        return df is not None and set(cols) <= set(df.columns)

    def add(rule, table, violations, total, comment="", informative=False):
        results.append({
            "rule": rule, "table": table, "violations": int(violations), "total": int(total),
            "pct": round(int(violations) / int(total) * 100, 3) if total else 0.0,
            "comment": comment, "informative": informative,
        })

    def by_status(mask):
        vc = orders.loc[mask, "order_status"].value_counts().head(5)
        return "statuts : " + "; ".join(f"{k}: {int(v)}" for k, v in vc.items())

    def r_missing_delivery_by_status():
        if has(orders, "order_status", "order_delivered_customer_date"):
            mask = orders["order_delivered_customer_date"].isna()
            add("Dates de livraison client manquantes (répartition par statut)",
                "olist_orders_dataset", mask.sum(), len(orders), by_status(mask), informative=True)

    def r_delivered_without_date():
        if has(orders, "order_status", "order_delivered_customer_date"):
            d = orders[orders["order_status"] == "delivered"]
            add("Commandes 'delivered' sans date de livraison client",
                "olist_orders_dataset", d["order_delivered_customer_date"].isna().sum(), len(d))

    def r_nondelivered_with_date():
        if has(orders, "order_status", "order_delivered_customer_date"):
            d = orders[orders["order_status"] != "delivered"]
            add("Commandes non 'delivered' avec une date de livraison client",
                "olist_orders_dataset", d["order_delivered_customer_date"].notna().sum(), len(d))

    def r_orders_without_items():
        if has(orders, "order_id", "order_status") and has(items, "order_id"):
            mask = ~orders["order_id"].isin(set(items["order_id"]))
            add("Commandes sans aucune ligne d'article", "olist_orders_dataset",
                mask.sum(), len(orders), by_status(mask))

    def r_orders_without_payment():
        if has(orders, "order_id", "order_status") and has(pays, "order_id"):
            mask = ~orders["order_id"].isin(set(pays["order_id"]))
            add("Commandes sans aucun paiement", "olist_orders_dataset",
                mask.sum(), len(orders), by_status(mask))

    def r_delivery_before_purchase():
        if has(orders, "order_purchase_timestamp", "order_delivered_customer_date"):
            a = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
            b = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
            both = a.notna() & b.notna()
            add("Livraison client antérieure à la date d'achat", "olist_orders_dataset",
                (b[both] < a[both]).sum(), both.sum())

    def r_nonpositive_price():
        if has(items, "price"):
            add("Lignes d'article avec un prix <= 0", "olist_order_items_dataset",
                (items["price"] <= 0).sum(), items["price"].notna().sum())

    def r_zero_weight():
        if has(prods, "product_weight_g"):
            w = prods["product_weight_g"].dropna()
            add("Produits avec un poids égal à 0", "olist_products_dataset", (w == 0).sum(), len(w))

    def r_payment_reconciliation():
        if has(items, "order_id", "price", "freight_value") and has(pays, "order_id", "payment_value"):
            it = items.groupby("order_id")[["price", "freight_value"]].sum().sum(axis=1)
            pay = pays.groupby("order_id")["payment_value"].sum()
            gap = (it - pay.reindex(it.index)).abs().dropna() > 1
            add("Total articles + livraison différent du total payé (écart > 1)",
                "olist_order_items_dataset", gap.sum(), len(gap),
                "à investiguer avant de choisir la définition du chiffre d'affaires (articles ou paiements)")
    def r_late_delivery():
        if has(orders, "order_delivered_customer_date", "order_estimated_delivery_date"):
            a = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
            b = pd.to_datetime(orders["order_estimated_delivery_date"], errors="coerce")
            both = a.notna() & b.notna()
            add("Commandes livrées après la date de livraison estimée",
                "olist_orders_dataset", (a[both] > b[both]).sum(), both.sum(),
                "mesure des retards pour le KPI de délais de livraison", informative=True)

    for rule in (r_missing_delivery_by_status, r_delivered_without_date, r_nondelivered_with_date,
                 r_orders_without_items, r_orders_without_payment, r_delivery_before_purchase,
                 r_nonpositive_price, r_zero_weight, r_payment_reconciliation, r_late_delivery):
        try:
            rule()
        except Exception as e:  # une règle qui échoue ne doit pas bloquer le profil
            results.append({"rule": rule.__name__, "error": str(e)})
    return results


# ----------------------------------------------------------------------------
# 4. État des traductions de catégories
# ----------------------------------------------------------------------------
def category_translation_status(raw_dir=RAW_DIR) -> dict:
    p = pd.read_csv(Path(raw_dir) / "olist_products_dataset.csv", usecols=["product_category_name"])
    t = pd.read_csv(Path(raw_dir) / "product_category_name_translation.csv")
    cat = p["product_category_name"]
    present = cat.dropna()
    untranslated = present[~present.isin(set(t["product_category_name"]))]
    return {
        "products_total": int(len(p)),
        "products_without_category": int(cat.isna().sum()),
        "products_with_untranslated_category": int(len(untranslated)),
        "distinct_untranslated_categories": int(untranslated.nunique()),
        "untranslated_categories": {str(k): int(v) for k, v in untranslated.value_counts().items()},
        "translation_table_columns": list(t.columns),
    }


# ----------------------------------------------------------------------------
# 5. Rapport lisible
# ----------------------------------------------------------------------------
def write_report(profile: dict, path: Path) -> None:
    sd = profile["score_details"]
    L = ["# Data Quality Report - Olist", "",
         f"Score de fiabilité global : **{profile['global_score']}/100**", "",
         f"Détail : tables {sd['tables_mean']} (poids 60 %), relations {sd['relations']} (20 %), "
         f"cohérence métier {sd['consistency']} (20 %).", "",
         "## 1. Tables", "",
         "| Table | Lignes | Colonnes | Doublons | Manquants moy. | Atypiques moy. | Score |",
         "|---|---|---|---|---|---|---|"]
    for n, t in profile["tables"].items():
        L.append(f"| {n} | {t['rows']} | {t['columns_count']} | {t['duplicates']} "
                 f"| {t['avg_missing_pct']} % | {t['avg_outlier_pct']} % | {t['reliability_score']} |")

    L += ["", "## 2. Relations vérifiées", "",
          "| Enfant | Parent | Lignes orphelines | Valeurs distinctes orphelines | % |",
          "|---|---|---|---|---|"]
    for r in profile["relations"]:
        L.append(f"| {r['child']} | {r['parent']} | {r['orphan_rows']} "
                 f"| {r['orphan_distinct_values']} | {r['orphan_pct']} % |")

    ev = profile["relation_detection_eval"]
    L += ["", "## 3. Relations détectées automatiquement", "",
          f"Relations de référence retrouvées : {ev['retrieved']}/{ev['declared']} ; "
          f"relations supplémentaires proposées : {ev['extra']}.", "",
          "| Enfant | Parent | Correspondance | Type |", "|---|---|---|---|"]
    for r in profile["detected_relations"]:
        L.append(f"| {r['child']} | {r['parent']} | {r['match_pct']} % | {r['type']} |")

    L += ["", "## 4. Règles métier", "",
          "| Règle | Violations | Total | % | Commentaire |", "|---|---|---|---|---|"]
    for r in profile["business_rules"]:
        if "error" in r:
            L.append(f"| {r['rule']} | erreur | | | {r['error']} |")
        else:
            tag = " (info)" if r["informative"] else ""
            L.append(f"| {r['rule']}{tag} | {r['violations']} | {r['total']} | {r['pct']} % | {r['comment']} |")

    L += ["", "## 5. Couverture temporelle", ""]
    for n, t in profile["tables"].items():
        for col, i in t["columns"].items():
            m = i.get("monthly_counts")
            if m:
                ks = list(m)
                L.append(f"- **{n}.{col}** : {i['date_range']['min'][:10]} -> {i['date_range']['max'][:10]} ; "
                         f"premiers mois {[(k, m[k]) for k in ks[:3]]} ; derniers mois {[(k, m[k]) for k in ks[-3:]]}")

    L += ["", "## 6. Points d'attention", ""]
    for n, t in profile["tables"].items():
        for col, i in t["columns"].items():
            notes = list(i["hints"])
            if i["missing_pct"] > 1 and col not in OPTIONAL_COLUMNS:
                notes.append(f"{i['missing_count']} valeurs manquantes ({i['missing_pct']} %)")
            if i.get("negatives_unexpected"):
                notes.append(f"{i['negatives']} valeurs négatives inattendues")
            if i.get("outlier_pct", 0) > 5:
                notes.append(f"{i['outlier_pct']} % de valeurs atypiques (à examiner, pas forcément des erreurs)")
            if notes:
                L.append(f"- **{n}.{col}** : " + " ; ".join(notes))
        if t["duplicates_pct"] > 1:
            L.append(f"- **{n}** : {t['duplicates']} lignes en double ({t['duplicates_pct']} %)")
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
# 6. Orchestration du profilage
# ----------------------------------------------------------------------------
def profile_all(raw_dir=RAW_DIR, out_dir=OUT_DIR) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for f in sorted(Path(raw_dir).glob("*.csv")):
        print(f"Lecture de {f.name}...")
        tables[f.stem] = pd.read_csv(f)
    if not tables:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {raw_dir}")

    print("Profilage des tables...")
    profiles = {n: profile_table(df) for n, df in tables.items()}
    profile = {"tables": profiles}
    profile["relations"] = check_relations(tables)
    profile["detected_relations"] = detect_relations(tables, profiles)
    profile["business_rules"] = business_rules(tables)

    decl = {frozenset([r["child"], r["parent"]]) for r in profile["relations"]}
    det = {frozenset([r["child"], r["parent"]]) for r in profile["detected_relations"]}
    profile["relation_detection_eval"] = {
        "declared": len(decl), "retrieved": len(decl & det), "extra": len(det - decl),
        "missed": sorted(" <-> ".join(sorted(x)) for x in decl - det),
    }

    tables_mean = float(np.mean([t["reliability_score"] for t in profiles.values()]))
    rel_score = 100 - float(np.mean([r["orphan_pct"] for r in profile["relations"]] or [0]))
    rule_pcts = [r["pct"] for r in profile["business_rules"] if "pct" in r and not r["informative"]]
    cons_score = 100 - float(np.mean(rule_pcts or [0]))
    profile["score_details"] = {
        "tables_mean": round(tables_mean, 1), "relations": round(rel_score, 1),
        "consistency": round(cons_score, 1),
    }
    profile["global_score"] = round(0.6 * tables_mean + 0.2 * rel_score + 0.2 * cons_score, 1)

    def to_py(o):
        return o.item() if hasattr(o, "item") else str(o)

    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False, default=to_py),
                            encoding="utf-8")
    write_report(profile, out_dir / "data_quality_report.md")
    return profile


def profile_is_fresh(raw_dir=RAW_DIR) -> bool:
    """Le profil est périmé si un CSV ou ce fichier a été modifié après sa création."""
    if not PROFILE_PATH.exists():
        return False
    newest = max([f.stat().st_mtime for f in Path(raw_dir).glob("*.csv")]
                 + [Path(__file__).stat().st_mtime])
    return PROFILE_PATH.stat().st_mtime >= newest


def load_profile(refresh: bool = False) -> dict:
    if refresh or not profile_is_fresh():
        profile_all()
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    p = profile_all()
    print(f"\nScore global : {p['global_score']}/100  ({p['score_details']})")
    for name, t in p["tables"].items():
        print(f"  {name:40} {t['rows']:>8} lignes | doublons {t['duplicates']:>6} | score {t['reliability_score']}")
    print("\nRapport : data/profile/data_quality_report.md")