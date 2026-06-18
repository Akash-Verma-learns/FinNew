from __future__ import annotations

"""
FormulaGraph — directed graph of metric relationships sourced from the
FASB XBRL calculation linkbase. Supports multi-path resolution with
cycle detection so circular GAAP relationships (e.g. A derivable from B
and B derivable from A) never cause infinite recursion.

Resolution order:
  1. Company's own 10-K definition (XBRL-reported value)
  2. FASB standard formula (linkbase-derived)
  3. CFA/textbook formula (fallback)

When multiple definitions produce a value, all are returned with deltas
rather than silently picking one.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class FormulaNode:
    concept: str
    inputs: list[str]                        # dependency concept names
    compute: Callable[[dict[str, float]], float]  # function over resolved inputs
    source: str = "fasb_linkbase"            # "company_10k" | "fasb_linkbase" | "textbook"
    formula_str: str = ""                    # human-readable, e.g. "GrossProfit / Revenues × 100"


@dataclass
class ResolutionResult:
    concept: str
    value: float
    source: str
    inputs_used: dict[str, float]


@dataclass
class MultiResolution:
    concept: str
    results: list[ResolutionResult] = field(default_factory=list)

    @property
    def primary(self) -> Optional[ResolutionResult]:
        """Highest-priority result: company 10-K > FASB > textbook."""
        order = {"company_10k": 0, "fasb_linkbase": 1, "textbook": 2}
        return min(self.results, key=lambda r: order.get(r.source, 99), default=None)

    @property
    def delta(self) -> Optional[float]:
        """Difference between highest and lowest value across definitions."""
        if len(self.results) < 2:
            return None
        vals = [r.value for r in self.results]
        return max(vals) - min(vals)


class FormulaGraph:
    """
    Directed acyclic graph of financial metric derivations.

    Nodes are XBRL concepts. Edges represent 'A requires B to compute'.
    The graph is built from the FASB calculation linkbase at startup and
    held in memory (the linkbase is effectively static — FASB updates the
    taxonomy once a year).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, list[FormulaNode]] = {}  # concept -> [formula variants]

    def register(self, node: FormulaNode) -> None:
        self._nodes.setdefault(node.concept, []).append(node)

    def resolve(
        self,
        concept: str,
        known: dict[str, float],
        *,
        _visited: frozenset[str] = frozenset(),
        _stack: frozenset[str] = frozenset(),
    ) -> MultiResolution:
        """
        Recursively resolve `concept` from `known` values.

        `_visited` — concepts fully resolved in this call tree (skip re-work).
        `_stack`   — concepts currently on the active recursion path.
                     A concept appearing in its own stack means a cycle.
        """
        result = MultiResolution(concept=concept)

        if concept in _stack:
            logger.warning("Cycle detected resolving %s — path: %s", concept, _stack)
            return result

        if concept in known:
            result.results.append(ResolutionResult(
                concept=concept,
                value=known[concept],
                source="company_10k",
                inputs_used={},
            ))

        new_stack = _stack | {concept}

        for node in self._nodes.get(concept, []):
            resolved_inputs: dict[str, float] = {}
            ok = True
            for dep in node.inputs:
                if dep in known:
                    resolved_inputs[dep] = known[dep]
                elif dep not in _visited and dep not in _stack:
                    sub = self.resolve(dep, known, _visited=_visited, _stack=new_stack)
                    if sub.primary is not None:
                        resolved_inputs[dep] = sub.primary.value
                    else:
                        ok = False
                        break
                else:
                    ok = False
                    break
            if ok:
                try:
                    value = node.compute(resolved_inputs)
                    result.results.append(ResolutionResult(
                        concept=concept,
                        value=value,
                        source=node.source,
                        inputs_used=resolved_inputs,
                    ))
                except Exception as exc:
                    logger.debug("Formula compute failed for %s: %s", concept, exc)

        return result

    def has_cycle(self) -> bool:
        """Detect any cycle in the full graph using DFS."""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for formula in self._nodes.get(node, []):
                for dep in formula.inputs:
                    if dep not in visited:
                        if _dfs(dep):
                            return True
                    elif dep in rec_stack:
                        logger.error("Cycle found at: %s → %s", node, dep)
                        return True
            rec_stack.discard(node)
            return False

        for concept in list(self._nodes):
            if concept not in visited:
                if _dfs(concept):
                    return True
        return False


# ---------------------------------------------------------------------------
# Built-in textbook formulas (lowest priority — used when FASB linkbase and
# company-reported values are both absent)
# ---------------------------------------------------------------------------

def _build_default_graph() -> FormulaGraph:
    g = FormulaGraph()

    def _reg(concept: str, inputs: list[str], fn: Callable, formula_str: str, source: str = "textbook") -> None:
        g.register(FormulaNode(concept=concept, inputs=inputs, compute=fn, source=source, formula_str=formula_str))

    _reg("GrossProfit", ["Revenues", "CostOfRevenue"],
         lambda v: v["Revenues"] - v["CostOfRevenue"],
         "Revenues − CostOfRevenue")

    _reg("OperatingIncomeLoss", ["GrossProfit", "OperatingExpenses"],
         lambda v: v["GrossProfit"] - v["OperatingExpenses"],
         "GrossProfit − OperatingExpenses")

    _reg("EBITDA", ["OperatingIncomeLoss", "DepreciationDepletionAndAmortization"],
         lambda v: v["OperatingIncomeLoss"] + v["DepreciationDepletionAndAmortization"],
         "OperatingIncome + D&A")

    _reg("GrossMarginPct", ["GrossProfit", "Revenues"],
         lambda v: v["GrossProfit"] / v["Revenues"] * 100,
         "GrossProfit ÷ Revenues × 100")

    _reg("OperatingMarginPct", ["OperatingIncomeLoss", "Revenues"],
         lambda v: v["OperatingIncomeLoss"] / v["Revenues"] * 100,
         "OperatingIncome ÷ Revenues × 100")

    _reg("NetMarginPct", ["NetIncomeLoss", "Revenues"],
         lambda v: v["NetIncomeLoss"] / v["Revenues"] * 100,
         "NetIncome ÷ Revenues × 100")

    _reg("EBITDAMarginPct", ["EBITDA", "Revenues"],
         lambda v: v["EBITDA"] / v["Revenues"] * 100,
         "EBITDA ÷ Revenues × 100")

    _reg("FreeCashFlow",
         ["NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment"],
         lambda v: (v["NetCashProvidedByUsedInOperatingActivities"]
                    - v["PaymentsToAcquirePropertyPlantAndEquipment"]),
         "OperatingCashFlow − CapEx")

    _reg("NetDebt", ["LongTermDebt", "CashAndCashEquivalentsAtCarryingValue"],
         lambda v: v["LongTermDebt"] - v["CashAndCashEquivalentsAtCarryingValue"],
         "LongTermDebt − Cash")

    _reg("WorkingCapital", ["AssetsCurrent", "LiabilitiesCurrent"],
         lambda v: v["AssetsCurrent"] - v["LiabilitiesCurrent"],
         "CurrentAssets − CurrentLiabilities")

    _reg("CurrentRatio", ["AssetsCurrent", "LiabilitiesCurrent"],
         lambda v: v["AssetsCurrent"] / v["LiabilitiesCurrent"],
         "CurrentAssets ÷ CurrentLiabilities")

    _reg("DebtToEquity", ["LongTermDebt", "StockholdersEquity"],
         lambda v: v["LongTermDebt"] / v["StockholdersEquity"],
         "LongTermDebt ÷ StockholdersEquity")

    _reg("InterestCoverage", ["OperatingIncomeLoss", "InterestExpense"],
         lambda v: v["OperatingIncomeLoss"] / v["InterestExpense"],
         "OperatingIncome ÷ InterestExpense")

    _reg("ROE", ["NetIncomeLoss", "StockholdersEquity"],
         lambda v: v["NetIncomeLoss"] / v["StockholdersEquity"] * 100,
         "NetIncome ÷ Equity × 100")

    _reg("ROA", ["NetIncomeLoss", "Assets"],
         lambda v: v["NetIncomeLoss"] / v["Assets"] * 100,
         "NetIncome ÷ TotalAssets × 100")

    _reg("FCFMarginPct", ["FreeCashFlow", "Revenues"],
         lambda v: v["FreeCashFlow"] / v["Revenues"] * 100,
         "FreeCashFlow ÷ Revenues × 100")

    _reg("ROIC", ["OperatingIncomeLoss", "LongTermDebt", "StockholdersEquity"],
         lambda v: v["OperatingIncomeLoss"] * 0.80 / max(abs(v["LongTermDebt"] + v["StockholdersEquity"]), 1) * 100,
         "NOPAT(est.) ÷ InvestedCapital × 100  [NOPAT ≈ OperatingIncome × 0.80]")

    return g


DEFAULT_GRAPH: FormulaGraph = _build_default_graph()


def graph_to_json(graph: FormulaGraph, active_concepts: set[str] | None = None) -> dict:
    """
    Serialize the FormulaGraph as nodes + edges for frontend rendering.

    active_concepts: set of concept names that were resolved in the current
                     validation run — used to highlight the active subgraph.
    """
    active = active_concepts or set()

    # Collect all unique node IDs (both computed and raw/leaf)
    all_node_ids: set[str] = set()
    edges: list[dict] = []

    for concept, nodes in graph._nodes.items():
        all_node_ids.add(concept)
        for node in nodes:
            for dep in node.inputs:
                all_node_ids.add(dep)
                edges.append({
                    "id": f"{concept}->{dep}",
                    "source": concept,
                    "target": dep,
                    "formula_str": node.formula_str,
                    "source_type": node.source,
                })

    # Deduplicate edges (same concept-dep pair may appear across multiple formula variants)
    seen_edges: set[str] = set()
    unique_edges = []
    for e in edges:
        key = f"{e['source']}->{e['target']}"
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    computed_ids = set(graph._nodes.keys())
    nodes = []
    for nid in all_node_ids:
        is_computed = nid in computed_ids
        formulas = [
            {"formula_str": n.formula_str, "source": n.source, "inputs": n.inputs}
            for n in graph._nodes.get(nid, [])
        ]
        nodes.append({
            "id": nid,
            "label": nid,
            "node_type": "computed" if is_computed else "raw",
            "active": nid in active,
            "formulas": formulas,
        })

    return {
        "nodes": nodes,
        "edges": unique_edges,
        "has_cycle": graph.has_cycle(),
        "stats": {
            "total_nodes": len(nodes),
            "computed_nodes": len(computed_ids),
            "raw_nodes": len(all_node_ids) - len(computed_ids),
            "edges": len(unique_edges),
        },
    }
