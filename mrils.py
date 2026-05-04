"""
mrils.py – Multi‑Restart Iterated Local Search for Influence Maximization

Copyright (c) 2026 Sheikh Khalif and Mohammed Adeeb.
National Institute of Technology Srinagar, India.
Licensed under the MIT License. See LICENSE file for details.

Author contacts:
  Sheikh Khalif   – khalif.123.sk@gmail.com

This module implements the MRILS and MRILS‑Enhanced algorithms,
as described in the paper "MRILS and Enhanced MRILS: Multi‑Restart
Iterated Local Search for Influence Maximization" (2026).
"""

import math
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


# ===========================================================================
#  IMResult  –  simple data container for algorithm output
# ===========================================================================

class IMResult:
    """Stores the result of an influence maximisation run."""

    def __init__(
        self,
        seeds: List[Any],
        spread: float,
        runtime: float,
        method: str,
        extras: Optional[Dict[str, Any]] = None,
    ):
        self.seeds = seeds
        self.spread = spread
        self.runtime = runtime
        self.method = method
        self.extras = extras if extras is not None else {}

    def __repr__(self) -> str:
        return (
            f"{self.method} | " 
            f"k={len(self.seeds)} | "
            f"spread={self.spread:.2f} | "
            f"runtime={self.runtime:.2f}s"
        )


# ===========================================================================
#  Monte Carlo spread estimation
# ===========================================================================

def mc_spread(
    graph: nx.Graph,
    seeds: List[Any],
    n_sim: int = 400,
    rng_seed: Optional[int] = None,
) -> float:
    """
    Estimate the expected spread of `seeds` under the Independent Cascade model
    using `n_sim` Monte Carlo simulations.

    Parameters
    ----------
    graph : networkx.DiGraph or Graph
        Graph with edge attribute 'weight' (activation probability).
    seeds : list of nodes
        Initially active nodes.
    n_sim : int
        Number of independent cascades to simulate.
    rng_seed : int, optional
        If given, seeds the random number generator to obtain reproducible results.

    Returns
    -------
    float
        Average number of activated nodes across all simulations.
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    total_spread = 0
    seed_set = set(seeds)

    for _ in range(n_sim):
        active = set(seed_set)
        new_active = set(seed_set)

        while new_active:
            next_new = set()
            for u in new_active:
                # On a DiGraph neighbours are successors; on Graph all neighbours.
                for v in graph.neighbors(u):
                    if v not in active:
                        prob = graph[u][v].get("weight", 0.1)
                        if random.random() < prob:
                            next_new.add(v)
                            active.add(v)
            new_active = next_new

        total_spread += len(active)

    return total_spread / n_sim


# ===========================================================================
#  RIS internals
# ===========================================================================

def _sample_rrset(graph: nx.DiGraph, root: Any) -> Set[Any]:
    """
    Sample one random Reverse Reachable (RR) set rooted at `root`.

    Starting from the root, we walk backwards along incoming edges according
    to their activation probabilities.  The RR set contains all nodes that
    could possibly activate the root via a live‑edge path.
    """
    rr_set = {root}
    queue = [root]
    while queue:
        v = queue.pop()
        # In a DiGraph, predecessors are nodes u with edge u→v.
        for u in graph.predecessors(v):
            if u not in rr_set:
                prob = graph[u][v].get("weight", 0.1)
                if random.random() < prob:
                    rr_set.add(u)
                    queue.append(u)
    return rr_set


def _imm_theta(
    n: int,
    k: int,
    eps: float = 0.1,
    l: float = 1.0,
) -> int:
    """
    Compute a simplified IMM‑like bound for the number of RR‑sets needed
    to obtain a (1‑1/e‑ε) approximation with probability at least 1‑n^{‑l}.

    The formula is a heuristic that shares the qualitative behaviour of the
    exact IMM bound (grows with n and k, shrinks with ε).  A floor of 2 000
    and a ceiling of 50 000 are enforced to keep test runtimes practical.
    """
    # Very rough log{C(n,k)} ≈ k * log(n)  (valid when k << n)
    log_c = k * math.log(n) if n > k else math.log(2) * n
    # Mimic the (2n / ε²) * (log C(n,k) + log(1/l) + log log n) term
    bound = (2 * n / (eps * eps)) * (log_c + math.log(1 / l) + math.log(max(1, math.log(n))))
    theta = int(math.ceil(bound))
    # Hard limits for practical testing
    theta = max(theta, 2_000)
    theta = min(theta, 50_000)
    return theta


def _swap_delta(
    seeds: Set[Any],
    v_out: Any,
    v_in: Any,
    node_to_rr: Dict[Any, Set[int]],
) -> int:
    """
    Change in the number of covered RR‑sets when replacing `v_out` with `v_in`.

    Parameters
    ----------
    seeds : set of current seeds (must contain v_out, must not contain v_in)
    v_out : node to be removed
    v_in  : node to be added
    node_to_rr : mapping from node → set of RR‑set indices that contain it

    Returns
    -------
    int
        (coverage after swap) - (coverage before swap)
    """
    # Coverage without v_out
    covered_without = set()
    for s in seeds:
        if s != v_out:
            covered_without |= node_to_rr.get(s, set())

    gain = len(node_to_rr.get(v_in, set()) - covered_without)
    loss = len(node_to_rr.get(v_out, set()) - covered_without)
    return gain - loss


# ===========================================================================
#  RIS‑based greedy maximum coverage
# ===========================================================================

def _ris_greedy(
    graph: nx.DiGraph,
    k: int,
    eps: float = 0.1,
    rng_state: Optional[int] = None,
) -> Tuple[List[Any], Dict[Any, Set[int]], int]:
    """
    Sample θ RR‑sets, then greedily pick k seeds that maximise coverage.

    Returns
    -------
    seeds : list of k nodes
    node_to_rr : dict mapping each node to the set of RR‑set indices that contain it
    theta : number of RR‑sets that were actually sampled
    """
    if rng_state is not None:
        random.seed(rng_state)

    n = graph.number_of_nodes()
    theta = _imm_theta(n, k, eps=eps)

    # Sample RR‑sets
    rr_sets: List[Set[Any]] = []
    for _ in range(theta):
        root = random.choice(list(graph.nodes()))
        rr_sets.append(_sample_rrset(graph, root))

    # Invert index: node → set of RR‑set indices
    node_to_rr: Dict[Any, Set[int]] = defaultdict(set)
    for idx, rr in enumerate(rr_sets):
        for node in rr:
            node_to_rr[node].add(idx)

    # Greedy maximum coverage
    seeds: List[Any] = []
    covered: Set[int] = set()

    for _ in range(k):
        best_node = None
        best_gain = -1
        for node, rr_indices in node_to_rr.items():
            if node in seeds:
                continue
            gain = len(rr_indices - covered)
            if gain > best_gain:
                best_gain = gain
                best_node = node
        if best_node is None:
            # Fallback: pick any unused node
            for node in graph.nodes():
                if node not in seeds:
                    seeds.append(node)
                    break
            continue
        seeds.append(best_node)
        covered |= node_to_rr[best_node]

    return seeds, dict(node_to_rr), theta


# ===========================================================================
#  RR‑based local search (swap refinement)
# ===========================================================================

def rr_local_search(
    seeds: List[Any],
    graph: nx.DiGraph,
    node_to_rr: Dict[Any, Set[int]],
    passes: int = 1,
) -> List[Any]:
    """
    Improve `seeds` by iterated swap moves on the RR‑set coverage landscape.

    A swap (remove a seed, add a non‑seed) is accepted as soon as a strictly
    positive delta is found (first‑improvement strategy).  The process is
    repeated `passes` times.
    """
    seed_set = set(seeds)
    all_nodes = set(graph.nodes())

    for _ in range(passes):
        improved = False
        # Shuffle seed order to avoid bias
        seed_order = list(seed_set)
        random.shuffle(seed_order)

        for v_out in seed_order:
            # Try every possible non‑seed
            non_seeds = all_nodes - seed_set
            # Randomise order to avoid systematic bias
            shuffled_nonseeds = list(non_seeds)
            random.shuffle(shuffled_nonseeds)
            for v_in in shuffled_nonseeds:
                if _swap_delta(seed_set, v_out, v_in, node_to_rr) > 0:
                    seed_set.remove(v_out)
                    seed_set.add(v_in)
                    improved = True
                    break   # first improvement
        if not improved:
            break
    return list(seed_set)


# ===========================================================================
#  Baseline algorithms
# ===========================================================================

class SingleRIS:
    """Single run of RIS‑greedy, no refinement."""

    def select_seeds(self, graph: nx.DiGraph, k: int) -> IMResult:
        t0 = time.time()
        seeds, _, _ = _ris_greedy(graph, k, eps=0.1, rng_state=None)
        spread = mc_spread(graph, seeds, n_sim=400, rng_seed=None)
        runtime = time.time() - t0
        return IMResult(seeds, spread, runtime, method="RIS-single")


class DegreeDiscount:
    """
    Fast heuristic based on weighted out‑degree with a simple discount
    for shared neighbours.
    """

    def select_seeds(self, graph: nx.DiGraph, k: int) -> IMResult:
        t0 = time.time()

        # Initial weighted out‑degree
        dd = {v: 0.0 for v in graph.nodes()}
        for u in graph.nodes():
            for v in graph.successors(u):
                dd[u] += graph[u][v].get("weight", 0.1)

        seeds = []
        selected = set()

        for _ in range(k):
            # Pick node with largest dd not yet selected
            best_node = None
            best_val = -1.0
            for node, val in dd.items():
                if node not in selected and val > best_val:
                    best_val = val
                    best_node = node
            if best_node is None:
                # Remaining nodes (all discarded), pick any
                for node in graph.nodes():
                    if node not in selected:
                        seeds.append(node)
                        selected.add(node)
                        break
                continue
            seeds.append(best_node)
            selected.add(best_node)

            # Discount nodes that share an out‑neighbour with the new seed
            for w in graph.successors(best_node):
                for v in graph.predecessors(w):
                    if v not in selected:
                        dd[v] -= graph[v][w].get("weight", 0.1)
                        if dd[v] < 0:
                            dd[v] = 0.0

        spread = mc_spread(graph, seeds, n_sim=400, rng_seed=None)
        runtime = time.time() - t0
        return IMResult(seeds, spread, runtime, method="DegreeDiscount")


class GreedyMC:
    """
    Gold‑standard greedy Monte Carlo algorithm.

    At each step the node that yields the largest marginal spread increase
    is chosen.  This is the provably (1‑1/e)‑optimal greedy strategy for
    submodular functions, evaluated with MC simulations.
    """

    def __init__(self, n_sim: int = 400, rng_seed: Optional[int] = None):
        self.n_sim = n_sim
        self.rng_seed = rng_seed

    def select_seeds(self, graph: nx.DiGraph, k: int) -> IMResult:
        t0 = time.time()

        if self.rng_seed is not None:
            random.seed(self.rng_seed)

        seeds: List[Any] = []
        nodes = list(graph.nodes())

        for _ in range(k):
            best_node = None
            best_gain = -float("inf")
            current_spread = mc_spread(graph, seeds, self.n_sim, rng_seed=None)

            for v in nodes:
                if v in seeds:
                    continue
                cand_spread = mc_spread(graph, seeds + [v], self.n_sim, rng_seed=None)
                gain = cand_spread - current_spread
                if gain > best_gain:
                    best_gain = gain
                    best_node = v
            seeds.append(best_node)

        spread = mc_spread(graph, seeds, self.n_sim, rng_seed=None)
        runtime = time.time() - t0
        return IMResult(seeds, spread, runtime, method="Greedy-MC")


# ===========================================================================
#  MRILS  (Multi‑Restart Iterated Local Search)
# ===========================================================================

class MRILS:
    """
    Multi‑Restart RIS + RR‑guided Local Search.

    Parameters
    ----------
    n_restarts : int
        Number of independent RIS‑greedy → local‑search pipelines.
    ls_passes : int
        Number of swap passes during local search.
    eval_sims : int
        Number of MC simulations for final spread evaluation.
    rng_seed : int, optional
        Base seed for reproducibility.
    """

    def __init__(
        self,
        n_restarts: int = 5,
        ls_passes: int = 5,
        eval_sims: int = 400,
        rng_seed: Optional[int] = None,
    ):
        self.n_restarts = n_restarts
        self.ls_passes = ls_passes
        self.eval_sims = eval_sims
        self.rng_seed = rng_seed

    def select_seeds(self, graph: nx.DiGraph, k: int) -> IMResult:
        t0 = time.time()

        best_seeds: List[Any] = []
        best_spread = -1.0
        all_spreads: List[float] = []

        # Seed the outer random state if requested
        if self.rng_seed is not None:
            random.seed(self.rng_seed)

        for i in range(self.n_restarts):
            # Use a fresh rng state per restart to get diversity
            local_seed = random.randint(0, 2**31 - 1)
            seeds0, node_to_rr, _ = _ris_greedy(graph, k, eps=0.1, rng_state=local_seed)

            # Local search
            improved = rr_local_search(seeds0, graph, node_to_rr, passes=self.ls_passes)

            # Evaluate with MC
            spread = mc_spread(graph, improved, n_sim=self.eval_sims, rng_seed=None)
            all_spreads.append(spread)

            if spread > best_spread:
                best_spread = spread
                best_seeds = improved

        runtime = time.time() - t0
        return IMResult(
            best_seeds,
            best_spread,
            runtime,
            method="MRILS",
            extras={"all_spreads": all_spreads},
        )


# ===========================================================================
#  MRILS‑Enhanced  (MC‑validated swaps + biased re‑seeding)
# ===========================================================================

class MRILSEnhanced:
    """
    Enhanced MRILS with MC‑validated swap moves and optional re‑seeding
    on biased RR‑sets.

    Parameters
    ----------
    n_restarts : int
        Number of independent restarts.
    ls_passes : int
        Number of swap passes during local search.
    mc_val_sims : int
        Number of MC simulations to validate a candidate swap.
    eval_sims : int
        Number of MC simulations for final seed set evaluation.
    re_seed : bool
        If True, generate an extra candidate from a biased RIS instance.
    rng_seed : int, optional
        Base seed for reproducibility.
    """

    def __init__(
        self,
        n_restarts: int = 5,
        ls_passes: int = 5,
        mc_val_sims: int = 200,
        eval_sims: int = 400,
        re_seed: bool = True,
        rng_seed: Optional[int] = None,
    ):
        self.n_restarts = n_restarts
        self.ls_passes = ls_passes
        self.mc_val_sims = mc_val_sims
        self.eval_sims = eval_sims
        self.re_seed = re_seed
        self.rng_seed = rng_seed

    def select_seeds(self, graph: nx.DiGraph, k: int) -> IMResult:
        t0 = time.time()

        best_seeds: List[Any] = []
        best_spread = -1.0
        all_spreads: List[float] = []

        if self.rng_seed is not None:
            random.seed(self.rng_seed)

        nodes_list = list(graph.nodes())

        for _ in range(self.n_restarts):
            # --------  1. Initial RIS‑greedy solution  --------
            local_seed = random.randint(0, 2**31 - 1)
            seeds, node_to_rr, _ = _ris_greedy(graph, k, eps=0.1, rng_state=local_seed)

            # --------  2. MC‑validated local search  --------
            for _pass in range(self.ls_passes):
                improved = False
                seed_order = list(seeds)
                random.shuffle(seed_order)
                for v_out in seed_order:
                    # Collect all possible swaps and their RR‑deltas
                    swap_candidates = []
                    non_seeds = set(nodes_list) - set(seeds)
                    for v_in in non_seeds:
                        delta = _swap_delta(set(seeds), v_out, v_in, node_to_rr)
                        if delta > 0:
                            swap_candidates.append((delta, v_in))
                    if not swap_candidates:
                        continue
                    # Sort by delta descending, take top 3
                    swap_candidates.sort(reverse=True, key=lambda x: x[0])
                    top3 = swap_candidates[:3]

                    # Validate each with quick MC, accept first that increases spread
                    current_spread = mc_spread(graph, seeds, self.mc_val_sims, rng_seed=None)
                    for _, v_in in top3:
                        candidate = [v_in if x == v_out else x for x in seeds]
                        new_spread = mc_spread(graph, candidate, self.mc_val_sims, rng_seed=None)
                        if new_spread > current_spread:
                            seeds = candidate
                            improved = True
                            break   # out of v_in loop
                if not improved:
                    break

            # Evaluate this candidate
            spread = mc_spread(graph, seeds, self.eval_sims, rng_seed=None)
            all_spreads.append(spread)
            if spread > best_spread:
                best_spread = spread
                best_seeds = seeds

            # --------  3. Biased re‑seeding (optional)  --------
            if self.re_seed:
                # Build a new RIS instance with roots biased toward the current seeds
                theta_new = _imm_theta(graph.number_of_nodes(), k, eps=0.1)
                rr_sets_new = []
                for _ in range(theta_new):
                    # 50 % chance to pick from seeds, 50 % from all nodes
                    if random.random() < 0.5 and seeds:
                        root = random.choice(seeds)
                    else:
                        root = random.choice(nodes_list)
                    rr_sets_new.append(_sample_rrset(graph, root))

                # Invert to node → RR indices
                node_to_rr_new: Dict[Any, Set[int]] = defaultdict(set)
                for idx, rr in enumerate(rr_sets_new):
                    for node in rr:
                        node_to_rr_new[node].add(idx)

                # Greedy coverage on this biased instance
                seeds_re: List[Any] = []
                covered: Set[int] = set()
                for _ in range(k):
                    best_node = None
                    best_gain = -1
                    for node, rr_indices in node_to_rr_new.items():
                        if node in seeds_re:
                            continue
                        gain = len(rr_indices - covered)
                        if gain > best_gain:
                            best_gain = gain
                            best_node = node
                    if best_node is None:
                        # fallback
                        for n in nodes_list:
                            if n not in seeds_re:
                                seeds_re.append(n)
                                break
                        continue
                    seeds_re.append(best_node)
                    covered |= node_to_rr_new[best_node]

                spread_re = mc_spread(graph, seeds_re, self.eval_sims, rng_seed=None)
                all_spreads.append(spread_re)
                if spread_re > best_spread:
                    best_spread = spread_re
                    best_seeds = seeds_re

        runtime = time.time() - t0
        return IMResult(
            best_seeds,
            best_spread,
            runtime,
            method="MRILS-Enhanced",
            extras={"all_spreads": all_spreads},
        )


# ===========================================================================
#  Convenience runner
# ===========================================================================

def run_comparison(
    graph: nx.DiGraph,
    k: int,
    methods: Optional[List[str]] = None,
) -> Dict[str, IMResult]:
    """
    Run a selection of influence maximisation methods and return their results.

    Parameters
    ----------
    graph : DiGraph
    k : int
        Number of seeds to select.
    methods : list of str, optional
        Subset of {"mrils", "ris", "degree", "enhanced", "greedy"}.
        Defaults to ["mrils", "ris", "degree"] (fast methods).

    Returns
    -------
    dict mapping method abbreviation (e.g. "mrils") to IMResult
    """
    if methods is None:
        methods = ["mrils", "ris", "degree"]

    results: Dict[str, IMResult] = {}

    for method in methods:
        m = method.lower()
        if m == "mrils":
            alg = MRILS(n_restarts=6, ls_passes=6, eval_sims=500)
        elif m == "enhanced":
            alg = MRILSEnhanced(
                n_restarts=6, ls_passes=4, mc_val_sims=100, eval_sims=500,
                re_seed=True,
            )
        elif m == "ris":
            alg = SingleRIS()
        elif m == "degree":
            alg = DegreeDiscount()
        elif m == "greedy":
            alg = GreedyMC(n_sim=400)
        else:
            raise ValueError(f"Unknown method: {method}")

        res = alg.select_seeds(graph, k)
        results[m] = res

    return results
