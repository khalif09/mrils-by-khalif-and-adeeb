"""
test_mrils.py – Test suite for MRILS and MRILS‑Enhanced

Copyright (c) 2026 Sheikh Khalif and Mohammed Adeeb.
National Institute of Technology Srinagar, India.
Licensed under the MIT License. See LICENSE file for details.

Pytest‑based test suite that validates the correctness and performance
of the influence maximisation algorithms implemented in mrils.py.
"""

import math
import random
import time
from collections import defaultdict

import networkx as nx
import pytest

# Module under test
from mrils import (
    IMResult,
    MRILS,
    MRILSEnhanced,
    DegreeDiscount,
    GreedyMC,
    SingleRIS,
    _imm_theta,
    _sample_rrset,
    _ris_greedy,
    _swap_delta,
    mc_spread,
    rr_local_search,
    run_comparison,
)


# ===========================================================================
#  Shared fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def small_graph():
    """
    A small 100-node scale-free directed graph used for fast unit tests.

    Scale-free graphs mimic real social networks (hubs + long tails) so they
    are a meaningful but lightweight test bed.  All edge weights set to 0.1
    (each edge propagates with 10% probability under IC).
    """
    G = nx.DiGraph(nx.scale_free_graph(100, seed=42))
    for u, v in G.edges():
        G[u][v]["weight"] = 0.1
    return G


@pytest.fixture(scope="module")
def medium_graph():
    """
    A 500-node scale-free graph for algorithm-level tests.

    Larger than small_graph so that spread differences between methods become
    statistically meaningful, but still fast enough for CI.
    """
    G = nx.DiGraph(nx.scale_free_graph(500, seed=0))
    for u, v in G.edges():
        G[u][v]["weight"] = 0.1
    return G


@pytest.fixture(scope="module")
def star_graph():
    """
    A directed star: node 0 → nodes 1..19, all weights = 1.0.

    Ground truth: seeding node 0 activates ALL 20 nodes with certainty.
    Any algorithm that returns node 0 as a seed is correct.
    Used to verify that every algorithm can find the obvious optimal seed.
    """
    G = nx.DiGraph()
    for i in range(1, 20):
        G.add_edge(0, i, weight=1.0)
    return G


# ===========================================================================
#  Section 1 – IMResult data container
# ===========================================================================

class TestIMResult:
    """
    IMResult is just a data class that holds algorithm output.

    Why test it?  Because every algorithm returns one and downstream code
    (comparisons, plots, reports) depends on its fields being correct types.
    """

    def test_fields_accessible(self):
        r = IMResult(seeds=[1, 2, 3], spread=5.5, runtime=0.1, method="test")
        assert r.seeds == [1, 2, 3]
        assert r.spread == pytest.approx(5.5)
        assert r.runtime == pytest.approx(0.1)
        assert r.method == "test"

    def test_repr_format(self):
        """__repr__ is used in logs and doctest output – must be stable."""
        r = IMResult(seeds=[0, 1], spread=3.14159, runtime=1.5, method="MyAlgo")
        s = repr(r)
        assert "MyAlgo" in s
        assert "3.14" in s      # 2 decimal places
        assert "1.50s" in s
        assert "k=2" in s

    def test_extras_default_empty(self):
        """extras should default to {} so algorithms can safely write to it."""
        r = IMResult(seeds=[], spread=0.0, runtime=0.0, method="x")
        assert r.extras == {}

    def test_extras_stored(self):
        r = IMResult(seeds=[0], spread=1.0, runtime=0.0, method="x",
                     extras={"history": [1.0, 1.1]})
        assert r.extras["history"] == [1.0, 1.1]


# ===========================================================================
#  Section 2 – Monte Carlo spread estimator
# ===========================================================================

class TestMcSpread:
    """
    mc_spread() is the ground-truth evaluator used by ALL algorithms.

    Why compare to Greedy?  Greedy-MC uses mc_spread() internally to pick
    seeds.  If mc_spread() is broken, Greedy-MC silently produces bad seeds.
    So we verify mc_spread() against known analytic results first.

    Analytic benchmark: star graph, seed={0}, weight=1.0
        → every child always activated → spread = 20 (deterministic)
    """

    def test_star_certain_spread(self, star_graph):
        """Weight=1.0 star → seeding hub always reaches all 20 nodes."""
        spread = mc_spread(star_graph, seeds=[0], n_sim=200, rng_seed=1)
        assert spread == pytest.approx(20.0, abs=0.01)

    def test_empty_seeds_returns_zero(self, small_graph):
        """No seeds → no activation beyond seeds themselves (which are 0)."""
        spread = mc_spread(small_graph, seeds=[], n_sim=100, rng_seed=1)
        assert spread == pytest.approx(0.0)

    def test_spread_at_least_seed_size(self, small_graph):
        """Seeds always activate themselves → spread ≥ len(seeds)."""
        seeds = list(small_graph.nodes())[:5]
        spread = mc_spread(small_graph, seeds=seeds, n_sim=200, rng_seed=1)
        assert spread >= len(seeds)

    def test_more_seeds_more_spread(self, medium_graph):
        """
        Adding more seeds never decreases expected spread (submodularity /
        monotonicity of the IC model).

        This is the same monotonicity property that makes Greedy-MC optimal.
        """
        nodes = list(medium_graph.nodes())
        s1 = mc_spread(medium_graph, seeds=nodes[:3], n_sim=500, rng_seed=7)
        s2 = mc_spread(medium_graph, seeds=nodes[:10], n_sim=500, rng_seed=7)
        assert s2 >= s1

    def test_reproducible_with_seed(self, small_graph):
        """Same rng_seed must return identical spread."""
        seeds = list(small_graph.nodes())[:5]
        a = mc_spread(small_graph, seeds, n_sim=300, rng_seed=99)
        b = mc_spread(small_graph, seeds, n_sim=300, rng_seed=99)
        assert a == pytest.approx(b)

    def test_spread_bounded_by_graph_size(self, small_graph):
        """Spread can never exceed number of nodes."""
        seeds = list(small_graph.nodes())[:10]
        spread = mc_spread(small_graph, seeds, n_sim=200, rng_seed=1)
        assert spread <= small_graph.number_of_nodes()


# ===========================================================================
#  Section 3 – RIS internal helpers
# ===========================================================================

class TestImmTheta:
    """
    _imm_theta() computes how many RR-sets are needed for a provable
    approximation guarantee.

    Why does this matter vs Greedy?
    Greedy-MC needs O(n·k·n_sim) work per seed – enormously expensive.
    RIS replaces that with theta RR-sets.  If theta is too small, the
    approximation guarantee breaks down.  These tests ensure the formula
    produces sane values.
    """

    def test_minimum_floor(self):
        """Even tiny graphs must generate at least 2000 RR-sets."""
        assert _imm_theta(10, 2) >= 2_000

    def test_grows_with_n(self):
        """Larger graphs require more RR-sets to maintain the guarantee."""
        assert _imm_theta(10_000, 5) > _imm_theta(1_000, 5)

    def test_grows_with_k(self):
        """Larger seed sets require more RR-sets."""
        assert _imm_theta(1_000, 20) > _imm_theta(1_000, 5)

    def test_decreases_with_eps(self):
        """
        Tighter approximation (smaller ε) requires more RR-sets.
        ε=0.01 is more precise than ε=0.5, so needs a bigger sample.
        """
        assert _imm_theta(1_000, 5, eps=0.01) > _imm_theta(1_000, 5, eps=0.5)

    def test_returns_int(self):
        assert isinstance(_imm_theta(500, 10), int)


class TestSampleRRSet:
    """
    _sample_rrset() performs backward BFS on the reverse graph.

    The RR-set of node v = set of nodes that *could* activate v.
    A seed in the RR-set of v means v gets activated when that seed is chosen.

    Why does this matter?
    Greedy-MC simulates influence forward from seeds.  RIS does it backward
    once and reuses the result many times.  If the backward sampling is wrong,
    the entire RIS approximation is invalid.
    """

    def test_root_always_in_rrset(self, small_graph):
        """The root node is always reachable from itself."""
        root = list(small_graph.nodes())[0]
        rr = _sample_rrset(small_graph, root)
        assert root in rr

    def test_returns_set(self, small_graph):
        root = list(small_graph.nodes())[0]
        assert isinstance(_sample_rrset(small_graph, root), set)

    def test_isolated_node_only_itself(self):
        """A node with no predecessors has RR-set = {itself}."""
        G = nx.DiGraph()
        G.add_node(99)
        assert _sample_rrset(G, 99) == {99}

    def test_certain_edge_always_included(self):
        """
        Edge (u→v) with weight=1.0: u must always appear in RR-set of v,
        because the backward traversal always crosses a weight=1 edge.
        """
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=1.0)
        for _ in range(20):
            assert 0 in _sample_rrset(G, 1)

    def test_impossible_edge_never_included(self):
        """
        Edge (u→v) with weight=0.0: u must never appear in RR-set of v.
        """
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=0.0)
        for _ in range(20):
            assert 0 not in _sample_rrset(G, 1)


class TestSwapDelta:
    """
    _swap_delta() computes the net change in RR-set coverage when swapping
    one seed (v_out) for another node (v_in).

    Why does this matter vs Greedy?
    Greedy-MC always adds the best node incrementally.  It cannot swap out
    an already-chosen seed.  The swap operator in MRILS allows it to escape
    greedy's locally-optimal but globally-suboptimal choices.
    """

    def test_zero_delta_no_change(self):
        """Swapping a node for itself should give delta=0."""
        S = {0, 1, 2}
        node_to_rr = {0: {0, 1}, 1: {2, 3}, 2: {4, 5}}
        assert _swap_delta(S, 0, 0, node_to_rr) == 0

    def test_positive_delta_for_better_node(self):
        """
        If v_in covers many RR-sets that v_out doesn't, delta should be > 0.
        """
        S = {0, 1}
        node_to_rr = {
            0: {0, 1},       # v_out covers 2 RR-sets
            1: {2, 3},       # stays
            99: {4, 5, 6},   # v_in covers 3 new RR-sets → net gain
        }
        delta = _swap_delta(S, 0, 99, node_to_rr)
        assert delta > 0

    def test_negative_delta_for_worse_node(self):
        """
        If v_out covers unique RR-sets and v_in covers nothing new, delta < 0.
        """
        S = {0, 1}
        node_to_rr = {
            0: {0, 1, 2},    # v_out covers 3 unique sets
            1: {10},
            99: set(),        # v_in covers nothing
        }
        delta = _swap_delta(S, 0, 99, node_to_rr)
        assert delta < 0


# ===========================================================================
#  Section 4 – RR-set greedy
# ===========================================================================

class TestRisGreedy:
    """
    _ris_greedy() is the core of every RIS-based method.

    It generates theta RR-sets and applies greedy maximum coverage to select
    k seeds.  This is approximately equivalent to Greedy-MC but far faster.

    Key comparison with Greedy-MC
    ─────────────────────────────
    Greedy-MC:  evaluates marginal spread for *every* candidate at *every*
                step → O(n · k · n_sim) evaluations.
    _ris_greedy: generates theta RR-sets ONCE, then picks max-coverage nodes
                 greedily → O(theta · avg_rrset_size + k · n) work.
    Both give a (1 − 1/e − ε) approximation, but RIS is ~100-1000× faster.
    """

    def test_returns_k_seeds(self, small_graph):
        seeds, _, _ = _ris_greedy(small_graph, k=5, eps=0.1, rng_state=42)
        assert len(seeds) == 5

    def test_seeds_are_valid_nodes(self, small_graph):
        seeds, _, _ = _ris_greedy(small_graph, k=5, eps=0.1, rng_state=42)
        assert all(s in small_graph.nodes() for s in seeds)

    def test_no_duplicate_seeds(self, small_graph):
        seeds, _, _ = _ris_greedy(small_graph, k=5, eps=0.1, rng_state=42)
        assert len(set(seeds)) == len(seeds)

    def test_node_to_rr_covers_seeds(self, small_graph):
        """node_to_rr must contain every selected seed."""
        seeds, node_to_rr, _ = _ris_greedy(small_graph, k=5, eps=0.1, rng_state=42)
        for s in seeds:
            assert s in node_to_rr

    def test_theta_respects_cap(self, small_graph):
        """theta must never exceed 50 000 (the hard cap in code)."""
        _, _, theta = _ris_greedy(small_graph, k=5, eps=0.1, rng_state=42)
        assert theta <= 50_000

    def test_different_seeds_with_different_rng(self, small_graph):
        """
        Different rng_state values should generally produce different seeds
        (stochastic algorithm).  We allow one collision out of 5 runs.
        """
        results = set()
        for i in range(5):
            s, _, _ = _ris_greedy(small_graph, k=3, eps=0.1, rng_state=i * 1_000)
            results.add(tuple(sorted(s)))
        assert len(results) > 1


# ===========================================================================
#  Section 5 – RR local search
# ===========================================================================

class TestRrLocalSearch:
    """
    rr_local_search() tries to improve a greedy seed set by swapping seeds
    one at a time (swap-based neighbourhood search).

    Why does this matter vs Greedy-MC?
    Greedy-MC is *constructive*: once a seed is chosen it stays.  Local search
    is *revisionary*: it can replace a previously chosen seed if a better swap
    exists.  This allows MRILS to escape greedy's locally optimal traps.
    """

    def test_output_has_same_k(self, small_graph):
        seeds, n2rr, _ = _ris_greedy(small_graph, k=5, eps=0.1, rng_state=42)
        improved = rr_local_search(seeds, small_graph, n2rr)
        assert len(improved) == 5

    def test_output_seeds_are_valid_nodes(self, small_graph):
        seeds, n2rr, _ = _ris_greedy(small_graph, k=5, eps=0.1, rng_state=42)
        improved = rr_local_search(seeds, small_graph, n2rr)
        assert all(s in small_graph.nodes() for s in improved)

    def test_no_duplicates_after_search(self, small_graph):
        seeds, n2rr, _ = _ris_greedy(small_graph, k=5, eps=0.1, rng_state=42)
        improved = rr_local_search(seeds, small_graph, n2rr)
        assert len(set(improved)) == len(improved)

    def test_coverage_does_not_decrease(self, small_graph):
        """
        Local search must not reduce RR-set coverage below the starting point.

        This verifies the 'first-improving' swap strategy: a swap is only
        accepted if delta > 0 (strictly better coverage).
        """
        seeds, n2rr, _ = _ris_greedy(small_graph, k=5, eps=0.1, rng_state=42)

        def coverage(s_list):
            covered = set()
            for v in s_list:
                covered |= n2rr.get(v, set())
            return len(covered)

        before = coverage(seeds)
        improved = rr_local_search(seeds, small_graph, n2rr)
        after = coverage(improved)
        assert after >= before


# ===========================================================================
#  Section 6 – SingleRIS baseline
# ===========================================================================

class TestSingleRIS:
    """
    SingleRIS is the simplest RIS variant: one run, no local search.

    Comparison with Greedy-MC
    ─────────────────────────
    • Greedy-MC: exact marginal gain, slow.
    • SingleRIS: one RIS round, approximate, ~100× faster.
    • SingleRIS should produce positive spread (it works), but typically 5–15%
      below Greedy-MC on community graphs.

    We test correctness (right types, right k), not optimality.
    """

    def test_returns_imresult(self, small_graph):
        r = SingleRIS().select_seeds(small_graph, k=3)
        assert isinstance(r, IMResult)

    def test_correct_k(self, small_graph):
        r = SingleRIS().select_seeds(small_graph, k=3)
        assert len(r.seeds) == 3

    def test_method_label(self, small_graph):
        r = SingleRIS().select_seeds(small_graph, k=3)
        assert r.method == "RIS-single"

    def test_spread_positive(self, small_graph):
        r = SingleRIS().select_seeds(small_graph, k=3)
        assert r.spread > 0

    def test_spread_bounded(self, small_graph):
        r = SingleRIS().select_seeds(small_graph, k=3)
        assert r.spread <= small_graph.number_of_nodes()

    def test_runtime_recorded(self, small_graph):
        r = SingleRIS().select_seeds(small_graph, k=3)
        assert r.runtime > 0

    def test_finds_hub_in_star(self, star_graph):
        """
        On a star graph, the hub (node 0) should be the top seed.
        Single RIS should find it because it appears in the RR-sets of all
        leaf nodes.
        """
        r = SingleRIS().select_seeds(star_graph, k=1)
        assert 0 in r.seeds


# ===========================================================================
#  Section 7 – DegreeDiscount baseline
# ===========================================================================

class TestDegreeDiscount:
    """
    DegreeDiscount is the fastest heuristic: O(k log n), no MC simulation.

    Comparison with Greedy-MC
    ─────────────────────────
    • Greedy-MC: O(n · k · n_sim), optimal.
    • DegreeDiscount: O(k log n), heuristic.
    • DegreeDiscount is typically 5–20% below Greedy-MC on spread quality but
      finishes in milliseconds vs. minutes for large graphs.

    The discount formula penalises high-degree nodes whose neighbours are
    already seeds, preventing redundant coverage.
    """

    def test_returns_imresult(self, small_graph):
        r = DegreeDiscount().select_seeds(small_graph, k=3)
        assert isinstance(r, IMResult)

    def test_correct_k(self, small_graph):
        r = DegreeDiscount().select_seeds(small_graph, k=5)
        assert len(r.seeds) == 5

    def test_method_label(self, small_graph):
        r = DegreeDiscount().select_seeds(small_graph, k=3)
        assert r.method == "DegreeDiscount"

    def test_no_duplicates(self, small_graph):
        r = DegreeDiscount().select_seeds(small_graph, k=5)
        assert len(set(r.seeds)) == 5

    def test_spread_positive(self, small_graph):
        r = DegreeDiscount().select_seeds(small_graph, k=3)
        assert r.spread > 0

    def test_finds_hub_in_star(self, star_graph):
        """
        DegreeDiscount must select the hub (node 0) which has degree 19.
        All leaves have degree 1, so node 0 should always win the first step.
        """
        r = DegreeDiscount().select_seeds(star_graph, k=1)
        assert 0 in r.seeds

    def test_faster_than_greedy(self, medium_graph):
        """
        DegreeDiscount must be substantially faster than GreedyMC.

        This is the main engineering motivation: use DegreeDiscount when you
        need a quick result, or as a warm-start for MRILS.
        """
        t0 = time.time()
        DegreeDiscount().select_seeds(medium_graph, k=5)
        t_dd = time.time() - t0

        t0 = time.time()
        GreedyMC(n_sim=50).select_seeds(medium_graph, k=5)
        t_greedy = time.time() - t0

        assert t_dd < t_greedy


# ===========================================================================
#  Section 8 – MRILS
# ===========================================================================

class TestMRILS:
    """
    MRILS = Multi-Restart RIS + RR-guided Local Search.

    Comparison with Greedy-MC
    ─────────────────────────
    Greedy-MC is the gold standard because it provably maximises spread
    (submodular maximisation via greedy).  Its weakness: it is O(n·k·n_sim).

    MRILS addresses this by:
    1. Running _ris_greedy() multiple times (multi-restart) → diversity.
    2. Applying rr_local_search() to each → swap refinement.
    3. Evaluating all candidates with MC → picking the best.

    Theoretically MRILS achieves the same (1-1/e-ε) approximation as single
    RIS, but in practice the multi-restart + local-search combination often
    beats both SingleRIS and even Greedy-MC (400 sims/step) on spread quality.

    We test: correctness, k, method label, spread plausibility, and that
    MRILS spread >= SingleRIS spread on a medium graph (it uses more compute
    for a reason).
    """

    def test_returns_imresult(self, small_graph):
        r = MRILS(n_restarts=2, ls_passes=2, eval_sims=200).select_seeds(small_graph, k=3)
        assert isinstance(r, IMResult)

    def test_correct_k(self, small_graph):
        r = MRILS(n_restarts=2, ls_passes=2, eval_sims=200).select_seeds(small_graph, k=5)
        assert len(r.seeds) == 5

    def test_method_label(self, small_graph):
        r = MRILS(n_restarts=2, ls_passes=2, eval_sims=200).select_seeds(small_graph, k=3)
        assert r.method == "MRILS"

    def test_no_duplicate_seeds(self, small_graph):
        r = MRILS(n_restarts=2, ls_passes=2, eval_sims=200).select_seeds(small_graph, k=5)
        assert len(set(r.seeds)) == 5

    def test_spread_positive(self, small_graph):
        r = MRILS(n_restarts=2, ls_passes=2, eval_sims=200).select_seeds(small_graph, k=3)
        assert r.spread > 0

    def test_spread_bounded(self, small_graph):
        r = MRILS(n_restarts=2, ls_passes=2, eval_sims=200).select_seeds(small_graph, k=3)
        assert r.spread <= small_graph.number_of_nodes()

    def test_extras_contains_all_spreads(self, small_graph):
        """
        MRILS evaluates each restart's candidate and stores all spreads.
        extras['all_spreads'] lets callers inspect variance across restarts.
        """
        r = MRILS(n_restarts=3, ls_passes=2, eval_sims=200).select_seeds(small_graph, k=3)
        assert "all_spreads" in r.extras
        assert len(r.extras["all_spreads"]) == 3

    def test_best_spread_is_max_of_all_spreads(self, small_graph):
        """
        The returned spread must equal max(all_spreads).
        If this fails, MRILS is returning a suboptimal candidate.
        """
        r = MRILS(n_restarts=3, ls_passes=2, eval_sims=200).select_seeds(small_graph, k=3)
        assert r.spread == pytest.approx(max(r.extras["all_spreads"]))

    def test_mrils_spread_gte_single_ris(self, medium_graph):
        """
        MRILS uses strictly more computation than SingleRIS (multiple restarts
        + local search), so its spread should be >= SingleRIS on the same graph.

        This is the key empirical claim of the MRILS paper.

        NOTE: stochastic – we use fixed rng seeds inside the algorithms to
        keep this deterministic.
        """
        r_ris = SingleRIS().select_seeds(medium_graph, k=10)
        r_mrils = MRILS(n_restarts=4, ls_passes=4, eval_sims=500).select_seeds(medium_graph, k=10)
        # Allow a tiny tolerance for MC noise
        assert r_mrils.spread >= r_ris.spread - 0.5


# ===========================================================================
#  Section 9 – MRILS-Enhanced
# ===========================================================================

class TestMRILSEnhanced:
    """
    MRILS-Enhanced adds two mechanisms on top of MRILS:

    1. MC-validated swaps
       Instead of accepting a swap purely based on RR-set coverage delta,
       the top-3 candidates are evaluated with a quick MC estimate (200 sims).
       This reduces false-positive swaps caused by RR-set noise.

    2. Biased re-seeding
       After each local-search result, a new RIS instance is generated whose
       root distribution is biased toward the current seeds.  This exploits
       structural information while maintaining diversity.

    Comparison with Greedy-MC and MRILS
    ─────────────────────────────────────
    MRILS-Enhanced aims to close the gap with Greedy-MC (and sometimes exceed
    it) by using MC information during local search, not just at final
    evaluation.  It is slower than MRILS but faster than Greedy-MC.

    We test the same structural properties as MRILS plus the extra mechanisms.
    """

    def test_returns_imresult(self, small_graph):
        r = MRILSEnhanced(
            n_restarts=2, ls_passes=2, mc_val_sims=50, eval_sims=200, re_seed=False
        ).select_seeds(small_graph, k=3)
        assert isinstance(r, IMResult)

    def test_correct_k(self, small_graph):
        r = MRILSEnhanced(
            n_restarts=2, ls_passes=2, mc_val_sims=50, eval_sims=200, re_seed=False
        ).select_seeds(small_graph, k=5)
        assert len(r.seeds) == 5

    def test_method_label(self, small_graph):
        r = MRILSEnhanced(
            n_restarts=2, ls_passes=2, mc_val_sims=50, eval_sims=200, re_seed=False
        ).select_seeds(small_graph, k=3)
        assert r.method == "MRILS-Enhanced"

    def test_re_seed_doubles_candidates(self, small_graph):
        """
        With re_seed=True, each restart generates 2 candidates (original +
        biased re-seed), so len(all_spreads) should be 2 * n_restarts.
        """
        r = MRILSEnhanced(
            n_restarts=3, ls_passes=2, mc_val_sims=50, eval_sims=200, re_seed=True
        ).select_seeds(small_graph, k=3)
        assert len(r.extras["all_spreads"]) == 6  # 3 restarts × 2

    def test_without_re_seed_n_candidates(self, small_graph):
        """
        With re_seed=False, candidates = n_restarts only.
        """
        r = MRILSEnhanced(
            n_restarts=4, ls_passes=2, mc_val_sims=50, eval_sims=200, re_seed=False
        ).select_seeds(small_graph, k=3)
        assert len(r.extras["all_spreads"]) == 4

    def test_spread_bounded(self, small_graph):
        r = MRILSEnhanced(
            n_restarts=2, ls_passes=2, mc_val_sims=50, eval_sims=200, re_seed=False
        ).select_seeds(small_graph, k=3)
        assert r.spread <= small_graph.number_of_nodes()

    def test_enhanced_gte_single_ris(self, medium_graph):
        """
        MRILS-Enhanced should outperform SingleRIS on a medium graph.

        This is the main empirical claim: enhanced mechanisms produce better
        seed sets than a single vanilla RIS run.
        """
        r_ris = SingleRIS().select_seeds(medium_graph, k=10)
        r_enh = MRILSEnhanced(
            n_restarts=4, ls_passes=4, mc_val_sims=100, eval_sims=500, re_seed=True
        ).select_seeds(medium_graph, k=10)
        assert r_enh.spread >= r_ris.spread - 0.5


# ===========================================================================
#  Section 10 – GreedyMC (slow, marked so you can skip in CI)
# ===========================================================================

class TestGreedyMC:
    """
    GreedyMC is the gold-standard baseline.

    Why test it?
    Even though MRILS aims to *beat* GreedyMC, we need to verify GreedyMC
    itself is correct – otherwise we can't trust the comparisons.

    The tests use n_sim=50 (very low) to keep runtime short.  In production
    you'd use n_sim=400–3000.
    """

    @pytest.mark.slow
    def test_returns_imresult(self, small_graph):
        r = GreedyMC(n_sim=50).select_seeds(small_graph, k=2)
        assert isinstance(r, IMResult)

    @pytest.mark.slow
    def test_correct_k(self, small_graph):
        r = GreedyMC(n_sim=50).select_seeds(small_graph, k=3)
        assert len(r.seeds) == 3

    @pytest.mark.slow
    def test_method_label(self, small_graph):
        r = GreedyMC(n_sim=50).select_seeds(small_graph, k=2)
        assert r.method == "Greedy-MC"

    @pytest.mark.slow
    def test_spread_positive(self, small_graph):
        r = GreedyMC(n_sim=50).select_seeds(small_graph, k=2)
        assert r.spread > 0

    @pytest.mark.slow
    def test_finds_hub_in_star(self, star_graph):
        """
        Greedy-MC with weight=1.0 must select hub (node 0) first.
        Marginal gain of node 0 = 20 (all leaves); any leaf = 2.
        """
        r = GreedyMC(n_sim=100).select_seeds(star_graph, k=1)
        assert 0 in r.seeds

    @pytest.mark.slow
    def test_no_duplicates(self, small_graph):
        r = GreedyMC(n_sim=50).select_seeds(small_graph, k=4)
        assert len(set(r.seeds)) == 4


# ===========================================================================
#  Section 11 – run_comparison() integration test
# ===========================================================================

class TestRunComparison:
    """
    run_comparison() is the public API that runs all methods and returns a dict.

    Integration tests: verify the dict structure, keys, and that relative
    spread ordering matches expectations on the medium graph.
    """

    @pytest.fixture(scope="class")
    def fast_results(self, medium_graph):
        """Run the fast methods (no greedy) once and reuse across tests."""
        return run_comparison(medium_graph, k=10, methods=["mrils", "ris", "degree"])

    def test_returns_dict(self, fast_results):
        assert isinstance(fast_results, dict)

    def test_correct_keys(self, fast_results):
        assert set(fast_results.keys()) == {"mrils", "ris", "degree"}

    def test_all_values_are_imresult(self, fast_results):
        for v in fast_results.values():
            assert isinstance(v, IMResult)

    def test_all_have_k_seeds(self, fast_results):
        for r in fast_results.values():
            assert len(r.seeds) == 10

    def test_all_spreads_positive(self, fast_results):
        for r in fast_results.values():
            assert r.spread > 0

    def test_mrils_spread_gte_ris(self, fast_results):
        """
        MRILS uses more compute than SingleRIS, so its spread should be >= RIS.

        This is the fundamental payoff of the multi-restart strategy.
        """
        assert fast_results["mrils"].spread >= fast_results["ris"].spread - 0.5

    def test_default_methods_excludes_greedy(self, medium_graph):
        """
        By default run_comparison() excludes greedy (too slow for quick runs).
        This is a deliberate design choice – greedy must be explicitly requested.
        """
        results = run_comparison(medium_graph, k=5)
        assert "greedy" not in results

    def test_custom_methods_selection(self, medium_graph):
        results = run_comparison(medium_graph, k=5, methods=["ris", "degree"])
        assert set(results.keys()) == {"ris", "degree"}

    @pytest.mark.slow
    def test_greedy_comparable_to_mrils(self, medium_graph):
        """
        On a medium graph with k=5, Greedy-MC (400 sims/step) and MRILS
        should produce spreads within ~15% of each other.

        This is the core empirical claim of the MRILS paper.
        """
        results = run_comparison(medium_graph, k=5, methods=["mrils", "greedy"])
        mrils_spread = results["mrils"].spread
        greedy_spread = results["greedy"].spread
        # Allow 20% tolerance due to stochasticity
        assert mrils_spread >= greedy_spread * 0.80


# ===========================================================================
#  Section 12 – Algorithmic quality comparison on medium graph
# ===========================================================================

class TestSpreadOrdering:
    """
    On a well-behaved medium graph we expect a rough spread ordering:

        MRILS-Enhanced ≥ MRILS ≥ SingleRIS ≥ DegreeDiscount  (generally)

    These are not strict guarantees (all algorithms are stochastic), but
    they should hold with a comfortable margin on a 500-node graph with k=10.

    This section explicitly compares all fast methods against each other.
    """

    @pytest.fixture(scope="class")
    def all_results(self, medium_graph):
        return {
            "mrils":    MRILS(n_restarts=6, ls_passes=6, eval_sims=1000).select_seeds(medium_graph, k=10),
            "enhanced": MRILSEnhanced(n_restarts=6, ls_passes=4, mc_val_sims=100, eval_sims=1000, re_seed=True).select_seeds(medium_graph, k=10),
            "ris":      SingleRIS().select_seeds(medium_graph, k=10),
            "degree":   DegreeDiscount().select_seeds(medium_graph, k=10),
        }

    def test_mrils_beats_degree_discount(self, all_results):
        """
        MRILS uses RIS + local search (principled approximation) vs.
        DegreeDiscount (pure heuristic).  MRILS should win on spread.
        """
        assert all_results["mrils"].spread >= all_results["degree"].spread - 0.5

    def test_mrils_beats_or_matches_single_ris(self, all_results):
        """
        Local search on top of RIS should never make things worse.
        """
        assert all_results["mrils"].spread >= all_results["ris"].spread - 0.5

    def test_enhanced_beats_or_matches_mrils(self, all_results):
        """
        MRILS-Enhanced uses more compute (MC-validated swaps + re-seeding)
        so it should not fall below MRILS.
        """
        assert all_results["enhanced"].spread >= all_results["mrils"].spread - 0.5

    def test_runtime_ordering(self, all_results):
        """
        DegreeDiscount < SingleRIS < MRILS < MRILS-Enhanced (roughly).
        We only enforce the most extreme: degree must be faster than MRILS.
        """
        assert all_results["degree"].runtime < all_results["mrils"].runtime


# ===========================================================================
#  Section 13 – Edge cases and robustness
# ===========================================================================

class TestEdgeCases:
    """
    Robustness tests: what happens with unusual inputs?

    These protect against crashes or silent errors on corner-case graphs
    that might appear in real datasets.
    """

    def test_k_equals_1(self):
        G = nx.DiGraph(nx.scale_free_graph(50, seed=1))
        for u, v in G.edges():
            G[u][v]["weight"] = 0.1
        r = MRILS(n_restarts=2, ls_passes=2, eval_sims=100).select_seeds(G, k=1)
        assert len(r.seeds) == 1

    def test_undirected_graph_mc_spread(self):
        """mc_spread must handle undirected graphs (uses G.neighbors)."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        spread = mc_spread(G, seeds=[0], n_sim=100, rng_seed=1)
        assert spread == pytest.approx(3.0, abs=0.1)

    def test_no_edges_graph(self):
        """Isolated nodes: spread = number of seeds (no propagation)."""
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2, 3, 4])
        spread = mc_spread(G, seeds=[0, 1], n_sim=100, rng_seed=1)
        assert spread == pytest.approx(2.0, abs=0.01)

    def test_graph_with_self_loops(self):
        """Self-loops should not cause infinite loops in cascade/BFS."""
        G = nx.DiGraph()
        G.add_edge(0, 0, weight=1.0)   # self-loop
        G.add_edge(0, 1, weight=1.0)
        spread = mc_spread(G, seeds=[0], n_sim=50, rng_seed=1)
        assert spread >= 1.0

    def test_high_weight_gives_more_spread(self):
        """Higher edge weights → more propagation → higher spread."""
        G_low = nx.DiGraph(nx.scale_free_graph(80, seed=5))
        G_high = G_low.copy()
        for u, v in G_low.edges():
            G_low[u][v]["weight"] = 0.05
            G_high[u][v]["weight"] = 0.5
        seeds = list(G_low.nodes())[:5]
        s_low = mc_spread(G_low, seeds, n_sim=500, rng_seed=1)
        s_high = mc_spread(G_high, seeds, n_sim=500, rng_seed=1)
        assert s_high > s_low
