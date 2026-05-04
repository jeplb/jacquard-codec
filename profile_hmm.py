"""
profile HMM for tdt-synthesized DNA sequences.

each pattern gets its own HMM with self-loops on match states (geometric
hp-length), insert states for tdt carryover, delete states for skipped
bases, and a g-tail terminator. viterbi gives the MAP path; forward gives
the marginal likelihood; we use forward to identify which codebook entry
a read came from, viterbi to pull out per-position run-length counts.

implementation is plain python + numpy. inner loops walk the sparse
log-trans matrix (mostly -inf) so each step touches only a couple of
predecessors. fast enough for the paper's read counts.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum

import numba
from numba import njit

from logging_config import get_logger

logger = get_logger(__name__)


# sentinel used in the njit kernel; finite so numba doesn't choke on -inf
# arithmetic, and far enough below any realistic log-prob that real scores
# never reach it.
NEG_INF_SENTINEL = -1e308


BASES = ['A', 'C', 'G', 'T']
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
IDX_TO_BASE = {i: b for i, b in enumerate(BASES)}


def laplace_log_likelihood(observed: float, expected: float, scale: float) -> float:
    """laplace log-pdf, sans the -log(2b) constant.

    we use this for hp-length scoring in the run-based decoder. why
    laplace: nanopore hp-length errors are heavy-tailed (kurtosis ~4.3,
    shapiro-wilk p ~0). laplace beats gaussian by ~6000 AIC on the
    empirical fit. L1 penalty matches the long-tail behavior we see in
    the channel matrix.
    """
    return -abs(observed - expected) / scale


class StateType(Enum):
    START = "S"
    MATCH = "M"      # emits the expected base; self-loop carries hp extension
    INSERT = "I"     # emits leftover/carryover from the previous position
    DELETE = "D"     # silent; lets us skip a base
    GTAIL = "G"      # terminal poly-G anchor
    END = "E"


@dataclass
class ProfileHMMParameters:
    """transition + emission knobs.

    these are the defaults used in the paper. fitted by hand against the
    convergence/error-rate tables; nothing fancy. the run-based scorer uses
    a laplace likelihood (see hp_laplace_scale) but the base-by-base
    viterbi uses the geometric implicit in the self-loop, which is what
    everyone does and is what everyone gets dinged for at review.
    """
    p_match_extend: float = 0.70        # M_i -> M_i (hp extension)
    p_match_to_next: float = 0.25       # M_i -> M_{i+1}
    p_match_to_insert: float = 0.03     # M_i -> I_i (leftover)
    p_match_to_delete: float = 0.02     # M_i -> D_{i+1}

    p_insert_to_match: float = 0.65     # I_i -> M_{i+1}
    p_insert_to_insert: float = 0.35    # I_i -> I_i

    p_delete_to_match: float = 0.90     # D_i -> M_i
    p_delete_to_delete: float = 0.10    # D_i -> D_{i+1}

    p_gtail_extend: float = 0.80        # G -> G
    p_gtail_from_last: float = 0.95     # last M -> G (vs M -> I)

    p_emission_error: float = 0.02      # rough nanopore base-call error rate

    # used by score_runs (run-based scorer); tuned on real data
    hp_expected_length: float = 3.0
    hp_laplace_scale: float = 1.8


def _build_sparse(log_trans: np.ndarray):
    """sparse adjacency lists from a dense log-trans matrix.

    predecessors[j] = [(i, log_p), ...] with log_p > -inf, ascending i.
    successors[i]   = [(j, log_p), ...] with log_p > -inf, ascending j.
    iteration order matters for tie-breaking in viterbi.
    """
    n = log_trans.shape[0]
    predecessors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    successors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = float(log_trans[i, j])
            if v > -math.inf:
                predecessors[j].append((i, v))
                successors[i].append((j, v))
    return predecessors, successors


def _csr_from_lists(predecessors: List[List[Tuple[int, float]]],
                    successors: List[List[Tuple[int, float]]]):
    """CSR-style arrays from predecessor/successor adjacency lists.

    returns six arrays:
        pred_indptr     int64, shape (n + 1,)
        pred_indices    int64, shape (nnz,)   - predecessor i for each j-column
        pred_log_trans  float64, shape (nnz,)
        succ_indptr     int64, shape (n + 1,)
        succ_indices    int64, shape (nnz,)   - successor j for each i-row
        succ_log_trans  float64, shape (nnz,)

    entries inherit the ordering of the input lists (predecessors[j] in
    ascending i, successors[i] in ascending j), so the njit viterbi
    reproduces the lowest-i tie-breaking of the list-based version.
    """
    n = len(predecessors)
    pred_indptr = np.zeros(n + 1, dtype=np.int64)
    succ_indptr = np.zeros(n + 1, dtype=np.int64)
    for k in range(n):
        pred_indptr[k + 1] = pred_indptr[k] + len(predecessors[k])
        succ_indptr[k + 1] = succ_indptr[k] + len(successors[k])

    pred_nnz = int(pred_indptr[-1])
    succ_nnz = int(succ_indptr[-1])
    pred_indices = np.zeros(pred_nnz, dtype=np.int64)
    pred_log_trans = np.zeros(pred_nnz, dtype=np.float64)
    succ_indices = np.zeros(succ_nnz, dtype=np.int64)
    succ_log_trans = np.zeros(succ_nnz, dtype=np.float64)

    for k in range(n):
        base = int(pred_indptr[k])
        for off, (i, v) in enumerate(predecessors[k]):
            pred_indices[base + off] = i
            pred_log_trans[base + off] = v
        base = int(succ_indptr[k])
        for off, (j, v) in enumerate(successors[k]):
            succ_indices[base + off] = j
            succ_log_trans[base + off] = v

    return (pred_indptr, pred_indices, pred_log_trans,
            succ_indptr, succ_indices, succ_log_trans)


def _build_sparse_csr(log_trans: np.ndarray):
    """flatten a dense log-trans matrix into CSR-style numpy arrays.

    same six-array layout as _csr_from_lists. kept for callers that start
    from a dense matrix (e.g. the parity tests); production HMM construction
    goes through the edge-dict path in ProfileHMM._rebuild_sparse instead.
    """
    predecessors, successors = _build_sparse(log_trans)
    return _csr_from_lists(predecessors, successors)


def _logsumexp(values: List[float]) -> float:
    """numerically stable log-sum-exp. textbook."""
    if not values:
        return -math.inf
    max_val = max(values)
    if max_val == -math.inf:
        return -math.inf
    s = 0.0
    for v in values:
        s += math.exp(v - max_val)
    return max_val + math.log(s)


def _run_viterbi(seq_idx, n_states, predecessors, successors,
                 log_emit, emitting_mask, start_idx):
    """sparse viterbi forward pass. returns (V, bp).

    ties go to the lowest predecessor index because we iterate ascending
    and only replace on strict-greater.
    """
    t_len = len(seq_idx)
    v = np.full((t_len + 1, n_states), -math.inf, dtype=np.float64)
    bp = np.zeros((t_len + 1, n_states), dtype=np.int64)

    v[0, start_idx] = 0.0

    obs = int(seq_idx[0])
    for j, log_p in successors[start_idx]:
        emit = log_emit[j, obs]
        if emit > -math.inf:
            v[1, j] = log_p + emit
            bp[1, j] = start_idx

    for t in range(2, t_len + 1):
        obs = int(seq_idx[t - 1])
        for j in range(n_states):
            if not emitting_mask[j]:
                continue
            emit = log_emit[j, obs]
            if emit == -math.inf:
                continue

            best_score = -math.inf
            best_prev = 0
            for i, log_trans in predecessors[j]:
                prev_score = v[t - 1, i]
                if prev_score > -math.inf:
                    s = prev_score + log_trans
                    if s > best_score:  # strict >, ties stay with lowest i
                        best_score = s
                        best_prev = i

            if best_score > -math.inf:
                v[t, j] = best_score + emit
                bp[t, j] = best_prev

    return v, bp


@njit(cache=True)
def _run_viterbi_njit(seq_idx, n_states,
                      pred_indptr, pred_indices, pred_log_trans,
                      succ_indptr, succ_indices, succ_log_trans,
                      log_emit, emitting_mask, start_idx):
    """njit viterbi over CSR predecessors. mirrors _run_viterbi.

    log_emit and v use NEG_INF_SENTINEL (-1e308) instead of -inf so numba
    avoids -inf arithmetic. real log-probs never reach the sentinel.
    tie-breaking matches the list version: strict-greater update over
    ascending predecessor indices keeps the lowest i.
    """
    NEG = -1e308
    t_len = seq_idx.shape[0]
    v = np.full((t_len + 1, n_states), NEG, dtype=np.float64)
    bp = np.zeros((t_len + 1, n_states), dtype=np.int64)

    v[0, start_idx] = 0.0

    obs0 = seq_idx[0]
    for k in range(succ_indptr[start_idx], succ_indptr[start_idx + 1]):
        j = succ_indices[k]
        log_p = succ_log_trans[k]
        emit = log_emit[j, obs0]
        if emit > NEG:
            v[1, j] = log_p + emit
            bp[1, j] = start_idx

    for t in range(2, t_len + 1):
        obs = seq_idx[t - 1]
        for j in range(n_states):
            if not emitting_mask[j]:
                continue
            emit = log_emit[j, obs]
            if emit <= NEG:
                continue

            best_score = NEG
            best_prev = 0
            for k in range(pred_indptr[j], pred_indptr[j + 1]):
                i = pred_indices[k]
                log_trans = pred_log_trans[k]
                prev_score = v[t - 1, i]
                if prev_score > NEG:
                    s = prev_score + log_trans
                    if s > best_score:  # strict >, ties stay with lowest i
                        best_score = s
                        best_prev = i

            if best_score > NEG:
                v[t, j] = best_score + emit
                bp[t, j] = best_prev

    return v, bp


def _run_forward(seq_idx, n_states, predecessors, successors,
                 log_emit, emitting_mask, start_idx):
    """sparse forward pass. mirrors _run_viterbi structurally."""
    t_len = len(seq_idx)
    alpha = np.full((t_len + 1, n_states), -math.inf, dtype=np.float64)

    alpha[0, start_idx] = 0.0

    obs = int(seq_idx[0])
    for j, log_p in successors[start_idx]:
        emit = log_emit[j, obs]
        if emit > -math.inf:
            alpha[1, j] = log_p + emit

    for t in range(2, t_len + 1):
        obs = int(seq_idx[t - 1])
        for j in range(n_states):
            if not emitting_mask[j]:
                continue
            emit = log_emit[j, obs]
            if emit == -math.inf:
                continue

            preds = predecessors[j]

            max_val = -math.inf
            for i, log_trans in preds:
                prev = alpha[t - 1, i]
                if prev > -math.inf:
                    s = prev + log_trans
                    if s > max_val:
                        max_val = s

            if max_val > -math.inf:
                s = 0.0
                for i, log_trans in preds:
                    prev = alpha[t - 1, i]
                    if prev > -math.inf:
                        s += math.exp(prev + log_trans - max_val)
                alpha[t, j] = max_val + math.log(s) + emit

    return alpha


class ProfileHMM:
    """one HMM per pattern.

    structure for a pattern of length L:
        START -> M_0 (self-loop) -> M_1 (self-loop) -> ... -> M_{L-1} -> GTAIL (self-loop) -> END
    plus I_i parallel to each M_i (carryover) and D_i for skipping.
    """

    def __init__(self, pattern: str, params: Optional[ProfileHMMParameters] = None):
        self.pattern = pattern.upper()
        self.params = params or ProfileHMMParameters()
        self.L = len(pattern)

        self._build_states()
        self._build_transitions()
        self._build_emissions()
        self._rebuild_sparse()

    def _build_states(self):
        """state list: START, then M_i / I_i / D_i (D skipped at i=0), then GTAIL, END."""
        self.states: List[Tuple[StateType, int, Optional[str]]] = []
        self.state_to_idx: Dict[Tuple[StateType, int, Optional[str]], int] = {}

        self.states.append((StateType.START, -1, None))

        for i, base in enumerate(self.pattern):
            self.states.append((StateType.MATCH, i, base))
            self.states.append((StateType.INSERT, i, None))
            if i > 0:
                self.states.append((StateType.DELETE, i, None))

        self.states.append((StateType.GTAIL, self.L, 'G'))
        self.states.append((StateType.END, -1, None))

        for idx, state in enumerate(self.states):
            self.state_to_idx[state] = idx

        # (state_type, position) uniquely identifies a state, so this index
        # turns _get_state_idx into an O(1) lookup instead of a linear scan.
        self._idx_by_type_pos: Dict[Tuple[StateType, int], int] = {
            (state[0], state[1]): idx for idx, state in enumerate(self.states)
        }

        self.n_states = len(self.states)
        self.start_idx = self.state_to_idx[(StateType.START, -1, None)]
        self.end_idx = self.state_to_idx[(StateType.END, -1, None)]
        self.gtail_idx = self.state_to_idx[(StateType.GTAIL, self.L, 'G')]

    def _get_state_idx(self, state_type: StateType, position: int,
                       base: Optional[str] = None) -> int:
        idx = self._idx_by_type_pos.get((state_type, position), -1)
        if idx < 0:
            return -1
        if base is not None and self.states[idx][2] != base:
            return -1
        return idx

    def _set_trans(self, i: int, j: int, value: float):
        """set a transition log-prob and record the edge.

        recording edges as they are set lets _rebuild_sparse build the sparse
        adjacency and CSR arrays directly, skipping two full n-by-n scans of
        log_trans. last write wins, matching dense-matrix overwrite semantics.
        """
        self.log_trans[i, j] = value
        self._trans_edges[(i, j)] = value

    def _build_transitions(self):
        """fill log_trans from the parameter dataclass.

        we set entries explicitly per state type and trust that the implied
        outgoing mass sums to 1 (it does, for the defaults). don't normalize
        afterwards.
        """
        p = self.params
        self.log_trans = np.full((self.n_states, self.n_states), -np.inf)
        self._trans_edges: Dict[Tuple[int, int], float] = {}

        for state, idx in self.state_to_idx.items():
            state_type, pos, _ = state

            if state_type == StateType.START:
                m0_idx = self._get_state_idx(StateType.MATCH, 0, self.pattern[0])
                self._set_trans(idx, m0_idx, 0.0)  # log(1)

            elif state_type == StateType.MATCH:
                self._set_trans(idx, idx, math.log(p.p_match_extend))

                if pos < self.L - 1:
                    next_base = self.pattern[pos + 1]
                    m_next_idx = self._get_state_idx(StateType.MATCH, pos + 1, next_base)
                    self._set_trans(idx, m_next_idx, math.log(p.p_match_to_next))

                    i_idx = self._get_state_idx(StateType.INSERT, pos)
                    self._set_trans(idx, i_idx, math.log(p.p_match_to_insert))

                    if pos + 1 < self.L - 1:
                        d_idx = self._get_state_idx(StateType.DELETE, pos + 1)
                        if d_idx >= 0:
                            self._set_trans(idx, d_idx, math.log(p.p_match_to_delete))
                else:
                    # last match -> gtail or insert; split the leftover mass.
                    remaining = 1.0 - p.p_match_extend
                    self._set_trans(idx, self.gtail_idx, math.log(remaining * p.p_gtail_from_last))
                    i_idx = self._get_state_idx(StateType.INSERT, pos)
                    self._set_trans(idx, i_idx, math.log(remaining * (1 - p.p_gtail_from_last)))

            elif state_type == StateType.INSERT:
                self._set_trans(idx, idx, math.log(p.p_insert_to_insert))
                if pos < self.L - 1:
                    next_base = self.pattern[pos + 1]
                    m_next_idx = self._get_state_idx(StateType.MATCH, pos + 1, next_base)
                    self._set_trans(idx, m_next_idx, math.log(p.p_insert_to_match))
                else:
                    self._set_trans(idx, self.gtail_idx, math.log(p.p_insert_to_match))

            elif state_type == StateType.DELETE:
                m_idx = self._get_state_idx(StateType.MATCH, pos, self.pattern[pos])
                self._set_trans(idx, m_idx, math.log(p.p_delete_to_match))
                if pos < self.L - 1:
                    d_next_idx = self._get_state_idx(StateType.DELETE, pos + 1)
                    if d_next_idx >= 0:
                        self._set_trans(idx, d_next_idx, math.log(p.p_delete_to_delete))

            elif state_type == StateType.GTAIL:
                self._set_trans(idx, idx, math.log(p.p_gtail_extend))
                self._set_trans(idx, self.end_idx, math.log(1.0 - p.p_gtail_extend))

    def _build_emissions(self):
        """match emits its base; insert biases toward the previous base
        (because that's what tdt carryover looks like in real reads); gtail
        emits G."""
        p = self.params
        self.log_emit = np.full((self.n_states, 4), -np.inf)

        for state, idx in self.state_to_idx.items():
            state_type, pos, base = state

            if state_type == StateType.MATCH:
                for i, b in enumerate(BASES):
                    if b == base:
                        self.log_emit[idx, i] = math.log(1.0 - p.p_emission_error)
                    else:
                        self.log_emit[idx, i] = math.log(p.p_emission_error / 3)

            elif state_type == StateType.INSERT:
                if pos > 0:
                    prev_base = self.pattern[pos - 1]
                    for i, b in enumerate(BASES):
                        # 0.55 for the carryover base, 0.15 each for the rest.
                        # eyeballed from a few hundred reads; not worth fitting.
                        if b == prev_base:
                            self.log_emit[idx, i] = math.log(0.55)
                        else:
                            self.log_emit[idx, i] = math.log(0.15)
                else:
                    self.log_emit[idx, :] = math.log(0.25)

            elif state_type == StateType.GTAIL:
                for i, b in enumerate(BASES):
                    if b == 'G':
                        self.log_emit[idx, i] = math.log(1.0 - p.p_emission_error)
                    else:
                        self.log_emit[idx, i] = math.log(p.p_emission_error / 3)

    def _adjacency_from_edges(self):
        """sparse predecessors/successors from the recorded edge dict.

        reproduces the ordering _build_sparse derives from a dense scan:
        successors[i] ascending j, predecessors[j] ascending i. only finite
        edges are kept, matching the dense > -inf filter.
        """
        n = self.n_states
        predecessors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        successors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        edges = self._trans_edges
        for (i, j) in sorted(edges):  # ascending (i, j)
            v = edges[(i, j)]
            if v > -math.inf:
                successors[i].append((j, v))
        for (i, j) in sorted(edges, key=lambda e: (e[1], e[0])):  # ascending (j, i)
            v = edges[(i, j)]
            if v > -math.inf:
                predecessors[j].append((i, v))
        return predecessors, successors

    def _rebuild_sparse(self):
        """rebuild sparse adjacency + emitting mask after log_trans is touched.

        builds the list adjacency and CSR-flat copies from the edge dict
        recorded in _build_transitions (no dense n-by-n scan), plus the
        sentinel-converted log_emit + bool mask array consumed by
        _run_viterbi_njit. log_emit must be set before we get here (we read
        it for the sentinel copy).
        """
        self.predecessors, self.successors = self._adjacency_from_edges()
        self.emitting_mask = [
            s[0] != StateType.START and s[0] != StateType.END for s in self.states
        ]
        (self.pred_indptr, self.pred_indices, self.pred_log_trans,
         self.succ_indptr, self.succ_indices, self.succ_log_trans) = \
            _csr_from_lists(self.predecessors, self.successors)
        self.emitting_mask_arr = np.array(self.emitting_mask, dtype=np.bool_)
        self.log_emit_sentinel = np.where(
            np.isneginf(self.log_emit), NEG_INF_SENTINEL, self.log_emit
        ).astype(np.float64)

    def viterbi(self, sequence) -> Tuple[List[Tuple[StateType, int, Optional[str]]], float]:
        """MAP state path. accepts a DNA string OR a pre-converted int32
        index array (decoder.py and stress_test pass arrays directly).

        path is a list of (StateType, position, base_or_None) tuples in
        time order, length T (one per observation, no START/END).
        """
        if isinstance(sequence, str):
            seq_idx = np.array([BASE_TO_IDX[b] for b in sequence.upper()], dtype=np.int32)
        else:
            seq_idx = np.asarray(sequence, dtype=np.int32)
        v, bp = _run_viterbi_njit(
            seq_idx, self.n_states,
            self.pred_indptr, self.pred_indices, self.pred_log_trans,
            self.succ_indptr, self.succ_indices, self.succ_log_trans,
            self.log_emit_sentinel, self.emitting_mask_arr, self.start_idx,
        )
        # restore -inf so path reconstruction below behaves like the list version
        v = np.where(v == NEG_INF_SENTINEL, -np.inf, v)
        t_len = len(seq_idx)

        best_score = -math.inf
        best_state = 0
        for i, log_p in self.predecessors[self.end_idx]:
            score = v[t_len, i] + log_p
            if score > best_score:
                best_score = score
                best_state = i

        path: List[Tuple[StateType, int, Optional[str]]] = []
        curr = best_state
        for t in range(t_len, 0, -1):
            path.append(self.states[curr])
            curr = int(bp[t, curr])
        path.reverse()
        return path, best_score

    def viterbi_score(self, sequence) -> float:
        """best viterbi path log-score, without reconstructing the path.

        returns the same value as viterbi()[1] but skips the full-matrix
        sentinel conversion and the backtrace. callers that only need the
        score to rank candidates (HMMLibrary.viterbi_best) use this to avoid
        639 wasted path reconstructions per read, paying for the path only
        on the winning HMM.
        """
        if isinstance(sequence, str):
            seq_idx = np.array([BASE_TO_IDX[b] for b in sequence.upper()], dtype=np.int32)
        else:
            seq_idx = np.asarray(sequence, dtype=np.int32)
        v, _ = _run_viterbi_njit(
            seq_idx, self.n_states,
            self.pred_indptr, self.pred_indices, self.pred_log_trans,
            self.succ_indptr, self.succ_indices, self.succ_log_trans,
            self.log_emit_sentinel, self.emitting_mask_arr, self.start_idx,
        )
        last = v[len(seq_idx)]
        best_score = -math.inf
        for i, log_p in self.predecessors[self.end_idx]:
            val = last[i]
            if val == NEG_INF_SENTINEL:
                continue
            score = val + log_p
            if score > best_score:  # max value; matches viterbi()'s best_score
                best_score = score
        return best_score

    def forward(self, sequence) -> np.ndarray:
        """log-alpha matrix, shape (T+1, n_states)."""
        if isinstance(sequence, str):
            seq_idx = np.array([BASE_TO_IDX[b] for b in sequence.upper()], dtype=np.int32)
        else:
            seq_idx = np.asarray(sequence, dtype=np.int32)
        return _run_forward(
            seq_idx, self.n_states, self.predecessors, self.successors,
            self.log_emit, self.emitting_mask, self.start_idx,
        )

    def score_sequence(self, sequence) -> float:
        """log P(seq | model). forward + logsumexp at END."""
        if isinstance(sequence, str):
            seq_idx = np.array([BASE_TO_IDX[b] for b in sequence.upper()], dtype=np.int32)
        else:
            seq_idx = np.asarray(sequence, dtype=np.int32)
        alpha = _run_forward(
            seq_idx, self.n_states, self.predecessors, self.successors,
            self.log_emit, self.emitting_mask, self.start_idx,
        )
        t_len = len(seq_idx)
        terms = []
        for i, log_p in self.predecessors[self.end_idx]:
            v = alpha[t_len, i] + log_p
            if v > -math.inf:
                terms.append(v)
        return _logsumexp(terms)

    def decode_to_pattern(self, sequence: str) -> str:
        """run viterbi and pull out the unique match-state bases in order.
        cheap reconstruction of the source pattern from a noisy read."""
        path, _ = self.viterbi(sequence)
        decoded = []
        seen = set()
        for state_type, pos, base in path:
            if state_type == StateType.MATCH and pos not in seen:
                decoded.append(base)
                seen.add(pos)
        return ''.join(decoded)

    def score_runs(self, runs: List[Tuple[str, int]]) -> float:
        """run-based scorer. operates on the run-length-encoded read.

        loops over runs, expects them to match the pattern in order. if a
        run is the wrong base but matches the previous pattern position, we
        count it as carryover and apply the laplace length penalty with a
        smaller expected length. unmatched pattern positions get penalized
        once at the end.
        """
        if not runs:
            return -np.inf

        score = 0.0
        pattern_pos = 0
        p = self.params

        for run_base, run_len in runs:
            if pattern_pos >= self.L:
                if run_base == 'G':
                    score += laplace_log_likelihood(
                        run_len, p.hp_expected_length * 2, p.hp_laplace_scale * 2
                    )
                else:
                    score += math.log(p.p_emission_error)
                continue

            expected_base = self.pattern[pattern_pos]

            if run_base == expected_base:
                score += math.log(1.0 - p.p_emission_error)
                score += laplace_log_likelihood(
                    run_len, p.hp_expected_length, p.hp_laplace_scale
                )
                pattern_pos += 1

            elif pattern_pos > 0 and run_base == self.pattern[pattern_pos - 1]:
                score += math.log(p.p_match_to_insert)
                score += laplace_log_likelihood(
                    run_len, p.hp_expected_length * 0.5, p.hp_laplace_scale
                )

            else:
                score += math.log(p.p_emission_error / 3)

        unmatched = self.L - pattern_pos
        if unmatched > 0:
            score += unmatched * math.log(p.p_match_to_delete)
        return score


class ProfileHMMDecoder:
    """multi-pattern wrapper. holds one ProfileHMM per name, runs forward
    on each to identify which pattern a read belongs to."""

    def __init__(self, patterns: Dict[str, str],
                 params: Optional[ProfileHMMParameters] = None):
        self.patterns = patterns
        self.params = params or ProfileHMMParameters()
        self.hmms: Dict[str, ProfileHMM] = {
            name: ProfileHMM(pattern, self.params)
            for name, pattern in patterns.items()
        }

    def identify(self, sequence: str, strip_primers: bool = False,
                 forward_primer: str = "", reverse_primer: str = ""
                 ) -> Tuple[str, float, Dict[str, float]]:
        """forward-score the read against every pattern, return the argmax."""
        seq = sequence.upper()
        if strip_primers and forward_primer and reverse_primer:
            if seq.startswith(forward_primer):
                seq = seq[len(forward_primer):]
            if seq.endswith(reverse_primer):
                seq = seq[:-len(reverse_primer)]

        scores = {name: hmm.score_sequence(seq) for name, hmm in self.hmms.items()}
        best_name = max(scores, key=scores.get)
        return best_name, scores[best_name], scores

    def decode(self, sequence: str, strip_primers: bool = False,
               forward_primer: str = "", reverse_primer: str = ""
               ) -> Tuple[str, str, float]:
        """identify + viterbi-decode in one shot."""
        seq = sequence.upper()
        if strip_primers and forward_primer and reverse_primer:
            if seq.startswith(forward_primer):
                seq = seq[len(forward_primer):]
            if seq.endswith(reverse_primer):
                seq = seq[:-len(reverse_primer)]
        name, score, _ = self.identify(seq)
        decoded = self.hmms[name].decode_to_pattern(seq)
        return name, decoded, score


class HMMLibrary:
    """list of named ProfileHMMs. used by the stress test (and analyses)
    to score a read against many candidate patterns and pick the best.

    ties go to the earliest-inserted name; we iterate insertion order and
    only replace on strict-greater, so this is deterministic.
    """

    def __init__(self):
        self.hmms: List[ProfileHMM] = []
        self.names: List[str] = []

    def add_hmm(self, name: str, hmm: ProfileHMM):
        self.hmms.append(hmm)
        self.names.append(name)

    def __len__(self):
        return len(self.hmms)

    def viterbi_best(self, seq_idx
                     ) -> Tuple[str, List[Tuple[StateType, int, Optional[str]]], float, Dict[str, float]]:
        """viterbi against all HMMs. returns (best_name, best_path,
        best_score, all_scores). seq_idx can be a string or int32 array.

        ranks all HMMs by score-only viterbi (no path), then reconstructs the
        path for the winner alone. identical results to scoring every HMM with
        full viterbi -- scores and the strict-greater argmax tie-break match --
        at a fraction of the cost.
        """
        best_idx = 0
        best_score = -math.inf
        all_scores: Dict[str, float] = {}
        for i, hmm in enumerate(self.hmms):
            score = hmm.viterbi_score(seq_idx)
            all_scores[self.names[i]] = score
            if score > best_score:
                best_score = score
                best_idx = i
        path, score = self.hmms[best_idx].viterbi(seq_idx)
        return self.names[best_idx], path, score, all_scores

    def score_all(self, seq_idx) -> Dict[str, float]:
        """forward-score one sequence against every HMM."""
        return {name: hmm.score_sequence(seq_idx)
                for name, hmm in zip(self.names, self.hmms)}


def demo():
    """sanity-check viterbi + identify on synthetic CATCAT-style reads."""
    pattern = "CATCAT"
    logger.info(f"target: {pattern}")
    hmm = ProfileHMM(pattern)

    test_sequences = [
        "CCCAAAATTTTCCCAAAATTTTGGGGGG",     # clean expansion
        "CCCCCAAATTTTTTCCCCCAAATTTGGGGGGG",  # variable runs
        "CCAATTCCAATTGGGG",                  # short runs
    ]
    for seq in test_sequences:
        path, score = hmm.viterbi(seq)
        decoded = hmm.decode_to_pattern(seq)
        logger.info(f"  seq {seq} -> {decoded} (score {score:.2f}, match={decoded == pattern})")

    patterns = {
        "pattern_A": "CATCAT",
        "pattern_B": "TACTAC",
        "pattern_C": "ACGTAC",
    }
    decoder = ProfileHMMDecoder(patterns)
    test_seqs = [
        ("CCCAAAATTTTCCCAAAATTTTGGGGGG", "pattern_A"),
        ("TTTTAAAACCCTTTTAAAACCCGGGGG", "pattern_B"),
        ("AAAACCCCGGGGTTTTAAAACCCGGGG", "pattern_C"),
    ]
    for seq, expected in test_seqs:
        name, score, _ = decoder.identify(seq)
        logger.info(
            f"  {seq[:30]}... -> {name} (expected {expected}, score {score:.2f})"
        )


if __name__ == "__main__":
    demo()
