# Vendored from PyTensor's test suite (tests/unittest_tools.py at commit 0685ee164), which is not part of the installed
# package. Keep it in sync by copying, not by editing.
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pytensor

from pytensor.compile import optdb
from pytensor.compile.mode import Mode, predefined_optimizers
from pytensor.graph.basic import Variable, equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import NodeRewriter, dfs_rewriter
from pytensor.graph.rewriting.db import RewriteDatabaseQuery

if TYPE_CHECKING:
    from pytensor.graph.rewriting.basic import GraphRewriter


def assert_equal_computations(rewritten, expected, *args, original=None, **kwargs):
    """
    Assert that `rewritten` computes the same as `expected`.

    Parameters
    ----------
    rewritten
        The expression after the rewrite pass.
    expected
        The reference expression to compare against.
    *args, **kwargs
        Extra arguments forwarded to equal_computations.
    original : optional
        If given, will be printed in the error message.
    """
    __tracebackhide__ = True  # Hide traceback for py.test

    ok = equal_computations(rewritten, expected, *args, **kwargs)

    if not ok:
        parts = []

        def _dprint(expr):
            return pytensor.dprint(expr, print_type=True, file="str")

        if original is not None:
            parts.append(f"\nOriginal:\n{_dprint(original)}")
        parts.append(f"\nRewritten:\n{_dprint(rewritten)}")
        parts.append(f"\nExpected:\n{_dprint(expected)}")

        raise AssertionError("equal_computations failed\n" + "".join(parts))


class RewriteTester:
    """Apply rewrites to a graph and provide assertions for correctness.

    Rewrite tests should verify **two independent properties**:

    1. **Graph structure** — the rewritten graph matches the expected symbolic
       expression.  This is the primary check.  Prefer this over Op-count
       assertions, which can silently pass when one Op is replaced by a
       specialised variant of the same kind (e.g. ``AdvancedSubtensor`` →
       ``AdvancedSubtensor1``, or ``Blockwise(Dot)`` → ``Dot``).  Writing
       out the expected graph is harder, but it makes the test explicit about
       *exactly* what the rewrite should produce.

    2. **Numerical equivalence** — the original and rewritten graphs produce
       the same values when evaluated.  This is an independent safety net: a
       graph can look structurally correct yet compute something subtly
       different from the original (e.g. operand order swapped in a
       non-commutative Op).  Both graphs are compiled with ``optimizer=None``
       so no further rewrites can mask a problem.  If you are confident the
       structural check is sufficient, the numerical check can be skipped.

    Example — canonicalize removes redundant transposes (``x.T.T → x``)::

        x = pt.dmatrix("x")
        out = x.T.T

        result = RewriteTester([x], [out])
        result.assert_graph(x)
        result.assert_eval(np.eye(3))

    **include / exclude** control which passes from the rewrite database are
    applied.  They are orthogonal to ``custom_rewrite`` — both can be used
    together or independently.  The default ``include="canonicalize"``
    applies only the canonicalization pass, which is usually enough for
    targeted rewrite tests.  ``include`` also accepts a predefined optimizer
    name as a single string — e.g. ``include="o3"`` applies ``fast_run``
    minus ``inplace``.  Pass ``include=None`` to disable database rewrites
    entirely.

    **custom_rewrite** applies a specific rewriter after the database
    passes, mirroring the ``canonicalize``-then-specialize order of a real
    compilation (and matching ``rewrite_graph``): the database first puts the
    graph in canonical form, then the rewriter under test runs on it.
    ``NodeRewriter`` instances are automatically wrapped in a DFS
    ``WalkingGraphRewriter``.  For example, testing ``log(1 + x) → log1p(x)``
    with the ``local_log1p`` node rewriter::

        from pytensor.tensor.math import log, log1p
        from pytensor.tensor.rewriting.math import local_log1p

        x = pt.dscalar("x")
        out = log(1 + x)

        result = RewriteTester(
            [x],
            [out],
            include=None,
            custom_rewrite=local_log1p,
        )
        result.assert_graph(log1p(x))
        result.assert_eval(0.5)

    **linker** controls the backend used when evaluating both graphs
    (default ``"py"``).  Use ``"c"`` or ``"numba"`` to test rewrites that
    target a specific backend.

    Parameters
    ----------
    inputs
        Input variables of the graph.
    outputs
        Output variables of the graph.
    include
        Rewrite database tags to include (default: ``"canonicalize"``),
        or a predefined optimizer name as a string (e.g. ``"o3"``,
        ``"fast_run"``).  Pass ``None`` to disable database rewrites.
    exclude
        Rewrite database tags to exclude.
    custom_rewrite
        A ``GraphRewriter`` or ``NodeRewriter`` to apply after the database
        rewrites.  ``NodeRewriter`` instances are automatically wrapped in a
        DFS ``WalkingGraphRewriter``.
    linker
        Linker used when evaluating both graphs (default: ``"py"``).
    **kwargs
        Extra arguments forwarded to ``RewriteDatabaseQuery``.
    """

    def __init__(
        self,
        inputs: Sequence[Variable],
        outputs: Sequence[Variable],
        *,
        include: str | Sequence[str] | None = "canonicalize",
        exclude: Sequence[str] | None = (),
        custom_rewrite: "GraphRewriter | NodeRewriter | None" = None,
        linker: str = "py",
        **kwargs,
    ):
        inputs = list(inputs)
        outputs = list(outputs)
        self.orig_fg = FunctionGraph(inputs=inputs, outputs=outputs, clone=True)
        self.rewr_fg = FunctionGraph(inputs=inputs, outputs=outputs, clone=True)
        self._orig_inputs = inputs
        self._no_opt = Mode(linker=linker, optimizer=None)
        self._orig_fn = None
        self._rewr_fn = None

        if include:
            if isinstance(include, str) and (query := predefined_optimizers.get(include)) is not None:
                pass
            else:
                if isinstance(include, str):
                    include = [include]
                query = RewriteDatabaseQuery(include=list(include), **kwargs)
            if exclude:
                query = query.excluding(*exclude)
            optdb.query(query).rewrite(self.rewr_fg)

        if custom_rewrite is not None:
            if isinstance(custom_rewrite, NodeRewriter):
                custom_rewrite = dfs_rewriter(custom_rewrite)
            custom_rewrite.rewrite(self.rewr_fg)

    @property
    def orig_fn(self):
        if self._orig_fn is None:
            self._orig_fn = pytensor.function(
                self.orig_fg.inputs,
                self.orig_fg.outputs,
                mode=self._no_opt,
                on_unused_input="ignore",
            )
        return self._orig_fn

    @property
    def rewr_fn(self):
        if self._rewr_fn is None:
            self._rewr_fn = pytensor.function(
                self.rewr_fg.inputs,
                self.rewr_fg.outputs,
                mode=self._no_opt,
                on_unused_input="ignore",
            )
        return self._rewr_fn

    def assert_eval(self, *test_values, rtol=None, atol=None):
        __tracebackhide__ = True
        orig_out = self.orig_fn(*test_values)
        rewr_out = self.rewr_fn(*test_values)
        if not isinstance(orig_out, list | tuple):
            orig_out = [orig_out]
        if not isinstance(rewr_out, list | tuple):
            rewr_out = [rewr_out]
        for i, (a, b) in enumerate(zip(orig_out, rewr_out, strict=True)):
            np.testing.assert_allclose(
                a,
                b,
                rtol=rtol or 1e-7,
                atol=atol or 0,
                err_msg=f"Output {i} mismatch between original and rewritten graph",
            )

    def assert_graph(self, *expected_outputs, strict_dtype=False, **kwargs):
        __tracebackhide__ = True
        assert_equal_computations(
            self.rewr_fg.outputs,
            list(expected_outputs),
            in_xs=list(self.rewr_fg.inputs),
            in_ys=self._orig_inputs,
            original=self.orig_fg.outputs,
            strict_dtype=strict_dtype,
            **kwargs,
        )
