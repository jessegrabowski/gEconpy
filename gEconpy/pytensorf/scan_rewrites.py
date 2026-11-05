import dataclasses

from pytensor.compile import optdb
from pytensor.compile.ops import ViewOp
from pytensor.conditional import IfElse
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import copy_stack_trace, dfs_rewriter, node_rewriter
from pytensor.scan.op import Scan

# JAX has no while scan, so the rewrite is kept out of fast_run. The Numba mode picks it up through its own "numba"
# tag. The C VM keeps scan states in preallocated buffers, which leaves them traced, so opting it in also needs the
# untraced storage Numba uses:
#     get_mode("CVM").including(SCAN_UNTIL_TAG, "scan_save_mem_no_prealloc").excluding("scan_save_mem_prealloc")
SCAN_UNTIL_TAG = "scan_until"


def _strip_views(variable):
    while variable.owner is not None and isinstance(variable.owner.op, ViewOp):
        variable = variable.owner.inputs[0]
    return variable


def _converged_ifelse(op, inner_inputs, inner_outputs):
    """Return the convergence test of a step ``ifelse(converged, state, f(state))``, or None for any other step."""
    state_inputs = op.inner_untraced_sit_sot(inner_inputs)
    branch_outputs = [_strip_views(output) for output in op.inner_untraced_sit_sot_outs(inner_outputs)]

    ifelse_node = branch_outputs[0].owner
    if ifelse_node is None or not isinstance(ifelse_node.op, IfElse):
        return None
    if any(output.owner is not ifelse_node for output in branch_outputs):
        return None

    converged_branch = ifelse_node.inputs[1 : 1 + ifelse_node.op.n_outs]
    passes_state_through = all(
        converged_branch[ifelse_node.outputs.index(output)] is state
        for output, state in zip(branch_outputs, state_inputs, strict=True)
    )
    condition = ifelse_node.inputs[0]
    if not passes_state_through or condition.type.ndim != 0:
        return None
    return condition


@node_rewriter([Scan])
def scan_until_converged(_fgraph, node):
    r"""
    Stop a fixed-length scan at convergence when its steps become no-ops.

    A scan whose step is ``ifelse(converged(state), state, f(state))`` runs all of its ``n_steps``, paying for each
    no-op step after convergence. When the converged branch passes every state through unchanged, a converged state
    is a fixed point, so stopping at the first step whose outputs satisfy ``converged`` gives the same final state.
    The rewrite keeps the ``ifelse`` and adds ``converged(outputs)`` as the ``until`` condition.

    It only applies when every state is untraced, meaning nothing downstream reads the intermediate steps. A gradient
    that backpropagates through the scan reads them, so it keeps the fixed-length scan it was built against.
    """
    op = node.op
    info = op.info

    only_untraced_states = info.n_untraced_sit_sot > 0 and not (
        info.n_seqs or info.n_mit_mot or info.n_mit_sot or info.n_sit_sot or info.n_nit_sot
    )
    if info.as_while or not only_untraced_states:
        return False

    inner_graph = op.fgraph.unfreeze()
    inner_inputs = list(inner_graph.inputs)
    inner_outputs = list(inner_graph.outputs)

    condition = _converged_ifelse(op, inner_inputs, inner_outputs)
    if condition is None:
        return False

    state_inputs = op.inner_untraced_sit_sot(inner_inputs)
    state_outputs = op.inner_untraced_sit_sot_outs(inner_outputs)
    stop = clone_replace(condition, replace=list(zip(state_inputs, state_outputs, strict=True)))

    new_op = Scan(
        inner_inputs,
        [*inner_outputs, stop],
        dataclasses.replace(info, as_while=True),
        mode=op.mode,
        profile=op.profile,
        truncate_gradient=op.truncate_gradient,
        name=op.name,
        allow_gc=op.allow_gc,
    )

    new_outputs = new_op(*node.inputs, return_list=True)
    copy_stack_trace(node.outputs, new_outputs)
    return new_outputs


# After scan_sit_sot_to_untraced (1.62), which marks the states whose history nothing reads, and before the inplace
# rewrites.
optdb.register(
    "scan_until_converged",
    dfs_rewriter(scan_until_converged, ignore_newtrees=True),
    "numba",
    SCAN_UNTIL_TAG,
    use_db_name_as_tag=False,
    position=1.625,
    overwrite_existing=True,
)
