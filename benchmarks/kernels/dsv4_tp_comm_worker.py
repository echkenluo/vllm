# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Development RPC extension for quiescent mode changes and TP8 wire checks.

Only use with VLLM_SERVER_DEV_MODE=1 on an isolated benchmark server. Call
dsv4_comm_mode after all requests finish, flush the prefix cache, then warm
up before timing. All workers acknowledge the mode before requests resume.
"""

import torch

from vllm.distributed import get_tp_group
from vllm.models.common.ops.sequence_parallel import sp_shard
from vllm.models.deepseek_v4.nvidia.tp_comm import GROUP, MODES, _pack, _unpack


class CommWorker:
    def dsv4_comm_mode(self, mode: str = "", reset: str = "0"):
        model = self.model_runner.get_model().model
        comm = model.tp_comm
        assert comm.enabled
        torch.accelerator.synchronize()
        if mode:
            if mode not in MODES:
                raise ValueError(mode)
            comm.mode = mode
        if reset == "1":
            comm.stats.clear()
        return {
            "rank": get_tp_group().rank_in_group,
            "mode": comm.mode,
            "min_tokens": comm.min_tokens,
            "stats": dict(comm.stats),
            "aux_layers": list(model.aux_hidden_state_layers),
            "layers": model.end_layer - model.start_layer,
        }

    def dsv4_comm_wire_gate(self):
        """Check row order, padding, zero scales and independent FP8 arithmetic."""
        comm = self.model_runner.get_model().model.tp_comm
        group = get_tp_group()
        records = []
        with torch.random.fork_rng(devices=[torch.accelerator.current_device_index()]):
            torch.manual_seed(92010 + group.rank_in_group)
            for rows in (1, 7, 8, 511, 512, 513, 4096):
                for hidden in (128, 4096):
                    local = torch.randn(rows, hidden, device="cuda").bfloat16()
                    local[0].zero_()
                    full = group.all_reduce(local.clone())
                    shard = sp_shard(full).contiguous()
                    for mode in MODES[1:]:
                        actual = comm.reduce(local, mode, "gate")
                        error = (actual.float() - shard.float()).norm() / (
                            shard.float().norm() + 1e-12
                        )
                        limit = (
                            0
                            if mode == "preserve_ar"
                            else (0.06 if mode in ("fp8_rs", "fp8_both") else 0.015)
                        )
                        assert torch.isfinite(actual).all() and error <= limit
                        gathered = comm.gather(shard, rows, mode, "attn")
                        ag_error = (gathered.float() - full.float()).norm() / (
                            full.float().norm() + 1e-12
                        )
                        assert ag_error <= (
                            0.04 if mode in ("fp8_ag", "fp8_both") else 0
                        )
                        records.append(
                            [rows, hidden, mode, error.item(), ag_error.item()]
                        )
                    for site in ("dspark_aux", "head"):
                        state = (
                            full
                            if site == "dspark_aux"
                            else (full[:, None].expand(-1, 4, -1).contiguous())
                        )
                        result = comm.gather(
                            sp_shard(state).contiguous(), rows, "fp8_both", site
                        )
                        assert torch.equal(result, state)
                    decoded = _unpack(_pack(local), hidden, rows, torch.float32)
                    blocks = local.float().view(rows, hidden // GROUP, GROUP)
                    # Tensor / constant in FP32 can multiply a rounded
                    # reciprocal. CUDA computes absmax / max_8bit instead;
                    # one ULP in scale changes FP8 midpoint rounding.
                    scale = blocks.double().abs().amax(-1, keepdim=True)
                    scale = (scale.clamp_min(1e-10) / 448.0).float()
                    oracle = (blocks / scale).to(torch.float8_e4m3fn).float() * scale
                    torch.testing.assert_close(
                        decoded, oracle.reshape(rows, hidden), rtol=1e-5, atol=1e-6
                    )
        return {"rank": group.rank_in_group, "passed": True, "records": records}


if __name__ == "__main__":
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.deepseek_v4.nvidia.tp_comm import TPComm

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    with set_current_vllm_config(config):
        init_distributed_environment(world, rank, "env://", local_rank)
        initialize_model_parallel(world, 1)
        try:
            # Direct operator calls do not use model forward admission.
            os.environ["DSV4_TP_COMM"] = "0"
            controller = TPComm(None)
            worker = CommWorker()
            worker.model_runner = SimpleNamespace(
                get_model=lambda: SimpleNamespace(
                    model=SimpleNamespace(tp_comm=controller)
                )
            )
            result = worker.dsv4_comm_wire_gate()
        finally:
            destroy_distributed_environment()
    output = Path(os.environ["COMM_GATE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"WIRE_GATE_PASS rank={rank}", flush=True)
