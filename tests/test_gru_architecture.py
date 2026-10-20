"""Synthetic tests for the GRU architecture and learning-rate search."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neural_decoding.dataset import compute_sha256
from neural_decoding.gru import (
    GruDecoder,
    GruError,
    load_checkpoint,
    load_protocol,
    predict_normalized,
    save_checkpoint,
)
from neural_decoding.gru_architecture import (
    ArchitectureScore,
    GruArchitecture,
    baseline_architecture,
    load_architecture_search,
    plan_runs,
    rank_architectures,
    text_matches_sha256,
    verify_artifact_hashes,
)

_ROOT = Path(__file__).resolve().parents[1]
_SEARCH = _ROOT / "configs" / "gru_architecture_selection.yaml"


def test_one_and_two_layer_decoders_follow_the_stacked_gru_rule() -> None:
    torch.manual_seed(0)
    single = GruDecoder(n_units=3, hidden_size=8)
    torch.manual_seed(0)
    stacked = GruDecoder(n_units=3, hidden_size=8, num_layers=2, dropout=0.2)

    assert (single.num_layers, stacked.num_layers) == (1, 2)
    assert sorted(single.state_dict()) == [
        "gru.bias_hh_l0",
        "gru.bias_ih_l0",
        "gru.weight_hh_l0",
        "gru.weight_ih_l0",
        "head.bias",
        "head.weight",
    ]
    assert stacked.upper[0].input_size == 8
    assert torch.equal(single.gru.weight_ih_l0, stacked.gru.weight_ih_l0)

    reference = torch.nn.GRU(input_size=3, hidden_size=8, num_layers=2, batch_first=True)
    with torch.no_grad():
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(reference, f"{name}_l0").copy_(getattr(stacked.gru, f"{name}_l0"))
            getattr(reference, f"{name}_l1").copy_(getattr(stacked.upper[0], f"{name}_l0"))

    sequences = torch.randn(6, 4, 3)
    stacked.eval()
    with torch.no_grad():
        batched = stacked(sequences)
        one_at_a_time = torch.cat([stacked(sequences[i : i + 1]) for i in range(6)])
        _, final = reference(sequences, torch.zeros(2, 6, 8))
        expected = stacked.head(final[-1])
        repeated = stacked(sequences)
    assert batched.shape == (6, 2)
    assert torch.allclose(batched, expected, atol=1e-6)
    assert torch.allclose(batched, one_at_a_time, atol=1e-6)
    assert torch.equal(repeated, batched)

    stacked.train()
    with torch.no_grad():
        assert not torch.equal(stacked(sequences), batched)


def test_dropout_is_refused_where_it_cannot_apply(tmp_path) -> None:
    with pytest.raises(GruError, match="one-layer"):
        GruDecoder(n_units=3, hidden_size=8, num_layers=1, dropout=0.2)
    with pytest.raises(GruError, match="one-layer"):
        GruArchitecture(1, 64, 0.2, 0.001)
    with pytest.raises(GruError, match=r"\[0, 1\)"):
        GruArchitecture(2, 64, 1.0, 0.001)
    assert GruArchitecture(2, 64, 0.2, 0.001).dropout == 0.2

    text = _SEARCH.read_text(encoding="utf-8").replace("1: [0.0]", "1: [0.0, 0.2]")
    altered = tmp_path / "search.yaml"
    altered.write_text(text, encoding="utf-8")
    with pytest.raises(GruError, match="one-layer"):
        load_architecture_search(altered)


def test_checkpoints_without_architecture_metadata_load_as_one_layer(tmp_path) -> None:
    sequences = torch.randn(5, 4, 3)
    torch.manual_seed(3)
    single = GruDecoder(n_units=3, hidden_size=8)
    torch.manual_seed(4)
    stacked = GruDecoder(n_units=3, hidden_size=8, num_layers=2, dropout=0.2)

    one_layer = tmp_path / "one_layer.pt"
    save_checkpoint(one_layer, single.state_dict(), {"n_units": 3, "hidden_size": 8, "seed": 0})
    model, metadata = load_checkpoint(one_layer)
    assert model.num_layers == 1 and "num_layers" not in metadata
    assert torch.equal(
        torch.from_numpy(predict_normalized(model, sequences, batch_size=2)),
        torch.from_numpy(predict_normalized(single, sequences, batch_size=2)),
    )

    two_layer = tmp_path / "two_layer.pt"
    save_checkpoint(
        two_layer,
        stacked.state_dict(),
        {"n_units": 3, "hidden_size": 8, "num_layers": 2, "dropout": 0.2},
    )
    model, _ = load_checkpoint(two_layer)
    assert model.num_layers == 2 and model.dropout.p == 0.2
    assert torch.equal(
        torch.from_numpy(predict_normalized(model, sequences, batch_size=2)),
        torch.from_numpy(predict_normalized(stacked, sequences, batch_size=2)),
    )

    mislabeled = tmp_path / "mislabeled.pt"
    save_checkpoint(mislabeled, stacked.state_dict(), {"n_units": 3, "hidden_size": 8})
    with pytest.raises(GruError, match="architecture"):
        load_checkpoint(mislabeled)


def test_artifact_verification_reports_missing_and_altered_files(tmp_path) -> None:
    (tmp_path / "runs.csv").write_text("seed\n0\n", encoding="utf-8")
    (tmp_path / "checkpoint.pt").write_bytes(b"\x00\x01")
    expected = {name: compute_sha256(tmp_path / name) for name in ("runs.csv", "checkpoint.pt")}
    assert verify_artifact_hashes(tmp_path, expected) == []

    (tmp_path / "checkpoint.pt").write_bytes(b"\x00\x02")
    recorded = {**expected, "summary.json": "0" * 64, "../runs.csv": expected["runs.csv"]}
    assert verify_artifact_hashes(tmp_path, recorded) == [
        "../runs.csv",
        "checkpoint.pt",
        "summary.json",
    ]

    protocol = tmp_path / "protocol.yaml"
    protocol.write_bytes(b"seeds: [0, 1, 2]\r\nlag: 50\r\n")
    frozen = compute_sha256(protocol)
    protocol.write_bytes(b"seeds: [0, 1, 2]\nlag: 50\n")
    assert text_matches_sha256(protocol, frozen)
    protocol.write_bytes(b"seeds: [0, 1, 2]\nlag: 60\n")
    assert not text_matches_sha256(protocol, frozen)


def test_the_plan_reuses_the_baseline_runs_without_retraining_them() -> None:
    search = load_architecture_search(_SEARCH)
    assert text_matches_sha256(_ROOT / search.input_protocol, search.input_protocol_sha256)
    protocol = load_protocol(_ROOT / search.input_protocol)
    baseline = baseline_architecture(protocol)
    plan = plan_runs(search, protocol.seeds, baseline)

    assert baseline == GruArchitecture(1, 64, 0.0, 0.001)
    assert len(search.architectures) == 18
    assert sum(item.num_layers == 1 for item in search.architectures) == 6
    assert not any(item.num_layers == 1 and item.dropout for item in search.architectures)
    assert len(plan.new) == 51
    assert plan.reused == tuple((baseline, seed) for seed in protocol.seeds)
    assert all(architecture != baseline for architecture, _ in plan.new)
    assert set(plan.new) | set(plan.reused) == {
        (architecture, seed) for architecture in search.architectures for seed in protocol.seeds
    }
    assert len(set(plan.new)) == len(plan.new)

    with pytest.raises(GruError, match="not in the search grid"):
        plan_runs(search, protocol.seeds, GruArchitecture(1, 256, 0.0, 0.001))


def test_ranking_uses_seed_means_then_spread_layers_hidden_dropout_and_rate() -> None:
    def score(layers: int, hidden: int, dropout: float, rate: float, values) -> ArchitectureScore:
        return ArchitectureScore(GruArchitecture(layers, hidden, dropout, rate), tuple(values))

    # Dyadic values keep every mean and spread exact; the leader is ahead by less
    # than the tolerance, so every 0.5-level configuration ties.
    leader = 0.5 + 2.0**-22
    slow = score(1, 32, 0.0, 0.0003, [0.5] * 3)
    fast = score(1, 32, 0.0, 0.001, [0.5] * 3)
    wider = score(1, 64, 0.0, 0.0003, [leader] * 3)
    deeper = score(2, 32, 0.0, 0.0003, [0.5] * 3)
    regularized = score(2, 32, 0.2, 0.0003, [0.5] * 3)
    noisy = score(1, 128, 0.0, 0.001, [0.5, 0.25, 0.75])
    worse = score(2, 128, 0.2, 0.001, [0.25] * 3)

    tied = rank_architectures(
        [worse, noisy, regularized, deeper, wider, fast, slow], tolerance=1e-6
    )
    assert tied == [slow, fast, wider, deeper, regularized, noisy, worse]

    best_alone = score(2, 128, 0.0, 0.001, [0.625] * 3)
    assert rank_architectures([slow, best_alone, worse], tolerance=1e-6)[0] is best_alone
