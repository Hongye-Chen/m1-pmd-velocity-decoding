"""Tests for GRU training, prediction, and model selection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_decoding.gru import (
    EarlyStopping,
    GruDecoder,
    GruError,
    RepresentationScore,
    load_checkpoint,
    load_protocol,
    predict_normalized,
    rank_representations,
    save_checkpoint,
    train_run,
)
from neural_decoding.sequences import InputRepresentation

_PROTOCOL = Path(__file__).resolve().parents[1] / "configs" / "gru_input_selection.yaml"


def _small_protocol():
    return replace(load_protocol(_PROTOCOL), max_epochs=6, patience=2, batch_size=16)


def _synthetic_data():
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(96, 5, 3, generator=generator)
    y = torch.stack([x[:, -1, 0] - x[:, -2, 1], x[:, :, 2].mean(dim=1)], dim=1)
    return x[:64], y[:64], x[64:], y[64:].double().numpy()


def test_every_sample_starts_from_a_zero_hidden_state() -> None:
    torch.manual_seed(0)
    model = GruDecoder(n_units=3, hidden_size=8).eval()
    sequences = torch.randn(6, 4, 3)

    with torch.no_grad():
        batched = model(sequences)
        one_at_a_time = torch.cat([model(sequences[i : i + 1]) for i in range(6)])
        permuted = model(sequences.flip(0))
        repeated = model(sequences)

    assert torch.allclose(batched, one_at_a_time, atol=1e-6)
    assert torch.allclose(permuted, batched.flip(0), atol=1e-6)
    assert torch.equal(repeated, batched)


def test_early_stopping_requires_an_absolute_improvement() -> None:
    stopping = EarlyStopping(patience=2, min_delta=1e-5)

    assert stopping.update(1, 1.0)
    assert stopping.update(2, 0.9)
    assert not stopping.update(3, 0.9 - 5e-6)
    assert not stopping.should_stop
    assert not stopping.update(4, 0.95)

    assert stopping.should_stop
    assert stopping.best_epoch == 2 and stopping.best_value == 0.9


def test_ranking_uses_seed_means_then_spread_history_and_width() -> None:
    def score(history: int, width: int, values: tuple[float, ...]) -> RepresentationScore:
        return RepresentationScore(InputRepresentation(history, width), values)

    # Dyadic values keep every mean and spread exact; the narrow representation
    # leads by less than the tolerance, so all four 0.5-level entries tie.
    leader = 0.5 + 2.0**-22
    noisy = score(100, 10, (0.5, 0.25, 0.75))
    steady_long = score(200, 50, (0.5, 0.5, 0.5))
    steady_short_narrow = score(100, 20, (leader, leader, leader))
    steady_short_wide = score(100, 50, (0.5, 0.5, 0.5))
    worse = score(200, 10, (0.25, 0.25, 0.25))
    best_alone = score(200, 20, (0.625, 0.625, 0.625))

    tied = rank_representations(
        [noisy, steady_long, steady_short_narrow, steady_short_wide, worse], tolerance=1e-6
    )
    assert tied[0] is steady_short_wide
    assert tied[1:] == [steady_short_narrow, steady_long, noisy, worse]

    clear = rank_representations([noisy, best_alone, worse], tolerance=1e-6)
    assert clear[0] is best_alone


def test_the_restored_checkpoint_is_the_best_epoch_and_survives_reload(tmp_path) -> None:
    protocol = _small_protocol()
    train_x, train_y, validation_x, validation_y = _synthetic_data()

    result = train_run(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        seed=1,
        protocol=protocol,
        hidden_size=8,
        learning_rate=0.01,
    )

    best_record = result.history[result.best_epoch - 1]
    assert best_record["improved"]
    assert best_record["validation_mse"] == result.best_validation_mse
    assert all(not record["improved"] for record in result.history[result.best_epoch :])
    assert result.restored_validation_mse == result.best_validation_mse

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, result.state, {"n_units": 3, "hidden_size": 8, "seed": 1})
    model, metadata = load_checkpoint(path)
    reloaded = predict_normalized(model, validation_x, batch_size=protocol.batch_size)
    assert metadata["seed"] == 1
    assert np.array_equal(reloaded, result.normalized_prediction)


def test_seeded_training_is_bitwise_reproducible() -> None:
    protocol = _small_protocol()
    train_x, train_y, validation_x, validation_y = _synthetic_data()

    def run(seed: int):
        return train_run(
            train_x=train_x,
            train_y=train_y,
            validation_x=validation_x,
            validation_y=validation_y,
            seed=seed,
            protocol=protocol,
            hidden_size=8,
            learning_rate=0.01,
        )

    first, second, other = run(2), run(2), run(3)

    assert first.history == second.history
    assert all(torch.equal(first.state[name], second.state[name]) for name in first.state)
    assert not all(torch.equal(first.state[name], other.state[name]) for name in first.state)


def test_a_protocol_with_an_unsupported_setting_is_refused(tmp_path) -> None:
    text = _PROTOCOL.read_text(encoding="utf-8").replace("dropout: 0.0", "dropout: 0.2")
    altered = tmp_path / "protocol.yaml"
    altered.write_text(text, encoding="utf-8")

    with pytest.raises(GruError, match="dropout"):
        load_protocol(altered)
