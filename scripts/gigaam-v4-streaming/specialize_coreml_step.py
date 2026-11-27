"""Specialize the v4 640 ms step encoder to one physical 16-frame chunk."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx


def fix_shape(value, shape):
    dimensions = value.type.tensor_type.shape.dim
    if len(dimensions) != len(shape):
        raise ValueError(f"Unexpected rank for {value.name}")
    for dimension, length in zip(dimensions, shape, strict=True):
        if dimension.HasField("dim_value") and dimension.dim_value != length:
            raise ValueError(f"Unexpected dimension for {value.name}")
        dimension.dim_value = length


def specialize(source, output):
    model = onnx.load(source, load_external_data=False)
    graph = model.graph
    if (
        [value.name for value in graph.input[:4]]
        != ["features", "feature_lengths", "position", "cache_valid"]
        or [value.name for value in graph.output[:4]]
        != ["encoded", "encoded_lengths", "next_position", "next_cache_valid"]
        or len(graph.input) != 58
        or len(graph.output) != 58
        or any(onnx.external_data_helper.uses_external_data(x) for x in graph.initializer)
    ):
        raise ValueError("Expected a self-contained, single-stream GigaAM-v4 step")

    # Physical width is fixed. feature_lengths, position and cache_valid stay live.
    fix_shape(graph.input[0], (1, 16, 768))
    fix_shape(graph.output[0], (1, 768, 16))
    for state, updated in zip(graph.input[4:], graph.output[4:], strict=True):
        if updated.name != "next_" + state.name:
            raise ValueError(f"Unexpected streaming cache: {updated.name}")
        shape = [dim.dim_value for dim in state.type.tensor_type.shape.dim]
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"Expected a fixed input cache: {state.name}")
        fix_shape(updated, shape)

    with TemporaryDirectory(dir=output.parent) as directory:
        fixed = Path(directory) / "fixed.onnx"
        onnx.save_model(model, fixed)
        onnx.checker.check_model(fixed)
        output.hardlink_to(fixed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    specialize(args.source, args.output)
