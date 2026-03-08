import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill_factory.data.selection import (
    combine_position_masks,
    mask_to_windows,
    merge_overlapping_windows,
    select_positions,
    select_positions_by_entropy,
    select_positions_by_top1_gap,
)


def test_entropy_only_selection() -> None:
    entropy = [0.1, 0.7, 1.0, 0.3]
    mask = select_positions_by_entropy(entropy, threshold=0.7)
    assert mask == [False, True, True, False]



def test_gap_only_selection() -> None:
    gap = [0.5, 0.1, 0.3, 0.05]
    mask = select_positions_by_top1_gap(gap, threshold=0.1)
    assert mask == [False, True, False, True]



def test_combined_selection_union_and_intersection() -> None:
    entropy_mask = [False, True, True, False]
    gap_mask = [True, True, False, False]

    union = combine_position_masks(entropy_mask, gap_mask, mode="union")
    intersection = combine_position_masks(entropy_mask, gap_mask, mode="intersection")

    assert union == [True, True, True, False]
    assert intersection == [False, True, False, False]



def test_window_expansion_and_merge_overlaps() -> None:
    mask = [False, True, False, False, True, False]
    windows = mask_to_windows(mask, radius=1)
    assert windows == [(0, 2), (3, 5)]

    merged = merge_overlapping_windows([(0, 2), (2, 4), (8, 9)])
    assert merged == [(0, 4), (8, 9)]



def test_empty_selection_behavior() -> None:
    mask = select_positions(
        per_token_entropy=[0.1, 0.2, 0.3],
        per_token_top1_gap=[0.4, 0.5, 0.6],
        entropy_threshold=0.9,
        top1_gap_threshold=0.01,
        combine_mode="union",
    )
    assert mask == [False, False, False]
    assert mask_to_windows(mask, radius=2) == []



def test_minimum_selected_positions_enforced() -> None:
    mask = select_positions(
        per_token_entropy=[0.1, 0.2, 0.3, 0.4],
        per_token_top1_gap=None,
        entropy_threshold=1.0,
        top1_gap_threshold=None,
        minimum_selected_positions=2,
    )
    assert sum(1 for v in mask if v) == 2
    assert mask[:2] == [True, True]
