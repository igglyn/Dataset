from distill_factory.data.chunking import build_long_context_records, chunk_document_bytes
from distill_factory.data.splits import split_records_by_doc_id


def test_chunking_with_overlap_metadata():
    doc = {"doc_id": "doc-1", "text": "abcdefghij"}
    chunks = chunk_document_bytes(doc, chunk_bytes=4, overlap_bytes=1)

    assert [c["raw_bytes"] for c in chunks] == [b"abcd", b"defg", b"ghij", b"j"]
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2, 3]
    assert [c["prev_chunk_index"] for c in chunks] == [None, 0, 1, 2]
    assert [c["next_chunk_index"] for c in chunks] == [1, 2, 3, None]


def test_chunking_byte_offsets_consistent():
    doc = {"doc_id": "doc-2", "text": "abcdefghij"}
    chunks = chunk_document_bytes(doc, chunk_bytes=4, overlap_bytes=1)

    for chunk in chunks:
        start = chunk["byte_start"]
        end = chunk["byte_end"]
        assert chunk["raw_bytes"] == doc["text"].encode("utf-8")[start:end]

    assert [(c["byte_start"], c["byte_end"]) for c in chunks] == [
        (0, 4),
        (3, 7),
        (6, 10),
        (9, 10),
    ]


def test_train_eval_split_deterministic():
    records = [
        {"doc_id": "doc-a", "chunk_index": 0},
        {"doc_id": "doc-a", "chunk_index": 1},
        {"doc_id": "doc-b", "chunk_index": 0},
        {"doc_id": "doc-c", "chunk_index": 0},
    ]

    train_1, eval_1 = split_records_by_doc_id(records, eval_fraction=0.34, seed=123, method="shuffle")
    train_2, eval_2 = split_records_by_doc_id(records, eval_fraction=0.34, seed=123, method="shuffle")

    assert train_1 == train_2
    assert eval_1 == eval_2

    train_doc_ids = {r["doc_id"] for r in train_1}
    eval_doc_ids = {r["doc_id"] for r in eval_1}
    assert train_doc_ids.isdisjoint(eval_doc_ids)


def test_long_context_windows_and_target_location():
    doc = {"doc_id": "doc-ctx", "text": "abcdefghijklmnop"}
    chunks = chunk_document_bytes(doc, chunk_bytes=4, overlap_bytes=0)
    views = build_long_context_records(chunks, context_window=8, stride=1)

    # target chunk index 1 is bytes [4:8] -> "efgh"
    target = views[1]
    # deterministic with current implementation: neighborhood around idx=1 is chunks 0..2 => bytes 0..12,
    # capped to 8 bytes centered on target => bytes 2..10
    assert target["window_byte_start"] == 2
    assert target["window_byte_end"] == 10
    assert target["window_raw_bytes"] == b"cdefghij"
    assert target["target_byte_start_in_window"] == 2
    assert target["target_byte_end_in_window"] == 6


def test_long_context_never_crosses_documents():
    doc_a = {"doc_id": "doc-a", "text": "AAAAAAAABBBBBBBB"}
    doc_b = {"doc_id": "doc-b", "text": "ccccccccdddddddd"}
    records = chunk_document_bytes(doc_a, chunk_bytes=4, overlap_bytes=0) + chunk_document_bytes(
        doc_b, chunk_bytes=4, overlap_bytes=0
    )

    views = build_long_context_records(records, context_window=12, stride=2)
    by_doc = {}
    for view in views:
        by_doc.setdefault(view["doc_id"], []).append(view)

    for view in by_doc["doc-a"]:
        assert b"c" not in view["window_raw_bytes"]
        assert b"d" not in view["window_raw_bytes"]
    for view in by_doc["doc-b"]:
        assert b"A" not in view["window_raw_bytes"]
        assert b"B" not in view["window_raw_bytes"]
