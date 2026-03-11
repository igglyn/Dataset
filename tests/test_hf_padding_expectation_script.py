from scripts.check_hf_padding_expectation import summarize_padding_expectation


class _TokenizerStub:
    vocab_size = 10
    pad_token_id = None
    pad_token = None
    eos_token_id = 2
    eos_token = "<eos>"
    unk_token_id = 3
    unk_token = "<unk>"


def test_summarize_padding_expectation_reports_configured_range_check() -> None:
    out = summarize_padding_expectation(_TokenizerStub(), configured_pad_token_id=0)
    assert out["vocab_size"] == 10
    assert out["configured_hf_pad_token_id"] == 0
    assert out["configured_hf_pad_token_id_in_vocab_range"] is True


def test_summarize_padding_expectation_reports_out_of_range_config() -> None:
    out = summarize_padding_expectation(_TokenizerStub(), configured_pad_token_id=99)
    assert out["configured_hf_pad_token_id_in_vocab_range"] is False
