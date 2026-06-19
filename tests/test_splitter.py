from src.splitter import split_by_divider, split_text, stitch_chunks


class TestSplitText:
    def test_shorter_than_chunk_size_returns_single_chunk(self):
        result = split_text("Short text.", chunk_size=1000)
        assert len(result) == 1
        assert result[0] == "Short text."

    def test_longer_than_chunk_size_returns_multiple_chunks(self):
        text = (
            "Sentence A. Sentence B. Sentence C. Sentence D. "
            "Sentence E. Sentence F. Sentence G. Sentence H."
        )
        result = split_text(text, chunk_size=50)
        assert len(result) > 1
        reconstructed = "".join(result)
        assert reconstructed == text

    def test_respects_paragraph_boundaries(self):
        para1 = "This is the first paragraph with some content."
        para2 = "This is the second paragraph with more content."
        text = f"{para1}\n\n{para2}"
        result = split_text(text, chunk_size=80)
        assert len(result) == 2
        assert result[0].strip() == para1
        assert result[1].strip() == para2

    def test_respects_sentence_boundaries(self):
        s1 = "First fairly long sentence."
        s2 = "Second fairly long sentence."
        s3 = "Third fairly long sentence."
        text = f"{s1} {s2} {s3}"
        chunk_size = len(s1) + len(s2) - 10
        result = split_text(text, chunk_size=chunk_size)
        assert len(result) == 3
        reconstructed = "".join(result)
        assert reconstructed == text

    def test_empty_string(self):
        result = split_text("", chunk_size=200)
        assert result == []

    def test_whitespace_only_text(self):
        result = split_text("   \n\n   ", chunk_size=200)
        assert result == []

    def test_single_character_text(self):
        result = split_text("X", chunk_size=200)
        assert result == ["X"]

    def test_chunk_size_zero(self):
        import pytest

        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            split_text("Some text here", chunk_size=0)


class TestSplitByDivider:
    def test_divider_present_splits_correctly(self):
        divider = "-"
        chunk1 = "Content of the first section"
        chunk2 = "Content of the second section"
        chunk3 = "Content of the third section"
        text = f"{chunk1}\n----------\n{chunk2}\n--------------\n{chunk3}"
        result = split_by_divider(text, divider)
        assert result == [chunk1, chunk2, chunk3]

    def test_divider_absent_returns_single_chunk(self):
        text = "Some text without any divider line"
        result = split_by_divider(text, "-")
        assert result == [text]

    def test_divider_at_start_no_empty_leading_chunk(self):
        divider = "-"
        chunk = "Content after divider"
        text = f"----------\n{chunk}"
        result = split_by_divider(text, divider)
        assert result == [chunk]

    def test_divider_at_end_no_empty_trailing_chunk(self):
        divider = "-"
        chunk = "Content before divider"
        text = f"{chunk}\n----------"
        result = split_by_divider(text, divider)
        assert result == [chunk]

    def test_short_divider_line_not_treated_as_divider(self):
        divider = "-"
        chunk1 = "Section one content"
        chunk2 = "Section two content"
        text = f"{chunk1}\n---\n{chunk2}"
        result = split_by_divider(text, divider)
        assert result == [text]

    def test_uses_custom_divider_character(self):
        divider = "="
        chunk1 = "First block"
        chunk2 = "Second block"
        text = f"{chunk1}\n============\n{chunk2}"
        result = split_by_divider(text, divider)
        assert result == [chunk1, chunk2]

    def test_empty_text_returns_empty_list(self):
        result = split_by_divider("", "-")
        assert result == []

    def test_only_divider_lines_returns_empty_list(self):
        result = split_by_divider("\n----------\n--------------\n", "-")
        assert result == []

    def test_nine_repetitions_not_treated_as_divider(self):
        divider = "-"
        text = "Before\n---------\nAfter"
        result = split_by_divider(text, divider)
        assert result == [text]

    def test_dot_as_divider_re_escaped(self):
        divider = "."
        chunk1 = "First"
        chunk2 = "Second"
        text = f"{chunk1}\n..........\n{chunk2}"
        result = split_by_divider(text, divider)
        assert result == [chunk1, chunk2]

    def test_star_as_divider_re_escaped(self):
        divider = "*"
        chunk1 = "First"
        chunk2 = "Second"
        text = f"{chunk1}\n**********\n{chunk2}"
        result = split_by_divider(text, divider)
        assert result == [chunk1, chunk2]

    def test_plus_as_divider_re_escaped(self):
        divider = "+"
        chunk1 = "First"
        chunk2 = "Second"
        text = f"{chunk1}\n++++++++++\n{chunk2}"
        result = split_by_divider(text, divider)
        assert result == [chunk1, chunk2]


class TestStitchChunks:
    def test_empty_list_returns_empty_string(self):
        assert stitch_chunks([]) == ""

    def test_single_chunk_returns_that_chunk(self):
        chunk = "Only chunk"
        assert stitch_chunks([chunk]) == chunk

    def test_multiple_chunks_joined_with_newline(self):
        chunks = ["First chunk", "Second chunk", "Third chunk"]
        result = stitch_chunks(chunks)
        assert result == "First chunk\nSecond chunk\nThird chunk"
