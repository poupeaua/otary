"""
Unit tests for the heuristic module
"""

from otary.vision.kie.heuristic import HeuristicKeyInformationExtractor
from otary.vision.ocr.ocr_multi_output import OcrMultiOutput

class TestHeuristicKeyInformationExtractor:
    
    def test_extract_exact(self, ocrmultioutput_from_example1: OcrMultiOutput):
        result = HeuristicKeyInformationExtractor.extract(
            ocr_outputs=ocrmultioutput_from_example1, 
            key="you",
            closest_word_dist_thresh=5,
            exact_key_match=True
        )
        assert len(result) == 2
        assert result[0].text == "can"
        assert result[1].text == "have"

    def test_extract_approx_leventshtein(self, ocrmultioutput_from_example1: OcrMultiOutput):
        result = HeuristicKeyInformationExtractor.extract(
            ocr_outputs=ocrmultioutput_from_example1, 
            key="you",
            closest_word_dist_thresh=5,
            exact_key_match=False,
            levenshtein_threshold=0.75
        )
        print(result)
        assert len(result) == 3
        assert result[0].text == "can"
        assert result[1].text == "have"
        assert result[2].text == "computer."
