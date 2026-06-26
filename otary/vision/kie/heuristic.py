"""
Key Information Extraction (KIE) heuristics for the vision module of Otary.
"""

from otary.vision.kie.utils import Levenshtein
from otary.vision.ocr import OcrMultiOutput, OcrSingleOutput


class HeuristicKeyInformationExtractor:
    @staticmethod
    def extract(
        ocr_outputs: OcrMultiOutput,
        key: str,
        closest_word_dist_thresh: float,
        levenshtein_threshold: float = 0.9,
        exact_key_match: bool = False,
    ) -> list[OcrSingleOutput]:
        """
        Extracts key-value pairs from OCR outputs by matching keys heuristically
        using Levenshtein distance.

        Parameters:
            ocr_outputs (List[OCRSingleOutput]): OCR results, each containing a value
                and access to closest_word(to="left").
            key (str): expected keys to match.
            threshold (float): Minimum normalized distance for a valid match.

        Returns:
            Dict[str, str]: Dictionary mapping expected keys to matched values.
        """

        def is_valid_ocrso_and_key(ocrso: OcrSingleOutput) -> bool:
            if ocrso.text is None:
                return False
            if exact_key_match:
                return ocrso.text.lower() == key.lower()
            else:
                levenshtein_score = Levenshtein.similarity(
                    key.lower(), ocrso.text.lower()
                )
                return levenshtein_score >= levenshtein_threshold

        # find the good candidates for the key
        candidates = [
            ocrso for ocrso in ocr_outputs.ocrsos if is_valid_ocrso_and_key(ocrso)
        ]

        # find the value associated for each candidate
        result: list[OcrSingleOutput] = []
        for candidate in candidates:
            ocrso = ocr_outputs.closest_word(
                word=candidate, _to="right", dist_thresh=closest_word_dist_thresh
            )
            if ocrso is not None:
                result.append(ocrso)

        return result
