"""
OCR Output used to gather information about the output of any OCR model
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, TYPE_CHECKING

import math
import numpy as np

import otary.geometry as geo
from otary.vision.ocr import OcrSingleOutput

if TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import Self
else:  # pragma: no cover
    try:
        from typing import Self
    except ImportError:  # make Self available in Python <= 3.10
        from typing_extensions import Self


class OcrMultiOutput:
    """Class for keeping track of multiple OCR extracted information from an image

    The idea is to make the code agnostic of the OCR engine used.
    The OCR could be EasyOCR, Tesserocr, KerasOCR, etc... this would not affect the
    program.

    Already handles:
    - EasyOCR (https://github.com/JaidedAI/EasyOCR)
    - DocTR (https://github.com/mindee/doctr)
    """

    def __init__(self, ocrsos: list[OcrSingleOutput]) -> None:
        """Initialize an OcrMultiOutput object

        Args:
            ocrsos (list[OcrSingleOutput]): list of OcrSingleOutput objects
        """
        self.ocrsos = ocrsos

    @classmethod
    def from_pytesseract(cls, data: dict, min_conf: float = 0.0) -> OcrMultiOutput:
        """
        Convert a pytesseract image_to_data dictionary into a list of ``OcrSingleOutput`` objects.

        Args:
            data (dcit): Dictionary returned by ``pytesseract.image_to_data(..., output_type=pytesseract.Output.DICT)``.
            min_conf (int): Minimum confidence threshold on the Tesseract scale [0, 100].
                Words below this value are dropped. Defaults to ``0`` (keep all
                valid words). Tesseract uses ``-1`` as a sentinel for non-word
                layout tokens; those are always dropped regardless.

        Returns:
            OcrMultiOutput: an OcrMultiOutput object, one per recognized word.
        """
        ocrsos = []

        for i in range(len(data["text"])):
            word = data["text"][i]
            conf = int(data["conf"][i])

            # Skip layout-level tokens (conf == -1) and empty strings
            if not word or conf < 0:
                continue

            # Apply optional confidence filter (Tesseract scale 0–100)
            if conf < min_conf:
                continue

            x, y, width, height = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )

            # top-left → top-right → bottom-right → bottom-left
            bbox = geo.Rectangle.from_topleft(
                topleft=np.array([x, y]), width=width, height=height
            )

            ocrsos.append(
                OcrSingleOutput(
                    bbox=bbox,
                    text=word,
                    confidence=conf / 100.0,
                )
            )

        return cls(ocrsos=ocrsos)

    @classmethod
    def from_easyocr(
        cls, easyocr_output: list, is_bbox_cast_int_enabled: bool = False
    ) -> OcrMultiOutput:
        """Convert an easyocr formatted output from the read method into a common
        OcrMultiOutput format.

        Args:
            easyocr_output (list): In easyocr package currently the output is a list of
                tuples that contains in this order the bounding box, the text and
                the confidence about the text read.
            is_bbox_cast_int_enabled (bool, optional): whether to cast all bounding
                boxes coordinates into integers.

        Returns:
            OcrMultiOutput: OcrMultiOutput object
        """
        ocrsos: list[OcrSingleOutput] = []
        for e in easyocr_output:
            try:
                ocrso = OcrSingleOutput(
                    bbox=geo.Rectangle(
                        points=e[0],
                        is_cast_int=is_bbox_cast_int_enabled,
                        regularity_rtol=0.05,
                    ),
                    text=e[1],
                    confidence=e[2],
                )
                ocrsos.append(ocrso)
            except ValueError:
                continue  # skip invalid boxes
        return cls(ocrsos=ocrsos)

    @classmethod
    def from_doctr(
        cls,
        doctr_output: dict,
        assume_straight_pages: bool,
        is_bbox_cast_int_enabled: bool = True,
    ) -> OcrMultiOutput:
        """Transform a single page DocTR output into a OcrMultiOutput object.

        Args:
            doctr_output (dict): the output of the DocTR OCR pipeline.
            assume_straight_pages (bool): whether to assume that the pages are straight
                or not. This has an impact on the geometry bounding boxes coordinates
                output representation.
            is_bbox_cast_int_enabled (bool, optional): whether to cast all bounding
                boxes coordinates into integers.

        Returns:
            OcrMultiOutput: OcrMultiOutput object
        """
        # get the first page of the document
        page = doctr_output["pages"][0]

        # get the dimensions of the image
        height, width = page["dimensions"]
        arr_dim = np.array([width, height])

        # extract OCR single outputs information
        ocrsos: list[OcrSingleOutput] = []
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    try:
                        bbox_arr = np.array(word["geometry"], dtype=float) * arr_dim
                        if not assume_straight_pages:
                            bbox = geo.Rectangle(
                                points=bbox_arr,
                                regularity_rtol=0.1,
                                is_cast_int=is_bbox_cast_int_enabled,
                            )
                        else:
                            bbox = geo.Rectangle.from_topleft_bottomright(
                                topleft=bbox_arr[0],
                                bottomright=bbox_arr[1],
                                is_cast_int=is_bbox_cast_int_enabled,
                            )
                        orcso = OcrSingleOutput(
                            text=word["value"],
                            bbox=bbox,
                            confidence=word["confidence"],
                            objectness=word["objectness_score"],
                        )
                        ocrsos.append(orcso)
                    except ValueError:
                        continue  # skip invalid boxes

        return cls(ocrsos=ocrsos)

    @classmethod
    def merge(cls, ocrmos: Sequence[OcrMultiOutput]) -> OcrMultiOutput:
        """Generate one single OcrMultiOutput object from a list of OcrMultiOutput

        Args:
            ocrmos (list[OcrMultiOutput]): list of OcrMultiOutput

        Returns:
            OcrMultiOutput: one single OcrMultiOutput that contains all the
                OcrSingleOutput objects of all the OcrMultiOutput
        """
        ocrsos = []
        for ocrmo in ocrmos:
            ocrsos.extend(ocrmo.ocrsos)
        return cls(ocrsos=ocrsos)

    def confidence_mean(self, count_none: bool = True) -> float:
        """Compute the confidence mean of all the OCR single outputs that
        compose the OcrMultiOutput as the mean of all the confidence scores.

        Args:
            count_none (bool, optional): whether to count the None confidence
                values or not. Defaults to True.

        Returns:
            float: confidence mean total score
        """
        if len(self) == 0:
            return 0

        score = 0.0
        n_none = 0
        for ocrso in self.ocrsos:
            if ocrso.confidence is not None:
                confidence = ocrso.confidence
            else:
                confidence = 0.0
                n_none += 1
            score += confidence

        n = len(self) if count_none else len(self) - n_none
        return score / n

    def words_in(
        self, box: geo.Rectangle, box_expand_scale: float = 1.0
    ) -> list[OcrSingleOutput]:
        """Return a list of OcrSingleOutput that are in the box.

        Args:
            box (geo.Rectangle): rectangle box where we look for words
            box_expand_scale (float, optional): extend the width and heigth of the
                box by this value. Defaults to 0 meaning no extension just use the box
                as-is.

        Returns:
            list[OcrSingleOutput]: list of OcrSingleOutput found in the box.
        """
        _box = box.copy().expand(scale=box_expand_scale)

        ocrsos_contained = []
        for ocrso in self.ocrsos:
            if ocrso.bbox is not None and _box.contains(ocrso.bbox):
                ocrsos_contained.append(ocrso)
        return ocrsos_contained

    def closest_word(
        self,
        word: OcrSingleOutput,
        dist_thresh: float,
        _to: str = "right",
        enforce_horizontal_alignment: bool = True,
        alignment_angle_error: float = math.pi / 50,
    ) -> Optional[OcrSingleOutput]:
        """Given a OcrSingleOutput object, get the closest word in the image to the
        right or to the left.

        Args:
            word (Word): input OcrsingleOutput object
            _to (str, optional): whether right or left. Defaults to "right".
            enforce_horizontal_alignment (bool, optional): Whether to only consider
                words that are horizontally aligned. This is particularly useful
                because in some case the closest word can be slightly shifted and
                therefore is not a humanly logically correlated word. Defaults to True.
            alignment_angle_error (float, optional): the angle
                error margin to consider two close word as following one
                from each other. Default to pi / 50.
            dist_thresh (float): maximum distance value to accept a close word

        Returns:
            Optional[Word]: None if no OcrSingleOutput can be found else a
                OcrSingleOutput.
        """
        valid_direction = ["right", "left"]
        if _to not in valid_direction:
            raise ValueError(
                f"The direction {_to} parameter is expected to be in "
                f"{valid_direction}"
            )

        # expose interesting points based on the current definition of coords
        xbottom = (
            word.bbox.get_vertice_from_topleft(0, "bottomright")
            if _to == "right"
            else word.bbox.get_vertice_from_topleft(0, "bottomleft")
        )

        if _to == "right":
            ocrsos = [
                ocrso
                for ocrso in self.ocrsos
                if ocrso.bbox.xmin >= word.bbox.centroid[0]
            ]
        else:
            ocrsos = [
                ocrso
                for ocrso in self.ocrsos
                if ocrso.bbox.xmax <= word.bbox.centroid[0]
            ]

        # gather the other points
        xbottom2 = np.zeros(shape=(len(ocrsos), 2))
        for i, w in enumerate(ocrsos):
            xbottom2[i] = (
                w.bbox.get_vertice_from_topleft(0, "bottomleft")
                if _to == "right"
                else w.bbox.get_vertice_from_topleft(0, "bottomright")
            )

        diff = xbottom2 - xbottom
        dist_bottom = np.linalg.norm(diff, axis=1)
        idxs_valid: list[int] = list(np.nonzero(dist_bottom < dist_thresh)[0])

        if len(idxs_valid) == 0:
            return None

        if enforce_horizontal_alignment:
            idxs_valid2: list[int] = []
            for idx in idxs_valid:
                cur_word: OcrSingleOutput = ocrsos[idx]
                seg1 = geo.Segment(
                    [
                        word.bbox.get_vertice_from_topleft(0, "bottomleft"),
                        word.bbox.get_vertice_from_topleft(0, "bottomright"),
                    ]
                )
                seg2 = geo.Segment(
                    [
                        cur_word.bbox.get_vertice_from_topleft(0, "bottomleft"),
                        cur_word.bbox.get_vertice_from_topleft(0, "bottomright"),
                    ]
                )
                abs_slope_diff = np.abs(seg1.slope_angle() - seg2.slope_angle())
                if abs_slope_diff < alignment_angle_error:
                    idxs_valid2.append(idx)
            idxs_valid = idxs_valid2

            if len(idxs_valid) == 0:
                return None

        closest_word_idx = np.argmin(dist_bottom[idxs_valid])
        return ocrsos[idxs_valid[closest_word_idx]]

    def _separate_groupwords_by_symbol(
        self,
        groupwords: list[OcrSingleOutput],
        symbol: str = ":",
        min_area_score: float = 0.8,
    ) -> list[OcrSingleOutput]:
        """
        Splits group words containing a symbol into two separate `OcrSingleOutput`
        instances, one for the part before the symbol (including the symbol) and one for
        the part after the symbol.
        The bounding boxes are adjusted to match the split text regions.

        Args:
            groupwords (list[OcrSingleOutput]): List of OCR output objects to process.
            symbol (str, optional): The symbol to split on. Defaults to ":".

        Returns:
            list[OcrSingleOutput]: A new list of `OcrSingleOutput` objects with words
            containing colons split into two, and all other group words included
            unchanged.
        """
        words_with_symbol = [
            ocrso
            for ocrso in self.ocrsos
            if ocrso.text is not None and symbol in ocrso.text
        ]

        new_groupwords: list[OcrSingleOutput] = []
        for gw in groupwords:
            if gw.text is None:
                new_groupwords.append(gw)
                continue

            # Find all symbol-words that overlap sufficiently with this groupword
            remaining_words_with_symbol = []
            matching_words: list[OcrSingleOutput] = []
            for w in words_with_symbol:
                if w.bbox.inter_area(other=gw.bbox) / w.bbox.area >= min_area_score:
                    matching_words.append(w)
                else:
                    remaining_words_with_symbol.append(w)
            words_with_symbol = remaining_words_with_symbol

            if not matching_words:
                new_groupwords.append(gw)
                continue

            # Sort matches left-to-right so we can split the groupword in order
            matching_words = sorted(matching_words, key=lambda w: w.bbox.xmax)

            # Split the groupword text on each symbol occurrence, pairing each
            # segment with the x-boundary of the corresponding symbol-word
            parts = gw.text.split(symbol)
            xseps = [w.bbox.xmax for w in matching_words]

            for i, (part, xsep) in enumerate(zip(parts[:-1], xseps)):
                x_start = xseps[i - 1] if i > 0 else gw.bbox.xmin
                new_groupwords.append(
                    OcrSingleOutput(
                        text=(part + symbol).strip(),
                        bbox=geo.Rectangle.from_topleft_bottomright(
                            topleft=np.array([x_start, gw.bbox.ymin]),
                            bottomright=np.array([xsep, gw.bbox.ymax]),
                        ),
                        confidence=gw.confidence,
                    )
                )

            # Trailing segment after the last symbol (if any text remains)
            last_xsep = xseps[-1]
            if last_xsep < gw.bbox.xmax:
                trailing_text = parts[-1].strip()
                if trailing_text:
                    new_groupwords.append(
                        OcrSingleOutput(
                            text=trailing_text,
                            bbox=geo.Rectangle.from_topleft_bottomright(
                                topleft=np.array([last_xsep, gw.bbox.ymin]),
                                bottomright=gw.bbox.get_vertice_from_topleft(
                                    0, "bottomright"
                                ),
                            ),
                            confidence=gw.confidence,
                        )
                    )

        return new_groupwords

    def group_words(
        self,
        dist_thresh: float,
        max_n_words: int = 1_000_000,
        min_n_words: int = 2,
        symbol_splitter: Optional[str] = None,
        restrict_word_definition: bool = False,
        word_definition_regex: str = "[a-zA-Z0-9]+",
    ) -> tuple[OcrMultiOutput, OcrMultiOutput]:
        """Groups words into sentences based on their spatial proximity,
        simulating how a human would read lines from left to right.

        This method iterates through OCR word outputs, grouping together words that are
        close enough horizontally (within `dist_thresh`) to be considered part of the
        same group of words. The grouping respects constraints on the minimum and
        maximum number of words per group.

        We can optionally restrict group of words formation
        based on a regular expression (regex) defining valid words.

        Args:
            dist_thresh (float): Maximum allowed distance between consecutive words to
                be grouped together.
            max_n_words (int, optional): Maximum number of words allowed in a group.
                Defaults to 1000000.
            min_n_words (int, optional): Minimum number of words required to form a
                group. Defaults to 2.
            symbol_splitter (str, optional): If provided, each group of words will be
                split by this symbol. Each part will be treated as a separate group.
                Defaults to None which implies no split.
            restrict_word_definition (bool, optional): If True, only groups matching
                the `word_definition_regex` pattern are considered valid.
                Defaults to False.
            word_definition_regex (str, optional): Regular expression pattern that
                defines a valid word. Used when `restrict_word_definition` is True.
                Defaults to "[a-zA-Z0-9]+".

        Returns:
            tuple[OcrMultiOutput, OcrMultiOutput]:
                - The first element is an `OcrMultiOutput` containing grouped sentences
                    (as `OcrSingleOutput` objects).
                - The second element is an `OcrMultiOutput` containing words that were
                    not used in any group.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        assert max_n_words > min_n_words > 1

        unused_words: list[OcrSingleOutput] = []
        groupwords: list[OcrSingleOutput] = []
        for word in self.ocrsos:
            left_word = self.closest_word(
                word=word, _to="left", dist_thresh=dist_thresh
            )

            if left_word is not None:
                # no need to iterate on this word since we have an existing left word
                continue

            right_word = self.closest_word(
                word=word, _to="right", dist_thresh=dist_thresh
            )
            if right_word is None:
                # no need to iterate as we have no right word
                unused_words.append(word)
                continue

            words: list[OcrSingleOutput] = [word, right_word]
            while right_word is not None and len(words) < max_n_words + 1:
                right_word = self.closest_word(
                    word=right_word, _to="right", dist_thresh=dist_thresh
                )
                if right_word is not None:
                    words.append(right_word)

            # compute utils attributes
            groupwords_txt: str = " ".join(
                [w.text for w in words if w.text is not None]
            )
            n_words = len(groupwords_txt.split(" "))

            # ensure words quality within the group of words
            regex = (f"{word_definition_regex} " * min_n_words)[:-1]
            cond_regex = re.search(pattern=regex, string=groupwords_txt) is None
            if (max_n_words < n_words or n_words < min_n_words) or (
                restrict_word_definition and cond_regex
            ):
                unused_words.extend(words)
                continue

            # compute new bbox
            new_bbox = geo.Rectangle.from_topleft_bottomright(
                topleft=word.bbox[0],
                bottomright=words[-1].bbox.get_vertice_from_topleft(
                    topleft_index=0, vertice="bottomright"
                ),
            )

            # new ocrso
            new_ocrso = OcrSingleOutput(
                text=groupwords_txt,
                bbox=new_bbox,
                confidence=np.min(np.array([o.confidence for o in words])),
            )
            groupwords.append(new_ocrso)

        if symbol_splitter is not None:
            new_groupwords = self._separate_groupwords_by_symbol(
                groupwords=groupwords, symbol=symbol_splitter
            )
            groupwords = new_groupwords

        return OcrMultiOutput(ocrsos=groupwords), OcrMultiOutput(ocrsos=unused_words)

    def drop_duplicates(self, dist_thresh: float, criteria: str = "max_area") -> Self:
        """Drop duplicates bbox in the OcrMultiOutput object. The criteria is used
        to decide which bbox to keep when multiple bbox are close to each other.

        Currently the drop duplicates is done only on the bounding box without
        considering the text or the confidence.
        The idea is to keep the best bounding boxes.

        Args:
            dist_thresh (float): minimum distance threshold between two centers
                of bounding boxes to be considered as duplicates.
            criteria (str, optional): criteria to keep the best bounding box.
                Defaults to "max_area" which means that the bounding box with the
                maximum area will be kept.

        Returns:
            Self: returns the OcrMultiOutput object itself without duplicates
        """
        valid_criterion = ["max_area"]
        if criteria not in valid_criterion:
            raise ValueError(f"Criteria is expected to be in {valid_criterion}")

        ocrsos_duplicates: list[list[OcrSingleOutput]] = []
        remaining = self.ocrsos.copy()
        self.ocrsos = []

        while remaining:
            ocrso = remaining.pop(0)
            cur_duplicates = [ocrso]
            keep = []

            for ocrso2 in remaining:
                dist = math.dist(ocrso.bbox.centroid, ocrso2.bbox.centroid)
                cond_bbox2_in_bbox = (
                    ocrso.bbox.iou(other=ocrso2.bbox) > 0.1
                    and ocrso2.bbox.area < ocrso.bbox.area
                )
                if dist < dist_thresh or cond_bbox2_in_bbox:
                    cur_duplicates.append(ocrso2)
                else:
                    keep.append(ocrso2)

            remaining = keep
            ocrsos_duplicates.append(cur_duplicates)

        # Second pass: drop duplicates using IoU, keeping the one with the biggest area
        final_ocrsos = []
        while ocrsos_duplicates:
            group = ocrsos_duplicates.pop(0)
            merged = False
            for i, other_group in enumerate(ocrsos_duplicates):
                for ocrso1 in group:
                    for ocrso2 in other_group:
                        if ocrso1.bbox.iou(ocrso2.bbox) > 0.1:
                            # Merge groups if any bbox overlaps
                            ocrsos_duplicates[i] = other_group + group
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            if not merged:
                final_ocrsos.append(group)
        ocrsos_duplicates = final_ocrsos

        ocrsos = []
        for dup in ocrsos_duplicates:
            if criteria == "max_area":
                max_area = 0.0
                best_idx = 0
                for i, ocrso in enumerate(dup):
                    if ocrso.bbox.area > max_area:
                        max_area = ocrso.bbox.area
                        best_idx = i

                ocrsos.append(dup[best_idx])

        self.ocrsos = ocrsos

        return self

    def copy(self) -> OcrMultiOutput:
        """Copy the OcrMultiOutput object

        Returns:
            OcrMultiOutput: new MultiOutput object
        """
        return OcrMultiOutput(ocrsos=[ocrso.copy() for ocrso in self.ocrsos])

    def __len__(self) -> int:
        """Number of elements in the ocrsos attribute

        Returns:
            int: number of ocrsos elements
        """
        return len(self.ocrsos)

    def __str__(self) -> str:
        return "OcrMultiOutput(" + str([str(ocrso for ocrso in self.ocrsos)]) + ")"

    def __repr__(self) -> str:
        return "OcrMultiOutput(" + str([str(ocrso for ocrso in self.ocrsos)]) + ")"
