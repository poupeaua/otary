"""
Utils for KIE (Key Information Extraction) in the vision module of Otary.
"""


class Levenshtein:
    @staticmethod
    def distance(s1: str, s2: str) -> int:
        """Calculates the Levenshtein distance between two strings.

        Args:
            s1 (str): First string.
            s2 (str): Second string.

        Returns:
            int: Levenshtein distance.
        """
        # Ensure s2 is the shorter string to minimize space usage
        if len(s1) < len(s2):
            return Levenshtein.distance(s2, s1)

        if not s2:
            return len(s1)

        # Initialize the previous row (distances from an empty s2)
        previous_row = list(range(len(s2) + 1))

        for i, char1 in enumerate(s1):
            # Current row starts with the distance from empty s1 (index + 1)
            current_row = [i + 1]

            for j, char2 in enumerate(s2):
                # Cost of insertion, deletion, or substitution
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (char1 != char2)

                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def similarity(s1: str, s2: str) -> float:
        """Normalized Levenshtein similarity between two strings.

        Args:
            s1 (str): First string.
            s2 (str): Second string.

        Returns:
            float: Similarity score in [0, 1], where 1 means identical strings
                and 0 means maximally different.
        """
        denominator = max(len(s1), len(s2))
        if denominator == 0:
            return 1.0
        return 1.0 - Levenshtein.distance(s1, s2) / denominator
