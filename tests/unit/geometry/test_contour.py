"""
Unit tests for Contour geometry class
"""

from src.geometry import Contour

class TestContourIsEqual:
    
    def test_contour_is_equal_same_exact_contours(self):
        cnt1 = Contour([[0, 0], [1, 0], [1, 1], [0, 1]])
        cnt2 = Contour([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert cnt1.is_equal(cnt2) == True
    
    def test_contour_is_equal_very_close_contours(self):
        cnt1 = Contour([[0, 0], [100, 0], [100, 100], [0, 100]])
        cnt2 = Contour([[1, -2], [99, -2], [103, 98], [-2, 101]])
        assert cnt1.is_equal(cnt2, dist_margin_error=5) == True
    
    def test_contour_is_equal_false(self):
        cnt1 = Contour([[0, 0], [100, 0], [100, 100], [0, 100]])
        cnt2 = Contour([[1, -2], [95, -3], [103, 98], [-2, 101]])
        assert cnt1.is_equal(cnt2, dist_margin_error=5) == False