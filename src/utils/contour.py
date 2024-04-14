"""
Contour class to handle complexity with contour calculation
"""

import cv2
import copy
import numpy as np
from sympy.geometry import Line
from src.utils.geometry import assert_is_array_of_lines
from src.utils.image import show_image_and_lines

class Contour:
    
    def __init__(self, points: np.ndarray) -> None:
        self.points = copy.deepcopy(points)
        self.lines = Contour.points_to_lines(self.points)
        self.area = cv2.contourArea(self.points)
        self.perimeter = cv2.arcLength(self.points, True)
    
    def points_to_lines(points: np.ndarray) -> np.ndarray:
        """Static method to convert a contour described by points to lines

        Args:
            points (np.ndarray): array of points of shape (n, 2)

        Returns:
            np.ndarray: array of lines of shape (n, 2, 2)
        """
        return np.stack([points, np.roll(points, shift=-1, axis=0)], axis=1)
    
    @classmethod
    def from_lines(cls, lines: np.ndarray):
        """The lines should describe a perfect closed shape contour

        Args:
            lines (np.ndarray): array of lines of shape (n, 2, 2)
        """
        # assert lines quality
        nlines = len(lines)
        shifted_lines = np.roll(
            np.array(lines).reshape(nlines*2, 2), shift=1, axis=0).reshape(nlines, 2, 2)
        distances = np.linalg.norm(np.diff(shifted_lines, axis=1), axis=2)
        if not np.any(distances): # all distances must be equal to 0
            bad_idxs = np.nonzero(distances > 0)
            raise RuntimeError(
                f"Could not construct the contour from the given lines."
                f"Please check at those indices: {bad_idxs}"
            )
        
        contour_points = lines[:, 0]
        return Contour(points=contour_points)
    
    @classmethod
    def from_unordered_lines_approx(
            cls,
            img,
            lines: np.ndarray,
            min_dist_threshold: float=50,
            start_index: int=0
        ):
        """_summary_

        Args:
            lines (np.array): _description_
        """
        img = img.copy()
        _lines = copy.deepcopy(lines)
        construct_contour = []
        is_contour_found = False
        idx_line_closest_point = start_index
        i = 0
        while not is_contour_found and i < 50:
            cur_line = _lines[idx_line_closest_point]
            cur_geoline = Line(cur_line[0], cur_line[1])
            cur_point = cur_line[1]
            construct_contour.append(cur_line)
            _lines = np.delete(_lines, idx_line_closest_point, axis=0)
            
            if len(_lines) == 0:
                print("No more lines will do the same operation as no point detected")
                        
            #show_image_and_lines(image=img, lines=_lines, colors_lines=[(50 * i, 255 - 50 * i, 255) for i in range(len(_lines))])
            
            # find the closest point to the current one and associated line 
            lines2points = _lines.reshape(len(_lines)*2, 2)
            distances_from_cur_point = np.linalg.norm(lines2points - cur_point, axis=1)
            #print(len(_lines), len(distances_from_cur_point), distances_from_cur_point)
            idx_closest_points = np.nonzero(distances_from_cur_point < min_dist_threshold)[0]
            
            if len(idx_closest_points) > 1: # more than one point close to the current point
                #TODO
                raise RuntimeError("More than one point close to the current point")
            elif len(idx_closest_points) == 0:
                first_line = construct_contour[0]
                first_point = first_line[0]
                distance_end_to_first_points = np.linalg.norm(first_point - cur_point)
                if distance_end_to_first_points < min_dist_threshold:
                    first_geoline = Line(first_line[0], first_line[1])
                    intersect_point = np.array(cur_geoline.intersection(first_geoline)[0].evalf(n=7))
                    construct_contour[-1][1] = intersect_point
                    is_contour_found = True
                    break
                else:
                    raise RuntimeError("No point detected close to the current point")
            
            idx_closest_point = int(idx_closest_points[0])
            idx_line_closest_point = int(np.floor(idx_closest_point / 2))
            
            # arrange the line so that the closest point is in the first place
            idx_point_in_line = 0 if (idx_closest_point / 2).is_integer() else 1
            line_closest_point = _lines[idx_line_closest_point]
            if idx_point_in_line == 1: # flip points positions
                _lines[idx_line_closest_point] = np.flip(line_closest_point, axis=0)
            
            #show_image_and_lines(image=img, lines=[cur_line, line_closest_point], colors_lines=[(0, 255, 255), (255, 0, 255)])
            
            # find intersection point between the two lines
            geoline_closest_point = Line(line_closest_point[0], line_closest_point[1])
            intersect_point = np.array(cur_geoline.intersection(geoline_closest_point)[0].evalf(n=7))
            
            # update values in arrays
            cur_line[1] = intersect_point
            line_closest_point[0] = intersect_point
            construct_contour[i][1] = intersect_point
            
            #show_image_and_lines(image=img, lines=[cur_line, line_closest_point], colors_lines=[(0, 255, 255), (255, 0, 255)])
            
            i += 1
            
        contour_lines = np.array(construct_contour)
        show_image_and_lines(image=img, lines=contour_lines)
        contour_points = contour_lines[:, 0]
        return Contour(points=contour_points)
    
    def as_lines(self):
        """Return contour as a array of successive points
        """
        #TODO
        pass
 