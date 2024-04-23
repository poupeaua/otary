"""
Contour class to handle complexity with contour calculation
"""

import copy
import numpy as np
from sympy.geometry import Line
from src.geometry import GeometryEntity, Segment

class Contour(GeometryEntity):
          
    def __init__(self, points: np.ndarray, reduce: bool=True) -> None:
        if reduce: # remove consecutive very close points
            points = Contour.__reduce(points)
        super().__init__(points)
        
    @property
    def lines(self):
        return Contour.points_to_lines(self.points)
    
    @property
    def lengths(self):
        return np.linalg.norm(np.diff(self.lines, axis=1), axis=2)
        
    # ---------------------------------- OTHER CONSTRUCTORS --------------------------------------
    
    @classmethod
    def from_lines(cls, lines: np.ndarray):
        """The lines should describe a perfect closed shape contour

        Args:
            lines (np.ndarray): array of lines of shape (n, 2, 2)
            
        Returns:
            (Contour): a Contour object
        """
        # assert lines quality
        nlines = len(lines)
        shifted_lines = np.roll(
            np.array(lines).reshape(nlines*2, 2), shift=1, axis=0).reshape(nlines, 2, 2)
        distances = np.linalg.norm(np.diff(shifted_lines, axis=1), axis=2)
        if np.any(distances): 
            # a distance is different from 0
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
            lines: np.ndarray,
            min_dist_threshold: float=50,
            max_iteration: int=50,
            start_line_index: int=0,
            img: np.ndarray=None,
            debug: bool=False
        ):
        """Create a Contour object from an unordered list of lines that approximate a closed-shape.
        They approximate in the sense that they do not necessarily share common points.
        We have to extract the intersection point

        Args:
            img (_type_): array of shape (lx, ly)
            lines (np.ndarray): array of lines of shape (n, 2, 2)
            min_dist_threshold (float, optional): For any given point, the minimum distance . Defaults to 50.
            max_iteration (float, optional): Maximum number of iterations before finding a
                contour. It defines also the maximum number of lines in the contour to find.
            start_line_index (int, optional): The starting line to find searching for the
                contour. Defaults to 0.

        Returns:
            (Contour): a Contour object
        """
        def display(lines):
            if debug:
                image.show_image_and_lines(
                    image=img, 
                    lines=lines, 
                    colors_lines=[(50 * i, 255 - 50 * i, 255) for i in range(len(_lines))]
                )

        if debug:
            assert img is not None
            img = img.copy()
            
        _lines = copy.deepcopy(np.array(lines))
        construct_contour = []
        is_contour_found = False
        idx_line_closest_point = start_line_index
        i = 0
        while not is_contour_found and i < max_iteration:
            cur_line = _lines[idx_line_closest_point]
            cur_geoline = Line(cur_line[0], cur_line[1])
            cur_point = cur_line[1]
            construct_contour.append(cur_line)
            _lines = np.delete(_lines, idx_line_closest_point, axis=0)
            
            if len(_lines) == 0:
                print("No more lines will do the same operation as no point detected")
                        
            display(lines=_lines)
            
            # find the closest point to the current one and associated line 
            lines2points = _lines.reshape(len(_lines)*2, 2)
            distances_from_cur_point = np.linalg.norm(lines2points - cur_point, axis=1)
            #print(len(_lines), len(distances_from_cur_point), distances_from_cur_point)
            idx_closest_points = np.nonzero(distances_from_cur_point < min_dist_threshold)[0]
            
            if len(idx_closest_points) > 1: # more than one point close to the current point
                #TODO
                # maybe just take the closest may be more complicated than that
                raise RuntimeError("More than one point close to the current point")
            elif len(idx_closest_points) == 0:
                first_line = construct_contour[0]
                first_point = first_line[0]
                distance_end_to_first_points = np.linalg.norm(first_point - cur_point)
                if distance_end_to_first_points < min_dist_threshold:
                    first_geoline = Line(first_line[0], first_line[1])
                    # TODO sometimes multiples intersection example 7
                    # print(cur_geoline.intersection(first_geoline)[0].evalf(n=7))
                    intersect_point = np.array(cur_geoline.intersection(first_geoline)[0].evalf(n=7))
                    construct_contour[-1][1] = intersect_point
                    construct_contour[0][0] = intersect_point
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
            
            display(lines=[cur_line, line_closest_point])
            
            # find intersection point between the two lines
            geoline_closest_point = Line(line_closest_point[0], line_closest_point[1])
            intersect_point = np.array(cur_geoline.intersection(geoline_closest_point)[0].evalf(n=7))
            
            # update values in arrays
            cur_line[1] = intersect_point
            line_closest_point[0] = intersect_point
            construct_contour[i][1] = intersect_point
            
            display(lines=[cur_line, line_closest_point])
            
            i += 1
            
        contour_lines = np.array(construct_contour)
        display(lines=[cur_line, line_closest_point])
        cnt = Contour.from_lines(contour_lines)
        return cnt
    
    # ---------------------------------- STATIC METHODS -----------------------------------------
    
    def points_to_lines(points: np.ndarray) -> np.ndarray:
        """Static method to convert a contour described by points to lines

        Args:
            points (np.ndarray): array of points of shape (n, 2)

        Returns:
            np.ndarray: array of lines of shape (n, 2, 2)
        """
        return np.stack([points, np.roll(points, shift=-1, axis=0)], axis=1)
    
    def __reduce(points: np.ndarray, max_dist_threshold: float=10):
        # remove consecutive very close points
        idx_to_remove = []
        for i, cur_point in enumerate(points):
            if i == len(points)-1:
                i = -1 # so that next point is the first in that case
            next_point = points[i+1]
            distance = Segment(points=[cur_point, next_point]).length
            if distance < max_dist_threshold:
                idx_to_remove.append(i)
                
        reduced_points = np.delete(np.asarray(points), idx_to_remove, 0)
            
        return reduced_points
    
    # ---------------------------------- CLASSIC METHODS -----------------------------------------
    
    def is_auto_intersected(self) -> bool:
        """Whether any of the segments intersect another segment in the same set

        Returns:
            bool: True if at least two lines intersect, False otherwise
        """
        #TODO
        pass
    
    def add_point(self, point: np.ndarray, index: int):
        self.points = np.concatenate([self.points[:index], [point], self.points[index:]])
        return self
        
    def rearrange_first_point_is_at_index(
            self,
            index: int
        ): 
        self.points = np.concatenate([self.points[index:], self.points[:index]])
        return self
        
    def rearrange_first_point_closest_to_reference_point(
            self,
            reference_point: np.ndarray=np.zeros(shape=(2,))
        ):
        shifted_points = self.points - reference_point
        distances = np.linalg.norm(shifted_points, axis=1)
        idx_min_dist = np.argmin(distances)
        return self.rearrange_first_point_is_at_index(index=idx_min_dist)