from collections import namedtuple

import numpy as np
from bitsea.commons.geodistances import compute_great_circle_distance
from scipy.optimize import bisect


class Point(namedtuple("Point", ["lat", "lon"])):
    """
    Represents a point on the Earth's surface defined by its latitude and
    longitude.

    This class follows the convention to have latitude first and longitude
    second. The interface allows you to access the coordinates by name,
    avoiding the ambiguity, but when the output is a Numpy array, you have
    to take into account the order of the coordinates.

    Attributes:
        lat (float): The latitude of the point.
        lon (float): The longitude of the point.

    Methods:
        midpoint(other): Calculates the midpoint between this point and
            another point.
        __sub__(other): Computes the difference (as a NumPy array) between
            this point and another point.
    """

    def midpoint(self, other):
        return Point(
            lat=(self.lat + other.lat) / 2.0, lon=(self.lon + other.lon) / 2.0
        )

    def __sub__(self, other):
        return np.array([self.lat - other.lat, self.lon - other.lon])


class Segment:
    """
    Represents a segment defined by a starting point and an ending point.

    Attributes:
        start (Point): The starting point of the segment.
        end (Point): The ending point of the segment.
    """

    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    def length(self):
        return compute_great_circle_distance(
            lat1=self.start.lat,
            lon1=self.start.lon,
            lat2=self.end.lat,
            lon2=self.end.lon,
        )

    @staticmethod
    def build_from_center_and_length(
        center: Point,
        direction: np.ndarray,
        length: float,
        tolerance: float = 1e-8,
    ):
        """
        Builds a segment of a specified length centered around a given point.

        Args:
            center (Point): The center point around which the segment is built.
            direction (np.ndarray): A unit vector indicating the segment's
                orientation and direction. direction[0] is latitude and
                direction[1] is longitude.
            length (float): The total length of the segment (in metres).
            tolerance (float, optional): The numerical tolerance for bisecting
                to find segment boundaries.

        Returns:
            Segment: The constructed segment with its starting and ending
                points.
        """
        # Ensure direction is a vector of length 1
        direction = direction / np.linalg.norm(direction)

        # Compute the length of a movement that follows direction
        new_point = Point(
            lat=center.lat + direction[0], lon=center.lon + direction[1]
        )
        linearized_unit_distance = Segment(center, new_point).length()

        # Define a segment whose middle point is `center` and oriented in the
        # same direction as `direction`, so that the vector that goes between
        # the boundaries of the segments has norm `direction_norm` (respect
        # to its coordinates, not on the sphere)
        def define_segment(direction_norm):
            _p1 = Point(
                lat=center.lat - direction[0] * direction_norm * 0.5,
                lon=center.lon - direction[1] * direction_norm * 0.5,
            )
            _p2 = Point(
                lat=center.lat + direction[0] * direction_norm * 0.5,
                lon=center.lon + direction[1] * direction_norm * 0.5,
            )
            return Segment(_p1, _p2)

        # Given a multiplicative factor `guess` that we will use to increase
        # the size of the direction, we check if this produces a segment that
        # is too long or to short. If this function would return 0 for a guess
        # `t`, it means that `define_segment(t)` is the segment we are looking
        # for
        def check_guess(guess):
            return define_segment(guess).length() - length

        # Our first guess is to pretend that the size of the vector changes
        # linearly
        first_guess = length / linearized_unit_distance

        # If we produce a segment that is too long, we look for a lower bound
        if check_guess(first_guess) > 0:
            upper_bound = first_guess
            lower_bound = first_guess / 2.0
            while check_guess(lower_bound) > 0:
                lower_bound /= 2.0
        # Otherwise we look for an upper bound (trying to avoid an exponential
        # growth)
        else:
            lower_bound = first_guess
            upper_bound = min(lower_bound * 2.0, lower_bound + 10)
            while check_guess(upper_bound) < 0:
                upper_bound = min(lower_bound * 2.0, lower_bound + 10)

        # Use the bisect algorithm to find a zero of the check_guess function
        v_length = bisect(
            check_guess, lower_bound, upper_bound, xtol=tolerance
        )

        return define_segment(v_length)

    def project(self, point: Point):
        """
        Determines the closest point on the segment.

        Args:
            point (Point): The point to project onto the segment.

        Returns:
            tuple: A tuple containing the closest point (as a Point object)
            and a floating point number representing the relative position on
            the segment (0 for start, 1 for end, and a value between 0 and 1
            for intermediate positions).
        """
        # Represent the segment as a vector
        segment_vector = self.end - self.start
        segment_length_squared = np.dot(segment_vector, segment_vector)

        # Handle the case where the segment is nearly a point
        if segment_length_squared == 0:
            return self.start, 0.0

        # Vector from the start of the segment to the point
        point_vector = np.array(
            [point.lat - self.start.lat, point.lon - self.start.lon]
        )

        # Calculate projection scalar (t) onto the segment
        t = np.dot(point_vector, segment_vector) / segment_length_squared

        # Clamp t to the [0, 1] range to stay within the segment
        t = max(0.0, min(1.0, t))

        # Determine the closest point on the segment
        closest_point = Point(
            lat=self.start.lat + t * segment_vector[0],
            lon=self.start.lon + t * segment_vector[1],
        )

        return closest_point, t
