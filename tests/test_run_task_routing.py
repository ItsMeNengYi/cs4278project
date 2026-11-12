"""Regressions for getting around table legs to a collision-free grasp stance."""
import unittest
import numpy as np

from run_task import route


class TableRouteTests(unittest.TestCase):
    def setUp(self):
        self.boxes = [(np.array([-.5,-.75,0.]),np.array([.5,.75,.626]))]

    def assert_clear_transit(self, path):
        # The last leg is the deliberate approach under the table's overhang.
        for a,b in zip(path[:-2],path[1:-1]):
            points = np.linspace(a,b,200)
            inside = np.all((points>[-.9,-1.15]) & (points<[.9,1.15]),axis=1)
            self.assertFalse(inside.any(), (a,b))

    def test_north_stance_approaches_normally_between_the_legs(self):
        goal = np.array([-.02,.85])
        path = route([-2,0],goal,self.boxes)
        np.testing.assert_allclose(path[-2],[-.02,1.35])
        np.testing.assert_allclose(path[-1],goal)
        self.assert_clear_transit(path)

    def test_far_side_stance_goes_around_the_table(self):
        path = route([-2,0],[.65,0],self.boxes)
        np.testing.assert_allclose(path[-2],[1.1,0])
        self.assertGreaterEqual(len(path),5)
        self.assert_clear_transit(path)

    def test_departure_from_an_under_overhang_stance_is_outward(self):
        path = route([-.65,0],[0,.85],self.boxes)
        np.testing.assert_allclose(path[1],[-1.1,0])


if __name__ == '__main__':
    unittest.main()
