import unittest
import random
import math
from unittest.case import expectedFailure
import model 
import numpy as np
from unittest.mock import Mock

from model import logic

ATTR_COUNT = 7 # Number of attributes associated with a Particle
# For reference:
        # [0] = x-coord
        # [1] = diameter,
        # [2] = y-coord (elevation),
        # [3] = uid,
        # [4] = active (boolean)
        # [5] = age counter
        # [6] = loop age counter
  
class TestGetEventParticlesWithOneSubregion(unittest.TestCase):

    def setUp(self):
        self.test_length = 10
        self.num_particles = 3

        mock_subregion = Mock()
        mock_subregion.leftBoundary.return_value = 0
        mock_subregion.rightBoundary.return_value = self.test_length
        mock_subregion.getName.return_value = 'Mock_Subregion'
        self.mock_sub_list = [mock_subregion]

        self.entrainment_events = 3
        self.level_limit = np.random.randint(0, np.random.randint(2, 10))


    def test_all_active_returns_valid_list(self): 
        model_particles = np.zeros((self.num_particles, ATTR_COUNT))
        model_particles[:,3] = np.arange(self.num_particles) # unique ids
        model_particles[:,4] = np.ones(self.num_particles) # all active
        model_particles[:,0] = np.random.randint(
                                                self.test_length, 
                                                size=self.num_particles ) # random placement

        list = logic.get_event_particles(
                                        self.entrainment_events, 
                                        self.mock_sub_list, 
                                        model_particles, 
                                        self.level_limit )
        self.assertCountEqual(list, model_particles[:,3])

        # Height dependancy should not effect list results here
        hp_list = list = logic.get_event_particles(
                                        self.entrainment_events, 
                                        self.mock_sub_list, 
                                        model_particles, 
                                        self.level_limit,
                                        height_dependant=True )
        self.assertCountEqual(hp_list, model_particles[:,3])
        self.assertCountEqual(hp_list, list)
    
    # TODO: Mock logging object and assert warning is logged
    def test_not_all_active_returns_list_of_2(self):

        mp_one_inactive = np.zeros((self.num_particles, ATTR_COUNT))
        mp_one_inactive[:,3] = np.arange(self.num_particles) 
        mp_one_inactive[0][4] = 1
        mp_one_inactive[1][4] = 1
        mp_one_inactive[:,0] = np.random.randint(self.test_length, size=self.num_particles)

        list = logic.get_event_particles(
                                        self.entrainment_events, 
                                        self.mock_sub_list, 
                                        mp_one_inactive, 
                                        self.level_limit )
        self.assertEqual(len(list), self.num_particles - 1)
        active_list = mp_one_inactive[mp_one_inactive[:,4] != 0]
        self.assertCountEqual(list, active_list[:,3])

    # TODO: Mock logging object and assert warning is logged
    def test_none_active_returns_empty_list(self):
        
        np_none_active = np.zeros((self.num_particles, ATTR_COUNT))
        np_none_active[:,3] = np.arange(self.num_particles) 
        np_none_active[:,0] = np.random.randint(self.test_length, size=self.num_particles)

        empty_list = logic.get_event_particles(
                                        self.entrainment_events, 
                                        self.mock_sub_list, 
                                        np_none_active, 
                                        self.level_limit )
        self.assertEqual(len(empty_list), 0)
    

    def test_all_ghost_particles_returns_ghost_particles(self):
        
        np_all_ghost = np.zeros((self.num_particles, ATTR_COUNT))
        np_all_ghost[:,3] = np.arange(self.num_particles) 
        np_all_ghost[:,0] = -1

        ghost_list = logic.get_event_particles(
                                        self.entrainment_events, 
                                        self.mock_sub_list, 
                                        np_all_ghost, 
                                        self.level_limit )
        self.assertCountEqual(ghost_list, np_all_ghost[:,3])

    def test_some_ghost_particles_returns_ghost_and_regular(self):
        
        np_all_ghost = np.zeros((self.num_particles, ATTR_COUNT))
        np_all_ghost[:,3] = np.arange(self.num_particles) 
        np_all_ghost[:,0] = -1

        ghost_list = logic.get_event_particles(
                                        self.entrainment_events, 
                                        self.mock_sub_list, 
                                        np_all_ghost, 
                                        self.level_limit )
        self.assertCountEqual(ghost_list, np_all_ghost[:,3])     

    # Do we need a tear down method?

class TestGetEventParticlesWithNSubregions(unittest.TestCase):
    
    def setUp(self):
        self.test_length = 20
        self.num_particles = 6
        mock_subregion_0 = Mock()
        mock_subregion_0.leftBoundary.return_value = 0
        mock_subregion_0.rightBoundary.return_value = self.test_length / 2
        mock_subregion_0.getName.return_value = 'Mock_Subregion_0'

        mock_subregion_1 = Mock()
        mock_subregion_1.leftBoundary.return_value = self.test_length / 2
        mock_subregion_1.rightBoundary.return_value = self.test_length
        mock_subregion_1.getName.return_value = 'Mock_Subregion_1'

        self.mock_sub_list_2 = [mock_subregion_0, mock_subregion_1]

        self.entrainment_events = 3
        self.level_limit = np.random.randint(0, np.random.randint(2, 10))


    def test_all_active_returns_3_per_subregion(self):
        
        model_particles = np.zeros((self.num_particles, ATTR_COUNT))
        model_particles[:,3] = np.arange(self.num_particles) # unique ids
        model_particles[:,4] = np.ones(self.num_particles) # all active
        # Randomly place first three particles in Subregion 1 
        model_particles[0:3, 0] = np.random.randint(
                                                9, 
                                                size=3 )
        # Randomly place last three particles in Subregion 2
        model_particles[3:6, 0] = np.random.randint(
                                                11,
                                                self.test_length, 
                                                size=3 )

        list = logic.get_event_particles(
                                        self.entrainment_events, 
                                        self.mock_sub_list_2, 
                                        model_particles, 
                                        self.level_limit )

        self.assertCountEqual(list, model_particles[:,3])
        self.assertEqual(len(list), self.entrainment_events * 2)

        # Height dependancy should not effect list results here
        hp_list = list = logic.get_event_particles(
                                        self.entrainment_events, 
                                        self.mock_sub_list_2, 
                                        model_particles, 
                                        self.level_limit,
                                        height_dependant=True )
        self.assertCountEqual(hp_list, model_particles[:,3])
        self.assertCountEqual(hp_list, list)


    def test_active_in_1_subregion_returns_only_active(self):
        
        mp_half_active = np.zeros((self.num_particles, ATTR_COUNT))
        mp_half_active[:,3] = np.arange(self.num_particles) 
        mp_half_active[0:3, 4] = np.ones(int((self.num_particles/2))) # First half active

        mp_half_active[0:3, 0] = np.random.randint(10,size=3 )
        mp_half_active[3:6, 0] = np.random.randint(10, self.test_length, size=3 ) 

        list = logic.get_event_particles(
                                        self.entrainment_events, 
                                        self.mock_sub_list_2, 
                                        mp_half_active, 
                                        self.level_limit )

        active_particles = mp_half_active[mp_half_active[:,4] != 0] 

        self.assertCountEqual(list, active_particles[:,3])
        self.assertEqual(len(list), 3)
    
    def test_particle_on_boundary_is_not_returned_twice(self):
        
        one_particle_on_boundary = np.zeros((1, ATTR_COUNT))
        one_particle_on_boundary[0][4] = 1
        one_particle_on_boundary[0][0] = 10
        # Use custom entrainment_event count for simplicity
        entrainment_events = 1

        list = logic.get_event_particles(
                                        entrainment_events, 
                                        self.mock_sub_list_2, 
                                        one_particle_on_boundary, 
                                        self.level_limit )
        self.assertEqual(len(list), 1)

# Test Define Subregions
class TestDefineSubregions(unittest.TestCase):

    def setUp(self):
        self.bed_length = 10
        self.iterations = 10

    def test_good_parameters_return_good_subregion_list(self):
        subregion_count_even = 2
        left_boundary = 0
        middle_boundary = self.bed_length / 2
        right_boundary = self.bed_length

        subregion_list = logic.define_subregions(self.bed_length, 
                                                subregion_count_even, 
                                                self.iterations)
        # Check number of subregions
        self.assertEqual(len(subregion_list), 2) 
        # Check boundary definitions
        self.assertEqual(subregion_list[0].leftBoundary(), left_boundary)
        self.assertEqual(subregion_list[0].rightBoundary(), middle_boundary)
        self.assertEqual(subregion_list[1].leftBoundary(), middle_boundary)
        self.assertEqual(subregion_list[1].rightBoundary(), right_boundary)

        subregion_count_odd = 5
        sub_length = self.bed_length / subregion_count_odd

        left_boundary = 0
        middle_boundary_1 = left_boundary + sub_length*1
        middle_boundary_2 = left_boundary + sub_length*2
        middle_boundary_3 = left_boundary + sub_length*3
        middle_boundary_4 = left_boundary + sub_length*4
        right_boundary = self.bed_length

        subregion_list_odd = logic.define_subregions(self.bed_length, 
                                                subregion_count_odd, 
                                                self.iterations)
        # Check number of subregions
        self.assertEqual(len(subregion_list_odd), 5) 
        # Check boundary definitions
        self.assertEqual(subregion_list_odd[0].leftBoundary(), left_boundary)
        self.assertEqual(subregion_list_odd[0].rightBoundary(), middle_boundary_1)

        self.assertEqual(subregion_list_odd[1].leftBoundary(), middle_boundary_1)
        self.assertEqual(subregion_list_odd[1].rightBoundary(), middle_boundary_2)

        self.assertEqual(subregion_list_odd[2].leftBoundary(), middle_boundary_2)
        self.assertEqual(subregion_list_odd[2].rightBoundary(), middle_boundary_3)

        self.assertEqual(subregion_list_odd[3].leftBoundary(), middle_boundary_3)
        self.assertEqual(subregion_list_odd[3].rightBoundary(), middle_boundary_4)

        self.assertEqual(subregion_list_odd[4].leftBoundary(), middle_boundary_4)
        self.assertEqual(subregion_list_odd[4].rightBoundary(), right_boundary)
    
    def test_all_subregion_flux_are_init_0(self):
        subregion_count_even = 2
        empty_list = np.zeros(self.iterations, dtype=np.int64)
        subregion_list_even = logic.define_subregions(self.bed_length, 
                                                subregion_count_even, 
                                                self.iterations)
        
        for subregion in subregion_list_even:
            self.assertEqual(len(subregion.getFluxList()), self.iterations)
            self.assertCountEqual(subregion.getFluxList(), empty_list)

    # TODO: test that incrementFlux() works

# Test Build Streambed
class TestBuildStreambed(unittest.TestCase):
    # compatibiliy is assured by validation at start of model
    def test_compat_diam_returns_good_particles(self):
        stream_length = 100
        diameter = 0.5
        expected_number_particles = stream_length / diameter

        bed_particles = logic.build_streambed(stream_length, diameter)

        self.assertEqual(len(bed_particles), expected_number_particles)

        expected_centres = np.arange(diameter/2, expected_number_particles*diameter, step=diameter)
        expected_ids = np.arange(1, int(expected_number_particles)+1)*-1
        expected_attr = np.zeros(int(expected_number_particles))
        expected_diam = np.ones(int(expected_number_particles))*diameter

        self.assertIsNone(np.testing.assert_array_equal(expected_centres[::-1], bed_particles[:,0]))
        self.assertIsNone(np.testing.assert_array_equal(expected_ids[::-1], bed_particles[:,3]))
        self.assertIsNone(np.testing.assert_array_equal(expected_diam[::-1], bed_particles[:,1]))
        for attribute_idx in [2, 4, 5, 6]:
            self.assertIsNone(np.testing.assert_array_equal(expected_attr, bed_particles[:,attribute_idx]))
        
        final_particle_idx = -len(bed_particles)
        final_particle_extent = bed_particles[final_particle_idx][0] + diameter/2
        self.assertEqual(final_particle_extent, stream_length)

class TestSetModelParticles(unittest.TestCase):

    def setUp(self):
        stream_length = 10
        self.diam = 0.5
        self.pack_fraction = 0.8
        # Directly from https://math.stackexchange.com/questions/2293201/
        # Variables used for geometric placement
        d = np.divide(np.multiply(np.divide(self.diam, 2), 
                                            self.diam), 
                                            self.diam)
        self.h = np.sqrt(np.square(self.diam) - np.square(d))

        # Mock a full bed_particles array
        num_bed_particles = int(stream_length/self.diam)
        bed_particles = np.zeros([num_bed_particles, ATTR_COUNT], dtype=float)
        bed_particles[:,0] = np.arange(self.diam/2, stream_length+(self.diam/2), step=self.diam)
        bed_particles[:,3] = np.arange(1, num_bed_particles+1)*-1
        self.bed_particles = bed_particles
        # Make all vertices created by the touching bed particles available
        # -----> 0.5, 1.0, 1.5, ... , 9.5 (with stream length 10)
        self.available_vertices = np.arange(self.diam, stream_length, step=self.diam)

    def test_model_particles_placed_at_valid_locations(self):
        model_particles, model_supports = logic.set_model_particles(self.bed_particles,   
                                                    self.available_vertices, 
                                                    self.diam, 
                                                    self.pack_fraction, 
                                                    self.h)
        # Particles should only be placed at available vertices
        self.assertTrue(set(model_particles[:,0]).issubset(self.available_vertices))  
        # All placements should be unique
        self.assertTrue(len(model_particles[:,0]) == len(set(model_particles[:,0])))
        # All ids should be unique
        # There should be no stacking
        self.assertEqual(len(set(model_particles[:,2])), 1)

    def test_all_model_particles_have_valid_initial_attributes(self):
        model_particles, model_supports = logic.set_model_particles(self.bed_particles,   
                                                    self.available_vertices, 
                                                    self.diam, 
                                                    self.pack_fraction, 
                                                    self.h)
        # all diam = self.diam
        expected_diam = np.ones(len(model_particles)) * self.diam
        self.assertCountEqual(model_particles[:,1], expected_diam)

        # unique id's
        self.assertTrue(len(model_particles[:,3]) == len(set(model_particles[:,3])))
        
        # all model are active
        expected_activity = np.ones(len(model_particles)) 
        self.assertCountEqual(model_particles[:,4], expected_activity)

        # 0 age counter and loop age
        expected_age_and_loop = np.zeros(len(model_particles))
        self.assertCountEqual(model_particles[:,5], expected_age_and_loop)
        self.assertCountEqual(model_particles[:,6], expected_age_and_loop)

        # Supports should all be negative (resting on the bed)
        self.assertTrue(len(model_particles), len(model_particles[model_particles[:,3] < 0]))


class TestComputeAvailableVerticesLifted(unittest.TestCase):
    def setUp(self):
        # make bed particles
        self.stream_length = 5
        self.diam = 0.5

        # Mock a full bed_particles array
        num_bed_particles = int(self.stream_length/self.diam) # 10 bed particles
        bed_particles = np.zeros([num_bed_particles, ATTR_COUNT], dtype=float)
        bed_particles[:,0] = np.arange(self.diam/2, self.stream_length+(self.diam/2), step=self.diam)
        bed_particles[:,3] = np.arange(num_bed_particles) # unique ids
        self.bed_particles = bed_particles

        self.expected_bed_vertices = np.arange(self.diam, self.stream_length, step=self.diam)

    def test_only_bed_and_empty_lifted_returns_expected_bed_vert(self):
        level_limit = 3 # Arbitrary level limit
        empty_model_particles = np.empty((0, ATTR_COUNT))

        # Bed of length n should return n-1 available vertices
        available_vertices = logic.compute_available_vertices(empty_model_particles, self.bed_particles, self.diam, 
                                            level_limit=level_limit, lifted_particles=[])
        self.assertEqual(len(self.bed_particles)-1, len(available_vertices))
        self.assertCountEqual(available_vertices, self.expected_bed_vertices)
    
    def test_bed_and_all_model_lifted_returns_expected_bed_vertices(self):
        level_limit = 3
        num_model_particles = 3

        model_particles = np.zeros([num_model_particles, ATTR_COUNT], dtype=float)
        # Particles will be at the first 3 available vertices
        model_particles[:,0] = self.expected_bed_vertices[0:3]
        model_particles[:,3] = np.arange(num_model_particles) 
        # Bed of length n should return n-1 available vertices
        available_vertices = logic.compute_available_vertices(model_particles, self.bed_particles, self.diam, 
                                            level_limit=level_limit, lifted_particles=model_particles[:,3].astype(int))
        
        self.assertEqual(len(available_vertices), len(self.bed_particles)-1)
        self.assertCountEqual(available_vertices, self.expected_bed_vertices)

    def test_not_touching_and_one_lifted_model_returns_valid_vertices(self):
        level_limit = 3
        num_model_particles = 3

        model_particles = np.zeros([num_model_particles, ATTR_COUNT], dtype=float)
        # Particles will be at the first 3 available vertices
        model_particles[:,0] = self.expected_bed_vertices[0:3] 
        model_particles[:,3] = np.arange(num_model_particles)

        # Lift first particle, keep later 2 particles -- t/f locations of first particles should be be available
        # and locations of second and third particle should not be avaliable
        available_vertices = logic.compute_available_vertices(model_particles, self.bed_particles, self.diam, 
                                        level_limit=level_limit, lifted_particles=model_particles[0][3].astype(int))

        expected_vertices = np.delete(self.expected_bed_vertices, [1,2])
        self.assertEqual(len(available_vertices), len(expected_vertices))
        self.assertCountEqual(available_vertices, expected_vertices)

class TestComputeAvailableVerticesNotLifted(unittest.TestCase):
   
    def setUp(self):
        # make bed particles
        self.stream_length = 5
        self.diam = 0.5

        # Mock a full bed_particles array
        num_bed_particles = int(self.stream_length/self.diam) # 10 bed particles
        bed_particles = np.zeros([num_bed_particles, ATTR_COUNT], dtype=float)
        bed_particles[:,0] = np.arange(self.diam/2, self.stream_length+(self.diam/2), step=self.diam)
        bed_particles[:,3] = np.arange(num_bed_particles) # unique ids
        self.bed_particles = bed_particles

        self.expected_bed_vertices = np.arange(self.diam, self.stream_length, step=self.diam)
    
    def test_only_bed_returns_expected_bed_vertices(self):
        level_limit = 3 # Arbitrary level limit
        empty_model_particles = np.empty((0, ATTR_COUNT))

        # Bed of length n should return n-1 available vertices
        available_vertices = logic.compute_available_vertices(empty_model_particles, self.bed_particles, self.diam, 
                                            level_limit=level_limit)
        
        self.assertEqual(len(self.bed_particles)-1, len(available_vertices))
        self.assertCountEqual(available_vertices, self.expected_bed_vertices)

    def test_one_model_particle_returns_bed_available_minus_one(self):
        level_limit = 3 # Arbitrary level limit
        one_particle = np.array([[self.diam, 0, 0, 0, 0, 0, 0]]) # at first resting spot
        available_vertices = logic.compute_available_vertices(one_particle, self.bed_particles, self.diam, 
                                            level_limit=level_limit)
        
        # Assert there is no available vertex at one_particle[0][0]
        self.assertNotIn(one_particle[0][0], available_vertices)
        self.assertEqual(len(self.bed_particles)-2, len(available_vertices))
        
        expected_vertices = np.delete(self.expected_bed_vertices, 0) # bed minus first available vertex
        self.assertCountEqual(available_vertices, expected_vertices)

    def test_m_model_particles_return_bed_available_minus_m(self):
        level_limit = 3 # Arbitrary level limit
        m_particles = 4
        model_particles = np.zeros([m_particles, ATTR_COUNT], dtype=float)
        # Place m model particles 2 resting spots away from each other
        placement_idxs = [0, 2, 4, 6]
        model_particles[0][0] = self.expected_bed_vertices[placement_idxs[0]]
        model_particles[1][0] = self.expected_bed_vertices[placement_idxs[1]]
        model_particles[2][0] = self.expected_bed_vertices[placement_idxs[2]]
        model_particles[3][0] = self.expected_bed_vertices[placement_idxs[3]]

        available_vertices = logic.compute_available_vertices(model_particles, self.bed_particles, self.diam, 
                                            level_limit=level_limit)
        
        self.assertEqual(len(self.bed_particles)-m_particles-1, len(available_vertices))
        
        expected_vertices = np.delete(self.expected_bed_vertices, placement_idxs)
        self.assertCountEqual(available_vertices, expected_vertices)

    def test_no_available_vertices_returns_empty_array(self):
        # The bed resting spots are fully saturated with model particles 
        # BUT none are touching so no new vertices are being made

        level_limit = 3 # Arbitrary level limit
        model_particles = np.zeros([len(self.bed_particles)-1, ATTR_COUNT], dtype=float)
        model_particles[:,0] = self.expected_bed_vertices

        available_vertices = logic.compute_available_vertices(model_particles, self.bed_particles, self.diam, 
                                            level_limit=level_limit)
        self.assertEqual(0, len(available_vertices))
    
    def test_two_touching_model_and_empty_bed_return_one_valid_vertex(self):
        # Test without the bed particles
        level_limit = 3 # Arbitrary level limit

        model_particles = np.zeros([2, ATTR_COUNT], dtype=float)
        model_particles[:,0] = np.arange(0.5, 1.5, step=self.diam) # These particles will be touching
        empty_bed = np.empty((0, ATTR_COUNT))

        available_vertices = logic.compute_available_vertices(model_particles, empty_bed, self.diam, 
                                            level_limit=level_limit)
        expected_new_vertex = 0.75
        self.assertEqual(len(available_vertices), 1)
        self.assertEqual(available_vertices, expected_new_vertex)

    def test_3triangle_and_empty_bed_returns_empty_array(self):
        level_limit = 3 # Level limit > 2
        model_particles = np.zeros([3, ATTR_COUNT], dtype=float)
        # 3 triangle: 2 particles touching, 1 particle resting above/between 
        model_particles[0:2][:,0] = np.arange(0.5, 1.5, step=self.diam) 
        model_particles[2:][:,0] = np.add(0.5, np.divide(0.5, 2))

        d = np.divide(np.multiply(np.divide(self.diam, 2), 
                                        self.diam), 
                                        self.diam)
        h = np.sqrt(np.square(self.diam) - np.square(d))
        elevation = round(np.add(h, 0), 2)
        model_particles[2:][:,2] = elevation

        empty_bed = np.empty((0, ATTR_COUNT))

        available_vertices = logic.compute_available_vertices(model_particles, empty_bed, self.diam, 
                                            level_limit=level_limit)

        self.assertEqual(0, len(available_vertices))                    

    def test_above_level_limit_returns_empty_array(self):
        level_limit = 1 # Only one level of stacking allowed
        model_particles = np.zeros([5, ATTR_COUNT], dtype=float)
        # 3 triangle: 2 particles touching, 1 particle resting above/between 
        model_particles[0:3][:,0] = np.arange(0.5, 2.0, step=self.diam) 
        model_particles[3:][:,0] = np.arange(0.75, 1.5, step=self.diam)
        
        d = np.divide(np.multiply(np.divide(self.diam, 2), 
                                        self.diam), 
                                        self.diam)
        h = np.sqrt(np.square(self.diam) - np.square(d))
        elevation = round(np.add(h, 0), 2)
        model_particles[3:][:,2] = elevation

        empty_bed = np.empty((0, ATTR_COUNT))

        available_vertices = logic.compute_available_vertices(model_particles, empty_bed, self.diam, 
                                            level_limit=level_limit)
        self.assertEqual(0, len(available_vertices))

class TestFindSupports(unittest.TestCase):
    def setUp(self):
        self.diam = 0.5

    def test_placement_returns_highest_elevation_supports(self):
        particle_placement = 1.0
        particle_height = 1.0

        particles = np.zeros((5, ATTR_COUNT), dtype=float)
        particles[:,1] = self.diam
        particles[4,0] = particle_placement
        particles[4,2] = particle_height
        particles[0:2,0] = particle_placement - self.diam/2
        particles[2:4,0] = particle_placement + self.diam/2
        particles[[0,2],2] = particle_height - 1.0
        particles[[1,3],2] = particle_height + 1.0
        empty_bed = np.empty((0, ATTR_COUNT))

        left_support, right_support = logic.find_supports(particles[4], particles, empty_bed)

        expected_left = particles[1]
        expected_right = particles[3]
        self.assertIsNone(np.testing.assert_array_equal(expected_left, left_support))
        self.assertIsNone(np.testing.assert_array_equal(expected_right, right_support))
    
    def test_no_supports_available_returns_value_error(self):
        particle_placement = 1.0
        particle_height = 1.0

        particles = np.zeros((3, ATTR_COUNT), dtype=float)
        particles[:,1] = self.diam
        particles[2,0] = particle_placement
        particles[2,2] = particle_height
        particles[0,0] = particle_placement + self.diam
        particles[1,0] = particle_placement - self.diam # Centres lie outside threshold
        particles[[0,1],2] = particle_height - 1.0
        empty_bed = np.empty((0, ATTR_COUNT))

        # Both potential supporting particles outisde support threshold
        with self.assertRaises(ValueError):
            _, _ = logic.find_supports(particles[2], particles, empty_bed)
        
        # Only 1 potential supporting particles outisde support threshold
        particles[0,0] = particle_placement + self.diam/2 # right support within threshold

        with self.assertRaises(ValueError):
            _, _ = logic.find_supports(particles[2], particles, empty_bed)

        particles[0,0] = particle_placement + self.diam # right support outside threshold again
        particles[1,0] = particle_placement + self.diam/2 # left support within threshold

        with self.assertRaises(ValueError):
            _, _ = logic.find_supports(particles[2], particles, empty_bed)

# NOTE: This will probably also be subject to the whole pass model-particle-array-in thing
class TestUpdateParticleStates(unittest.TestCase):

    def setUp(self):
        self.diam = 0.5 

    def test_no_piles_returns_all_active(self): 
        no_pile_particles = np.zeros((7, ATTR_COUNT))
        no_pile_particles[:,1] = self.diam
        no_pile_particles[:,2] = 1.0
        no_pile_particles[:,0] = np.arange(self.diam, 3.5+(self.diam), step=self.diam)

        bed = np.zeros([8, ATTR_COUNT], dtype=float)
        bed[:,1] = self.diam
        bed[:,3] = np.arange(1,9)*-1
        bed[:,0] = np.arange(self.diam/2, 4.0+(self.diam/2), step=self.diam)

        # hard-coded supports array. If there is only one layer of n model particles
        # resting on n+1 bed particles, then this is the form the supports will take
        # assuming model particle 0 is placed between bed particle 0 and 1, model 
        # particle 1 is placed between bed particle 1 and 2, etc.
        no_pile_supports = np.array([[-1, -2],[-2, -3],[-3, -4],[-4, -5],[-5, -6],[-6, -7],[-7, -8]])

        returned_particles = logic.update_particle_states(no_pile_particles, no_pile_supports)

        expected_active = np.ones((7, ATTR_COUNT))
        self.assertIsNone(np.testing.assert_array_equal(expected_active[:,4], returned_particles[:,4]))
    
    def test_2_layers_returns_top_layer_active(self):
        # Test perfect stacks only return the top layer active
        # TODO: defince perfect stack? maybe call tight layers?
        two_layer_particles = np.zeros((13, ATTR_COUNT))
        two_layer_particles[:,1] = self.diam
        # First layer of particles set at elevation 1
        two_layer_particles[0:7,2] = 1.0
        two_layer_particles[0:7,0] = np.arange(self.diam, 3.5+(self.diam), step=self.diam)
        # Second layer of particles set at elevation 2
        two_layer_particles[7:13,2] = 2.0
        two_layer_particles[7:13,0] = np.arange(self.diam+self.diam/2, 3+(self.diam), step=self.diam)
        two_layer_particles[:,3] = np.arange(13)
    
        # Bed particles set at elevation 0
        bed = np.zeros([9, ATTR_COUNT], dtype=float)
        bed[:,3] = np.arange(1,10)*-1
        bed[:,1] = self.diam
        bed[:,0] = np.arange(self.diam/2, 4.5+(self.diam/2), step=self.diam)

        # the 7 particle in the first model particles layer will have bed particles as supports. The 6 
        # particles in the second layer will have the first 7 model particles as supports
        two_layer_supports = np.array([[-1, -2],[-2, -3],[-3, -4],[-4, -5],[-5, -6],[-6, -7], [-7, -8],
                                            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

        returned_particles = logic.update_particle_states(two_layer_particles, two_layer_supports)
        expected_inactive = np.zeros(7)
        expected_active = np.ones(6)
        expected_active_state = np.concatenate((expected_inactive, expected_active))
        self.assertIsNone(np.testing.assert_array_equal(expected_active_state, returned_particles[:,4]))

    def test_some_piles_return_valid_active(self):
        some_piles_particles = np.zeros((9, ATTR_COUNT))
        some_piles_particles[:,1] = self.diam
        some_piles_particles[:,3] = np.arange(9)
        # First 7 partciles set at elevation 1
        some_piles_particles[0:7,2] = 1.0
        some_piles_particles[0:7,0] = np.arange(self.diam, 3.5+(self.diam), step=self.diam)
        # Final 2 particles set at elevation 2
        some_piles_particles[7:9,2] = 2.0
        # Place first particle at first vertex formed by the first layer of model particles
        # Place second particle at 
        some_piles_particles[7:9,0] = np.array([some_piles_particles[0][0]+(self.diam/2), some_piles_particles[4][0]+(self.diam/2)])

        bed = np.zeros([9, ATTR_COUNT], dtype=float)
        bed[:,1] = self.diam
        bed[:,3] = np.arange(1,10)*-1
        bed[:,0] = np.arange(self.diam/2, 4.5+(self.diam/2), step=self.diam)

        some_piles_supports = np.array([[-1, -2],[-2, -3],[-3, -4],[-4, -5],[-5, -6],[-6, -7],
                                            [-7, -8], [0, 1], [4, 5]])
        
        returned_particles = logic.update_particle_states(some_piles_particles, some_piles_supports)
        expected_active_state = np.zeros(9)
        # Particles are supported by 0,1, 4, and 5. Every other particle (0-8) should be available
        expected_active_state[[2, 3, 6, 7, 8]] = 1
        self.assertIsNone(np.testing.assert_array_equal(expected_active_state, returned_particles[:,4]))

    def test_triangle_returns_only_tip_active(self):
        triangle_particles = np.zeros((3, ATTR_COUNT))
        triangle_particles[:,1] = self.diam
        triangle_particles[0:2,2] = 1.0
        triangle_particles[0:2,0] = np.arange(self.diam, 1+(self.diam), step=self.diam)
        triangle_particles[2,2] = 2.0
        triangle_particles[2,0] = self.diam+self.diam/2
        triangle_particles[:,3] = np.arange(3)

        bed = np.zeros([3, ATTR_COUNT], dtype=float)
        bed[:,3] = np.arange(1,4)*-1
        bed[:,1] = self.diam
        bed[:,0] = np.arange(self.diam/2, 1.5+(self.diam/2), step=self.diam)

        triangle_supports = np.array([[-1, -2],[-2, -3],[0, 1]])

        returned_particles = logic.update_particle_states(triangle_particles, triangle_supports)
        expected_inactive = np.zeros(2)
        expected_active = np.ones(1)
        expected_active_state = np.concatenate((expected_inactive, expected_active))
        self.assertIsNone(np.testing.assert_array_equal(expected_active_state, returned_particles[:,4]))

# TODO: assert the error messages are logged
class TestPlaceParticle(unittest.TestCase): 

    def setUp(self):
        self.diam = 0.5
        d = np.divide(np.multiply(np.divide(self.diam, 2), 
                                        self.diam), 
                                        self.diam)
        self.h = np.sqrt(np.square(self.diam) - np.square(d))

    def test_bad_placement_raises_value_error(self):
        # A bad placement is any placement where a particle does
        # not have a particle directly beneath it on the left and/or right. 
        # In other words, a particle is not physically supported

        # Particle (arbitrary placement) with empty model and bed particles arrays
        bad_particle = np.zeros([1, ATTR_COUNT], dtype=float)
        bad_particle[:,1] = self.diam
        empty_bed = np.empty((0, ATTR_COUNT))
        empty_model = np.empty((0, ATTR_COUNT))
        
        with self.assertRaises(ValueError):
            placed_x, placed_y = logic.place_particle(bad_particle[0], empty_model, empty_bed, self.h)

        # Particle placed at invalid (unsupported) location
        unsupported_particle = np.zeros([1, ATTR_COUNT], dtype=float)
        unsupported_particle[:,0] = 0.9
        unsupported_particle[:,1] = self.diam

        unsupported_model = np.zeros((2, ATTR_COUNT))
        unsupported_model[:,0] = np.arange(0.5, 1.5, step=self.diam) 

        with self.assertRaises(ValueError):
            _, _, _, _ = logic.place_particle(unsupported_particle[0], unsupported_model, empty_bed, self.h)

        # Particle placed at location where all model particles' centres have the same location
        # (i.e placed on a tall single stack/tower of particles)
        stacked_unsupported_particle = np.zeros([1, ATTR_COUNT], dtype=float)
        stacked_unsupported_particle[:,0] = 0.5
        stacked_unsupported_particle[:,1] = self.diam

        single_stack_model = np.zeros((2, ATTR_COUNT))
        single_stack_model[:,0] = 0.5
        single_stack_model[:,2] = np.arange(0, 1.0, step=self.diam)

        with self.assertRaises(ValueError):
            _, _, _, _  = logic.place_particle(stacked_unsupported_particle[0], single_stack_model, empty_bed, self.h)

        # Particle and model particle array are identicle
        particle = np.zeros([1, ATTR_COUNT], dtype=float)
        particle[:,1] = self.diam

        empty_bed = np.empty((0, ATTR_COUNT))
        model = np.empty((1, ATTR_COUNT)) 
        model[0] = particle

        with self.assertRaises(ValueError):
            _, _, _, _  = logic.place_particle(particle[0], model, empty_bed, self.h)
        
    def test_good_placement_returns_valid_xy(self):
        base_elevation = 0
        left_id = 4.0
        right_id = 9.0
        expected_elevation = round(np.add(self.h, base_elevation), 2)

        placement = 0.75
        particle = np.zeros([1, ATTR_COUNT], dtype=float)
        particle[:,0] = placement

        # Create model particles with particle at index 0 being placed 
        # in between particles at index 1 and 2. 
        # particles at index 1 and 2 will be at the base_elevation
        model_particles = np.zeros([3, ATTR_COUNT], dtype=float)
        model_particles[:,1] = self.diam
        model_particles[0][0] = placement
        model_particles[1:][:,0] = np.array([placement - (self.diam/2), placement + (self.diam/2)]) 
        model_particles[1:][:,3] = np.array([left_id, right_id]) # set ids for expected supports
        model_particles[1:][:,2] = base_elevation
    
        particle = model_particles[0]
        # don't concern ourselves with the bed here
        empty_bed = np.empty((0, ATTR_COUNT))

        placed_x, placed_y, left_supp, right_supp = logic.place_particle(particle, model_particles, empty_bed, self.h)

        expected_left = left_id
        expected_right = right_id
        self.assertEqual(expected_left, left_supp)
        self.assertEqual(expected_right, right_supp)
        self.assertEqual(expected_elevation, placed_y)
        self.assertEqual(placement, placed_x)

class TestElevationList(unittest.TestCase):
    
    def test_all_same_elev_returns_one_elev(self):
        same_elev = np.array((12.3, 12.3, 12.3, 12.3, 12.3))
        elev_list = logic.elevation_list(same_elev)
        asc_elev_list = logic.elevation_list(same_elev, desc=False)
        
        expected_list = np.array(12.3)
        self.assertIsNone(np.testing.assert_array_equal(expected_list, elev_list))
        self.assertIsNone(np.testing.assert_array_equal(expected_list, asc_elev_list))

    def test_all_unique_returns_same_list(self):
        unique_elev = np.arange(5, dtype=float)
        elev_list = logic.elevation_list(unique_elev)
        asc_elev_list = logic.elevation_list(unique_elev, desc=False)

        self.assertIsNone(np.testing.assert_array_equal(unique_elev[::-1], elev_list))
        self.assertIsNone(np.testing.assert_array_equal(unique_elev, asc_elev_list))
    
    def test_elevation_list_returns_unique_set(self):
        elev = np.array((12.3, 5.0, 12.3, 5.0, 1.0, 3.0))
        elev_list = logic.elevation_list(elev)
        asc_elev_list = logic.elevation_list(elev, desc=False)

        expected_desc_elev = np.array((12.3, 5.0, 3.0, 1.0))
        expected_asc_elev = expected_desc_elev[::-1]
        self.assertIsNone(np.testing.assert_array_equal(expected_desc_elev, elev_list))
        self.assertIsNone(np.testing.assert_array_equal(expected_asc_elev, asc_elev_list))

class TestComputeHops(unittest.TestCase): 

    def test_empty_event_particles_raises_index_error(self):
        mu = 0 
        sigma = 1
        model_particles = np.zeros((4,ATTR_COUNT), dtype=float)
        empty_event_particles = np.empty((0, ATTR_COUNT))

        with self.assertRaises(IndexError):
            event_particles = logic.compute_hops(empty_event_particles, model_particles, mu, sigma)
    
    def test_normal_updates_event_locations(self):
        # Testable values: 
        #   >> np.random.seed(0)
        #   >> np.random.normal(0, 1, 3)
        #   array([1.76405235, 0.40015721, 0.97873798])
        mu = 0
        sigma = 1
        model_particles = np.zeros((4, ATTR_COUNT), dtype=float)
        event_particles_idx = [0, 2, 3]

        np.random.seed(0)
        event_particles = logic.compute_hops(event_particles_idx, model_particles, mu, sigma, normal=True)
        self.assertCountEqual(np.round([1.76405235, 0.40015721, 0.97873798], 1), event_particles[:,0])

    def test_lognormal_updates_event_locations(self):
        # Testable values: 
        #   >>> np.random.seed(0)
        #   >>> np.random.lognormal(0, 0.25, 3)
        #   array([1.55428104, 1.10521435, 1.27721828])
        mu = 0
        sigma = 0.25
        model_particles = np.zeros((4, ATTR_COUNT), dtype=float)
        event_particles_idx = [0, 2, 3]

        np.random.seed(0)
        event_particles = logic.compute_hops(event_particles_idx, model_particles, mu, sigma, normal=False)
        self.assertCountEqual(np.round([1.55428104, 1.10521435, 1.27721828], 1), event_particles[:,0])


class TestMoveModelParticles(unittest.TestCase):

    def setUp(self):
        self.diam = 0.5
        d = np.divide(np.multiply(np.divide(self.diam, 2), 
                                        self.diam), 
                                        self.diam)
        self.h = np.sqrt(np.square(self.diam) - np.square(d))

    def test_empty_event_particles_returns_no_changes(self):
        empty_event_particles = np.empty((0, ATTR_COUNT))
        model_particles = np.zeros((1, ATTR_COUNT), dtype=float)
        model_supports = np.array([[1.0 , 2.0]], dtype=float)
        empty_bed = np.empty((0, ATTR_COUNT))
        available_vertices = np.empty((0))

        moved_model, moved_supports = logic.move_model_particles(empty_event_particles, 
                                                                    model_particles,
                                                                    model_supports, 
                                                                    empty_bed, 
                                                                    available_vertices, 
                                                                    self.h)
        expected_supports = np.array([[1.0 , 2.0]], dtype=float)                                                       
        self.assertIsNone(np.testing.assert_array_equal(expected_supports, moved_supports))
        self.assertIsNone(np.testing.assert_array_equal(model_particles, moved_model))

    def test_event_particle_on_desired_vertex_returns_no_changes(self):
        placement = 5

        model_particles = np.zeros((1, ATTR_COUNT), dtype=float)
        model_particles[:,0] = placement
        model_particles[:,1] = self.diam

        one_event = model_particles[[0]]
        two_bed = np.zeros((2, ATTR_COUNT))
        two_bed[0][3] = -1
        two_bed[1][3] = -2
        two_bed[0][0] = placement - (self.diam/2)
        two_bed[1][0] = placement + (self.diam/2)
        model_supports = np.array([[two_bed[0][3], two_bed[1][3]]], dtype=float)
        available_vertices = np.array([placement])

        moved_model, moved_supports = logic.move_model_particles(one_event, 
                                                                model_particles,
                                                                model_supports, 
                                                                two_bed, 
                                                                available_vertices, 
                                                                self.h)
        expected_supports = np.array([[two_bed[0][3], two_bed[1][3]]], dtype=float)
        self.assertIsNone(np.testing.assert_array_equal(expected_supports, moved_supports))
        self.assertIsNone(np.testing.assert_array_equal(moved_model, model_particles))

    def test_looped_particle_returns_nan_supports_and_incremented_counter(self):
        
        empty_bed = np.empty((0, ATTR_COUNT))
        available_vertices = np.arange((3)) 

        model_particles = np.zeros((1, ATTR_COUNT), dtype=float)
        placement = np.max(available_vertices) + 1 # assure that the particle is being placed beyond the largest vertex
        model_particles[:,0] = placement
        ghost_event = model_particles[[0]]
        model_supports = np.array([[1.0, 5.0]], dtype=float) # random values
        moved_model, moved_supports = logic.move_model_particles(ghost_event, 
                                                                    model_particles, 
                                                                    model_supports,
                                                                    empty_bed, 
                                                                    available_vertices, 
                                                                    self.h)

        expected_counter = 1
        expected_supports = np.empty((1,2))
        expected_supports[:] = np.nan
        self.assertIsNone(np.testing.assert_array_equal(expected_supports, moved_supports))
        self.assertEqual(expected_counter, moved_model[0][6])


class TestUpdateFlux(unittest.TestCase): # Easy

    def setUp(self):

        self.test_length = 10

        # mock one subregion
        self.mock_subregion = Mock()
        self.mock_subregion.leftBoundary.return_value = 0
        self.mock_subregion.rightBoundary.return_value = self.test_length
        self.one_mock_subregion = [self.mock_subregion]
        # mock multiple (2) subregions    
        self.mock_subregion_1 = Mock()
        self.mock_subregion_1.leftBoundary.return_value = 0
        self.mock_subregion_1.rightBoundary.return_value = self.test_length/2
        self.mock_subregion_2 = Mock()
        self.mock_subregion_2.leftBoundary.return_value = self.test_length/2
        self.mock_subregion_2.rightBoundary.return_value = self.test_length
        self.two_mock_subregion = [self.mock_subregion_1, self.mock_subregion_2]

    def test_different_lengths_raise_value_error(self):

        iteration = 0
        init_pos = np.arange(2)
        final_pos = np.arange(3)

        with self.assertRaises(ValueError):
            _ = logic.update_flux(init_pos, final_pos, iteration, self.mock_subregion)

    def test_0_crossings_call_increment_0_times(self):

        iteration = 0
        # Over a single subregion (no internal crossings possible)
        # -- Only 1 particle moving
        init_pos = np.array([5])
        final_pos = init_pos + 1
        
        subregion = logic.update_flux(init_pos, final_pos, iteration, self.one_mock_subregion)
        self.mock_subregion.incrementFlux.assert_not_called()

        # -- Multiple particles moving
        init_pos = np.array([1, 3, 5, 7, 8])
        final_pos = init_pos + 1 # No final positions >= 10
        subregion = logic.update_flux(init_pos, final_pos, iteration, self.one_mock_subregion)
        self.mock_subregion.incrementFlux.assert_not_called()

        # Over multiple subregions (internal crossings possible)
        # -- Only 1 particle moving
        init_pos = np.array([7])
        final_pos = init_pos + 1
        
        subregion = logic.update_flux(init_pos, final_pos, iteration, self.two_mock_subregion)
        self.mock_subregion.incrementFlux.assert_not_called()

        # -- Multiple particles moving
        init_pos = np.array([1, 3, 6, 7, 8])
        final_pos = init_pos + 1 # No final positions >= 10
        subregion = logic.update_flux(init_pos, final_pos, iteration, self.one_mock_subregion)
        self.mock_subregion.incrementFlux.assert_not_called()

    def test_n_crossings_call_increment_n_times(self):
        
        # Over one subregion
        # -- Over 1 iteration
        iteration = 0
        init_pos = np.array([5, 6, 7, 8])
        final_pos = init_pos + 5 
        subregion = logic.update_flux(init_pos, final_pos, iteration, self.one_mock_subregion)
        self.assertEqual(4, self.mock_subregion.incrementFlux.call_count)
        self.mock_subregion.incrementFlux.assert_called_with(0) # Only iteration 0
    
        # -- Over multiple subregions
        iteration = 0
        init_pos = np.array([1, 2, 3, 6, 7, 8])
        final_pos = init_pos + 5 
        subregion = logic.update_flux(init_pos, final_pos, iteration, self.two_mock_subregion)
        
        self.assertEqual(3, self.mock_subregion_1.incrementFlux.call_count)
        self.assertEqual(3, self.mock_subregion_2.incrementFlux.call_count)
        
        self.mock_subregion_1.incrementFlux.assert_called_with(0)
        self.mock_subregion_2.incrementFlux.assert_called_with(0) 

    def test_particle_crossing_multiple_subregions_calls_increment_on_each(self):
        
        iteration = 0
        init_pos = np.array([1])
        final_pos = np.array([11]) # Will cross 5 and 10
        subregion = logic.update_flux(init_pos, final_pos, iteration, self.two_mock_subregion)
        self.mock_subregion_1.incrementFlux.assert_called_once_with(0)
        self.mock_subregion_2.incrementFlux.assert_called_once_with(0) 

    def test_ghost_particle_calls_increment_on_final_subregion(self):

        iteration = 0
        init_pos = np.array([1, 6])
        final_pos = np.array([-1, -1])
        subregion = logic.update_flux(init_pos, final_pos, iteration, self.one_mock_subregion)
        self.assertEqual(2, self.mock_subregion.incrementFlux.call_count)
        self.mock_subregion.incrementFlux.assert_called_with(0) 

    def tearDown(self):
        self.mock_subregion.reset_mock()
        self.mock_subregion_1.reset_mock()
        self.mock_subregion_2.reset_mock()


class TestFindClosestVertex(unittest.TestCase): # Easy 
    
    def test_empty_available_vertices_returns_value_error(self):
        hop = 12.3
        empty_vertices = np.empty(0, dtype=float)
        with self.assertRaises(ValueError):
            _ = logic.find_closest_vertex(hop, empty_vertices)

    def test_negative_desired_hop_returns_value_error(self):
        neg_hop = -4.0
        avail_vert = np.arange(5, dtype=float)
        with self.assertRaises(ValueError):
            _ = logic.find_closest_vertex(neg_hop, avail_vert)

    def test_particle_exceeding_vertices_returns_negative_1(self):
        hop = 5.2
        avail_vert = np.arange(5, dtype=float)
        closest_vertex = logic.find_closest_vertex(hop, avail_vert)
        self.assertEqual(-1, closest_vertex)
        
    def test_valid_hop_to_vertex_returns_vertex(self):
        hop = 3.0
        avail_vert = np.arange(6, dtype=float)
        closest_vertex = logic.find_closest_vertex(hop, avail_vert)
        self.assertEqual(3.0, closest_vertex)

        last_valid_hop = 5.0
        closest_vertex = logic.find_closest_vertex(last_valid_hop, avail_vert)
        self.assertEqual(5.0, closest_vertex)

    def test_valid_hop_to_nonvertex_returns_next_vertex(self):
        hop = 3.001
        avail_vert = np.arange(6, dtype=float)
        closest_vertex = logic.find_closest_vertex(hop, avail_vert)
        self.assertEqual(4.0, closest_vertex)


class TestIncrementAge(unittest.TestCase): 
    def test_empty_event_increases_all_ages_by_1(self):
        model_particles = np.zeros([3, ATTR_COUNT], dtype=float)
        model_particles[:,3] = np.arange(3)
        event_ids = []

        aged_model = logic.increment_age(model_particles, event_ids)
        # The unmoved particles start at age 0 and we test after one iteration
        # Therefore, ages should all be 1
        expected_age = np.ones(3, dtype=float) 
        self.assertCountEqual(expected_age, aged_model[:,5])

    def test_event_set_to_0_and_nonevent_to_1(self):
        starting_age = 2.0
        model_particles = np.zeros([3, ATTR_COUNT], dtype=float)
        model_particles[:,3] = np.arange(3)
        model_particles[:,5] = starting_age
        event_ids = np.array([0, 1])

        aged_model = logic.increment_age(model_particles, event_ids)
        self.assertEqual(starting_age + 1, aged_model[2,5])
        self.assertCountEqual([0.0, 0.0], aged_model[event_ids][:,5])
        

if __name__ == '__main__':
    unittest.main()