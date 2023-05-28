import numpy as np

# variables for the default_check test cases
m = 30
X = np.array([[ 0.3190391, -1.07296862, 0.86540763, -0.17242821, 1.14472371, 0.50249434,
               -2.3015387, -0.68372786, -0.38405435, -0.87785842, -2.06014071, -1.10061918,
               -1.09989127, 1.13376944, 1.74481176, -0.12289023, -0.93576943, 1.62434536,
               1.46210794, 0.90159072, -0.7612069, 0.53035547, -0.52817175, -0.26788808,
               0.58281521, 0.04221375, 0.90085595, -0.24937038, -0.61175641, -0.3224172]])
Y = np.array([[ -3.01854669, -65.65047675, 26.96755728, 8.70562603, 57.94332628,
               -0.69293498, -78.66594473, -12.73881492, -13.26721663, -24.80488085,
               -74.24484385, -39.99533724, -22.70174437, 73.46766345, 55.7257405,
               23.80417646, -13.45481508, 25.57952246, 75.91238321, 50.91155323,
               -43.7191551, -1.7025559, -16.44931235, -33.54041234,  20.4505961,
               18.35949302, 37.69029586, -1.04801683, -4.47915933, -20.89431647]])
n_x = X.shape[0]
n_y = Y.shape[0]

def test_shapes(target_shape_X, target_shape_Y, target_m):
    
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "expected": {"shape_X": (1, 30), "shape_Y": (1, 30), "m": 30},
        },
    ]

    for test_case in test_cases:

        try:
            assert target_shape_X == test_case["expected"]["shape_X"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["shape_X"],
                    "got": target_shape_X,
                }
            )
            print(
                f"Wrong shape_X.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert target_shape_Y == test_case["expected"]["shape_Y"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["shape_Y"],
                    "got": target_shape_Y,
                }
            )
            print(
                f"Wrong shape_Y.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert target_m == test_case["expected"]["m"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["m"],
                    "got": target_m,
                }
            )
            print(
                f"Wrong number of training examples m.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

def test_layer_sizes(target_layer_sizes):
    successful_cases = 0
    failed_cases = []
    
    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": X,
                "Y": Y
            },
            "expected": {
                "n_x": n_x,
                "n_y": n_y
            },
        },
        {
            "name": "extra_check",
            "input": {
                "X": np.ones((5, 100)),
                "Y": np.ones((3, 100))
            },
            "expected": {
                "n_x": 5,
                "n_y": 3
            },
        },
    ]

    for test_case in test_cases:
        (result_n_x, result_n_y) = target_layer_sizes(test_case["input"]["X"], test_case["input"]["Y"])

        try:
            assert (
                result_n_x == test_case["expected"]["n_x"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["n_x"],
                    "got": result_n_x,
                }
            )
            print(
                f"Wrong size of the input layer n_x for the test case, where array X has a shape {test_case['input']['X'].shape}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert (
                result_n_y == test_case["expected"]["n_y"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["n_y"],
                    "got": result_n_y,
                }
            )
            print(
                f"Wrong size of the output layer n_y for the test case, where array Y has a shape {test_case['input']['Y'].shape}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

def test_initialize_parameters(target_initialize_parameters):
    
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "n_x": n_x,
                "n_y": n_y,
            },
            "expected": {
                "W": np.zeros((n_y, n_x)), # no check of the actual values in the unit tests
                "b": np.zeros((n_y, 1))
            },
        },
        {
            "name": "extra_check",
            "input": {
                "n_x": 5,
                "n_y": 3,
            },
            "expected": {
                "W": np.zeros((3, 5)), # no check of the actual values in the unit tests
                "b": np.zeros((3, 1))
            },
        },
    ]

    for test_case in test_cases:
        result = target_initialize_parameters(test_case["input"]["n_x"], test_case["input"]["n_y"])

        try:
            assert result["W"].shape == test_case["expected"]["W"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W"].shape,
                    "got": result["W"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
         
        try:
            assert result["b"].shape == test_case["expected"]["b"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b"].shape,
                    "got": result["b"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(result["b"], test_case["expected"]["b"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b"],
                    "got": result["b"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong bias vector b. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

def test_forward_propagation(target_forward_propagation):
    
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": X,
                "parameters": {"W": np.array([[0.01788628]]), "b": np.zeros((n_y, 1))},
            },
            "expected": {
                "A": np.array([[0.00570642, -0.01919142, 0.01547892, -0.0030841, 0.02047485, 0.00898775,
                                -0.04116597, -0.01222935, -0.0068693, -0.01570162, -0.03684825, -0.01968598,
                                -0.01967296, 0.02027892, 0.03120819, -0.00219805, -0.01673743, 0.0290535,
                                0.02615167, 0.0161261, -0.01361516, 0.00948609, -0.00944703, -0.00479152,
                                0.0104244, 0.00075505, 0.01611296, -0.00446031, -0.01094205, -0.00576684]])
            },
        },
        {
            "name": "change_weights_check",
            "input": {
                "X": X,
                "parameters": {"W": np.array([[-0.00768836]]), "b": np.zeros((n_y, 1))},
            },
            "expected": {
                "A": np.array([[-0.00245289, 0.00824937, -0.00665357, 0.00132569, -0.00880105, -0.00386336,
                                0.01769506, 0.00525675, 0.00295275, 0.00674929, 0.0158391, 0.00846196,
                                0.00845636, -0.00871683, -0.01341474, 0.00094482, 0.00719453, -0.01248855,
                                -0.01124121, -0.00693175, 0.00585243, -0.00407756, 0.00406077, 0.00205962,
                                -0.00448089, -0.00032455, -0.0069261, 0.00191725, 0.0047034, 0.00247886]])
            },
        },
        {
            "name": "change_dataset_check",
            "input": {
                "X": np.array([[0, 1, 0, 0, 1],[0, 0, 0, 0, 1]]),
                "parameters": {"W": np.array([[-0.00768836, -0.00230031]]), "b": np.zeros((n_y, 1))},
            },
            "expected": {
                "A": np.array([[0, -0.00768836, 0, 0, -0.00998867]])
            },
        },
    ]

    for test_case in test_cases:
        result = target_forward_propagation(test_case["input"]["X"], test_case["input"]["parameters"])

        try:
            assert result.shape == test_case["expected"]["A"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the array A. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")
            
        try:
            assert np.allclose(result, test_case["expected"]["A"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong array A. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}.")

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

def test_nn_model(target_nn_model):
    
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": X,
                "Y": Y,
                "num_iterations": 10,
            },
            "expected": {
                "W": np.array([[0.19732238]]), # no check of the actual values in the unit tests
                "b": np.array([[-0.2]]) # no check of the actual values in the unit tests
            },
        },
        {
            "name": "extra_check",
            "input": {
                "X": np.array([[0, 1, 0, 0, 1],[0, 0, 0, 0, 1]]),
                "Y": np.array([[0, 0, 0, 0, 1]]),
                "num_iterations": 100,
            },
            "expected": {
                "W": np.array([[-0.23147202,  0.49108187]]), # no check of the actual values in the unit tests
                "b": np.array([[-0.24]]) # no check of the actual values in the unit tests
            },
        },
    ]

    for test_case in test_cases:
        
        result = target_nn_model(test_case["input"]["X"], test_case["input"]["Y"], test_case["input"]["num_iterations"], False)

        try:
            assert result["W"].shape == test_case["expected"]["W"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W"].shape,
                    "got": result["W"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert result["b"].shape == test_case["expected"]["b"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b"].shape,
                    "got": result["b"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_multi(target_nn_model, target_X_multi_norm, target_Y_multi_norm, target_parameters):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "num_iterations": 100,
            },
        },
    ]

    for test_case in test_cases:
        # no check of the actual values in the unit tests
        expected_parameters = target_nn_model(target_X_multi_norm, target_Y_multi_norm, 
                                              test_case["input"]["num_iterations"], False)

        try:
            assert target_parameters["W"].shape == expected_parameters["W"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_parameters["W"].shape,
                    "got": target_parameters["W"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert target_parameters["b"].shape == expected_parameters["b"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_parameters["b"].shape,
                    "got": target_parameters["b"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
