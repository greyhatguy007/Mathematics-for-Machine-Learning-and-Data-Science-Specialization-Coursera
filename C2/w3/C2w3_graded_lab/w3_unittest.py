import numpy as np
from sklearn.datasets import make_blobs

# +
# variables for the default_check test cases
m = 2000
samples, labels = make_blobs(n_samples=m, 
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]), 
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0) | (labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 0
X = np.transpose(samples)
Y = labels.reshape((1, m))

n_x = X.shape[0]
n_h = 2
n_y = Y.shape[0]


# -

def test_sigmoid(target_sigmoid):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"z": -2,},
            "expected": {"sigmoid": 0.11920292202211755,},
        },
        {
            "name": "extra_check_1",
            "input": {"z": 0,},
            "expected": {"sigmoid": 0.5,},
        },
        {
            "name": "extra_check_2",
            "input": {"z": 3.5,},
            "expected": {"sigmoid": 0.9706877692486436,},
        },
    ]

    for test_case in test_cases:
        result = target_sigmoid(test_case["input"]["z"])
            
        try:
            assert np.allclose(result, test_case["expected"]["sigmoid"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["sigmoid"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of sigmoid for z = {test_case['input']['z']}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
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
                "n_h": n_h,
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
                "n_h": 2,
                "n_y": 3
            },
        },
    ]

    for test_case in test_cases:
        (result_n_x, result_n_h, result_n_y) = target_layer_sizes(test_case["input"]["X"], test_case["input"]["Y"])

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
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong size of the input layer n_x for the test case, where array X has a shape {test_case['input']['X'].shape}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert (
                result_n_h == test_case["expected"]["n_h"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["n_h"],
                    "got": result_n_h,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong size of the hidden layer n_h. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
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
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong size of the output layer n_y for the test case, where array Y has a shape {test_case['input']['Y'].shape}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
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
                "n_h": n_h,
                "n_y": n_y,
            },
            "expected": {
                "W1": np.zeros((n_h, n_x)), # no check of the actual values in the unit tests
                "b1": np.zeros((n_h, 1)),
                "W2": np.zeros((n_y, n_h)), # no check of the actual values in the unit tests
                "b2": np.zeros((n_y, 1)),
            },
        },
        {
            "name": "extra_check",
            "input": {
                "n_x": 5,
                "n_h": 4,
                "n_y": 3,
            },
            "expected": {
                "W1": np.zeros((4, 5)), # no check of the actual values in the unit tests
                "b1": np.zeros((4, 1)),
                "W2": np.zeros((3, 4)), # no check of the actual values in the unit tests
                "b2": np.zeros((3, 1)),
            },
        },
    ]

    for test_case in test_cases:
        result = target_initialize_parameters(test_case["input"]["n_x"], test_case["input"]["n_h"], test_case["input"]["n_y"])

        try:
            assert result["W1"].shape == test_case["expected"]["W1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W1"].shape,
                    "got": result["W1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
         
        try:
            assert result["b1"].shape == test_case["expected"]["b1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b1"].shape,
                    "got": result["b1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(result["b1"], test_case["expected"]["b1"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b1"],
                    "got": result["b1"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong bias vector b1. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )
            
        try:
            assert result["W2"].shape == test_case["expected"]["W2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W2"].shape,
                    "got": result["W2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
         
        try:
            assert result["b2"].shape == test_case["expected"]["b2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b2"].shape,
                    "got": result["b2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(result["b2"], test_case["expected"]["b2"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b2"],
                    "got": result["b2"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong bias vector b2. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
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
                "parameters": {
                    "W1": np.array([[0.01788628, 0.0043651], [0.00096497, -0.01863493]]), 
                    "b1": np.zeros((n_h, 1)),
                    "W2": np.array([[-0.00277388, -0.00354759]]), 
                    "b2": np.zeros((n_y, 1)),
                },
            },
            "expected": {
                "Z1_array": {
                    "shape": (2, 2000), 
                    "Z1": [
                        {"i": 0, "j": 0, "Z1_i_j": 0.11050400276471689,},
                        {"i": 1, "j": 1999, "Z1_i_j": -0.11866556808051022,},
                        {"i": 0, "j": 100, "Z1_i_j": 0.08570563958483839,},
                    ],},
                "A1_array": {
                    "shape": (2, 2000), 
                    "A1": [
                        {"i": 0, "j": 0, "A1_i_j": 0.5275979229090347,},
                        {"i": 1, "j": 1999, "A1_i_j": 0.47036837134568177,},
                        {"i": 0, "j": 100, "A1_i_j": 0.521413303959268,},
                    ],},
                "Z2_array": {
                    "shape": (1, 2000), 
                    "Z2": [
                        {"i": 0, "Z2_i": -0.003193737045395555,},
                        {"i": 400, "Z2_i": -0.003221924688299396,},
                        {"i": 1999, "Z2_i": -0.00317339213692169,},
                    ],},
                "A2_array": {
                    "shape": (1, 2000), 
                    "A2": [
                        {"i": 0, "A2_i": 0.4992015664173166,},
                        {"i": 400, "A2_i": 0.49919451952471916,},
                        {"i": 1999, "A2_i": 0.4992066526315478,},
                    ],},
            },
        },
        {
            "name": "change_weights_check",
            "input": {
                "X": X,
                "parameters": {
                    "W1": np.array([[-0.00082741, -0.00627001], [-0.00043818, -0.00477218]]), 
                    "b1": np.zeros((n_h, 1)),
                    "W2": np.array([[-0.01313865, 0.00884622]]), 
                    "b2": np.zeros((n_y, 1)),
                },
            },
            "expected": {
                "Z1_array": {
                    "shape": (2, 2000), 
                    "Z1": [
                        {"i": 0, "j": 0, "Z1_i_j": -0.022822666781157443,},
                        {"i": 1, "j": 1999, "Z1_i_j": -0.03577862954823146,},
                        {"i": 0, "j": 100, "Z1_i_j": -0.05872690929597483,},
                    ],},
                "A1_array": {
                    "shape": (2, 2000), 
                    "A1": [
                        {"i": 0, "j": 0, "A1_i_j": 0.49429458095298745,},
                        {"i": 1, "j": 1999, "A1_i_j": 0.4910562966698409,},
                        {"i": 0, "j": 100, "A1_i_j": 0.4853224908106953,},
                    ],},
                "Z2_array": {
                    "shape": (1, 2000), 
                    "Z2": [
                        {"i": 0, "Z2_i": -0.0021073530226891173,},
                        {"i": 400, "Z2_i": -0.002110285690191978,},
                        {"i": 1999, "Z2_i": -0.0020644562143733863,},
                    ],},
                "A2_array": {
                    "shape": (1, 2000), 
                    "A2": [
                        {"i": 0, "A2_i": 0.4994731619392989,},
                        {"i": 400, "A2_i": 0.49947242877323833,},
                        {"i": 1999, "A2_i": 0.49948388612971223,},
                    ],},
            },
        },
        {
            "name": "change_dataset_check",
            "input": {
                "X": np.array([[0, 1, 0, 0, 1], [0, 0, 0, 0, 1]]),
                "parameters": {
                    "W1": np.array([[-0.00082741, -0.00627001], [-0.00043818, -0.00477218]]), 
                    "b1": np.zeros((n_h, 1)),
                    "W2": np.array([[-0.01313865, 0.00884622]]), 
                    "b2": np.zeros((n_y, 1)),
                },
            },
            "expected": {
                "Z1_array": {
                    "shape": (2, 5), 
                    "Z1": [
                        {"i": 0, "j": 0, "Z1_i_j": 0.0,},
                        {"i": 1, "j": 4, "Z1_i_j": -0.00521036,},
                        {"i": 0, "j": 4, "Z1_i_j": -0.00709742,},
                    ],},
                "A1_array": {
                    "shape": (2, 5), 
                    "A1": [
                        {"i": 0, "j": 0, "A1_i_j": 0.5,},
                        {"i": 1, "j": 4, "A1_i_j": 0.49869741294686865,},
                        {"i": 0, "j": 4, "A1_i_j": 0.49822565244831607,},
                    ],},
                "Z2_array": {
                    "shape": (1, 5), 
                    "Z2": [
                        {"i": 0, "Z2_i": -0.002146215,},
                        {"i": 1, "Z2_i": -0.0021444662967103198,},
                        {"i": 4, "Z2_i": -0.00213442544018122,},
                    ],},
                "A2_array": {
                    "shape": (1, 5), 
                    "A2": [
                        {"i": 0, "A2_i": 0.4994634464559578,},
                        {"i": 1, "A2_i": 0.4994638836312772,},
                        {"i": 4, "A2_i": 0.49946639384253705,},
                    ],},
            },
        },
    ]

    for test_case in test_cases:
        result_A2, result_cache = target_forward_propagation(test_case["input"]["X"], test_case["input"]["parameters"])

        try:
            assert result_cache["Z1"].shape == test_case["expected"]["Z1_array"]["shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Z1_array"]["shape"],
                    "got": result_cache["Z1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the array Z1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")
            
        for test_case_i_j in test_case["expected"]["Z1_array"]["Z1"]:
            i = test_case_i_j["i"]
            j = test_case_i_j["j"]
                
            try:
                assert result_cache["Z1"][i, j] == test_case_i_j["Z1_i_j"]
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i_j["Z1_i_j"],
                        "got": result_cache["Z1"][i, j],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of Z1 for X = \n{test_case['input']['X']}\nTest for i = {i}, j = {j}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )
        
        try:
            assert result_cache["A1"].shape == test_case["expected"]["A1_array"]["shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A1_array"]["shape"],
                    "got": result_cache["A1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the array A1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")
            
        for test_case_i_j in test_case["expected"]["A1_array"]["A1"]:
            i = test_case_i_j["i"]
            j = test_case_i_j["j"]
                
            try:
                assert result_cache["A1"][i, j] == test_case_i_j["A1_i_j"]
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i_j["A1_i_j"],
                        "got": result_cache["A1"][i, j],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of A1 for X = \n{test_case['input']['X']}\nTest for i = {i}, j = {j}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )
        
        try:
            assert result_cache["Z2"].shape == test_case["expected"]["Z2_array"]["shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Z2_array"]["shape"],
                    "got": result_cache["Z2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the array Z2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")
            
        for test_case_i in test_case["expected"]["Z2_array"]["Z2"]:
            i = test_case_i["i"]
                
            try:
                assert result_cache["Z2"][0, i] == test_case_i["Z2_i"]
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i["Z2_i"],
                        "got": result_cache["Z2"][0, i],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of Z2. Test for i = {i}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )

        try:
            assert result_A2.shape == test_case["expected"]["A2_array"]["shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A2_array"]["shape"],
                    "got": result_A2.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the array A2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")
            
        for test_case_i in test_case["expected"]["A2_array"]["A2"]:
            i = test_case_i["i"]
                
            try:
                assert result_A2[0, i] == test_case_i["A2_i"]
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i["A2_i"],
                        "got": result_A2[0, i],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of A2. Test for i = {i}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )
                
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

def test_compute_cost(target_compute_cost, input_A2):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "A2": input_A2,
                "Y": Y,
            },
            "expected": {"cost": 0.6931477703826823,},
        },
        {
            "name": "extra_check",
            "input": {
                "A2": np.array([[0.64, 0.60, 0.35, 0.15, 0.95]]), 
                "Y": np.array([[0.58, 0.01, 0.42, 0.24, 0.99]])
            },
            "expected": {"cost": 0.5901032749748385,},
        },
    ]

    for test_case in test_cases:
        result = target_compute_cost(test_case["input"]["A2"], test_case["input"]["Y"])
            
        try:
            assert np.allclose(result, test_case["expected"]["cost"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["cost"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of compute_cost. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

def test_update_parameters(target_update_parameters):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "parameters": {
                    "W1": np.array([[0.01788628, 0.0043651], [0.00096497, -0.01863493]]), 
                    "b1": np.zeros((n_h, 1)),
                    "W2": np.array([[-0.00277388, -0.00354759]]), 
                    "b2": np.zeros((n_y, 1)),
                },
                "grads": {
                    "dW1": np.array([[-1.49856632e-05, 1.67791519e-05], [-2.12394543e-05, 2.43895135e-05]]), 
                    "db1": np.array([[5.11207671e-07], [7.06236219e-07]]),
                    "dW2": np.array([[-0.00032641, -0.0002606]]), 
                    "db2": np.array([[-0.00078732]]),
                },
                "learning_rate": 1.2,
            },
            "expected": {
                "parameters": {
                    "W1": np.array([[0.01790426, 0.00434497], [0.00099046, -0.0186642]]), 
                    "b1": np.array([[-6.13449205e-07], [-8.47483463e-07]]), 
                    "W2": np.array([[-0.00238219, -0.00323487]]), 
                    "b2": np.array([[0.00094478]]), 
                },
            }
        },
        {
            "name": "extra_check",
            "input": {
                "parameters": {
                    "W1": np.array([[-0.00082741, -0.00627001], [-0.00043818, -0.00477218]]), 
                    "b1": np.zeros((n_h, 1)),
                    "W2": np.array([[-0.01313865, 0.00884622]]), 
                    "b2": np.zeros((n_y, 1)),
            },
                "grads": {
                    "dW1": np.array([[-7.56054712e-05, 8.48587435e-05], [5.05322772e-05, -5.72665231e-05]]), 
                    "db1": np.array([[1.68002224e-06], [-1.14292837e-06]]),
                    "dW2": np.array([[-0.0002246, -0.00023206]]), 
                    "db2": np.array([[-0.000521]]),
                },
                "learning_rate": 0.1,
            },
            "expected": {
                "parameters": {
                    "W1": np.array([[-0.00081985, -0.0062785], [-0.00044323, -0.00476645]]), 
                    "b1": np.array([[-1.68002224e-07], [1.14292837e-07]]),
                    "W2": np.array([[-0.01311619, 0.00886943]]), 
                    "b2": np.array([[5.21e-05]]),
                },
            }
        },
    ]

    for test_case in test_cases:
        result = target_update_parameters(test_case["input"]["parameters"], test_case["input"]["grads"], test_case["input"]["learning_rate"])
        
        try:
            assert result["W1"].shape == test_case["expected"]["parameters"]["W1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["W1"].shape,
                    "got": result["W1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output array W1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(result["W1"], test_case["expected"]["parameters"]["W1"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["W1"],
                    "got": result["W1"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output array W1. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )
            
        try:
            assert result["b1"].shape == test_case["expected"]["parameters"]["b1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["b1"].shape,
                    "got": result["b1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output array b1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(result["b1"], test_case["expected"]["parameters"]["b1"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["b1"],
                    "got": result["b1"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output array b1. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"

            )
            
        try:
            assert result["W2"].shape == test_case["expected"]["parameters"]["W2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["W2"].shape,
                    "got": result["W2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output array W2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(result["W2"], test_case["expected"]["parameters"]["W2"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["W2"],
                    "got": result["W2"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output array W2. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )
            
        try:
            assert result["b2"].shape == test_case["expected"]["parameters"]["b2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["b2"].shape,
                    "got": result["b2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output array b2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(result["b2"], test_case["expected"]["parameters"]["b2"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["b2"],
                    "got": result["b2"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output array b2. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )
            

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
                "n_h": 2,
                "num_iterations": 3000,
                "learning_rate": 1.2,
            },
            "expected": {
                "W1": np.zeros((n_h, n_x)), # no check of the actual values in the unit tests
                "b1": np.zeros((n_h, 1)), # no check of the actual values in the unit tests
                "W2": np.zeros((n_y, n_h)), # no check of the actual values in the unit tests
                "b2": np.zeros((n_y, 1)), # no check of the actual values in the unit tests
            },
        },
        {
            "name": "extra_check",
            "input": {
                "X": np.array([[0, 1, 0, 0, 1],[0, 0, 0, 0, 1]]),
                "Y": np.array([[0, 0, 0, 0, 1]]),
                "n_h": 3,
                "num_iterations": 100,
                "learning_rate": 0.1,
            },
            "expected": {
                "W1": np.zeros((3, 2)), # no check of the actual values in the unit tests
                "b1": np.zeros((3, 1)), # no check of the actual values in the unit tests
                "W2": np.zeros((1, 3)), # no check of the actual values in the unit tests
                "b2": np.zeros((1, 1)), # no check of the actual values in the unit tests
            },
        },
    ]

    for test_case in test_cases:
        
        result = target_nn_model(test_case["input"]["X"], test_case["input"]["Y"], test_case["input"]["n_h"], 
                                 test_case["input"]["num_iterations"], test_case["input"]["learning_rate"], False)

        try:
            assert result["W1"].shape == test_case["expected"]["W1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W1"].shape,
                    "got": result["W1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert result["b1"].shape == test_case["expected"]["b1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b1"].shape,
                    "got": result["b1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result["W2"].shape == test_case["expected"]["W2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W2"].shape,
                    "got": result["W2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert result["b2"].shape == test_case["expected"]["b2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b2"].shape,
                    "got": result["b2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_predict(target_predict):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": np.array([[2, 8, 2, 8], [2, 8, 8, 2]]),
                "parameters": {
                    "W1": np.array([[2.14274251, -1.93155541], [2.20268789, -2.1131799]]), 
                    "b1": np.array([[-4.83079243], [6.2845223]]),
                    "W2": np.array([[-7.21370685, 7.0898022]]), 
                    "b2": np.array([[-3.48755239]]),
                },
            },
            "expected": {
                "predictions": np.array([[True, True, False, False]]),
            },
        },
        {
            "name": "extra_check",
            "input": {
                "X": np.array([[0, 10, 0, 0, 10],[0, 0, 0, 0, 10]]),
                "parameters": {
                    "W1": np.array([[2.15345603, -2.02993877], [2.24191569, -1.89471923]]), 
                    "b1": np.array([[6.29905582], [-4.80909975]]),
                    "W2": np.array([[7.07457688, -7.23061969]]), 
                    "b2": np.array([[-3.50971507]]),
                },
            },
            "expected": {
                "predictions": np.array([[True, False, True, True, True]]),
            },
        },
    ]

    for test_case in test_cases:
        
        result = target_predict(test_case["input"]["X"], test_case["input"]["parameters"])

        try:
            assert result.shape == test_case["expected"]["predictions"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["predictions"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output array. Input: X = \n{test_case['input']['X']},\nparameters = {test_case['input']['parameters']}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(result, test_case["expected"]["predictions"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["predictions"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output array. Input: X = \n{test_case['input']['X']},\nparameters = {test_case['input']['parameters']}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
