import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def test_load_data(target_adv):
    successful_cases = 0
    failed_cases = []
    
    try:
        assert type(target_adv) == pd.DataFrame
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "default_check",
                "expected": pd.DataFrame,
                "got": type(target_adv),
            }
        )
        print(
            f"Test case \"{failed_cases[-1].get('name')}\". Object adv has incorrect type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
        )

    # Not all of the values of the output array will be checked - only some of them.
    # In graders all of the output array gets checked.
    test_cases = [{
            "name": "default_check",
            "expected": {"shape": (200, 2), 
                         "adv": [
                             {"i": 0, "TV": 230.1, "Sales": 22.1},
                             {"i": 4, "TV": 180.8, "Sales": 12.9},
                             {"i": 40, "TV": 202.5, "Sales": 16.6},
                             {"i": 199, "TV": 232.1, "Sales": 13.4},
                         ],}
    },]

    for test_case in test_cases:
        result = target_adv

        try:
            assert result.shape == test_case["expected"]["shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["shape"],
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of adv. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
        
        for test_case_i in test_case["expected"]["adv"]:
            i = test_case_i["i"]
                
            try:
                assert float(result.iloc[i]["TV"]) == test_case_i["TV"]
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i["TV"],
                        "got": float(result.iloc[i]["TV"]),
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong value of TV in the adv. Test for index i = {i}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )
                
            try:
                assert float(result.iloc[i]["Sales"]) == test_case_i["Sales"]
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i["Sales"],
                        "got": float(result.iloc[i]["Sales"]),
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong value of Sales in the adv. Test for index i = {i}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_pred_numpy(target_pred_numpy):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "m": 0.04753664043301975, 
                "b": 7.0325935491276965,
                "X": np.array([50, 120, 280]),
            },
            "expected": {
                "Y": np.array([9.40942557, 12.7369904, 20.34285287]),
            }
        },
        {
            "name": "extra_check",
            "input": {
                "m": 2,
                "b": 10,
                "X": np.array([-5, 0, 1, 5])
            },
            "expected": {
                "Y": np.array([0, 10, 12, 20]),
            }
        },
    ]

    for test_case in test_cases:
        result = target_pred_numpy(test_case["input"]["m"], test_case["input"]["b"], test_case["input"]["X"])

        try:
            assert result.shape == test_case["expected"]["Y"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Y"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of pred_numpy output. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
        
        try:
            assert np.allclose(result, test_case["expected"]["Y"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Y"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of pred_numpy for X = {test_case['input']['X']}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_sklearn_fit(target_lr_sklearn):
    successful_cases = 0
    failed_cases = []

    # Not all of the values of the output array will be checked - only some of them.
    # In graders all of the output array gets checked.
    test_cases = [
        {
            "name": "default_check",
            "expected": {
                "coef_": np.array([[0.04753664]]),
                "intercept_": np.array([7.03259355]),
            }
        },
    ]

    for test_case in test_cases:
        result = target_lr_sklearn

        try:
            assert isinstance(result, LinearRegression)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": LinearRegression,
                    "got": type(result),
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Object lr_sklearn has incorrect type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
        
        try:
            assert hasattr(result, 'coef_')
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": "coef_ attribute of the lr_sklearn model",
                    "got": None,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". lr_sklearn has no attribute coef_. Check if you have fitted the linear regression model correctly. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
        
        try:
            assert hasattr(result, 'intercept_')
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": "intercept_ attribute of the lr_sklearn model",
                    "got": None,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". lr_sklearn has no attribute intercept_. Check if you have fitted the linear regression model correctly. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
        
        try:
            assert np.allclose(result.coef_, test_case["expected"]["coef_"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["coef_"],
                    "got": result.coef_,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong slope. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )
            
        try:
            assert np.allclose(result.intercept_, test_case["expected"]["intercept_"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["intercept_"],
                    "got": result.intercept_,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong intercept. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_sklearn_predict(target_pred_sklearn, input_lr_sklearn):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": np.array([50, 120, 280]),
            },
            "expected": {
                "Y": np.array([[9.40942557], [12.7369904], [20.34285287]]),
            }
        },
        {
            "name": "extra_check",
            "input": {
                "X": np.array([-5, 0, 1, 5])
            },
            "expected": {
                "Y": np.array([[6.79491035], [7.03259355], [7.08013019], [7.27027675]]),
            }
        },
    ]

    for test_case in test_cases:
        result = target_pred_sklearn(test_case["input"]["X"], input_lr_sklearn)

        try:
            assert result.shape == test_case["expected"]["Y"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Y"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of pred_sklearn output. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
        
        try:
            assert np.allclose(result, test_case["expected"]["Y"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Y"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of pred_sklearn for X = {test_case['input']['X']}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_partial_derivatives(target_dEdm, target_dEdb, input_X_norm, input_Y_norm):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "m": 0, 
                "b": 0,
            },
            "expected": {
                "dEdm": -0.7822244248616065,
                "dEdb": 1.687538997430238e-16,
            }
        },
        {
            "name": "extra_check",
            "input": {
                "m": 1,
                "b": 5,
            },
            "expected": {
                "dEdm": 0.21777557513839416,
                "dEdb": 5.000000000000001,
            }
        },
    ]

    for test_case in test_cases:
        result_dEdm = target_dEdm(test_case["input"]["m"], test_case["input"]["b"], input_X_norm, input_Y_norm)
        result_dEdb = target_dEdb(test_case["input"]["m"], test_case["input"]["b"], input_X_norm, input_Y_norm)
        
        try:
            assert np.allclose(result_dEdm, test_case["expected"]["dEdm"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["dEdm"],
                    "got": result_dEdm,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of dEdm for m = {test_case['input']['m']}, b = {test_case['input']['b']}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )
        
        try:
            assert np.allclose(result_dEdb, test_case["expected"]["dEdb"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["dEdb"],
                    "got": result_dEdb,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of dEdb for m = {test_case['input']['m']}, b = {test_case['input']['b']}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_gradient_descent(target_gradient_descent, input_dEdm, input_dEdb, input_X_norm, input_Y_norm):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "m": 0, 
                "b": 0,
                "learning_rate": 0.001,
                "num_iterations": 1000,
            },
            "expected": {
                "m": 0.49460408269589484,
                "b": -1.367306268207353e-16,
            }
        },
        {
            "name": "extra_check",
            "input": {
                "m": 1,
                "b": 5,
                "learning_rate": 0.01,
                "num_iterations": 10,
            },
            "expected": {
                "m": 0.9791767513915026,
                "b": 4.521910375044022,
            }
        },
    ]

    for test_case in test_cases:
        result_m, result_b = target_gradient_descent(
            input_dEdm, input_dEdb, test_case["input"]["m"], test_case["input"]["b"], 
            input_X_norm, input_Y_norm, test_case["input"]["learning_rate"], test_case["input"]["num_iterations"]
        )
        
        try:
            assert np.allclose(result_m, test_case["expected"]["m"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["m"],
                    "got": result_m,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output value m of the function gradient_descent.\nm = {test_case['input']['m']}, b = {test_case['input']['b']}, learning_rate = {test_case['input']['learning_rate']}, num_iterations = {test_case['input']['num_iterations']}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )
        
        try:
            assert np.allclose(result_b, test_case["expected"]["b"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b"],
                    "got": result_b,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output value b of the function gradient_descent.\nm = {test_case['input']['m']}, b = {test_case['input']['b']}, learning_rate = {test_case['input']['learning_rate']}, num_iterations = {test_case['input']['num_iterations']}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
