# +
import jax.numpy as np
from math import isclose

# Variables for the default_check test cases.
prices_A = np.array([
    104., 108., 101., 104., 102., 105., 114., 102., 105., 101., 109., 103., 93., 98., 92., 97., 96.,
    94., 97., 93., 99., 93., 98., 94., 93., 92., 96., 98., 98., 93., 97., 102., 103., 100., 100., 104.,
    100., 103., 104., 101., 102., 100., 102., 108., 107., 107., 103., 109., 108., 108.,
])
prices_B = np.array([
    76., 76., 84., 79., 81., 84., 90., 93., 93., 99., 98., 96., 94., 104., 101., 102., 104., 106., 105.,
    103., 106., 104., 113., 115., 114., 124., 119., 115., 112., 111., 106., 107., 108., 108., 102., 104.,
    101., 101., 100., 103., 106., 100., 97., 98., 90., 92., 92., 99., 94., 91.
])


# -

def test_load_and_convert_data(target_A, target_B):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "expected": {"prices_A": prices_A,
                         "prices_B": prices_B,},
        },
    ]

    for test_case in test_cases:

        try:
            assert type(target_A) == type(test_case["expected"]["prices_A"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["prices_A"]),
                    "got": type(target_A),
                }
            )
            print(
                f"prices_A has incorrect type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break
        
        try:
            assert type(target_B) == type(test_case["expected"]["prices_B"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["prices_B"]),
                    "got": type(target_B),
                }
            )
            print(
                f"prices_B has incorrect type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            # Check only one element - no need to check all array.
            assert type(target_A[0].item()) == type(test_case["expected"]["prices_A"][0].item())
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["prices_A"][0].item()),
                    "got": type(target_A[0].item()),
                }
            )
            print(
                f"Elements of prices_A array have incorrect type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            # Check only one element - no need to check all array.
            assert type(target_B[0].item()) == type(test_case["expected"]["prices_B"][0].item())
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["prices_B"][0].item()),
                    "got": type(target_B[0].item()),
                }
            )
            print(
                f"Elements of prices_B array have incorrect type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert target_A.shape == test_case["expected"]["prices_A"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["prices_A"].shape,
                    "got": target_A.shape,
                }
            )
            print(
                f"Wrong shape of prices_A array. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert target_B.shape == test_case["expected"]["prices_B"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["prices_B"].shape,
                    "got": target_B.shape,
                }
            )
            print(
                f"Wrong shape of prices_B array. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break
            
        try:
            assert np.allclose(target_A, test_case["expected"]["prices_A"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["prices_A"],
                    "got": target_A,
                }
            )
            print(
                f"Wrong array prices_A. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(target_B, test_case["expected"]["prices_B"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["prices_B"],
                    "got": target_B,
                }
            )
            print(
                f"Wrong array prices_B. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_f_of_omega(target_f_of_omega):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"omega": 0,},
            "expected": {"f_of_omega": prices_B,},
        },
        {
            "name": "extra_check_1",
            "input": {"omega": 0.2,},
            "expected": {"f_of_omega": prices_A * 0.2 + prices_B * (1-0.2),},
        },
        {
            "name": "extra_check_2",
            "input": {"omega": 0.8,},
            "expected": {"f_of_omega": prices_A * 0.8 + prices_B * (1-0.8),},
        },
        {
            "name": "extra_check_3",
            "input": {"omega": 1,},
            "expected": {"f_of_omega": prices_A,},
        },
    ]

    for test_case in test_cases:
        result = target_f_of_omega(test_case["input"]["omega"])

        try:
            assert result.shape == test_case["expected"]["f_of_omega"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["f_of_omega"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of f_of_omega output for omega = {test_case['input']['omega']}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(result, test_case["expected"]["f_of_omega"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["f_of_omega"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of f_of_omega for omega = {test_case['input']['omega']}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_L_of_omega_array(target_L_of_omega_array):
    successful_cases = 0
    failed_cases = []

    # Not all of the values of the output array will be checked - only some of them.
    # In graders all of the output array gets checked.
    test_cases = [
        {
            "name": "default_check",
            "input": {"omega_array": np.linspace(0, 1, 1001, endpoint=True),},
            "expected": {"shape": (1001,),
                         "L_of_omega_array": [
                             {"i": 0, "L_of_omega": 110.72,},
                             {"i": 1000, "L_of_omega": 27.48,},
                             {"i": 400, "L_of_omega": 28.051199,},
                         ],}
        },
        {
            "name": "extra_check",
            "input": {"omega_array": np.linspace(0, 1, 11, endpoint=True),},
            "expected": {"shape": (11,),
                         "L_of_omega_array": [
                             {"i": 0, "L_of_omega": 110.72,},
                             {"i": 11, "L_of_omega": 27.48,},
                             {"i": 5, "L_of_omega": 17.67,},
                         ],}
        },
    ]

    for test_case in test_cases:
        result = target_L_of_omega_array(test_case["input"]["omega_array"])

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
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of L_of_omega_array output. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
        
        for test_case_i in test_case["expected"]["L_of_omega_array"]:
            i = test_case_i["i"]
                
            try:
                assert isclose(result[i], test_case_i["L_of_omega"], abs_tol=1e-5)
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i["L_of_omega"],
                        "got": result[i],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of L_of_omega_array for omega_array = \n{test_case['input']['omega_array']}\nTest for index i = {i}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_dLdOmega_of_omega_array(target_dLdOmega_of_omega_array):
    successful_cases = 0
    failed_cases = []

    # Not all of the values of the output array will be checked - only some of them.
    # In graders all of the output array gets checked.
    test_cases = [
        {
            "name": "default_check",
            "input": {"omega_array": np.linspace(0, 1, 1001, endpoint=True),},
            "expected": {"shape": (1001,),
                         "dLdOmega_of_omega_array": [
                             {"i": 0, "dLdOmega_of_omega": -288.96,},
                             {"i": 1000, "dLdOmega_of_omega": 122.47999,},
                             {"i": 400, "dLdOmega_of_omega": -124.38398,},
                         ],}
        },
        {
            "name": "extra_check",
            "input": {"omega_array": np.linspace(0, 1, 11, endpoint=True),},
            "expected": {"shape": (11,),
                         "dLdOmega_of_omega_array": [
                             {"i": 0, "dLdOmega_of_omega": -288.96,},
                             {"i": 11, "dLdOmega_of_omega": 122.47999,},
                             {"i": 5, "dLdOmega_of_omega": -83.240036,},
                         ],}
        },
    ]

    for test_case in test_cases:
        result = target_dLdOmega_of_omega_array(test_case["input"]["omega_array"])

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
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of dLdOmega_of_omega_array output. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
        
        for test_case_i in test_case["expected"]["dLdOmega_of_omega_array"]:
            i = test_case_i["i"]
                
            try:
                assert isclose(result[i], test_case_i["dLdOmega_of_omega"], abs_tol=1e-5)
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i["dLdOmega_of_omega"],
                        "got": result[i],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of dLdOmega_of_omega_array for omega_array = \n{test_case['input']['omega_array']}\nTest for index i = {i}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
