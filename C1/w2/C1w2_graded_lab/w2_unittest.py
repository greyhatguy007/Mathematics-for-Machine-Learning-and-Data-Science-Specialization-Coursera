import numpy as np
import math


def test_matrix(target_A, target_b):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "expected": {
                "A": np.array(
                    [[2, -1, 1, 1], [1, 2, -1, -1], [-1, 2, 2, 2], [1, -1, 2, 1]]
                ),
                "b": np.array([6, 3, 14, 8]),
            },
        },
    ]

    for test_case in test_cases:

        try:
            assert target_A.shape == test_case["expected"]["A"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A"].shape,
                    "got": target_A.shape,
                }
            )
            print(
                f"Wrong shape of matrix A. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_A, test_case["expected"]["A"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(np.isclose(target_A, test_case["expected"]["A"]))
                    ),
                    "got": target_A,
                }
            )
            print(
                f"Wrong matrix of coefficients corresponding to the system of linear equations.\nCheck the coefficient in the row {failed_cases[-1].get('expected')[0][0] + 1}, column {failed_cases[-1].get('expected')[1][0] + 1}."
            )

        try:
            assert target_b.shape == test_case["expected"]["b"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b"].shape,
                    "got": target_b.shape,
                }
            )
            print(
                f"Wrong shape of vector b. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_b, test_case["expected"]["b"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(np.isclose(target_b, test_case["expected"]["b"]))
                    ),
                    "got": target_b,
                }
            )
            print(
                f"Wrong vector of free coefficients corresponding to the system of linear equations.\nCheck element {failed_cases[-1].get('expected')[0][0] + 1} in the vector b."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_det_and_solution_scipy(target_d, target_x):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {"name": "default_check", "expected": {
            "d": -17.0,
            "x": np.array([2, 3, 4, 1])},
        },
    ]

    for test_case in test_cases:

        try:
            assert math.isclose(target_d, test_case["expected"]["d"], abs_tol=1e-04) 
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["d"],
                    "got": target_d,
                }
            )
            print(
                f"Wrong determinant of matrix A.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
        
        try:
            assert target_x.shape == test_case["expected"]["x"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["x"].shape,
                    "got": target_x.shape,
                }
            )
            print(
                f"Wrong shape of the solution vector. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_x, test_case["expected"]["x"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["x"],
                    "got": target_x,
                }
            )
            print(
                f"Wrong solution vector.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_elementary_operations(target_MultiplyRow, target_AddRows, target_SwapRows):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "A": np.array(
                    [[1, -2, 3, -4], [-5, 6, -7, 8], [-4, 3, -2, 1], [8, -7, 6, -5]]
                )
            },
            "expected": {
                "A_MultiplyRow": np.array(
                    [[1, -2, 3, -4], [-5, 6, -7, 8], [8, -6, 4, -2], [8, -7, 6, -5]]
                ),
                "A_AddRows": np.array(
                    [[1, -2, 3, -4], [-5, 6, -7, 8], [0, -5, 10, -15], [8, -7, 6, -5]]
                ),
                "A_SwapRows": np.array(
                    [[-4, 3, -2, 1], [-5, 6, -7, 8], [1, -2, 3, -4], [8, -7, 6, -5]]
                ),
            },
        },
    ]

    for test_case in test_cases:
        result_MultiplyRow = target_MultiplyRow(test_case["input"]["A"], 2, -2)
        result_AddRows = target_AddRows(test_case["input"]["A"], 0, 2, 4)
        result_SwapRows = target_SwapRows(test_case["input"]["A"], 0, 2)

        try:
            assert (
                result_MultiplyRow.shape == test_case["expected"]["A_MultiplyRow"].shape
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_MultiplyRow"].shape,
                    "got": result_MultiplyRow.shape,
                }
            )
            print(
                f"Wrong shape of the output matrix. Check MultiplyRow function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_MultiplyRow, test_case["expected"]["A_MultiplyRow"]
            )
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_MultiplyRow"],
                    "got": result_MultiplyRow,
                }
            )
            print(
                f"Wrong output matrix. Check MultiplyRow function. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

        try:
            assert result_AddRows.shape == test_case["expected"]["A_AddRows"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_AddRows"].shape,
                    "got": result_AddRows.shape,
                }
            )
            print(
                f"Wrong shape of the output matrix. Check AddRows function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result_AddRows, test_case["expected"]["A_AddRows"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_AddRows"],
                    "got": result_AddRows,
                }
            )
            print(
                f"Wrong output matrix. Check AddRows function. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

        try:
            assert result_SwapRows.shape == test_case["expected"]["A_SwapRows"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_SwapRows"].shape,
                    "got": result_SwapRows.shape,
                }
            )
            print(
                f"Wrong shape of the output matrix. Check SwapRows function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result_SwapRows, test_case["expected"]["A_SwapRows"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_SwapRows"],
                    "got": result_SwapRows,
                }
            )
            print(
                f"Wrong output matrix. Check SwapRows function. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_augmented_to_ref(target):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "A": np.array(
                    [[2, -1, 1, 1], [1, 2, -1, -1], [-1, 2, 2, 2], [1, -1, 2, 1]]
                ),
                "b": np.array([6, 3, 14, 8]),
            },
            "expected": {
                "A_ref": np.array(
                    [
                        [1, 2, -1, -1, 3],
                        [0, 1, 4, 3, 22],
                        [0, 0, 1, 3, 7],
                        [0, 0, 0, 1, 1],
                    ]
                )
            },
        },
        {
            "name": "identity_matrix",
            "input": {
                "A": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                "b": np.array([1, 1, 1, 1]),
            },
            "expected": {
                "A_ref": np.array(
                    [
                        [0, 1, 0, 0, 1],
                        [0, 0, 1, 1, 2],
                        [2, -1, 1, -2, 0],
                        [0, 0, 0, -1, 0],
                    ]
                )
            },
        },
    ]

    for test_case in test_cases:
        result = target(test_case["input"]["A"], test_case["input"]["b"])

        try:
            assert result.shape == test_case["expected"]["A_ref"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_ref"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output matrix. Check horizontal stack of matrix A and vector b. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["A_ref"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_ref"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output matrix. Check row reduction operations. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_solution_elimination(target_x_1, target_x_2, target_x_3, target_x_4):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "expected": {"x_1": 2, "x_2": 3, "x_3": 4, "x_4": 1},
        },
    ]

    for test_case in test_cases:
        
        for i, target_x_i in enumerate([target_x_1, target_x_2, target_x_3, target_x_4]):
            try:
                assert target_x_i == test_case["expected"]["x_" + str(i+1)]
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case["expected"]["x_" + str(i+1)],
                        "got": target_x_i,
                    }
                )
                print(
                    f"Wrong value of x_{(i+1)}.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_ref_to_diagonal(target):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "A_ref": np.array(
                    [
                        [1, 2, -1, -1, 3],
                        [0, 1, 4, 3, 22],
                        [0, 0, 1, 3, 7],
                        [0, 0, 0, 1, 1],
                    ]
                )
            },
            "expected": {
                "A_diag": np.array(
                    [
                        [1, 0, 0, 0, 2],
                        [0, 1, 0, 0, 3],
                        [0, 0, 1, 0, 4],
                        [0, 0, 0, 1, 1],
                    ]
                )
            },
        },
        {
            "name": "identity_matrix",
            "input": {
                "A_ref": np.array(
                    [[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1]]
                )
            },
            "expected": {
                "A_diag": np.array(
                    [
                        [1, -2, 9, -20, -12],
                        [0, 1, -4, 9, 6],
                        [0, 0, 1, -3, -2],
                        [0, 0, 0, 1, 1],
                    ]
                )
            },
        },
    ]

    for test_case in test_cases:
        result = target(test_case["input"]["A_ref"])

        try:
            assert result.shape == test_case["expected"]["A_diag"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_diag"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["A_diag"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_diag"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output matrix. Check row reduction operations. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

