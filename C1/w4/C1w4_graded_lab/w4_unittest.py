import numpy as np


def test_A_reflection_yaxis(target_A, target_A_eig):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "expected": {"A_reflection_yaxis": np.array([[-1, 0], [0, 1]]),},
        },
    ]

    for test_case in test_cases:

        try:
            assert target_A.shape == test_case["expected"]["A_reflection_yaxis"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_reflection_yaxis"].shape,
                    "got": target_A.shape,
                }
            )
            print(
                f"Wrong shape of matrix A_reflection_yaxis. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_A, test_case["expected"]["A_reflection_yaxis"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(
                            np.isclose(
                                target_A, test_case["expected"]["A_reflection_yaxis"]
                            )
                        )
                    ),
                    "got": target_A,
                }
            )
            print(
                f"Wrong matrix A_reflection_yaxis.\nCheck the element in the row {failed_cases[-1].get('expected')[0][0] + 1}, column {failed_cases[-1].get('expected')[1][0] + 1}."
            )

        expected_A_eig = np.linalg.eig(test_case["expected"]["A_reflection_yaxis"])

        try:
            assert len(target_A_eig) == len(expected_A_eig)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(expected_A_eig),
                    "got": len(target_A_eig),
                }
            )
            print(
                f"Wrong size of tuple A_reflection_yaxis_eig. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert target_A_eig[0].shape == expected_A_eig[0].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_A_eig[0].shape,
                    "got": target_A_eig[0].shape,
                }
            )
            print(
                f"Wrong shape of the matrix containing eigenvalues in the A_reflection_yaxis_eig object. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_A_eig[0], expected_A_eig[0])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_A_eig[0],
                    "got": target_A_eig[0],
                }
            )
            print(
                f"Wrong matrix containing eigenvalues in the A_reflection_yaxis_eig object. Check that np.linalg.eig function is applied correctly."
            )

        try:
            assert target_A_eig[1].shape == expected_A_eig[1].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_A_eig[1].shape,
                    "got": target_A_eig[1].shape,
                }
            )
            print(
                f"Wrong shape of the matrix containing eigenvectors in the A_reflection_yaxis_eig object. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_A_eig[1], expected_A_eig[1])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_A_eig[1],
                    "got": target_A_eig[1],
                }
            )
            print(
                f"Wrong matrix containing eigenvectors in the A_reflection_yaxis_eig object. Check that np.linalg.eig function is applied correctly."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_A_shear_x(target_A, target_A_eig):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "expected": {"A_shear_x": np.array([[1, 0.5], [0, 1]]),},
        },
    ]

    for test_case in test_cases:

        try:
            assert target_A.shape == test_case["expected"]["A_shear_x"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A_shear_x"].shape,
                    "got": target_A.shape,
                }
            )
            print(
                f"Wrong shape of matrix A_shear_x. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_A, test_case["expected"]["A_shear_x"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(
                            np.isclose(target_A, test_case["expected"]["A_shear_x"])
                        )
                    ),
                    "got": target_A,
                }
            )
            print(
                f"Wrong matrix A_shear_x.\nCheck the element in the row {failed_cases[-1].get('expected')[0][0] + 1}, column {failed_cases[-1].get('expected')[1][0] + 1}."
            )

        expected_A_eig = np.linalg.eig(test_case["expected"]["A_shear_x"])

        try:
            assert len(target_A_eig) == len(expected_A_eig)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(expected_A_eig),
                    "got": len(target_A_eig),
                }
            )
            print(
                f"Wrong size of tuple A_shear_x_eig. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert target_A_eig[0].shape == expected_A_eig[0].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_A_eig[0].shape,
                    "got": target_A_eig[0].shape,
                }
            )
            print(
                f"Wrong shape of the matrix containing eigenvalues in the A_shear_x_eig object. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_A_eig[0], expected_A_eig[0])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_A_eig[0],
                    "got": target_A_eig[0],
                }
            )
            print(
                f"Wrong matrix containing eigenvalues in the A_shear_x_eig object. Check that np.linalg.eig function is applied correctly."
            )

        try:
            assert target_A_eig[1].shape == expected_A_eig[1].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_A_eig[1].shape,
                    "got": target_A_eig[1].shape,
                }
            )
            print(
                f"Wrong shape of the matrix containing eigenvectors in the A_shear_x_eig object. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_A_eig[1], expected_A_eig[1])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_A_eig[1],
                    "got": target_A_eig[1],
                }
            )
            print(
                f"Wrong matrix containing eigenvectors in the A_shear_x_eig object. Check that np.linalg.eig function is applied correctly."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_matrix(target_P, target_X0, target_X1):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "expected": {
                "P": np.array(
                    [
                        [0, 0.75, 0.35, 0.25, 0.85],
                        [0.15, 0, 0.35, 0.25, 0.05],
                        [0.15, 0.15, 0, 0.25, 0.05],
                        [0.15, 0.05, 0.05, 0, 0.05],
                        [0.55, 0.05, 0.25, 0.25, 0],
                    ]
                ),
                "X0": np.array([[0], [0], [0], [1], [0]]),
            },
        },
    ]

    for test_case in test_cases:

        try:
            assert target_P.shape == test_case["expected"]["P"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["P"].shape,
                    "got": target_P.shape,
                }
            )
            print(
                f"Wrong shape of matrix P. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(np.diagonal(target_P), np.diagonal(test_case["expected"]["P"]))
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(np.isclose(np.diagonal(target_P), np.diagonal(test_case["expected"]["P"])))
                    ),
                    "got": sum(target_P),
                }
            )
            print(
                f"Wrong matrix P. \nCheck the diagonal elements."
            )
            
        try:
            assert np.allclose(sum(target_P), sum(test_case["expected"]["P"]))
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(np.isclose(sum(target_P), sum(test_case["expected"]["P"])))
                    ),
                    "got": sum(target_P),
                }
            )
            print(
                f"Wrong matrix P. \nCheck the elements in the column {failed_cases[-1].get('expected')[0][0] + 1}."
            )

        try:
            assert target_X0.shape == test_case["expected"]["X0"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["X0"].shape,
                    "got": target_X0.shape,
                }
            )
            print(
                f"Wrong shape of vector X0. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_X0, test_case["expected"]["X0"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(
                            np.isclose(target_X0, test_case["expected"]["X0"])
                        )
                    ),
                    "got": target_X0,
                }
            )
            print(
                f"Wrong array X0.\nCheck element {failed_cases[-1].get('expected')[0][0] + 1} in the vector X0."
            )

        expected_X1 = np.matmul(target_P, target_X0)
            
        try:
            assert target_X1.shape == expected_X1.shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X1.shape,
                    "got": target_X1.shape,
                }
            )
            print(
                f"Wrong shape of vector X1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_X1, expected_X1)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X1,
                    "got": target_X1,
                }
            )
            print(
                f"Wrong vector X1. Check if matrix multiplication was performed correctly."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_check_eigenvector(target_T):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "P": np.array(
                    [
                        [0, 0.75, 0.35, 0.25, 0.85],
                        [0.15, 0, 0.35, 0.25, 0.05],
                        [0.15, 0.15, 0, 0.25, 0.05],
                        [0.15, 0.05, 0.05, 0, 0.05],
                        [0.55, 0.05, 0.25, 0.25, 0],
                    ]
                ),
            },
        },
        {"name": "extra_check", "input": {"P": np.array([[2, 3], [2, 1],]),},},
    ]

    for test_case in test_cases:

        X_inf = np.linalg.eig(test_case["input"]["P"])[1][:, 0]

        target_X_check = target_T(test_case["input"]["P"], X_inf)
        expected_X_check = test_case["input"]["P"] @ X_inf

        try:
            assert target_X_check.shape == expected_X_check.shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X_check.shape,
                    "got": target_X_check.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of output matrix in the check_eigenvector function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_X_check, expected_X_check)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X_check,
                    "got": target_X_check,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output matrix in the check_eigenvector function. Check if matrix multiplication was performed correctly."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
