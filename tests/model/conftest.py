import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-nk",
        action="store_true",
        default=False,
        help="skip tests that use full_nk.gcn",
    )


def pytest_runtest_setup(item):
    skip_nk = item.config.getoption("--skip-nk")
    if skip_nk and "skip_nk" in item.keywords:
        pytest.skip("skipped due to --skip-nk")


@pytest.fixture
def gcn_file_1():
    return """
                block HOUSEHOLD
                {
                    definitions
                    {
                        u[] = log(C[]);
                    };

                    objective
                    {
                        U[] = u[] + beta * E[][U[1]];
                    };

                    controls
                    {
                        C[], K[];
                    };

                    constraints
                    {
                        Y[] = K[-1] ^ alpha;
                        C[] = r[] * K[-1];
                        K[] = (1 - delta) * K[-1];
                        X[] = Y[] + C[];
                        Z[] = 3;
                    };

                    calibration
                    {
                        alpha = 0.33;
                        beta = 0.99;
                        delta = 0.035;
                    };
                };
                """
