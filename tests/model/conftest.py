import pytest


@pytest.fixture
def gcn_file_1():
    GCN_file = """
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
    return GCN_file
