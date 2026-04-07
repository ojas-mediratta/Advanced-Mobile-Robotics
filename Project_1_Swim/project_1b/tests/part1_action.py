import math
from controllers.ROV_controller.brain import Brain, wrench_to_thrusters, Wrench, Control


def test_wrench_to_thrusters():
    
    wrench = Wrench.force_only(10.0, 0.0, 5.0)
    u_lm, u_rm, u_vlm, u_vrm = wrench_to_thrusters(wrench)
    assert all(isinstance(x, float) for x in [u_lm, u_rm, u_vlm, u_vrm])
    assert all(math.isclose(a, b) for a, b in zip(
        [u_lm, u_rm, u_vlm, u_vrm],
        [-5.0, 5.0, 2.5, -2.5]
    ))
    
    wrench = Wrench.torque_only(0.0, 2.0, 3.0)
    u_lm, u_rm, u_vlm, u_vrm = wrench_to_thrusters(wrench)
    assert all(isinstance(x, float) for x in [u_lm, u_rm, u_vlm, u_vrm])
    assert all(math.isclose(a, b) for a, b in zip(
        [u_lm, u_rm, u_vlm, u_vrm],
        [15.0, 15.0, 0.0, 0.0]
    ))
    
    
def test_act_on_command():
    
    brain = Brain([])

    class KeyboardMock:
        UP = 315
        DOWN = 317
        LEFT = 314
        RIGHT = 316

    test_cases = {
        KeyboardMock.UP:     Control(-0.5, 0.5, 0.0, 0.0), 
        KeyboardMock.DOWN:   Control(0.5, -0.5, 0.0, 0.0), 
        KeyboardMock.LEFT:   Control(5.0, 5.0, 0.0, 0.0),  
        KeyboardMock.RIGHT:  Control(-5.0, -5.0, 0.0, 0.0), 
        ord("W"):            Control(0.0, 0.0, 1.0, -1.0), 
        ord("S"):            Control(0.0, 0.0, -1.0, 1.0), 
        ord("Q"):            Control(0.0, 0.0, -14.285714285714285, -14.285714285714285), 
        ord("E"):            Control(0.0, 0.0, 14.285714285714285, 14.285714285714285),   
    }

    for key, expected in test_cases.items():
        result = brain.act_on_command(key, KeyboardMock)

        # Type check
        assert isinstance(result, Control)

        # Aggregate value check
        actual_values = [result.u_lm, result.u_rm, result.u_vlm, result.u_vrm]
        expected_values = [expected.u_lm, expected.u_rm, expected.u_vlm, expected.u_vrm]

        assert all(
            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
            for a, b in zip(actual_values, expected_values)
        )

   