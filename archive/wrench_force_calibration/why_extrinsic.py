"""
Why does the extrinsic composition of Mark's angles fit, when he intended
intrinsic?

THE QUESTION
------------
Mark handwrote R = Rz(-135) @ Rx(45) and confirmed he meant intrinsic. The
recordings support Rx(45) @ Rz(-135) instead: 6/7 contact events against
1/7 for the intrinsic form. calibration.yaml has carried this as an
unexplained conflict since 2026-08-24.

Scoring wrench rays tells us WHICH matrix fits. It does not say why. This
does, and it uses no recordings at all -- only Mark's own drawing numbers
and the physical fact that the camera has to be able to see the gripper.

THE TEST
--------
A rotation that maps camera coordinates into the measurement frame has the
camera's optical axis as its THIRD COLUMN, since z_camera = R @ [0,0,1].
The camera position in the measurement frame is Mark's own t. The gripper
centre is 104 mm from the origin down the tool axis, which in the
measurement frame is +z because Mark states z points down.

So the direction from the camera to the gripper is fixed by his numbers
alone, and each candidate rotation predicts an optical axis. Whichever
rotation is correct must point the camera roughly AT the gripper, because
that is what the bracket is for and because every recording shows the
gripper and the held object in frame.

The ZED Mini's horizontal field of view is about 87 degrees, so anything
beyond roughly 44 degrees off axis is not in the image at all. That gives a
hard pass/fail rather than a judgement call.

WHAT WOULD FALSIFY THE CONCLUSION
---------------------------------
If both candidates landed inside the field of view, this test would be
uninformative and the conflict would stay open. If the INTRINSIC one landed
inside and the extrinsic outside, that would contradict the wrench-ray
result and mean one of the two is wrong. Both outcomes are reported.

Read only. No recordings, no calibration file, no masks.

Usage: .venv_analysis/bin/python Code/why_extrinsic.py
"""
import numpy as np

# ZED Mini, Stereolabs datasheet: ~87 deg horizontal FOV.
FOV_HALF_DEG = 87.0 / 2.0

# Mark's numbers, millimetres, in the measurement frame (z points DOWN).
T_CAMERA_MM = np.array([0.0, 120.99, -125.65])   # his corrected translation
D_GRIPPER_MM = 104.0                             # = 12.28 + 91.72, his sheet


def Rz(a):
    a = np.deg2rad(a)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a), 0],
                     [0, 0, 1]])


def Rx(a):
    a = np.deg2rad(a)
    return np.array([[1, 0, 0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a), np.cos(a)]])


def main():
    # gripper centre: 104 mm along +z, because Mark states z points down and
    # the gripper is below the sensor.
    p_gripper = np.array([0.0, 0.0, D_GRIPPER_MM])
    v = p_gripper - T_CAMERA_MM
    v_hat = v / np.linalg.norm(v)

    print("[setup] measurement frame, z points down (Mark, 2026-08-27)")
    print(f"  camera origin   {T_CAMERA_MM.round(2).tolist()} mm")
    print(f"  gripper centre  {p_gripper.round(2).tolist()} mm")
    print(f"  camera -> gripper direction {v_hat.round(4).tolist()}, "
          f"range {np.linalg.norm(v):.1f} mm\n")

    cands = [
        ("intrinsic  Rz(-135) @ Rx(45)   (what Mark intended)",
         Rz(-135) @ Rx(45)),
        ("extrinsic  Rx(45)  @ Rz(-135)  (what the data supports)",
         Rx(45) @ Rz(-135)),
    ]

    results = []
    for name, R in cands:
        z_cam = R[:, 2]                       # optical axis in measurement frame
        cos = float(np.clip(z_cam @ v_hat, -1, 1))
        off = np.degrees(np.arccos(cos))
        inside = off < FOV_HALF_DEG
        results.append((name, off, inside))
        print(f"[{name}]")
        print(f"  optical axis (3rd column)  {z_cam.round(4).tolist()}")
        print(f"  angle to the gripper       {off:6.1f} deg")
        print(f"  gripper inside {FOV_HALF_DEG:.1f} deg half-FOV? "
              f"{'YES' if inside else 'NO'}\n")

    ins = [r for r in results if r[2]]
    print("[verdict]")
    if len(ins) == 1:
        win = ins[0]
        lose = [r for r in results if not r[2]][0]
        print(f"  Only one candidate can see the gripper at all.")
        print(f"    keeps it in view : {win[0].split()[0]}  ({win[1]:.1f} deg)")
        print(f"    points away      : {lose[0].split()[0]}  ({lose[1]:.1f} deg)")
        print("  This is decided by Mark's own drawing numbers plus the ZED's")
        print("  field of view. No recordings are involved, so it is an")
        print("  independent confirmation of the wrench-ray result rather than")
        print("  a restatement of it.")
    elif len(ins) == 2:
        print("  Both candidates keep the gripper in view, so this test does")
        print("  not discriminate. The conflict stays open.")
    else:
        print("  NEITHER candidate sees the gripper. Something upstream is")
        print("  wrong -- most likely the assumed gripper direction or the")
        print("  sign of z. Do not use this result.")
    print("[done]")


if __name__ == "__main__":
    main()
