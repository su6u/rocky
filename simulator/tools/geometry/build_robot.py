"""Assemble Rocky's articulated description from the official kit.

Inputs: the official STL parts (kit/stl), joint features detected with primitives.py, and the statue
registrations produced by registration.py. Output: assets/robot.json (print-millimetre units; the simulator
scales to metres), containing:
  torso frame (from the ball-stud plane), per-limb joint axes (yaw/pitch through the ball centre, elbow along
  the kit hinge), part→body transforms, solid & shell mass properties, collision spheres, foot contact spheres,
  the sculpted statue pose as joint angles, and all measurement residuals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from massprops import shell_properties, solid_properties  # noqa: E402
from meshio import load_stl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]          # simulator/
SRC = ROOT / "kit" / "stl"
REG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "registration"

STUDS = np.array([[48.276, 13.375, 27.449], [51.277, 36.169, 21.176], [20.574, 12.553, 26.196], [14.699, 24.401, 25.006], [23.409, 46.102, 30.550]])
STUD_RADIUS = 2.3105
SOCKETS = {"1-A": [96.1142, -20.8828, 3.6828], "2-A": [-17.604, -14.3729, 33.6651], "3-A": [-33.269, -19.0039, 4.5322], "4-A": [40.941, -16.876, 4.3142], "5-A": [49.0903, -15.3003, 26.1403]}
HOLES = {"1-A": ([0.5588, -0.7956, 0.2342], [99.2734, -15.4902, 27.8946], 1.6547), "2-A": ([0.9823, 0.1871, 0.0066], [-14.3854, -20.3259, 4.9033], 1.5455),
         "3-A": ([0.4364, 0.8667, -0.2416], [-31.3093, -13.0897, 30.7402], 1.4169), "4-A": ([-0.9928, 0.0107, 0.1194], [40.38, -14.57, 35.026], 4.68),
         "5-A": ([0.99, -0.0005, -0.1414], [53.9051, -16.4435, 6.9615], 2.0426)}
PINS = {"1-B": ([0.0402, 0.9676, 0.2494], [10.7935, -10.2688, -0.8213], 1.6509), "1-C": ([-0.032, 0.9936, 0.1086], [78.9229, -14.2985, 3.0719], 1.651),
        "2-B": ([-0.8838, 0.2557, -0.3919], [66.384, -14.728, 5.146], 1.485), "3-B": ([-0.9062, -0.422, -0.025], [71.0581, -18.0092, 4.4371], 1.415),
        "4-B": ([0.995, -0.0673, -0.0741], [-2.041, -17.154, 27.837], 4.51), "5-B": ([0.0903, -0.9958, -0.0115], [28.2651, -8.212, 5.8969], 2.0301)}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rodrigues(axis, ang):
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def signed_angle(a, b, axis):
    return float(np.arctan2(np.dot(np.cross(a, b), unit(axis)), np.dot(a, b)))


def fit_sphere(p):
    A = np.c_[2 * p, np.ones(len(p))]
    s, *_ = np.linalg.lstsq(A, (p * p).sum(1), rcond=None)
    c = s[:3]
    return c, float(np.sqrt(max(s[3] + c @ c, 1e-12)))


def capsule_spheres(pts, count=4, shrink=0.85):
    """Chain of spheres along the principal axis; radius from the inner quantile of radial distances per slab."""
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = Vt[0]
    t = (pts - c) @ ax
    edges = np.linspace(t.min(), t.max(), count + 1)
    out = []
    for k in range(count):
        m = (t >= edges[k]) & (t <= edges[k + 1])
        if m.sum() < 20:
            continue
        sl = pts[m]
        ctr = sl.mean(0)
        r = np.quantile(np.linalg.norm((sl - ctr) - np.outer((sl - ctr) @ ax, ax), axis=1), 0.35) * shrink
        out.append(dict(center=ctr.tolist(), radius=float(max(r, 0.8))))
    return out


def rotating_calipers_width(xy):
    angles = np.radians(np.arange(0, 180, 0.5))
    w = [np.ptp(xy @ np.array([np.cos(a), np.sin(a)])) for a in angles]
    return float(max(w)), float(min(w))


def main():
    regA = json.loads((REG / "limbA_reg.json").read_text())
    regB = json.loads((REG / "limbB_reg.json").read_text())
    RT, tT = np.array(regA["torso"]["R"]), np.array(regA["torso"]["t"])
    parts = {name: load_stl(SRC / f"{name}.stl") for name in ["torso", "1-A", "1-B", "1-C", "2-A", "2-B", "3-A", "3-B", "4-A", "4-B", "5-A", "5-B"]}

    # ---- torso body frame: least-squares plane through the five ball centres ----
    o = STUDS.mean(0)
    _, _, Vt = np.linalg.svd(STUDS - o)
    z = Vt[2] * (1 if Vt[2][2] > 0 else -1)
    vt, _ = parts["torso"]
    if (vt.mean(0) - o) @ z < 0 and False:
        z = -z
    r1 = (STUDS[0] - o) - ((STUDS[0] - o) @ z) * z
    x = unit(r1); y = np.cross(z, x)
    E = np.c_[x, y, z]                               # columns: body axes in torso-part coordinates
    to_body = lambda p: (np.asarray(p) - o) @ E       # torso part coords → torso body coords
    studs_b = to_body(STUDS)
    plane_rms = float(np.sqrt(np.mean((studs_b[:, 2]) ** 2)))

    # carapace width without the studs, for the canon scale (18 in across, half as thick)
    vb = to_body(vt)
    keep = np.all(np.linalg.norm(vt[:, None, :] - STUDS[None], axis=2) > 4.5, axis=1)
    wmax, wmin = rotating_calipers_width(vb[keep][:, :2])
    thickness = float(np.ptp(vb[keep][:, 2]))

    # statue pose of any part in torso body coords: x_b = Eᵀ(R_Tᵀ(R_P x + t_P − t_T) − o)
    def statue_in_body(R, t):
        Rb = E.T @ RT.T @ np.asarray(R)
        tb = E.T @ (RT.T @ (np.asarray(t) - tT) - o)
        return Rb, tb
    statue_up_in_body = E.T @ RT.T @ np.array([0, 0, 1.0])

    robot = dict(
        format="rocky-robot/1", units="print_mm", source="Official Project Hail Mary Rocky action-figure STL kit (kit/stl)",
        torso=dict(part="torso", partToBody=dict(R=E.T.tolist(), t=(-(E.T @ o)).tolist()),
                   studPlaneRms_mm=plane_rms, carapaceWidthMax_mm=wmax, carapaceWidthMin_mm=wmin, carapaceThickness_mm=thickness,
                   statueUp=statue_up_in_body.tolist()),
        canon=dict(carapaceAcross_m=18 * 0.0254, note="Book: 'roughly a pentagon, 18 inches across and half as thick'; mass 168 kg",
                   printToCanonScale=float(18 * 0.0254 / (wmax / 1000.0))),
        limbs=[],
    )
    # torso mass properties (solid & shell), body coords
    for kind, fn in (("solid", solid_properties), ("shell", shell_properties)):
        m, c, I, q = fn(vb, parts["torso"][1], 1.0)
        robot["torso"][kind] = dict(mass_per_density=m, com=c.tolist(), inertia=I.tolist(), measure=q)
    # torso collision spheres: k-means on surface samples of the lower half + rim
    rng = np.random.default_rng(0)
    samp = vb[rng.choice(len(vb), 30000, replace=False)]
    K = 9
    cent = samp[rng.choice(len(samp), K, replace=False)]
    for _ in range(40):
        lab = np.argmin(((samp[:, None] - cent[None]) ** 2).sum(-1), 1)
        cent = np.array([samp[lab == k].mean(0) if np.any(lab == k) else cent[k] for k in range(K)])
    body_c = vb.mean(0)
    spheres = []
    for k in range(K):
        pts = samp[lab == k]
        inward = unit(body_c - cent[k])
        r = float(np.quantile(np.linalg.norm(pts - cent[k], axis=1), 0.5) * 0.9)
        spheres.append(dict(center=(cent[k] + inward * r * 0.6).tolist(), radius=r))
    robot["torso"]["collision"] = spheres

    for limb in range(1, 6):
        a_name = f"{limb}-A"
        stud_index = regA["limbsA"][a_name]["stud"]
        c = studs_b[stud_index]
        radial = unit(np.array([c[0], c[1], 0.0]))
        tangent = np.cross([0, 0, 1.0], radial)
        pitch_axis = np.cross(radial, [0, 0, 1.0])                       # positive pitch lifts the limb
        F = np.c_[radial, tangent, [0, 0, 1.0]]                           # yaw frame (columns in torso body coords)
        # ---- statue poses (free registrations) in torso body coords ----
        RAs, tAs = statue_in_body(regB["A_free"][a_name]["R"], regB["A_free"][a_name]["t"])
        RAc, tAc = statue_in_body(regA["limbsA"][a_name]["R"], regA["limbsA"][a_name]["t"])
        sock = np.array(SOCKETS[a_name])
        hole_ax, hole_pt, hole_r = HOLES[a_name]
        hole_ax = unit(hole_ax); hole_pt = np.array(hole_pt)
        b_names = ["1-C", "1-B"] if limb == 1 else [f"{limb}-B"]
        # refine the elbow axis where the kit feature fit is uncertain: average A-hole and B-pin axes (free fits)
        RBs, tBs = statue_in_body(regB["B"][b_names[0]]["R_free"], regB["B"][b_names[0]]["t_free"])
        pin_ax, pin_pt, pin_r = PINS[b_names[0]]
        pin_ax = unit(pin_ax); pin_pt = np.array(pin_pt)
        ax_hole_s = RAs @ hole_ax; ax_pin_s = RBs @ pin_ax
        if ax_pin_s @ ax_hole_s < 0:
            ax_pin_s = -ax_pin_s
        elbow_axis_s = unit(ax_hole_s + ax_pin_s)
        elbow_pt_s = 0.5 * ((RAs @ hole_pt + tAs) + (RBs @ pin_pt + tBs))
        # A's intrinsic axes in the statue pose
        sock_s = RAs @ sock + tAs
        u_s = unit(elbow_pt_s - sock_s)
        w_s = unit(elbow_axis_s - (elbow_axis_s @ u_s) * u_s)
        skew = float(np.degrees(np.arcsin(np.clip(abs(elbow_axis_s @ u_s), 0, 1))))
        # decompose statue orientation of A into yaw (about z at the stud) and pitch, then roll about the limb axis
        horiz = unit(np.array([u_s[0], u_s[1], 0.0]))
        yaw_s = signed_angle(radial, horiz, [0, 0, 1.0])
        pitch_s = float(np.arcsin(np.clip(u_s[2], -1, 1)))
        R_yaw = rodrigues([0, 0, 1.0], yaw_s)
        p_after_yaw = R_yaw @ pitch_axis
        R_pitch = rodrigues(p_after_yaw, pitch_s)
        R_yp = R_pitch @ R_yaw                                   # maps zero-pose directions to statue directions
        u0 = R_yp.T @ u_s                                        # ≈ radial
        w0 = R_yp.T @ w_s                                        # hinge axis with yaw/pitch removed
        w0p = unit(w0 - (w0 @ u0) * u0)
        if w0p @ pitch_axis < 0:
            flip_axis = True; w0p = -w0p; w_s = -w_s; elbow_axis_s = -elbow_axis_s
        else:
            flip_axis = False
        roll_s = signed_angle(pitch_axis, w0p, u0)
        # ---- robot zero pose for A: remove yaw, pitch and roll ----
        R_roll = rodrigues(u0, roll_s)
        R_A_zero = R_roll.T @ R_yp.T @ RAs                        # part → torso body at zero pose (rotation)
        t_A_zero = c - R_A_zero @ sock                            # socket centre sits exactly on the ball centre
        # A body frame = pitch frame F (origin at c); part → A body
        R_partA = F.T @ R_A_zero
        t_partA = F.T @ (t_A_zero - c)
        vA, fA = parts[a_name]
        vA_b = vA @ R_partA.T + t_partA
        hinge_A = R_partA @ hole_pt + t_partA
        elbow_axis_A = unit(R_partA @ (R_A_zero.T @ (R_roll.T @ R_yp.T @ elbow_axis_s)))
        # elbow frame H (in A coords): y = elbow axis, x = limb direction ⊥ y
        yH = elbow_axis_A
        if yH @ np.array([0, -1.0, 0]) < 0:
            yH = -yH
        xH = unit(hinge_A - (hinge_A @ yH) * yH)
        zH = np.cross(xH, yH)
        H = np.c_[xH, yH, zH]
        limb_rec = dict(
            limb=limb, stud=int(stud_index), shoulder=c.tolist(), radial=radial.tolist(), pitchAxis=pitch_axis.tolist(),
            A=dict(part=a_name, partToBody=dict(R=R_partA.tolist(), t=t_partA.tolist()), hinge=hinge_A.tolist(), elbowAxis=yH.tolist(),
                   hingeRadius_mm=hole_r, elbowFrame=H.tolist(), elbowSkew_deg=skew,
                   registration=dict(constrained_inlier=regA["limbsA"][a_name]["inlier"], free_inlier=regB["A_free"][a_name]["inlier"], free_rms_mm=regB["A_free"][a_name]["rms"],
                                     socket_offset_free_vs_ball_mm=float(np.linalg.norm((RAs @ sock + tAs) - c)))),
            statue=dict(yaw=yaw_s, pitch=pitch_s, roll=roll_s, hingeAxisFlipped=flip_axis),
            B={},
        )
        for kind, fn in (("solid", solid_properties), ("shell", shell_properties)):
            m, cc, I, q = fn(vA_b, fA, 1.0)
            limb_rec["A"][kind] = dict(mass_per_density=m, com=cc.tolist(), inertia=I.tolist(), measure=q)
        limb_rec["A"]["collision"] = capsule_spheres(vA_b, 4)
        for b_name in b_names:
            RBs, tBs = statue_in_body(regB["B"][b_name]["R_free"], regB["B"][b_name]["t_free"])
            vB, fB = parts[b_name]
            # B statue pose relative to A statue pose, then express in A body (zero) coords via A's zero transform
            # x_torso_statue = RBs x + tBs ;  A zero: x_Abody = R_partA R_As⁻¹ (x_torso_statue − tAs) + t_partA
            M = R_partA @ RAs.T
            R_B_in_A = M @ RBs
            t_B_in_A = M @ (tBs - tAs) + t_partA
            pin_ax, pin_pt, pin_r = PINS[b_name]
            vB_A = vB @ R_B_in_A.T + t_B_in_A
            # foot: farthest region from the hinge along B's principal direction
            ctrB = vB_A.mean(0)
            dirB = unit(ctrB - hinge_A)
            proj = (vB_A - hinge_A) @ dirB
            tip = vB_A[proj > proj.max() - 2.5]
            fc, fr = fit_sphere(tip)
            fr = float(np.clip(fr, 1.2, 3.5))
            tip_pt = vB_A[np.argmax(proj)]
            foot_center = tip_pt - dirB * fr
            # statue elbow angle: rotation about yH taking xH to B's direction projected ⊥ yH
            dperp = unit(dirB - (dirB @ yH) * yH)
            elbow_s = signed_angle(xH, dperp, yH)
            # B body frame at elbow zero: rotate by −elbow_s about yH through the hinge, then express in H frame
            Rz = rodrigues(yH, -elbow_s)
            R_partB = H.T @ Rz @ R_B_in_A
            t_partB = H.T @ (Rz @ (t_B_in_A - hinge_A))
            vB_b = vB @ R_partB.T + t_partB
            fc_b = H.T @ (Rz @ (foot_center - hinge_A))
            rec = dict(part=b_name, partToBody=dict(R=R_partB.tolist(), t=t_partB.tolist()), statueElbow=elbow_s,
                       foot=dict(center=fc_b.tolist(), radius=fr), pinRadius_mm=pin_r,
                       registration=dict(hinge_inlier=regB["B"][b_name]["hinge_inlier"], free_inlier=regB["B"][b_name]["free_inlier"], free_rms_mm=regB["B"][b_name]["free_rms"],
                                         pin_vs_hole_axis_deg=regB["B"][b_name]["axis_mismatch_deg"], pin_vs_hole_offset_mm=regB["B"][b_name]["axis_offset_mm"]))
            for kind, fn in (("solid", solid_properties), ("shell", shell_properties)):
                m, cc, I, q = fn(vB_b, fB, 1.0)
                rec[kind] = dict(mass_per_density=m, com=cc.tolist(), inertia=I.tolist(), measure=q)
            rec["collision"] = capsule_spheres(vB_b, 4)
            limb_rec["B"][b_name] = rec
        robot["limbs"].append(limb_rec)
    # statue part poses (free registrations) in torso body coords, for the sculpture reference view
    robot["statuePose"] = {}
    for name, (R, t) in {"torso": (RT, tT), **{k: (v["R"], v["t"]) for k, v in regB["A_free"].items()}, **{k: (v["R_free"], v["t_free"]) for k, v in regB["B"].items()}}.items():
        Rb, tb = statue_in_body(R, t)
        robot["statuePose"][name] = dict(R=Rb.tolist(), t=tb.tolist())
    out = ROOT / "assets" / "robot.json"
    out.write_text(json.dumps(robot, indent=1))
    # ---- report ----
    print(f"stud plane rms {plane_rms:.3f} mm; carapace width max {wmax:.2f} / min {wmin:.2f} mm, thickness {thickness:.2f} mm; canon scale ×{robot['canon']['printToCanonScale']:.3f}")
    for L in robot["limbs"]:
        s = L["statue"]; A = L["A"]
        print(f"limb {L['limb']} stud {L['stud']}: statue yaw {np.degrees(s['yaw']):6.1f}° pitch {np.degrees(s['pitch']):6.1f}° roll {np.degrees(s['roll']):6.1f}°  elbow skew {A['elbowSkew_deg']:.1f}°  "
              f"|A| {np.linalg.norm(A['hinge']):.2f} mm  socket offset(free) {A['registration']['socket_offset_free_vs_ball_mm']:.2f} mm")
        for bn, B in L["B"].items():
            print(f"     {bn}: statue elbow {np.degrees(B['statueElbow']):7.1f}°  foot r {B['foot']['radius']:.2f} mm at {np.round(B['foot']['center'], 2)}  |B| {np.linalg.norm(B['foot']['center']):.2f} mm")
    print("wrote", out)


if __name__ == "__main__":
    main()
