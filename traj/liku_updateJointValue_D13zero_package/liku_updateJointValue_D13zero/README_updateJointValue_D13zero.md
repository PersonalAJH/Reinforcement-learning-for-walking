# LIKU trajectory: updateJointValue exact, D13 zero

## Output
- `traj_updateJointValue_D13zero_60hz_rad.pt`
- `traj_updateJointValue_D13zero_60hz_rad.csv`
- `traj_updateJointValue_D13zero_21cols_no_header.csv`
- Shape: `(5121, 21)`
- Unit: radians
- Target Hz: `60.0`
- Duration / recommended `zmp_cycle_time`: `85.328` seconds

## Joint order
```python
['D13', 'F19', 'E15', 'F20', 'E16', 'E17', 'F21', 'F22', 'E18', 'A1', 'B7', 'B8', 'A2', 'B9', 'A3', 'A5', 'A4', 'B10', 'B11', 'B12', 'A6']
```

## Important
This version goes back to the current C++ `trMotionSystemV3::updateJointValue()` rule.
It is **not** the `nooffset_F19flip` variant.

The raw log values are interpreted as absolute motor write angles around 180 deg, so the status value entering `updateJointValue()` is:

```text
status_deg = raw_joint_write_deg - 180.0
```

Then the current C++ signs, offsets, and positive scaling are applied.
`D13` is fixed to `0.0 rad` as requested.

## Applied mapping
```text
D13 = 0.0

A1  = +(ID1  - 180) * deg2rad       # RHY
A2  = -(ID2  - 180) * deg2rad       # RHR
A3  = +(ID3  - 180) * deg2rad       # RHP
A4  = +(ID4  - 180) * deg2rad       # RKP
A5  = -(ID5  - 180) * deg2rad       # RAP
A6  = +(ID6  - 180) * deg2rad       # RAR

B7  = -(ID7  - 180) * deg2rad       # LHY
B8  = +(ID8  - 180) * deg2rad       # LHR
B9  = -(ID9  - 180) * deg2rad       # LHP
B10 = -(ID10 - 180) * deg2rad       # LKP
B11 = -(ID11 - 180) * deg2rad       # LAP
B12 = +(ID12 - 180) * deg2rad       # LAR

E15 = -(ID15 - 180) * deg2rad       # RSP
E16 = ((ID16 - 180) - 10) * deg2rad # RSR
E17 = +(ID17 - 180) * deg2rad       # REY, then *1.1 if > 0
E18 = -(ID18 - 180) * deg2rad       # RER

F19 = -(ID19 - 180) * deg2rad       # LSP
F20 = ((ID20 - 180) + 10) * deg2rad # LSR
F21 = -(ID21 - 180) * deg2rad       # LEY, then *1.1 if > 0
F22 = +(ID22 - 180) * deg2rad       # LER
```

## IsaacLab config
```python
zmp_traj_path: str = "D:/IsaacLab/traj/traj_updateJointValue_D13zero_60hz_rad.pt"
zmp_traj_in_degrees: bool = False
zmp_cycle_time: float = 85.328
```
