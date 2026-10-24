import json, sys, os
runs = sys.argv[1:]
keys = ['nominal/robocasa/zero','nominal/isaac/zero','strength085/robocasa/zero','strength075/robocasa/zero','gain080/robocasa/zero','torso5kg/robocasa/zero','hw/robocasa/zero','hw/isaac/zero','hang/robocasa/zero']
lkeys = ['nominal/robocasa/zero','nominal/isaac/zero','hw/robocasa/zero','hw/isaac/zero']
for r in runs:
    p=f'logs/hw_stress_mujoco/HWAbl_{r}/summary.json'
    if not os.path.exists(p): continue
    c=json.load(open(p))['cells']; L=json.load(open(f'logs/hw_stress_mujoco/HWAbl_{r}_long/summary.json'))['cells']
    print(f"== {r}")
    for k in keys:
        k = k if k in c else k.replace('hw/', 'hw_no_gyro/')
        x=c[k]; print(f"  {k:28s} fell {x['fell_pct']:4.0f}% floor {x['ankle_floor_pct']:4.0f}% knee_q {x['knee_q'] and [round(v,2) for v in x['knee_q']]} imb {x['imbalance'] and round(x['imbalance'],2)}")
    for k in lkeys:
        k = k if k in L else k.replace('hw/', 'hw_no_gyro/')
        x=L[k]; print(f"  180s {k:24s} fell {x['fell_pct']:4.0f}% t {x['fall_t_median'] and round(x['fall_t_median'],1)} lstm {x['lstm_c_5s']:.2f}->{x['lstm_c_end']:.2f}")
