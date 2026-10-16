"""Fit cmd->measured joint models on the 10-15 reseed tracking runs.
models: pure delay d; first-order lag tau after delay d; report per moving joint."""
import numpy as np, sys
K = np.array([240,240,240,240,100,60,20.])
def fo(t, u, tau):  # first-order filter of a signal sampled uniformly
    if tau <= 0: return u.copy()
    a = 1 - np.exp(-(t[1]-t[0]) / tau); y = np.empty_like(u); y[0] = u[0]
    for i in range(1, len(u)): y[i] = y[i-1] + a * (u[i] - y[i-1])
    return y
for run in sys.argv[1:]:
    d = np.load(run + '/joints.npz'); jt, jq, ct, cq = d['joint_t'], d['joint_q'], d['cmd_t'], d['cmd_q']
    dt = 0.002; t = np.arange(max(jt[0], ct[0]) + 0.5, min(jt[-1], ct[-1]), dt)
    print(f'\n== {run.split("/")[-1]}  cmd {len(ct)/(ct[-1]-ct[0]):.0f} Hz, joints {len(jt)/(jt[-1]-jt[0]):.0f} Hz')
    rng = np.ptp(cq, 0)
    for j in np.argsort(-rng)[:4]:
        real = np.interp(t, jt, jq[:, j]); cmdu = np.interp(t, ct, cq[:, j])
        best = {}
        for tau in [0] + list(np.arange(0.02, 0.80, 0.02)):
            f = fo(t, cmdu, tau)
            for dd in np.arange(0, 0.40, 0.008):
                sh = np.interp(t - dd, t, f)
                r = real - sh; off = np.median(r); e = np.sqrt(np.mean((r - off) ** 2))
                key = 'delay' if tau == 0 else 'fo'
                if key not in best or e < best[key][0]: best[key] = (e, dd, tau, off)
        e0 = best['delay']; e1 = best['fo']
        # static error (cmd - real) while holding: torque in deadband
        err = cmdu - real
        still = np.abs(np.gradient(real, dt)) < 1e-3
        print(f' j{j+1} range {rng[j]*1e3:6.1f} mrad | pure delay {e0[1]*1e3:4.0f} ms rms {e0[0]*1e3:.2f} mrad'
              f' | delay {e1[1]*1e3:3.0f} ms + tau {e1[2]*1e3:3.0f} ms rms {e1[0]*1e3:.2f} mrad'
              f' | |cmd-q| when still p50/p95 {np.median(np.abs(err[still]))*1e3:.1f}/{np.percentile(np.abs(err[still]),95)*1e3:.1f} mrad'
              f' = {K[j]*np.percentile(np.abs(err[still]),95):.2f} Nm')
