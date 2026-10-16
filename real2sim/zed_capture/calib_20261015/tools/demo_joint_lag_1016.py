import h5py, numpy as np, glob, json
def fo(u, a):
    y = np.empty_like(u); y[0] = u[0]
    for i in range(1, len(u)): y[i] = y[i-1] + a * (u[i] - y[i-1])
    return y
import sys; D = sys.argv[1].rstrip('/') + '/'
for fn in sorted(glob.glob(D + '*.h5')):
    f = h5py.File(fn, 'r'); t = f['time/steady'][:]; c = f['action/joint_positions'][:]; q = f['observation/joint_positions'][:]
    off = json.loads(f.attrs['gello_offset_rad'])
    dt = np.diff(t); fine = np.arange(t[0], t[-1], 0.002)
    out = []
    for j in range(7):
        r = np.interp(fine, t, q[:, j]); u = np.interp(fine, t, c[:, j])
        best = (1e9,)
        for tau in [0] + list(np.arange(0.02, 0.8, 0.02)):
            fu = u if tau == 0 else fo(u, 1 - np.exp(-0.002 / tau))
            for dd in np.arange(0, 0.4, 0.01):
                sh = np.interp(fine - dd, fine, fu); res = r - sh; o = np.median(res)
                e = np.sqrt(np.mean((res - o) ** 2))
                if e < best[0]: best = (e, dd, tau, o)
        raw = np.sqrt(np.mean((r - u - np.median(r - u)) ** 2))
        out.append(f'j{j+1}: d{best[1]*1e3:3.0f}+τ{best[2]*1e3:3.0f}ms rms {raw*1e3:4.1f}->{best[0]*1e3:3.1f} off {best[3]*1e3:+5.1f}')
    sp = np.abs(np.diff(c, axis=0)).max(0) / 0.1
    print(fn.split('/')[-1], f'dt {dt.mean()*1e3:.1f}±{dt.std()*1e3:.1f}ms', 'max|dcmd| rad/s', np.round(sp, 2))
    print('   ' + ' | '.join(out))
    print('   max |cmd-q| deg', np.round(np.degrees(np.abs(c - q).max(0)), 1), ' gello_offset', off)
