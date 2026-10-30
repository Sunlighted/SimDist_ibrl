import numpy as np, sys, json
def analyse(L):
    ring = L["ring_all"]                       # T x n x 3
    c = ring.mean(1)
    nrm = []
    for t in range(len(ring)):
        u, s, vt = np.linalg.svd(ring[t] - c[t]); nrm.append(vt[2])
    nrm = np.array(nrm)
    X = L["needle_xmat"]; sz = L["needle_size"]
    ax = int(np.argmax(sz)); half = sz[ax]
    a = X[:, :, ax]                            # shaft axis, world
    cen = L["needle_tip"]                      # geom centre (named needle_tip in my log)
    # orient axis toward the tip: away from the handle (body origin)
    s_ = np.sign(np.einsum("ij,ij->i", a, cen - L["needle"])); a = a * s_[:, None]
    tip = cen + a * half
    # orient ring normal along the needle travel (same side as axis)
    nrm = nrm * np.sign(np.einsum("ij,ij->i", nrm, a))[:, None]
    depth = np.einsum("ij,ij->i", tip - c, nrm)       # how far the tip is past the ring plane [m]
    # lateral offset of the shaft line at the ring plane
    lam = np.einsum("ij,ij->i", c - cen, nrm) / np.maximum(np.einsum("ij,ij->i", a, nrm), 1e-6)
    hit = cen + a * lam[:, None]
    lat = np.linalg.norm(hit - c, axis=1)
    centre_dist = np.linalg.norm(cen - c, axis=1)
    return dict(depth=depth, lat=lat, centre=centre_dist, half=half, tcp_err=L["tcp_err"])
if __name__ == "__main__":
    V = sys.argv[1]
    for i in range(6):
        L = np.load(f"{V}/demo_{i}_log.npz"); r = analyse(L); n = len(r["depth"])
        ins = (r["depth"] > 0) & (r["lat"] < 0.012)
        k = int(np.argmax(r["depth"] * ins)) if ins.any() else int(np.argmax(r["depth"]))
        print(f"demo_{i}: shaft half-length {1e3*r['half']:.0f} mm | max depth past ring {1e3*r['depth'][k]:6.1f} mm (lat {1e3*r['lat'][k]:4.1f}) at t={k}/{n} "
              f"| final depth {1e3*r['depth'][-1]:6.1f} lat {1e3*r['lat'][-1]:5.1f} | success needs centre<12: final centre {1e3*r['centre'][-1]:5.1f} "
              f"| tcp err last 50 steps median {1e3*np.median(r['tcp_err'][-70:-20]):5.1f} max {1e3*np.max(r['tcp_err']):5.1f} at t={int(np.argmax(r['tcp_err']))}")
