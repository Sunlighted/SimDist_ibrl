import numpy as np

def load_ply(path):
    """Minimal 3DGS ply reader (standard INRIA field layout, as written by
    train_3dgs.py). Avoids the gaussian_renderer import, which pulls torch and
    trips the login node's thread limit."""
    with open(path, "rb") as f:
        names, count = [], 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(count * 4 * len(names)),
                             dtype=np.float32).reshape(count, len(names))
    col = {n: i for i, n in enumerate(names)}
    get = lambda *k: data[:, [col[x] for x in k]]

    class G:
        pass
    g = G()
    g.xyz = get("x", "y", "z").astype(np.float64)
    g.opacity = 1.0 / (1.0 + np.exp(-get("opacity")))          # stored as logit
    g.scale = np.exp(get("scale_0", "scale_1", "scale_2"))      # stored as log
    return g
