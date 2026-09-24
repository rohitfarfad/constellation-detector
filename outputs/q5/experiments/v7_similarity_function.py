import numpy as np
import cv2
from constellation import geometric_matches
def identify_constellation(points, owners, probabilities, patterns, shape=(3000, 3000)):
    """Lecture 3 two-pair hypotheses, one-to-one consensus, and least-squares refit."""
    if len(set(owners)) < 3:
        return []
    points, owners, probabilities = np.array(points), np.array(owners), np.array(probabilities)
    # A coarse Voronoi map accelerates candidate scoring; finalists use exact distances.
    cells = np.rint(points/4).astype(int)
    representatives = {}
    for j in np.argsort(probabilities):
        representatives[tuple(cells[j])] = int(j)
    ordered = sorted(representatives, key=lambda p: (p[1], p[0]))
    indices = np.array([representatives[p] for p in ordered])
    mask = np.ones((shape[0]//4+1, shape[1]//4+1), np.uint8)
    for x, y in ordered:
        mask[y, x] = 0
    _, labels = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lookup = indices[labels-1]
    qi, qj = np.triu_indices(len(points), 1)
    complex_points = points[:, 0] + 1j*points[:, 1]
    valid = (np.abs(complex_points[qi]-complex_points[qj]) > 80) & (owners[qi] != owners[qj])
    qi, qj = qi[valid], qj[valid]
    if not len(qi):
        return []
    first = np.concatenate([complex_points[qi], complex_points[qj]])
    second = np.concatenate([complex_points[qj], complex_points[qi]])
    ranking = []
    for name, nodes in patterns.items():
        if len(nodes) < 3:
            continue  # Two nodes cannot distinguish identities under a similarity transform.
        proposals = []
        for reflection in [1, -1]:
            source = nodes * [reflection, 1]
            z = source[:, 0] + 1j*source[:, 1]
            pi, pj = np.triu_indices(len(z), 1)
            long = np.abs(z[pi]-z[pj]) > .25*np.ptp(source, axis=0).max()
            for i, j in zip(pi[long], pj[long]):
                a = (second-first)/(z[j]-z[i])
                b = first-a*z[i]
                moved = a[:, None]*z + b[:, None]
                x = np.clip(np.rint(moved.real/4).astype(int), 0, lookup.shape[1]-1)
                y = np.clip(np.rint(moved.imag/4).astype(int), 0, lookup.shape[0]-1)
                nearest = lookup[y, x]
                distance = np.abs(moved-complex_points[nearest])
                reward = np.maximum(0, 1-(distance/24)**2) * probabilities[nearest]**.25
                query_ids = owners[nearest]
                for k in range(len(nodes)):
                    same = query_ids == query_ids[:, k, None]
                    better = reward > reward[:, k, None]
                    tied = (reward == reward[:, k, None]) & (np.arange(len(nodes)) < k)
                    reward[np.any(same & (better | tied), axis=1), k] = 0
                scores = reward.sum(1)
                for k in np.argsort(scores)[-2:]:
                    if len(proposals) < 6 or scores[k] > proposals[-1][0]:
                        matrix = np.array([[a[k].real*reflection, -a[k].imag, b[k].real],
                                           [a[k].imag*reflection, a[k].real, b[k].imag]])
                        proposals.append((float(scores[k]), matrix))
                        proposals.sort(key=lambda p: p[0], reverse=True)
                        proposals = proposals[:6]
        finalists = []
        for _, matrix in proposals:
            for _ in range(3):
                src, dst = geometric_matches(nodes, points, owners, matrix)
                if len(src) < 3:
                    break
                if len(src) < 5:
                    reflection = np.sign(np.linalg.det(matrix[:, :2]))
                    z = reflection*nodes[src, 0] + 1j*nodes[src, 1]
                    w = points[dst, 0] + 1j*points[dst, 1]
                    centered = z-z.mean()
                    a = np.vdot(centered, w-w.mean()) / np.vdot(centered, centered)
                    b = w.mean()-a*z.mean()
                    candidate = np.array([[a.real*reflection, -a.imag, b.real],
                                          [a.imag*reflection, a.real, b.imag]])
                else:
                    candidate = np.linalg.lstsq(np.column_stack([nodes[src], np.ones(len(src))]),
                                                 points[dst], rcond=None)[0].T
                scales = np.linalg.svd(candidate[:, :2], compute_uv=False)
                if scales[-1] < .3 or scales[0]/scales[-1] > 3:
                    break
                matrix = candidate
            src, dst = geometric_matches(nodes, points, owners, matrix)
            if len(src) < 3:
                continue
            errors = np.linalg.norm(nodes[src]@matrix[:, :2].T + matrix[:, 2] - points[dst], axis=1)
            score = float(np.sum(np.maximum(0, 1-(errors/24)**2)*probabilities[dst]**.25))
            finalists.append(dict(name=name, score=score, support=len(src),
                error=float(np.median(errors)), rmse=float(np.sqrt(np.mean(errors**2))),
                matrix=matrix.tolist(),
                point_indices=dst.tolist(), node_indices=src.tolist()))
        if finalists:
            ranking.append(max(finalists, key=lambda r: r['score']))
    return sorted(ranking, key=lambda r: r['score'], reverse=True)

