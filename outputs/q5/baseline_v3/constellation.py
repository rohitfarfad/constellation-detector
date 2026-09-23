"""Classical Q5 matching. Used by ConstellationDetection.ipynb and the CLI."""
import argparse
import ast
import csv
import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np


def normalize(values):
    values = values - values.mean(axis=-1, keepdims=True)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-6)


def whiten(image):
    """Q3/Q4 Gaussian background removal with local contrast normalization."""
    signal = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .6)
    signal -= cv2.GaussianBlur(signal, (0, 0), 2)
    return signal / np.sqrt(9 + cv2.GaussianBlur(signal * signal, (0, 0), 3))


def dense_candidates(field, query):
    """Q4 correlation extended to rotation/scale; only valid central pixels."""
    query = whiten(query)
    candidates, scores = [], []
    for scale in np.arange(.7, 1.41, .1):
        width = int(round(22 * scale * .5))
        for angle in range(0, 360, 10):
            matrix = cv2.getRotationMatrix2D((15.5, 15.5), angle, scale * .5)
            matrix[:, 2] += (width - 1) / 2 - 15.5
            template = cv2.warpAffine(query, matrix, (width, width))
            response = cv2.matchTemplate(field, template, cv2.TM_CCOEFF_NORMED)
            for _ in range(10):
                _, score, _, (x, y) = cv2.minMaxLoc(response)
                theta = np.deg2rad(-angle)
                candidates.append([2*x + width - .5, 2*y + width - .5,
                                   scale*np.cos(theta), scale*np.sin(theta)])
                scores.append(score)
                response[max(0, y-5):y+6, max(0, x-5):x+6] = -1
    return np.float32(candidates), np.array(scores)


def sift_candidates(query, points, descriptors):
    """Lecture 2 descriptors propose transforms, including single-feature matches."""
    sift = cv2.SIFT_create(contrastThreshold=.005, edgeThreshold=15)
    candidates = []
    for blur in [0, .6, 1.0]:
        image = cv2.GaussianBlur(query, (0, 0), blur) if blur else query
        keys, desc = sift.detectAndCompute(cv2.resize(image, None, fx=3, fy=3), None)
        if desc is None or descriptors is None or len(descriptors) < 2:
            continue
        neighbors = cv2.BFMatcher().knnMatch(desc, descriptors, k=min(20, len(descriptors)))
        for key, matches in zip(keys, neighbors):
            target = points[[m.trainIdx for m in matches]]
            scale = target[:, 2] / key.size * 3
            angle = np.deg2rad(target[:, 3] - key.angle)
            a, b = scale*np.cos(angle), scale*np.sin(angle)
            dx, dy = (np.array([47.5, 47.5]) - key.pt) / 3
            x = target[:, 0] + a*dx - b*dy
            y = target[:, 1] + b*dx + a*dy
            candidates.extend(np.column_stack([x, y, a, b])[(scale > .55) & (scale < 1.8)])
    return np.float32(candidates).reshape(-1, 4)


def correlations(image, query, transforms):
    yy, xx = np.mgrid[-12:13, -12:13].astype(np.float32)
    mask = xx*xx + yy*yy <= 144
    x, y = xx[mask], yy[mask]
    target = normalize(cv2.remap(query, (x+15.5)[None], (y+15.5)[None],
                                 cv2.INTER_LINEAR).ravel())
    scores = []
    for block in np.array_split(transforms, max(1, len(transforms)//400)):
        cx, cy, a, b = block.T
        mx = cx[:, None] + a[:, None]*x - b[:, None]*y
        my = cy[:, None] + b[:, None]*x + a[:, None]*y
        crops = cv2.remap(image, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
        scores.extend(normalize(crops) @ target)
    return np.array(scores)


def refine(image, query, transform, texture=False):
    """Refine an affine warp by normalized correlation, then reject implausible fits."""
    if texture:
        original = transform
        left, top = original['origin']
        warp = np.float32(original['warp'])
        x, y = original['x'], original['y']
    else:
        x, y, a, b = transform
        left, top = int(x)-40, int(y)-40
        warp = np.float32([[a, -b, x-left-a*11.5+b*11.5],
                          [b, a, y-top-b*11.5-a*11.5]])
    if left < 0 or top < 0 or left+81 >= image.shape[1] or top+81 >= image.shape[0]:
        return None
    crop = image[top:top+81, left:left+81].astype(np.float32)
    source = whiten(crop) if texture else crop
    target = whiten(query)[4:28, 4:28] if texture else query[4:28, 4:28].astype(np.float32)
    try:
        score, warp = cv2.findTransformECC(target, source, warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60 if texture else 40, 1e-4),
            None, 3 if texture else 5)
    except cv2.error:
        return None
    scale = np.linalg.svd(warp[:, :2], compute_uv=False)
    center = warp @ np.array([11.5, 11.5, 1]) + [left, top]
    if (scale.min() < .5 or scale.max() > 2 or scale.max()/scale.min() > 1.3
            or np.linalg.det(warp[:, :2]) <= 0
            or np.linalg.norm(center-[x, y]) > (5 if texture else 10)):
        return None
    return dict(x=float(center[0]), y=float(center[1]), warp=warp.tolist(),
                origin=[left, top], ncc=float(score))


def match_features(image, query, match):
    """Q2 gain/illumination fit and independent fine-texture verification."""
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    interior = (xx-15.5)**2 + (yy-15.5)**2 < 14**2
    warp = np.float32(match['warp'])
    left, top = match['origin']
    source = image[top:top+81, left:left+81].astype(np.float32)
    query = query.astype(np.float32)
    mx = warp[0, 0]*(xx-4) + warp[0, 1]*(yy-4) + warp[0, 2]
    my = warp[1, 0]*(xx-4) + warp[1, 1]*(yy-4) + warp[1, 2]
    crop = cv2.remap(source, mx, my, cv2.INTER_LINEAR)
    texture = float(normalize(whiten(query)[interior]) @ normalize(whiten(crop)[interior]))
    ncc = float(normalize(query[interior]) @ normalize(crop[interior]))
    valid = interior & (query > 3) & (query < 252)
    if valid.sum() < 80:
        return None
    high = cv2.filter2D(query, -1, np.float32([[1,-2,1],[-2,4,-2],[1,-2,1]]))
    noise = max(1, float(np.median(np.abs(high[valid]))/.6745/6))
    plane = np.column_stack([np.ones(valid.sum()), xx[valid]-15.5, yy[valid]-15.5])
    best = (float('inf'), 0.)
    for blur in [0, .5, 1, 1.5]:
        softened = cv2.GaussianBlur(source, (0, 0), blur) if blur else source
        aligned = cv2.remap(softened, mx, my, cv2.INTER_LINEAR)
        design = np.column_stack([aligned[valid], plane])
        coefficients = np.linalg.lstsq(design, query[valid], rcond=None)[0]
        if coefficients[0] < .1:
            continue
        mse = float(np.mean((query[valid] - design @ coefficients)**2))
        if mse < best[0]:
            best = (mse, float(coefficients[0]))
    mse, gain = best
    if not np.isfinite(mse):
        return None
    match['mse'] = mse
    match['features'] = [float(np.arctanh(np.clip(texture, -.99, .99))),
        float(np.arctanh(np.clip(ncc, -.99, .999))), float(np.log1p(mse)),
        float(np.log1p(mse/(noise*noise+64))), float(abs(np.log(max(gain, .01))))]
    return match


def match_query(image, query, points, descriptors, field, coarse=None):
    candidates = sift_candidates(query, points, descriptors)
    candidates = candidates[(candidates[:, 0] > 16) & (candidates[:, 1] > 16)
        & (candidates[:, 0] < image.shape[1]-16) & (candidates[:, 1] < image.shape[0]-16)]
    selected, matches = [], []
    if len(candidates):
        smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .6)
        target = cv2.GaussianBlur(query.astype(np.float32), (0, 0), .6)
        for scores in [correlations(smooth, target, candidates),
                       correlations(whiten(image), whiten(query), candidates)]:
            count = 0
            for j in np.argsort(scores)[::-1]:
                if all(np.linalg.norm(candidates[j, :2]-candidates[k, :2]) > 3 for k in selected):
                    selected.append(j)
                    count += 1
                if count == 60:
                    break
        for j in selected:
            result = refine(image, query, candidates[j])
            if result is not None:
                matches.append(result)
                result = refine(image, query, result, texture=True)
                if result is not None:
                    matches.append(result)
    transforms, scores = dense_candidates(field, query) if coarse is None else coarse
    selected = []
    for j in np.argsort(scores)[::-1]:
        if all(np.linalg.norm(transforms[j, :2]-transforms[k, :2]) > 3 or
               abs(np.angle(complex(*transforms[j, 2:])/complex(*transforms[k, 2:]))) > .15
               for k in selected):
            selected.append(j)
        if len(selected) == 600:
            break
    for j in selected:
        result = refine(image, query, transforms[j])
        if result is not None:
            result = refine(image, query, result, texture=True)
            if result is not None:
                matches.append(result)
    matches = [r for m in matches if (r := match_features(image, query, m)) is not None]
    distinct = []
    for match in sorted(matches, key=lambda r: r['mse']):
        if all(np.hypot(match['x']-r['x'], match['y']-r['y']) > 8 for r in distinct):
            distinct.append(match)
        if len(distinct) == 12:
            break
    return distinct


def residual_evidence(image, paths, evidence):
    """Check clipped intensities and spatially coherent errors at cached image matches."""
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    interior = (xx-15.5)**2 + (yy-15.5)**2 < 14**2
    plane = np.stack([np.ones_like(xx), xx-15.5, yy-15.5], axis=-1)
    for path, matches in zip(paths, evidence):
        query = cv2.imread(str(path), 0).astype(np.float32)
        valid = interior & (query > 3) & (query < 252)
        for match in matches:
            warp = np.float32(match['warp'])
            left, top = match['origin']
            source = image[top:top+81, left:left+81].astype(np.float32)
            mx = warp[0, 0]*(xx-4) + warp[0, 1]*(yy-4) + warp[0, 2]
            my = warp[1, 0]*(xx-4) + warp[1, 1]*(yy-4) + warp[1, 2]
            best = float('inf')
            for blur in [0, .5, 1, 1.5]:
                smooth = cv2.GaussianBlur(source, (0, 0), blur) if blur else source
                aligned = cv2.remap(smooth, mx, my, cv2.INTER_LINEAR)
                design = np.concatenate([aligned[..., None], plane], axis=-1)
                coefficients = np.linalg.lstsq(design[valid], query[valid], rcond=None)[0]
                if coefficients[0] < .1:
                    continue
                residual = query - np.clip(design @ coefficients, 0, 255)
                mse = float(np.mean(residual[interior]**2))
                if mse < best:
                    best = mse
                    coherent = cv2.GaussianBlur(residual, (0, 0), 1)
                    features = [float(np.log1p(mse)),
                                float(np.log1p(np.mean(coherent[interior]**2)))]
            if not np.isfinite(best):
                raise ValueError('Cached match has no valid photometric fit')
            match['features'] = match['features'][:5] + features
    return evidence


def analyze_scene(folder, cache):
    """Cache image matching; recompute residual verification on every read."""
    folder, cache = Path(folder), Path(cache)
    paths = sorted((folder/'patches').glob('patch_*.png'))
    sky_paths = list(folder.glob('*_image.png'))
    if len(sky_paths) != 1 or not paths:
        raise ValueError(f'Expected one sky and query patches: {folder}')
    digest = hashlib.sha256()
    for path in [sky_paths[0], *paths]:
        digest.update(path.read_bytes())
    for function in [normalize, whiten, dense_candidates, sift_candidates, correlations,
                     refine, match_features, match_query]:
        digest.update(inspect.getsource(function).encode())
    digest.update((cv2.__version__ + np.__version__).encode())
    key = digest.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/f'{key}.json'
    image = cv2.imread(str(sky_paths[0]), cv2.IMREAD_GRAYSCALE)
    if target.exists():
        return residual_evidence(image, paths, json.loads(target.read_text()))
    keys, descriptors = cv2.SIFT_create(contrastThreshold=.005, edgeThreshold=15).detectAndCompute(image, None)
    points = np.array([(*k.pt, k.size, k.angle) for k in keys])
    field = cv2.resize(whiten(image), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    partial = cache/f'{key}.partial.json'
    results = json.loads(partial.read_text()) if partial.exists() else []
    if not isinstance(results, list) or len(results) > len(paths):
        raise ValueError(f'Invalid partial cache: {partial}')
    for path in paths[len(results):]:
        query = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if query.shape != (32, 32):
            raise ValueError(f'Expected a 32x32 query: {path}')
        results.append(match_query(image, query, points, descriptors, field))
        partial.write_text(json.dumps(results, allow_nan=False))
        print(f'{folder.name}: {len(results)}/{len(paths)} patches', flush=True)
    target.write_text(json.dumps(results, allow_nan=False))
    partial.unlink(missing_ok=True)
    return residual_evidence(image, paths, results)


def fit_calibration(features, labels):
    """Small regularized logistic calibration; no neural network or image classifier."""
    features, labels = np.array(features), np.array(labels)
    mean, std = features.mean(0), features.std(0) + 1e-6
    design = np.column_stack([np.ones(len(features)), (features-mean)/std])
    beta = np.zeros(design.shape[1])
    weights = np.where(labels, 1., .3)
    regularizer = np.diag([.01] + [2.]*(design.shape[1]-1))
    for _ in range(30):
        probability = 1/(1+np.exp(-np.clip(design@beta, -30, 30)))
        variance = weights*probability*(1-probability)
        step = np.linalg.solve((design.T*variance)@design + regularizer,
            design.T@(weights*(probability-labels)) + regularizer@beta)
        beta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return dict(mean=mean.tolist(), std=std.tolist(), beta=beta.tolist())


def candidate_probabilities(matches, calibration):
    if not matches:
        return np.array([])
    features = np.array([m['features'][:len(calibration['mean'])] for m in matches])
    design = np.column_stack([np.ones(len(matches)),
        (features-calibration['mean'])/calibration['std']])
    return 1/(1+np.exp(-np.clip(design@calibration['beta'], -30, 30)))


def calibration_data(rows, evidence):
    features, labels = [], []
    for row in rows:
        for i, matches in enumerate(evidence[row['Id']], 1):
            truth = ast.literal_eval(row[f'patch_{i:02}'])
            for match in matches:
                features.append(match['features'])
                labels.append(truth != -1 and np.hypot(match['x']-truth[0], match['y']-truth[1]) < 12)
    return features, labels


def read_patterns(folder):
    patterns = {}
    for path in sorted(Path(folder).glob('*_pattern.png')):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        mask = ((image[:, :, :3].min(2) > 180) & (image[:, :, 3] > 0)).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        peaks = (distance == cv2.dilate(distance, np.ones((3, 3), np.uint8))) & (distance >= 1.4)
        _, _, _, maxima = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
        nodes = []
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] >= 12:
                inside = [p for p in maxima[1:] if labels[int(round(p[1])), int(round(p[0]))] == i]
                kept = []
                for p in sorted(inside, key=lambda p: distance[int(round(p[1])), int(round(p[0]))],
                                reverse=True):
                    if all(np.linalg.norm(p-q) > 6 for q in kept):
                        kept.append(p)
                nodes.extend(kept if len(kept) > 1 else [centers[i]])
        patterns[path.name.removesuffix('_pattern.png')] = np.array(nodes)
    return patterns


def geometric_matches(nodes, points, owners, matrix, tolerance=24):
    if not len(points):
        return np.array([], int), np.array([], int)
    moved = nodes @ matrix[:, :2].T + matrix[:, 2]
    distances = np.linalg.norm(moved[:, None] - points[None], axis=2)
    source, target, used_queries = [], [], set()
    for flat in np.argsort(distances, axis=None):
        i, j = np.unravel_index(flat, distances.shape)
        if distances[i, j] > tolerance:
            break
        if i not in source and owners[j] not in used_queries:
            source.append(i)
            target.append(j)
            used_queries.add(owners[j])
    return np.array(source, int), np.array(target, int)


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
                if len(src) < 5:
                    break
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


def verify_constellations(ranking, patterns, image):
    """Verify unissued reference nodes against sky blobs and nearby control locations."""
    image = image.astype(np.float32)
    blobs = cv2.GaussianBlur(image, (0, 0), 3) - cv2.GaussianBlur(image, (0, 0), 8)
    peaks = cv2.dilate(blobs, np.ones((41, 41), np.uint8))
    offsets = [(80, 0), (-80, 0), (0, 80), (0, -80),
               (60, 60), (60, -60), (-60, 60), (-60, -60)]
    for model in ranking:
        matrix = np.array(model['matrix'])
        nodes = patterns[model['name']]
        points = nodes @ matrix[:, :2].T + matrix[:, 2]
        valid = (points[:, 0] > 100) & (points[:, 0] < image.shape[1]-100)
        valid &= (points[:, 1] > 100) & (points[:, 1] < image.shape[0]-100)
        points = np.rint(points[valid]).astype(int)
        ratio = 0.
        if len(points):
            signal = np.maximum(0, peaks[points[:, 1], points[:, 0]])
            control = np.array([peaks[points[:, 1]+dy, points[:, 0]+dx]
                                for dx, dy in offsets]).mean(0)
            ratio = float(np.exp(np.mean(np.log((signal+4)/(np.maximum(0, control)+4)))))
        model['query_score'] = model['score']
        model['sky_ratio'] = ratio
        # Discount the two seed pairs; penalize loose fits, including their non-seed points.
        independent = max(0, model['support']-2)/model['support']
        coherence = np.exp(-.5*(model['rmse']/6)**2)
        model['score'] *= independent * ratio * coherence / np.sqrt(len(nodes))
    return sorted(ranking, key=lambda r: r['score'], reverse=True)


def predict_scene(evidence, calibration, patterns, image, threshold=.15):
    probabilities = [candidate_probabilities(matches, calibration) for matches in evidence]
    gray = image.astype(np.float32)
    stars = np.maximum.reduce([cv2.GaussianBlur(gray, (0, 0), s)
                               - cv2.GaussianBlur(gray, (0, 0), 6*s) for s in [2., 6., 18.]])
    points, owners, confidence, alternatives = [], [], [], []
    for i, (matches, probs) in enumerate(zip(evidence, probabilities)):
        if not len(probs):
            continue
        for j in np.argsort(probs)[::-1][:6]:
            x, y = matches[j]['x'], matches[j]['y']
            strength = stars[round(y), round(x)]
            if probs[j] >= max(.001, .05*probs.max()) and strength > 25:
                points.append([x, y])
                owners.append(i)
                weight = max(probs[j], .05*probs[j]/probs.max())
                confidence.append(float(weight*np.clip(strength/70, 0, 1)**4))
                alternatives.append(int(j))
    ranking = identify_constellation(points, owners, confidence, patterns)
    ranking = verify_constellations(ranking, patterns, image)
    winner = ranking[0] if ranking else None
    chosen = {}
    if winner and winner['support'] >= 3:
        # Broader alternatives help identify the figure; localize using reliable image matches.
        points, owners, alternatives = [], [], []
        for i, (matches, probs) in enumerate(zip(evidence, probabilities)):
            for j in np.argsort(probs)[::-1][:2]:
                if probs[j] >= max(.04, .9*probs.max()):
                    points.append([matches[j]['x'], matches[j]['y']])
                    owners.append(i)
                    alternatives.append(int(j))
        # Schematics have larger geometric errors than the pixel-localization target.
        # A wider final association only selects existing verified image candidates.
        _, associated = geometric_matches(patterns[winner['name']], np.array(points),
            np.array(owners), np.array(winner['matrix']), tolerance=48)
        for j in associated:
            owner, alternative = owners[j], alternatives[j]
            if probabilities[owner][alternative] >= max(.04, .9*probabilities[owner].max()):
                chosen[owner] = alternative
    prediction = []
    for i, (matches, probs) in enumerate(zip(evidence, probabilities)):
        if not len(probs):
            prediction.append(-1)
            continue
        j = chosen.get(i, int(probs.argmax()))
        present = probs[j] >= threshold or (i in chosen and probs[j] >= .04)
        match = matches[j]
        prediction.append([round(match['x'], 2), round(match['y'], 2), int(i in chosen)] if present else -1)
    return dict(patches=prediction, constellation=winner['name'] if chosen else 'unknown',
                ranking=ranking, probabilities=[p.tolist() for p in probabilities])


def score_scene(row, prediction):
    """Handout-based local scorer; m/ties/empty-class conventions are not official."""
    truth = [ast.literal_eval(row[f'patch_{i:02}']) for i in range(1, int(row['n_patches'])+1)]
    guessed = prediction['patches']
    actual = np.array([p != -1 for p in truth])
    present = np.array([p != -1 for p in guessed])
    f1 = []
    for label in [False, True]:
        tp = np.sum((actual == label) & (present == label))
        denominator = np.sum(actual == label) + np.sum(present == label)
        f1.append(2*tp/denominator if denominator else 1.)
    def reward(distance):
        return float(np.clip((36-distance)/24, 0, 1))
    location = [reward(np.linalg.norm(np.array(p[:2])-q[:2])) if q != -1 else 0.
                for p, q in zip(truth, guessed) if p != -1]
    figures = np.array([p[:2] for p in truth if p != -1 and p[2] == 1])
    points = np.array([p[:2] for p in guessed if p != -1])
    geometry = 0.
    if len(figures) and len(points):
        distances = np.linalg.norm(figures[:, None]-points[None], axis=2)
        used_a, used_b = set(), set()
        for flat in np.argsort(distances, axis=None, kind='stable'):
            i, j = np.unravel_index(flat, distances.shape)
            if i not in used_a and j not in used_b:
                geometry += reward(distances[i, j])
                used_a.add(i)
                used_b.add(j)
        geometry /= len(figures)
    components = [float(np.mean(f1)), float(np.mean(location)) if location else 0.,
                  geometry, float(prediction['constellation'] == row['constellation'])]
    return dict(zip(['Presence', 'Localization', 'GeometricRecovery', 'Identification', 'Score'],
                    components + [float(np.dot(components, [.25, .20, .25, .30]))]))


def write_submission(sample, predictions, output, patterns):
    with Path(sample).open(newline='') as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    expected = ['Id', 'n_patches'] + [f'patch_{i:02}' for i in range(1, 88)] + ['constellation']
    assert columns == expected and set(predictions) == {r['Id'] for r in rows}
    assert len(rows) == len({r['Id'] for r in rows}), 'Duplicate scene IDs'
    for row in rows:
        result = predictions[row['Id']]
        assert len(result['patches']) == int(row['n_patches'])
        assert result['constellation'] in set(patterns) | {'unknown'}
        row['constellation'] = result['constellation']
        for i in range(1, 88):
            value = result['patches'][i-1] if i <= int(row['n_patches']) else -1
            if value != -1:
                x, y, membership = value
                assert np.isfinite(x+y) and 0 <= x < 3000 and 0 <= y < 3000 and membership in (0, 1)
                row[f'patch_{i:02}'] = f'({x:.2f}, {y:.2f}, {membership})'
            else:
                row[f'patch_{i:02}'] = '-1'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with output.open(newline='') as handle:
        written = list(csv.DictReader(handle))
    assert written == rows
    return output


def run(data='Data', output='outputs/q5/submission.csv', cache='outputs/q5/cache', calibration_file=None):
    data = Path(data)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(7)
    if calibration_file:
        calibration = json.loads(Path(calibration_file).read_text())
    else:
        rows = list(csv.DictReader((data/'train_ground_truth.csv').open()))
        evidence = {row['Id']: analyze_scene(data/'train'/row['Id'], cache) for row in rows}
        calibration = fit_calibration(*calibration_data(rows, evidence))
    patterns = read_patterns(data/'patterns')
    sample = list(csv.DictReader((data/'sample_submission.csv').open()))
    predictions = {row['Id']: predict_scene(analyze_scene(data/'validation'/row['Id'], cache),
                   calibration, patterns, cv2.imread(str(data/'validation'/row['Id']/
                   f"{row['Id']}_image.png"), cv2.IMREAD_GRAYSCALE)) for row in sample}
    write_submission(data/'sample_submission.csv', predictions, output, patterns)
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='Data')
    parser.add_argument('--output', default='outputs/q5/submission.csv')
    parser.add_argument('--cache', default='outputs/q5/cache')
    parser.add_argument('--calibration', help='Saved public-training calibration; no training folder needed')
    args = parser.parse_args()
    run(args.data, args.output, args.cache, args.calibration)
