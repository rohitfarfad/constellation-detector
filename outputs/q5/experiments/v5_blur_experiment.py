import numpy as np
import cv2
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
            # Include elliptical blur to model directional PSFs at fixed geometry.
            kernels = [None]
            grid_y, grid_x = np.mgrid[-6:7, -6:7].astype(np.float32)
            for major, minor in [(.5, .5), (1., 1.), (1.5, 1.5), (1.5, .5), (2.5, .7)]:
                for angle in ([0] if major == minor else [0, 45, 90, 135]):
                    theta = np.deg2rad(angle)
                    u = grid_x*np.cos(theta) + grid_y*np.sin(theta)
                    v = -grid_x*np.sin(theta) + grid_y*np.cos(theta)
                    kernel = np.exp(-.5*((u/major)**2 + (v/minor)**2))
                    kernels.append(kernel/kernel.sum())
            for kernel in kernels:
                smooth = cv2.filter2D(source, -1, kernel) if kernel is not None else source
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

