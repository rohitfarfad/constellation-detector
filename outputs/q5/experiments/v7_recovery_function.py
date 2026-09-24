import numpy as np
import cv2
from constellation import independent_transform
def recover_figure_queries(prediction, evidence, probabilities, nodes, points, owners, model):
    """Recover rejected queries using geometry fitted without their own candidates."""
    if model['support'] < 6:
        return []
    cv2.setRNGSeed(7)
    source, target = np.array(model['node_indices']), np.array(model['point_indices'])
    options = []
    for i, (matches, probs) in enumerate(zip(evidence, probabilities)):
        if prediction[i] != -1 or not len(probs):
            continue
        keep = owners[target] != i
        result = independent_transform(nodes[source[keep]], points[target[keep]])
        if result is None:
            continue
        matrix, support, error = result
        denominator = np.column_stack([nodes, np.ones(len(nodes))]) @ matrix[2]
        if (denominator.min()*denominator.max() <= 0
                or np.max(np.abs(denominator))/np.min(np.abs(denominator)) > 3):
            continue
        projected = cv2.perspectiveTransform(nodes[None], matrix)[0]
        for j, match in enumerate(matches):
            distances = np.linalg.norm(projected-[match['x'], match['y']], axis=1)
            node = int(distances.argmin())
            unclaimed = all(p == -1 or np.linalg.norm(projected[node]-p[:2]) > 12
                            for p in prediction)
            precise = support >= 8 and error <= 6 and distances[node] <= 2 and unclaimed
            floor = .001 if precise else .01
            if distances[node] <= 6 and probs[j] >= max(floor, .5*probs.max()):
                score = float(np.log(probs[j]) - .5*(distances[node]/6)**2)
                options.append((score, i, j, node, float(distances[node]), support, error))
    used_queries, used_nodes, recovered = set(), set(), []
    for _, i, j, node, distance, support, error in sorted(options, reverse=True):
        if i in used_queries or node in used_nodes:
            continue
        used_queries.add(i)
        used_nodes.add(node)
        match = evidence[i][j]
        prediction[i] = [round(match['x'], 2), round(match['y'], 2), 1]
        recovered.append(dict(query=i+1, probability=float(probabilities[i][j]),
                              distance=distance, independent_support=support, loo_max=error))
    return recovered

