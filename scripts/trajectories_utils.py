import random, bisect
import numpy as np

def pick_random_starting_pixel(graph):
    """
    Picks a random starting pixel from the graph, weighted by the volume of the edges.
    """
    edges = graph["links"]
    # Create weights for edge selection based on volume
    weights = [e['volume'] for e in edges]
    total_flow = sum(weights)
    cum_weights = np.cumsum(weights)
    r = random.uniform(0, total_flow)
    # Select an edge based on the cumulative weights
    edge_idx = bisect.bisect_right(cum_weights, r)
    edge_idx = min(edge_idx, len(edges) - 1)
    edge = edges[edge_idx]
    pixels = edge["path"]
    # Pick a random pixel from the edge's path
    pixel_idx = random.randrange(len(pixels))
    return pixel_idx, edge_idx, edge

def choose_downstream_edge(graph, edge_id):
    """
    Chooses a downstream edge based on the flow from the current edge.
    If there are multiple downstream edges, it selects one randomly weighted by flow.
    """
    out_edges = downstream_edges(graph, edge_id)
    if not out_edges:
        return None # Important - reached the end of the graph!
    if len(out_edges) == 1:
        return out_edges[0]
    
    # If there are multiple downstream edges, select one based on flow
    edges = graph["links"]
    weights = [edges[e]['flow'] for e in out_edges]
    total = sum(weights)
    r = random.uniform(0, total)
    cum = np.cumsum(weights)
    idx = bisect.bisect_right(cum, r)
    idx = min(idx, len(out_edges) - 1)
    return out_edges[idx]

def downstream_edges(graph, edge_id):
    """Finds downstream edges from the given edge_id."""
    links = graph['links']
    e = links[edge_id] # It is necessary that edge_id corresponds to the index in the list (hence the graph preprocessing and cleaning)

    # Flow orientation: if direction is 1, flow goes from source to target, if -1, flow goes from target to source
    exit_node = e['target'] if e['direction'] == 1 else e['source']
    out_edges = []
    # The same applies for neighboring edges (source is the real source if direction is 1, target if -1)
    for neighbor_eid in graph['nodes'][exit_node]['branches_ids']:
        if int(neighbor_eid) == int(edge_id): continue
        l = links[neighbor_eid]
        # CHeck if the node corresponds
        if (l['source'] == exit_node and l['direction'] == 1) or (l['target'] == exit_node and l['direction'] == -1):
            out_edges.append(neighbor_eid)
    return out_edges

def get_particle_path_within_edge_subpixel(edge, start_pos, start_ref_idx, time_step, velocity = None):
    """
    Gets one particle step within a single edge with subpixel accuracy.
    """
    if velocity is None:
        velocity = edge['velocity']
    if velocity <= 0:
        return False, start_pos, start_ref_idx, 0
    distance_to_travel = velocity * time_step
    path = [np.array(p) for p in edge['path']]
    step = 1 if edge['direction'] == 1 else -1
    end_idx = len(path) - 1 if edge['direction'] == 1 else 0
    current_pos = np.array(start_pos) # Does not have to be on the path - so that it does not skip nodes
    current_ref_idx = start_ref_idx

    # Steps pixel by pixel (or subpixel by subpixel)
    # Loop continues until particle reaches edge end or time step is fully used
    while distance_to_travel > 0:
        # If step is over the edge end
        if current_ref_idx == end_idx:
            return True, current_pos, current_ref_idx, distance_to_travel / velocity
        # Next pixel = next ref point
        next_ref_idx = current_ref_idx + step
        target_point = path[next_ref_idx]
        vec_to_target = target_point - current_pos
        dist_to_target = np.linalg.norm(vec_to_target)

        # Subpix accuracy - if the reference point is further than the distance to travel, interpolation occurs between current and reference
        if distance_to_travel <= dist_to_target:
            unit_vec = vec_to_target / dist_to_target
            return False, current_pos + unit_vec * distance_to_travel, current_ref_idx, 0
        
        # If the particle reaches the next pixel in the trajectory, it becomes the current position (loop continues)
        distance_to_travel -= dist_to_target
        current_pos = target_point
        current_ref_idx = next_ref_idx

def move_particle_subpixel(graph, edge_idx, current_pos, ref_idx, time):
    """
    One particle step with subpixel accuracy based on remaining time and velocity.
    """
    rem_time = time
    c_edge_idx = edge_idx
    c_pos = current_pos
    c_ref_idx = ref_idx
    
    while rem_time > 0:
        edge = graph['links'][c_edge_idx]
        velocity = edge['velocity']
        # Add velocty variability to simulate variability within the same vessel
        velocity_multiplier = random.uniform(0.9, 1.1)
        velocity *= velocity_multiplier
        finished, c_pos, c_ref_idx, rem_time = get_particle_path_within_edge_subpixel(
            edge, c_pos, c_ref_idx, rem_time, velocity
        )
        
        if finished:
            next_eid = choose_downstream_edge(graph, c_edge_idx)
            if next_eid is None: # Particle left the graph!
                return None
            # Move to the next edge
            c_edge_idx = next_eid
            next_edge = graph['links'][c_edge_idx]
            # Set the starting point in the new edge - we want it to move gradually to the first pixel of the new edge, not jump over the node (-1 because next ref pixel idx is c_ref_idx + 1)
            if next_edge['direction'] == 1:
                c_ref_idx = -1
            else:
                c_ref_idx = len(next_edge['path'])
        else:
            break
    return c_edge_idx, c_pos, c_ref_idx


def get_particle_path(graph, time, delta_time):
    """
    Simulates the path of a single particle through the graph for a given time and time step.
    Returns the path points and whether the particle left the graph.
    """
    # Find starting position
    start = pick_random_starting_pixel(graph)
    p_idx, e_idx, e_data = start
    c_pos = np.array(e_data['path'][p_idx])
    c_edge_idx = e_idx
    c_ref_idx = p_idx
    
    path_points = [c_pos.tolist()]
    left_graph = False

    # Simulate time steps and save points
    for _ in range(int(time / delta_time)):
        result = move_particle_subpixel(graph, c_edge_idx, c_pos, c_ref_idx, delta_time)
        if result is None:
            left_graph = True
            break
        new_edge_idx, c_pos, c_ref_idx = result
        if new_edge_idx != c_edge_idx:
            c_edge_idx = new_edge_idx
        path_points.append(c_pos.tolist())
    return np.array(path_points), left_graph

def get_D_tensor_from_eigvec(v1, lambdas = [1600, 250, 250]):
    """Constructs a diffusion tensor from a principal eigenvector and eigenvalues."""
    v1 /= np.linalg.norm(v1)
    tmp = np.random.randn(3)
    v2 = tmp - np.dot(tmp, v1) * v1
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    R = np.column_stack((v1, v2, v3))
    D_tensor = R @ np.diag(lambdas) @ R.T
    return D_tensor