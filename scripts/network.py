import json
import os
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod

# Functions for pressure boundary conditions

class PressureStrategy(ABC):
    """Abstract base class for different strategies to assign pressure boundary conditions in the vessel network."""
    @abstractmethod
    def get_boundary_pressures(self, G, node_data, boundary_nodes, pressure_diff):
        pass

    def _find_thickest_source(self, G, boundary_nodes):
        max_radius = -1
        source_node = None
        for n in boundary_nodes:
            edges = list(G.edges(n, data=True))
            if edges:
                r = edges[0][2]['radius']
                if r > max_radius:
                    max_radius = r
                    source_node = n
        return source_node
    

class RadialDistance(PressureStrategy):
    """Assigns boundary pressures based on the radial distance from the main source node, with pressure decreasing as distance increases."""
    def get_boundary_pressures(self, G, node_data, boundary_nodes, pressure_diff):
        source_node = self._find_thickest_source(G, boundary_nodes)
        source_coords = np.array(node_data[source_node]["center"])
        dists = {}
        for n in boundary_nodes:
            target_coords = np.array(node_data[n]["center"])
            # Eucledian distance
            dist = np.linalg.norm(target_coords - source_coords)
            dists[n] = dist
        max_reach = max(dists.values())
        boundary_pressures = {}
        for n in boundary_nodes:
            dist = dists[n]
            norm_dist = dist / max_reach
            # Pressure decrase based on distance from main source
            boundary_pressures[n] = pressure_diff * (1.0 - norm_dist)
        return boundary_pressures

class TopologicalDistance(PressureStrategy):
    """Assigns boundary pressures based on the topological distance from the main source node weighted by resistance, with pressure decreasing as distance increases."""
    def get_boundary_pressures(self, G, node_data, boundary_nodes, pressure_diff):
        source_node = self._find_thickest_source(G, boundary_nodes)
        topo_distances = nx.single_source_dijkstra_path_length(G, source_node, weight='res')
        print(topo_distances)
        true_max_reach = max(topo_distances.values())
        print(f"Max topological reach from source node {source_node}: {true_max_reach:.4f}")
        boundary_pressures = {}
        for n in boundary_nodes:
            dist = topo_distances[n]
            # Linear decrease based on topological distance from main source
            norm_dist = dist / true_max_reach
            boundary_pressures[n] = pressure_diff * ((1.0 - norm_dist))
        return boundary_pressures

class RadiusBased(PressureStrategy):
    """Assigns boundary pressures based on the radius of the connected edge, with thicker edges having input pressure = pressure difference and thinner edges having output pressure = 0."""
    def __init__(self, in_out_ratio=0.4):
        self.in_out_ratio = in_out_ratio

    def get_boundary_pressures(self, G, node_data, boundary_nodes, pressure_diff):
        boundary_with_radii = []
        # Identify inputs and outputs based on radius and in_out_ratio
        for n in boundary_nodes:
            _, _, edge_data = list(G.edges(n, data=True))[0]
            boundary_with_radii.append((n, edge_data['radius']))
        boundary_with_radii.sort(key=lambda x: x[1], reverse=True)
        num_inlets = int(len(boundary_with_radii) * self.in_out_ratio)
        inlets = [n for n, r in boundary_with_radii[:num_inlets]]
        boundary_pressures = {}
        for n in boundary_nodes:
            if n in inlets:
                boundary_pressures[n] = pressure_diff
            else:
                boundary_pressures[n] = 0.0
        return boundary_pressures

class LinearGradient(PressureStrategy):
    """Assigns boundary pressures based on a linear gradient along the main axis of the tissue, with pressure decreasing from one side to the other."""
    def get_boundary_pressures(self, G, node_data, boundary_nodes, pressure_diff):
        coords = np.array([node_data[n]["center"] for n in boundary_nodes])
        # PCA - find main axis of the tissue and project boundary nodes onto it
        centered_coords = coords - np.mean(coords, axis=0)
        cov_matrix = np.cov(centered_coords, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]
        projections = np.dot(coords, main_axis)
        proj_min = np.min(projections)
        proj_max = np.max(projections)
        proj_range = proj_max - proj_min
        # Linear decrease of pressure across the tissue based on position along main axis
        boundary_pressures = {}
        for idx, n in enumerate(boundary_nodes):
            norm_pos = (projections[idx] - proj_min) / proj_range
            boundary_pressures[n] = pressure_diff * (1.0 - norm_pos)
        return boundary_pressures


#================================================================================
# Main network class==============================================
#================================================================================

class VesselNetwork:
    """Class representing the vascular network, with methods for cleaning, orientational analysis, and flow calculation."""
    def __init__(self, file_path, mu_plasma=0.0012, pixel_size=1e-6):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.mu_plasma = mu_plasma
        self.pixel_size = pixel_size
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.node_data = {node['id']: node for node in self.data['nodes']}
        self._build_nx_graph()

    def _build_nx_graph(self):
        """Builds a MultiGraph from the current data in self.data."""
        self.G = nx.MultiGraph()
        for link in self.data['links']:
            u, v = link['source'], link['target']
            L, r = link['length'], link['radius']
            res = self._get_resistance(r, L)
            cond = 1 / res # conductivity
            self.G.add_edge(u, v, id=link['id'], cond=cond, radius=r, res = res,  length=L)



    #================================================================================
    # Network cleaning =========================================================
    #================================================================================
    def _explore_territory(self, start_link, node_to_links, all_unvisited):
        """Explores the territory connected to the start link."""
        valid_links = set()
        valid_nodes = set()
        stack = [start_link]

        while stack:
            current = stack.pop()
            if current['id'] in valid_links:
                continue
            valid_links.add(current['id'])
            valid_nodes.update([current['source'], current['target']])
            all_unvisited.discard(current['id'])
            for n in [current['source'], current['target']]:
                for neighbor in node_to_links.get(n, []):
                    if neighbor['id'] not in valid_links:
                        stack.append(neighbor)
        return valid_links, valid_nodes, all_unvisited
    
    def _find_biggest_territory(self):
        """Finds the biggest connected territory in the graph and returns its links and nodes."""
        all_links_map = {l['id']: l for l in self.data['links']}
        unvisited_ids = set(all_links_map.keys())
        node_to_links = {}
        # For each node make a list of connected links
        for l in self.data['links']:
            for n in [l['source'], l['target']]:
                if n not in node_to_links:
                    node_to_links[n] = []
                node_to_links[n].append(l)

        # FInd all territories in grapg (usually only one)
        territories = []
        while unvisited_ids:
            start_id = next(iter(unvisited_ids))
            t_links, t_nodes, unvisited_ids = self._explore_territory(all_links_map[start_id],node_to_links, unvisited_ids)
            territories.append((t_links, t_nodes))
        if not territories:
            print("No territories found in graph")
            return set(), set()
        # Biggest territory - has the most links
        biggest_territory = max(territories, key=lambda x: len(x[0]))
        return biggest_territory
    
    def clean_network(self):
        """Cleans the network by keeping the biggest territory and remapping IDs. 
        The network has to have only one territory for the flow extraction.""" 
        # Delete volume pixels from nodes and links - they take a lot of memory
        for edge in self.data["links"]:
            if "volume_pixels" in edge:
                del edge["volume_pixels"]
        for node in self.data["nodes"]:
            if "pixels" in node:
                del node["pixels"]

        valid_links, valid_nodes = self._find_biggest_territory()
        # Remapping IDs of nodes and links - to be consecutive without gaps, starting from 0
        node_map = {old: new for new, old in enumerate(sorted(valid_nodes))}
        link_map = {old: new for new, old in enumerate(sorted(valid_links))}
        
        # Remap links with correct references to source and target nodes
        remapped_links = []
        for link in self.data['links']:
            if link['id'] in valid_links:
                remapped_link = link.copy()
                remapped_link['id'] = link_map[link['id']]
                remapped_link['source'] = node_map[link['source']]
                remapped_link['target'] = node_map[link['target']]
                remapped_links.append(remapped_link)
        
        # Remap nodes with correct references to branches (links)
        remapped_nodes = []
        for node in self.data['nodes']:
            if node['id'] in valid_nodes:
                remapped_node = node.copy()
                remapped_node['id'] = node_map[node['id']]
                branches = []
                for b in remapped_node['branches_ids']:
                    if b in valid_links:
                        branches.append(link_map[b])
                remapped_node['branches_ids'] = branches
                remapped_nodes.append(remapped_node)
        
        # Sorting so that id = index in list
        remapped_links.sort(key=lambda x: x['id'])
        remapped_nodes.sort(key=lambda x: x['id'])
        self.data['links'] = remapped_links
        self.data['nodes'] = remapped_nodes
        # Add info that the network has been cleaned and remapped
        self.data['cleaned_remapped'] = True

        # Update variables for the cleaned and remapped graph
        self.node_data = {node['id']: node for node in self.data['nodes']}
        self._build_nx_graph()

    def save(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    #================================================================================
    # Network orientation analysis =========================================================
    #================================================================================
    def _get_edge_directions(self, edge_points):
        """Calculate unit direction vectors for each edge."""
        directions = []
        for start, end in edge_points:
            vec = np.array(end) - np.array(start)
            norm = np.linalg.norm(vec)
            if norm > 0:
                directions.append(vec / norm)
            else:
                directions.append(np.zeros_like(vec))
        return np.array(directions)

    def _get_edges_and_volumes(self):
        """Find start and end points of edges and their corresponding volumes."""    
        edge_points = []
        vols = []
        for link in self.data['links']:
            start = self.data['nodes'][link['source']]["center"]
            end = self.data['nodes'][link['target']]["center"]
            edge_points.append((start, end))
            vol_val = abs(link.get("volume", 1.0))
            vols.append(vol_val)
        edge_points = np.array(edge_points) 
        # Switch x and z axis (so that it fits with original tiff data)
        edge_points = edge_points[:, :, [2, 1, 0]]
        vols = np.array(vols)
        return edge_points, vols

    def _get_fa_tenzor(self, directions, volumes):
        """Calculate the orientation tensor from edge directions and volumes, and compute fractional anisotropy (FA) and principal direction."""
        # Weight calculation
        total_volume = np.sum(volumes)
        weights = volumes / total_volume

        # Weighted orientation tensor calculation!
        tensor = np.dot(directions.T, directions * weights[:, np.newaxis])

        # Compute eigenvalues and eigenvectors of the tensor
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        l1, l2, l3 = eigenvalues
        principal_direction = eigenvectors[:, 0]

        # FA calculation
        fa_numerator = np.sqrt((l1 - l2)**2 + (l2 - l3)**2 + (l3 - l1)**2)
        fa_denominator = np.sqrt(l1**2 + l2**2 + l3**2)
        fa = np.sqrt(1/2) * (fa_numerator / fa_denominator)

        # Save analysis results in the data dictionary
        self.data['fa'] = fa
        self.data['orientation_tensor'] = tensor.tolist()
        self.data['principal_direction'] = principal_direction.tolist()
        return fa, tensor, principal_direction

    def orientational_analysis(self):
        """Performs orientational analysis of the vascular network, calculating the orientation tensor, fractional anisotropy (FA), and principal direction based on edge directions and volumes.
        
        Returns:
            tuple: A tuple containing:
                - fa (float): Fractional anisotropy of the network.
                - tensor (np.ndarray): Orientation tensor of the network.
                - principal_direction (np.ndarray): Principal direction of the network.
        """
        edge_points, volumes = self._get_edges_and_volumes()
        directions = self._get_edge_directions(edge_points)
        fa, tensor, principal_direction = self._get_fa_tenzor(directions, volumes)
        return fa, tensor, principal_direction
    
    #================================================================================
    # Flow calculation ========================================================= 
    #================================================================================
    def _get_resistance(self, radius, length):
        """Hagen-Poiseull's law with Pries & Secomb correction."""
        radius = radius * self.pixel_size * 1e6 # Convert from pixels to microns
        correction = 1 + (110*np.exp(-1.424*radius*2) + 3 -3.45*np.exp(-0.035*radius*2))
        real_viscosity = correction * self.mu_plasma
        return (8 * real_viscosity * length) / (np.pi * (radius**4))


    def calculate_flows(self, strategy: PressureStrategy, pressure_diff=None, target_median=None):
        """Calculates flow and velocity for each link in the network based on the provided pressure boundary condition strategy.
        
        Parameters:
            strategy (PressureStrategy): An instance of a PressureStrategy subclass that defines how to assign boundary pressures.
            pressure_diff (float, optional): The pressure difference between nodes with lowest and highest pressure to use for boundary conditions. Required if target_median is not provided.
            target_median (float, optional): If provided, the function will rescale the calculated velocities to match this target median velocity. Required if pressure_diff is not provided.
        """
        if (pressure_diff is None) == (target_median is None):
            raise ValueError("Please provide either 'pressure_diff' or 'target_median' (but not both).")
        
        # Check if network is cleaned and remapped, if not, clean and remap it
        if not self.data.get('cleaned_remapped', False):
            print("Network is not cleaned and remapped. Cleaning and remapping...")
            self.clean_network()

        working_pressure_diff = pressure_diff if pressure_diff is not None else 1000

        boundary_nodes = [n for n, d in self.G.degree() if d == 1]
        boundary_pressures = strategy.get_boundary_pressures(self.G, self.node_data, boundary_nodes, working_pressure_diff)
        
        # Build system of equations based on Kirchhoff's law
        nodes = list(self.G.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}
        A = np.zeros((len(nodes), len(nodes)))
        B = np.zeros(len(nodes))
        for i, n in enumerate(nodes):
            # Boundary nodes have fixed pressures
            if n in boundary_pressures:
                A[i, i] = 1
                B[i] = boundary_pressures[n]
            else:
                for nbr in self.G.neighbors(n):
                    # The sum of flows at the node must be zero: sum((P_n - P_nbr) * G) = 0 -> so B is zero for non-boundary nodes
                    # sum((P_n - P_nbr) * G) = 0 => P_n * sum(G) - sum(P_nbr * G) = 0
                    total_g = sum(d['cond'] for d in self.G[n][nbr].values())
                    A[i, i] += total_g
                    A[i, node_idx[nbr]] -= total_g

        pressures = np.linalg.solve(A, B)
        node_p_map = {n: pressures[node_idx[n]] for n in nodes}
        
        # Velocity and flow calculation for each link
        for link in self.data['links']:
            p_u, p_v = node_p_map[link['source']], node_p_map[link['target']]
            resistance = self._get_resistance(link['radius'], link['length'])
            flow = abs(p_u - p_v) / resistance
            link['velocity'] = flow / (np.pi * (link['radius']**2))
            link['flow'] = flow
            link['direction'] = 1 if p_u >= p_v else -1
            link['resistance'] = resistance

        # If target median is provided, we need to rescale the velocities and flows to match it
        # Linear rescaling is ok, as velocity and pressure difference are linearly related in Hagen-Poiseull's law
        if target_median is not None:
            current_median = np.median([l['velocity'] for l in self.data['links']])
            ratio = target_median / current_median
            for link in self.data['links']:
                link['velocity'] *= ratio
                link['flow'] *= ratio

        # Add info about complete flow calculation to the data dictionary
        self.data['flow_calculated'] = True

    def print_summary(self):
        """Prints a summary of the network, including number of nodes and links, FA, and velocity statistics if available."""
        num_nodes = len(self.data.get('nodes', []))
        num_links = len(self.data.get('links', []))
        print("=" * 50)
        print(f" NETWORK SUMMARY: {self.file_name}")
        print("=" * 50)
        print(f" ▸ Nodes: {num_nodes:,}")
        print(f" ▸ Links: {num_links:,}")
        print(f" ▸ Average radius: {np.mean([l['radius'] for l in self.data['links']]):.4f}")
        print(f" ▸ Average segment length: {np.mean([l['length'] for l in self.data['links']]):.4f}")
        if self.data.get('fa') is not None:
            print(f" ▸ Fractional Anisotropy: {self.data['fa']:.4f}")
        if self.data.get('flow_calculated', False) and num_links > 0:
            velocities = [l['velocity'] for l in self.data['links']]
            flows = [l['flow'] for l in self.data['links']]
            v_min, v_max, v_med, v_mean = min(velocities), max(velocities), np.median(velocities), np.average(velocities)
            # f_min, f_max, f_med, f_mean = min(flows), max(flows), np.median(flows), np.average(flows)
            print("-" * 50)
            print(" HYDRODYNAMICS:")
            print("   Velocity (um/s):")
            print(f"     - Min:    {v_min:>10.4f}")
            print(f"     - Max:    {v_max:>10.4f}")
            print(f"     - Median: {v_med:>10.4f}")
            print(f"     - Mean:   {v_mean:>10.4f}")
            # print("   Flow (um³/s):")
            # print(f"     - Min:    {f_min:>10.1f}")
            # print(f"     - Max:    {f_max:>10.1f}")
            # print(f"     - Median: {f_med:>10.1f}")
            # print(f"     - Mean:   {f_mean:>10.1f}")
        print("=" * 50)

            
