"""
Poisson solver for electrostatic potential in quantum transport calculations.

This module provides finite difference methods for solving the Poisson equation
with appropriate boundary conditions for finite bias transport.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Optional, Union, Dict, Any
import warnings

# Optional GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    import cupyx.scipy.sparse.linalg as cpspla
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class PoissonSolver1D:
    """
    1D Poisson solver using finite differences.
    Solves: d²φ/dx² = -ρ(x)/ε
    """
    
    def __init__(self, x_grid: np.ndarray, epsilon: float = 1.0):
        """
        Initialize 1D Poisson solver.
        
        Parameters:
        -----------
        x_grid : np.ndarray
            1D spatial grid points
        epsilon : float
            Relative permittivity (ε_r)
        """
        self.x_grid = np.asarray(x_grid)
        self.n_points = len(self.x_grid)
        self.dx = self.x_grid[1] - self.x_grid[0] if self.n_points > 1 else 1.0
        self.epsilon = epsilon
        
        # Build finite difference matrix for d²/dx²
        self._build_laplacian_matrix()
    
    def _build_laplacian_matrix(self):
        """Build sparse matrix for second derivative operator."""
        n = self.n_points
        
        # Interior points: φ_{i-1} - 2φ_i + φ_{i+1} = -ρ_i * dx²/ε
        data = []
        row_ind = []
        col_ind = []
        
        for i in range(1, n-1):
            # Coefficients for interior points
            row_ind.extend([i, i, i])
            col_ind.extend([i-1, i, i+1])
            data.extend([1.0, -2.0, 1.0])
        
        # Boundary conditions (will be modified based on BC type)
        # For now, set identity at boundaries
        row_ind.extend([0, n-1]) 
        col_ind.extend([0, n-1])
        data.extend([1.0, 1.0])
        
        self.laplacian = sp.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
        
    def solve(self, charge_density: np.ndarray, 
             boundary_conditions: Dict[str, Any],
             reference_potential: float = 0.0) -> np.ndarray:
        """
        Solve Poisson equation with given boundary conditions.
        
        Parameters:
        -----------
        charge_density : np.ndarray
            Charge density ρ(x) at grid points
        boundary_conditions : dict
            Boundary conditions. Should contain:
            - 'type': 'dirichlet', 'neumann', or 'mixed'
            - 'left_value': value or derivative at left boundary
            - 'right_value': value or derivative at right boundary
        reference_potential : float
            Reference potential (for floating potential problems)
            
        Returns:
        --------
        np.ndarray
            Electrostatic potential φ(x)
        """
        n = self.n_points
        rhs = np.zeros(n)
        A = self.laplacian.copy()
        
        # Set up RHS for interior points
        rhs[1:-1] = -charge_density[1:-1] * self.dx**2 / self.epsilon
        
        # Apply boundary conditions
        bc_type = boundary_conditions.get('type', 'dirichlet')
        
        if bc_type == 'dirichlet':
            # φ(x_0) = V_left, φ(x_n) = V_right
            A[0, :] = 0; A[0, 0] = 1.0
            A[-1, :] = 0; A[-1, -1] = 1.0
            rhs[0] = boundary_conditions.get('left_value', 0.0)
            rhs[-1] = boundary_conditions.get('right_value', 0.0)
            
        elif bc_type == 'neumann':
            # dφ/dx|_left = E_left, dφ/dx|_right = E_right
            # Use one-sided finite differences
            A[0, :] = 0; A[0, 0] = -1.0; A[0, 1] = 1.0
            A[-1, :] = 0; A[-1, -2] = -1.0; A[-1, -1] = 1.0
            rhs[0] = boundary_conditions.get('left_value', 0.0) * self.dx
            rhs[-1] = boundary_conditions.get('right_value', 0.0) * self.dx
            
        elif bc_type == 'mixed':
            # Different BC types at each end
            left_type = boundary_conditions.get('left_type', 'dirichlet')
            right_type = boundary_conditions.get('right_type', 'dirichlet')
            
            if left_type == 'dirichlet':
                A[0, :] = 0; A[0, 0] = 1.0
                rhs[0] = boundary_conditions.get('left_value', 0.0)
            else:  # neumann
                A[0, :] = 0; A[0, 0] = -1.0; A[0, 1] = 1.0
                rhs[0] = boundary_conditions.get('left_value', 0.0) * self.dx
                
            if right_type == 'dirichlet':
                A[-1, :] = 0; A[-1, -1] = 1.0
                rhs[-1] = boundary_conditions.get('right_value', 0.0)
            else:  # neumann
                A[-1, :] = 0; A[-1, -2] = -1.0; A[-1, -1] = 1.0
                rhs[-1] = boundary_conditions.get('right_value', 0.0) * self.dx
        
        # Solve the linear system
        try:
            potential = spla.spsolve(A, rhs)
        except Exception as e:
            warnings.warn(f"Sparse solver failed: {e}. Trying dense solver.")
            potential = np.linalg.solve(A.toarray(), rhs)
        
        # Adjust potential to reference
        potential -= np.mean(potential) - reference_potential
        
        return potential

class PoissonSolver2D:
    """
    2D Poisson solver using finite differences.
    Solves: ∇²φ = -ρ(x,y)/ε
    """
    
    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, epsilon: float = 1.0):
        """
        Initialize 2D Poisson solver.
        
        Parameters:
        -----------
        x_grid, y_grid : np.ndarray
            2D spatial grid points
        epsilon : float
            Relative permittivity
        """
        self.x_grid = np.asarray(x_grid)
        self.y_grid = np.asarray(y_grid)
        self.nx = len(self.x_grid)
        self.ny = len(self.y_grid)
        self.n_total = self.nx * self.ny
        
        self.dx = self.x_grid[1] - self.x_grid[0] if self.nx > 1 else 1.0
        self.dy = self.y_grid[1] - self.y_grid[0] if self.ny > 1 else 1.0
        self.epsilon = epsilon
        
        # Build 2D Laplacian matrix
        self._build_laplacian_matrix_2d()
    
    def _build_laplacian_matrix_2d(self):
        """Build sparse matrix for 2D Laplacian operator."""
        nx, ny = self.nx, self.ny
        n_total = nx * ny
        
        # Create 2D Laplacian using Kronecker products
        # d²/dx² term
        d2dx2_1d = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx), format='csr')
        Ix = sp.eye(nx)
        Iy = sp.eye(ny)
        
        d2dx2_2d = sp.kron(Iy, d2dx2_1d) / self.dx**2
        
        # d²/dy² term  
        d2dy2_1d = sp.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny), format='csr')
        d2dy2_2d = sp.kron(d2dy2_1d, Ix) / self.dy**2
        
        # Full 2D Laplacian
        self.laplacian_2d = d2dx2_2d + d2dy2_2d
    
    def _index_2d_to_1d(self, i: int, j: int) -> int:
        """Convert 2D grid indices to 1D array index."""
        return i * self.ny + j
    
    def _index_1d_to_2d(self, idx: int) -> Tuple[int, int]:
        """Convert 1D array index to 2D grid indices."""
        i = idx // self.ny
        j = idx % self.ny
        return i, j
    
    def solve_2d(self, charge_density_2d: np.ndarray,
                boundary_conditions: Dict[str, Any],
                reference_potential: float = 0.0) -> np.ndarray:
        """
        Solve 2D Poisson equation.
        
        Parameters:
        -----------
        charge_density_2d : np.ndarray
            2D charge density array (nx × ny)
        boundary_conditions : dict
            Boundary conditions for 2D problem
        reference_potential : float
            Reference potential
            
        Returns:
        --------
        np.ndarray
            2D potential array (nx × ny)
        """
        nx, ny = self.nx, self.ny
        
        # Flatten charge density to 1D
        rho_1d = charge_density_2d.flatten()
        
        # Set up system matrix and RHS
        A = self.laplacian_2d.copy()
        rhs = -rho_1d / self.epsilon
        
        # Apply boundary conditions (simplified for now)
        # Set boundary points to specified values
        bc_value = boundary_conditions.get('boundary_value', 0.0)
        
        for i in range(nx):
            for j in range(ny):
                idx = self._index_2d_to_1d(i, j)
                
                # Check if point is on boundary
                if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                    A[idx, :] = 0
                    A[idx, idx] = 1.0
                    rhs[idx] = bc_value
        
        # Solve system
        try:
            potential_1d = spla.spsolve(A, rhs)
        except Exception as e:
            warnings.warn(f"2D Poisson solve failed: {e}")
            potential_1d = np.zeros_like(rho_1d)
        
        # Reshape back to 2D
        potential_2d = potential_1d.reshape((nx, ny))
        
        # Adjust to reference potential
        potential_2d -= np.mean(potential_2d) - reference_potential
        
        return potential_2d

class PoissonSolverFEM:
    """
    Finite Element Method Poisson solver with proper 2D scaling.
    
    Solves 2D Poisson equation ∇²φ = -ρ/(ε₀εᵣ) in SI units.
    For small/collinear systems, uses 1D FEM. For larger systems, uses 2D triangular FEM.
    """
    
    def __init__(self, lattice_sites: np.ndarray, epsilon: float = 11.7, use_gpu: bool = True):
        """
        Initialize FEM Poisson solver.
        
        Parameters:
        -----------
        lattice_sites : np.ndarray
            Array of (x,y) coordinates of lattice sites in meters
        epsilon : float  
            Relative permittivity
        use_gpu : bool
            Use GPU acceleration if available
        """
        self.lattice_sites = np.asarray(lattice_sites)
        self.n_sites = len(lattice_sites)
        self.epsilon = epsilon
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Physical constants  
        self.eps0 = 8.854e-12  # F/m
        self.e = 1.602e-19     # C
        
        if self.use_gpu and not GPU_AVAILABLE:
            warnings.warn("GPU requested but CuPy not available, using CPU")
            self.use_gpu = False
        
        # Determine if system is effectively 1D or 2D
        self._analyze_geometry()
        
        # Build appropriate finite element mesh
        if self.use_1d:
            self._build_1d_fem()
        else:
            self._build_2d_fem()
            
    def _analyze_geometry(self):
        """Determine if the system is effectively 1D or requires 2D treatment."""
        if self.n_sites < 10:
            self.use_1d = True
            return
            
        # Check if points are nearly collinear
        x_coords = self.lattice_sites[:, 0]
        y_coords = self.lattice_sites[:, 1]
        
        x_span = np.max(x_coords) - np.min(x_coords)
        y_span = np.max(y_coords) - np.min(y_coords)
        
        # Use 1D if aspect ratio > 10:1 or y-span very small
        aspect_ratio = max(x_span, y_span) / max(min(x_span, y_span), 1e-12)
        self.use_1d = (aspect_ratio > 10) or (y_span < 1e-10)
        
        print(f"FEM geometry: {'1D' if self.use_1d else '2D'} (aspect ratio {aspect_ratio:.1f})")
        
    def _build_1d_fem(self):
        """Build 1D finite element mesh along the x-direction."""
        # Project all sites onto x-axis and sort
        x_coords = self.lattice_sites[:, 0]
        self.sorted_indices = np.argsort(x_coords)
        self.x_sorted = x_coords[self.sorted_indices]
        
        # Build 1D mesh connectivity  
        self.n_nodes_1d = len(self.x_sorted)
        self.elements_1d = []
        for i in range(self.n_nodes_1d - 1):
            self.elements_1d.append([i, i+1])
        
        print(f"1D FEM: {self.n_nodes_1d} nodes, {len(self.elements_1d)} elements")
        
    def _build_2d_fem(self):
        """Build 2D triangular finite element mesh."""
        try:
            from scipy.spatial import Delaunay
            self.tri = Delaunay(self.lattice_sites)
            self.triangles = self.tri.simplices
            self._build_element_matrices()
            print(f"2D FEM: {self.n_sites} nodes, {len(self.triangles)} triangles")
        except Exception as e:
            warnings.warn(f"2D triangulation failed: {e}, falling back to 1D")
            self.use_1d = True
            self._build_1d_fem()
        
    def _build_fem_mesh(self):
        """Build triangular mesh from lattice sites using Delaunay triangulation."""
        try:
            from scipy.spatial import Delaunay
            
            # Create Delaunay triangulation
            self.tri = Delaunay(self.lattice_sites)
            self.triangles = self.tri.simplices
            self.n_triangles = len(self.triangles)
            
            # Precompute element matrices
            self._build_element_matrices()
            
            print(f"FEM mesh: {self.n_sites} nodes, {self.n_triangles} triangles")
            
        except Exception as e:
            warnings.warn(f"Delaunay triangulation failed: {e}")
            # Fallback: create a simple structured mesh
            try:
                self._build_structured_mesh()
            except Exception as mesh_error:
                warnings.warn(f"Structured mesh fallback also failed: {mesh_error}")
                self.triangles = None
                self.K_elements = []
                self.areas = []
    
    def _build_structured_mesh(self):
        """Fallback: create structured triangular mesh."""
        # Get bounding box
        x_coords = self.lattice_sites[:, 0]
        y_coords = self.lattice_sites[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Create regular grid
        nx = int(np.sqrt(self.n_sites)) + 1
        ny = nx
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        xx, yy = np.meshgrid(x, y)
        
        # Generate triangles from grid
        triangles = []
        for i in range(nx-1):
            for j in range(ny-1):
                # Two triangles per grid cell
                n1 = i * ny + j
                n2 = i * ny + (j + 1)
                n3 = (i + 1) * ny + j
                n4 = (i + 1) * ny + (j + 1)
                triangles.extend([[n1, n2, n3], [n2, n3, n4]])
        
        self.triangles = np.array(triangles)
        self.n_triangles = len(self.triangles)
        
        # Map lattice sites to grid nodes (nearest neighbor)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        self.site_to_node = self._map_sites_to_nodes(grid_points)
        
    def _map_sites_to_nodes(self, grid_points):
        """Map lattice sites to nearest grid nodes."""
        from scipy.spatial.distance import cdist
        distances = cdist(self.lattice_sites, grid_points)
        return np.argmin(distances, axis=1)
    
    def _build_element_matrices(self):
        """Precompute element stiffness matrices for all triangles."""
        self.K_elements = []
        self.areas = []
        
        if not hasattr(self, 'triangles') or self.triangles is None:
            warnings.warn("No triangles available for element matrices")
            return
        
        for tri_nodes in self.triangles:
            # Get triangle coordinates
            coords = self.lattice_sites[tri_nodes]
            x1, y1 = coords[0]
            x2, y2 = coords[1] 
            x3, y3 = coords[2]
            
            # Element area (2x signed area)
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            self.areas.append(area)
            
            if area < 1e-12:  # Degenerate triangle
                K_elem = np.zeros((3, 3))
            else:
                # Shape function derivatives
                b1 = y2 - y3
                b2 = y3 - y1  
                b3 = y1 - y2
                c1 = x3 - x2
                c2 = x1 - x3
                c3 = x2 - x1
                
                # Element stiffness matrix K_ij = ∫∇N_i·∇N_j dA
                B = np.array([[b1, b2, b3], [c1, c2, c3]]) / (2 * area)
                K_elem = (B.T @ B) * area / self.epsilon
            
            self.K_elements.append(K_elem)
    
    def _assemble_global_matrix(self, charge_density: np.ndarray):
        """Assemble global stiffness matrix and load vector."""
        n = self.n_sites
        
        if not hasattr(self, 'triangles') or self.triangles is None:
            raise ValueError("No triangulation available")
        
        # Initialize sparse matrix in COO format for efficiency
        row_idx = []
        col_idx = []
        data = []
        
        # Load vector
        f = np.zeros(n)
        
        # Assemble element contributions
        for elem_idx, tri_nodes in enumerate(self.triangles):
            K_elem = self.K_elements[elem_idx]
            area = self.areas[elem_idx]
            
            # Add to global matrix
            for i in range(3):
                for j in range(3):
                    row_idx.append(tri_nodes[i])
                    col_idx.append(tri_nodes[j])
                    data.append(K_elem[i, j])
            
            # Add to load vector (integrate charge over element)
            for i in range(3):
                # Linear interpolation: ∫ρN_i dA ≈ ρ_avg * area/3
                rho_avg = np.mean(charge_density[tri_nodes])
                f[tri_nodes[i]] += rho_avg * area / 3.0
        
        # Create sparse matrix
        K_global = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
        K_global = K_global.tocsr()
        
        return K_global, f
    
    def solve_fem(self, charge_density: np.ndarray, 
                  boundary_conditions: Dict[str, Any]) -> np.ndarray:
        """
        Solve Poisson equation using finite element method.
        
        Parameters:
        -----------
        charge_density : np.ndarray
            Charge density at lattice sites (electrons per site)  
        boundary_conditions : dict
            Must contain 'left_potential' and 'right_potential' in volts
            
        Returns:
        --------
        np.ndarray
            Electrostatic potential at lattice sites in volts
        """
        if self.use_1d:
            return self._solve_1d_fem(charge_density, boundary_conditions)
        else:
            return self._solve_2d_fem(charge_density, boundary_conditions)
    
    def _solve_1d_fem(self, charge_density: np.ndarray, 
                     boundary_conditions: Dict[str, Any]) -> np.ndarray:
        """
        Solve 1D Poisson equation using finite elements.
        Equation: -d²φ/dx² = ρ/(ε₀εᵣ)
        """
        # Convert charge density from electrons/site to C/m³
        a_cc = 1.42e-10  # Carbon-carbon distance (m)
        area_per_site = a_cc**2 * np.sqrt(3) / 2  # Hexagonal area
        thickness = 3.35e-10  # Graphene thickness (m)
        volume_per_site = area_per_site * thickness
        
        # Convert electrons per site to charge density in C/m³
        rho_3d = charge_density * self.e / volume_per_site  
        rho_sorted = rho_3d[self.sorted_indices]
        
        # Build 1D FEM system
        n = self.n_nodes_1d
        K = np.zeros((n, n))
        f = np.zeros(n)
        
        # Assemble stiffness matrix and load vector
        for elem_idx, (i, j) in enumerate(self.elements_1d):
            # Element length
            h = self.x_sorted[j] - self.x_sorted[i]
            if h <= 0:
                continue
                
            # Element stiffness matrix: K_e = (1/h) * [[1, -1], [-1, 1]]
            k_elem = np.array([[1, -1], [-1, 1]]) / h
            
            # Element load vector: f_e = ρh/2 * [1, 1] / (ε₀εᵣ)
            rho_elem = (rho_sorted[i] + rho_sorted[j]) / 2  # Average charge in element
            f_elem = np.array([1, 1]) * rho_elem * h / (2 * self.eps0 * self.epsilon)
            
            # Assemble into global system
            for ii in range(2):
                for jj in range(2):
                    K[i+ii, j+jj] += k_elem[ii, jj]
                f[i+ii] += f_elem[ii]
        
        # Apply boundary conditions
        V_left = boundary_conditions.get('left_potential', 0.0)
        V_right = boundary_conditions.get('right_potential', 0.0)
        
        # Left boundary (first node)
        K[0, :] = 0
        K[0, 0] = 1
        f[0] = V_left
        
        # Right boundary (last node)  
        K[-1, :] = 0
        K[-1, -1] = 1
        f[-1] = V_right
        
        # Solve system
        try:
            phi_sorted = np.linalg.solve(K, f)
        except np.linalg.LinAlgError:
            phi_sorted = np.linalg.lstsq(K, f, rcond=None)[0]
        
        # Unsort to match original site ordering
        phi = np.zeros(self.n_sites)
        phi[self.sorted_indices] = phi_sorted
        
        return phi
    
    def _solve_2d_fem(self, charge_density: np.ndarray, 
                     boundary_conditions: Dict[str, Any]) -> np.ndarray:
        """Solve 2D Poisson equation using triangular finite elements."""
        # Check if 2D mesh is available
        if not hasattr(self, 'K_elements') or not self.K_elements:
            raise ValueError("2D FEM mesh not available")
        
        # Convert charge density with proper 2D scaling
        a_cc = 1.42e-10  
        area_per_site = a_cc**2 * np.sqrt(3) / 2
        
        # For 2D: ρ is surface charge density (C/m²)
        rho_2d = charge_density * self.e / area_per_site
        
        # Assemble global system
        K, f = self._assemble_global_matrix_2d(rho_2d)

        # Apply Dirichlet boundary conditions
        V_left = boundary_conditions.get('left_potential', 0.0)
        V_right = boundary_conditions.get('right_potential', 0.0)
        
        # Find boundary nodes (leftmost and rightmost)
        x_coords = self.lattice_sites[:, 0]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        tol = (x_max - x_min) * 0.01  # 1% tolerance
        
        left_nodes = np.where(x_coords <= x_min + tol)[0]
        right_nodes = np.where(x_coords >= x_max - tol)[0]
        
        # Modify system for Dirichlet BCs
        boundary_nodes = np.concatenate([left_nodes, right_nodes])
        boundary_values = np.concatenate([
            np.full(len(left_nodes), V_left),
            np.full(len(right_nodes), V_right)
        ])
        
        # Interior nodes
        all_nodes = np.arange(self.n_sites)
        interior_nodes = np.setdiff1d(all_nodes, boundary_nodes)
        
        if len(interior_nodes) == 0:
            # All boundary - just return boundary values
            phi = np.zeros(self.n_sites)
            phi[left_nodes] = V_left
            phi[right_nodes] = V_right
            return phi
        
        # Reduced system: K_II * phi_I = f_I - K_IB * phi_B
        K_II = K[interior_nodes][:, interior_nodes]
        K_IB = K[interior_nodes][:, boundary_nodes]
        f_I = f[interior_nodes] - K_IB @ boundary_values
        
        # Solve system
        if self.use_gpu:
            phi_interior = self._solve_gpu(K_II, f_I)
        else:
            phi_interior = self._solve_cpu(K_II, f_I)
        
        # Reconstruct full solution
        phi = np.zeros(self.n_sites)
        phi[interior_nodes] = phi_interior
        phi[boundary_nodes] = boundary_values
        
        return phi
        
    def _assemble_global_matrix_2d(self, charge_density: np.ndarray):
        """Assemble 2D global stiffness matrix and load vector with proper scaling."""
        n = self.n_sites
        
        # Initialize sparse matrix in COO format
        row_idx = []
        col_idx = []
        data = []
        
        # Load vector  
        f = np.zeros(n)
        
        # Assemble element contributions
        for elem_idx, tri_nodes in enumerate(self.triangles):
            K_elem = self.K_elements[elem_idx]
            area = self.areas[elem_idx]
            
            # Add to global stiffness matrix
            for i in range(3):
                for j in range(3):
                    row_idx.append(tri_nodes[i])
                    col_idx.append(tri_nodes[j]) 
                    data.append(K_elem[i, j])
            
            # Add to load vector with proper 2D scaling
            # 2D Poisson: ∇²φ = -ρ/(ε₀εᵣ) where ρ is in C/m²
            for i in range(3):
                rho_avg = np.mean(charge_density[tri_nodes])
                # Integration: ∫ρN_i dA ≈ ρ_avg * area/3 / (ε₀εᵣ)
                f[tri_nodes[i]] += rho_avg * area / (3 * self.eps0 * self.epsilon)
        
        # Create sparse matrix
        K_global = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
        K_global = K_global.tocsr()
        
        return K_global, f
    
    def _solve_cpu(self, K, f):
        """Solve linear system on CPU."""
        try:
            return spla.spsolve(K, f)
        except Exception as e:
            warnings.warn(f"Direct solve failed: {e}, trying CG")
            phi, info = spla.cg(K, f, maxiter=1000, tol=1e-8)
            if info != 0:
                warnings.warn(f"CG did not converge (info={info})")
            return phi
    
    def _solve_gpu(self, K, f):
        """Solve linear system on GPU using CuPy."""
        try:
            # Transfer to GPU
            K_gpu = cpsp.csr_matrix(K)
            f_gpu = cp.asarray(f)
            
            # Solve using GPU CG (direct solvers are limited in CuPy)
            phi_gpu, info = cpspla.cg(K_gpu, f_gpu, maxiter=1000, tol=1e-8)
            
            if info != 0:
                warnings.warn(f"GPU CG did not converge (info={info})")
            
            # Transfer back to CPU
            return cp.asnumpy(phi_gpu)
            
        except Exception as e:
            warnings.warn(f"GPU solve failed: {e}, falling back to CPU")
            return self._solve_cpu(K, f)

class PoissonSolverGraphene:
    """
    Specialized Poisson solver for graphene nanoribbon geometry.
    
    Now uses finite element method as primary solver with graph-Laplacian fallback.
    """
    
    def __init__(self, lattice_sites: np.ndarray, epsilon: float = 11.7, use_gpu: bool = False):
        """
        Initialize graphene Poisson solver.
        
        Parameters:
        -----------
        lattice_sites : np.ndarray
            Array of (x,y) coordinates of graphene lattice sites
        epsilon : float
            Relative permittivity of graphene (≈11.7)
        use_gpu : bool
            Use GPU acceleration if available
        """
        self.lattice_sites = np.asarray(lattice_sites)
        self.n_sites = len(lattice_sites)
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        
        # Initialize FEM solver as primary method
        try:
            self.fem_solver = PoissonSolverFEM(lattice_sites, epsilon, use_gpu)
            self.fem_available = True
        except Exception as e:
            warnings.warn(f"FEM solver initialization failed: {e}")
            self.fem_available = False
        
        # Create regular grid encompassing the lattice
        x_coords = lattice_sites[:, 0]
        y_coords = lattice_sites[:, 1]
        
        self.x_min, self.x_max = np.min(x_coords), np.max(x_coords)
        self.y_min, self.y_max = np.min(y_coords), np.max(y_coords)
        
        # Ensure minimum grid spacing to avoid degenerate grids
        if self.x_max - self.x_min < 1e-10:
            self.x_min -= 0.5
            self.x_max += 0.5
        if self.y_max - self.y_min < 1e-10:
            self.y_min -= 0.5
            self.y_max += 0.5
        
        # Create regular grid (adjust resolution as needed)
        nx = max(int((self.x_max - self.x_min) * 2) + 1, 10)
        ny = max(int((self.y_max - self.y_min) * 2) + 1, 10)
        
        self.x_grid = np.linspace(self.x_min, self.x_max, nx)
        self.y_grid = np.linspace(self.y_min, self.y_max, ny)
        
        # Ensure grids are strictly monotonic 
        if len(self.x_grid) > 1 and np.any(np.diff(self.x_grid) <= 0):
            self.x_grid = np.linspace(self.x_min, self.x_max, nx)
        if len(self.y_grid) > 1 and np.any(np.diff(self.y_grid) <= 0):
            self.y_grid = np.linspace(self.y_min, self.y_max, ny)
        
        # Initialize 2D solver
        self.solver_2d = PoissonSolver2D(self.x_grid, self.y_grid, epsilon)
        
        # Precompute interpolation weights
        self._setup_interpolation()

        # Precompute graph Laplacian for meshless solve (robust fallback/default)
        self._build_graph_laplacian()

    def _build_graph_laplacian(self, k_neighbors: int = 4):
        """Build a k-NN graph Laplacian directly on lattice sites.

        This avoids Delaunay/Qhull and stays robust for quasi-1D ribbons.
        """
        pts = self.lattice_sites
        n = self.n_sites
        if n < 3:
            self.graph_L = sp.csr_matrix((n, n))
            return

        # Compute pairwise distances to estimate a neighbor length scale
        # Use a small subset for speed if very large
        try:
            from sklearn.neighbors import NearestNeighbors  # optional
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors+1, n), algorithm='auto').fit(pts)
            distances, indices = nbrs.kneighbors(pts)
        except Exception:
            # Fallback kNN: naive partial sort (O(n^2)) if sklearn not available
            dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            indices = np.argsort(dists, axis=1)[:, :min(k_neighbors+1, n)]
            rows = np.arange(n)[:, None]
            distances = dists[rows, indices]

        # Characteristic length scale from median of first neighbor distance (>0)
        nn = distances[:, 1] if distances.shape[1] > 1 else np.full(n, 1.0)
        ell = np.median(nn[nn > 0]) if np.any(nn > 0) else 1.0
        if not np.isfinite(ell) or ell <= 0:
            ell = 1.0

        # Build symmetric weight matrix using Gaussian kernel
        row_idx = []
        col_idx = []
        data = []
        for i in range(n):
            for j_idx in range(1, indices.shape[1]):  # skip self at 0
                j = int(indices[i, j_idx])
                if i == j:
                    continue
                # Symmetric entry
                rij = np.linalg.norm(pts[i] - pts[j])
                if rij <= 0:
                    continue
                w = np.exp(-(rij/ell)**2)
                row_idx.extend([i, j])
                col_idx.extend([j, i])
                data.extend([w, w])
        W = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
        d = np.array(W.sum(axis=1)).ravel()
        L = sp.diags(d) - W
        # Slight regularization to avoid singularity in floating potentials
        L = L + 1e-12 * sp.eye(n)
        self.graph_L = L.tocsr()
    
    def _setup_interpolation(self):
        """Set up bilinear interpolation from regular grid to lattice sites."""
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator for mapping grid back to lattice sites
        self.grid_to_lattice_interpolator = RegularGridInterpolator(
            (self.x_grid, self.y_grid), 
            np.zeros((len(self.x_grid), len(self.y_grid))),
            bounds_error=False, 
            fill_value=0.0
        )
    
    def map_charge_to_grid(self, charge_density_sites: np.ndarray) -> np.ndarray:
        """
        Map charge density from lattice sites to regular grid.
        
        Parameters:
        -----------
        charge_density_sites : np.ndarray
            Charge density at each lattice site
            
        Returns:
        --------
        np.ndarray
            Charge density on regular 2D grid
        """
        from scipy.interpolate import griddata
        
        # Create regular grid points
        xx, yy = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Interpolate from lattice sites to regular grid
        charge_grid = griddata(
            self.lattice_sites, 
            charge_density_sites,
            grid_points,
            method='linear',
            fill_value=0.0
        )
        
        return charge_grid.reshape(xx.shape)
    
    def map_potential_to_sites(self, potential_grid: np.ndarray) -> np.ndarray:
        """
        Map potential from regular grid back to lattice sites.
        
        Parameters:
        -----------
        potential_grid : np.ndarray
            Potential on regular 2D grid
            
        Returns:
        --------
        np.ndarray
            Potential at each lattice site
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (self.x_grid, self.y_grid),
            potential_grid,
            bounds_error=False,
            fill_value=0.0
        )
        
        # Interpolate to lattice sites
        potential_sites = interpolator(self.lattice_sites)
        
        return potential_sites
    
    def solve_graphene(self, charge_density_sites: np.ndarray,
                      bias_voltage: float = 0.0,
                      lead_potentials: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Solve Poisson equation for graphene nanoribbon with finite bias.
        
        Parameters:
        -----------
        charge_density_sites : np.ndarray
            Charge density at each lattice site
        bias_voltage : float
            Bias voltage between source and drain
        lead_potentials : tuple, optional
            (V_left, V_right) potentials of left and right leads
            
        Returns:
        --------
        np.ndarray
            Electrostatic potential at each lattice site
        """
        # Default lead potentials for finite bias
        if lead_potentials is None:
            V_left = bias_voltage / 2
            V_right = -bias_voltage / 2
            lead_potentials = (V_left, V_right)
        
        print(f"DEBUG: Poisson input charge range: [{np.min(charge_density_sites):.3e}, {np.max(charge_density_sites):.3e}]")
        
        # Try FEM solver first (preferred method for realistic scaling)
        if self.fem_available:
            try:
                boundary_conditions = {
                    'left_potential': lead_potentials[0],
                    'right_potential': lead_potentials[1]
                }
                phi = self.fem_solver.solve_fem(charge_density_sites, boundary_conditions)
                print(f"DEBUG: FEM Poisson output potential range: [{np.min(phi):.3e}, {np.max(phi):.3e}]")
                return phi
            except Exception as e:
                warnings.warn(f"FEM solver failed: {e}, falling back to graph-Laplacian")
        
        # Fallback to graph-Laplacian solve (with improved scaling)
        try:
            phi_graph = self._solve_on_graph_improved(charge_density_sites, lead_potentials)
            if np.any(np.isfinite(phi_graph)):
                print(f"DEBUG: Graph-Laplacian output potential range: [{np.min(phi_graph):.3e}, {np.max(phi_graph):.3e}]")
                return phi_graph
        except Exception as e:
            warnings.warn(f"Graph-Laplacian Poisson solve failed ({e}), trying grid-based solver")

        # For very small systems, use 1D approximation directly
        if self.n_sites < 200:
            warnings.warn("Using 1D Poisson approximation for small system")
            return self._solve_1d_approximation(charge_density_sites, bias_voltage, lead_potentials)
        
        # Map charge to regular grid
        try:
            charge_grid = self.map_charge_to_grid(charge_density_sites)
        except Exception as grid_error:
            warnings.warn(f"Grid mapping failed ({grid_error}), using 1D approximation")
            return self._solve_1d_approximation(charge_density_sites, bias_voltage, lead_potentials)
        
        # Set up boundary conditions for finite bias
        # Left edge at V_left, right edge at V_right, top/bottom grounded or periodic
        boundary_conditions = {
            'boundary_value': 0.0,  # Default value
            'left_potential': lead_potentials[0],
            'right_potential': lead_potentials[1]
        }
        
        # Modify boundary conditions for leads
        A = self.solver_2d.laplacian_2d.copy()
        rhs = -charge_grid.flatten() / self.epsilon
        
        nx, ny = len(self.x_grid), len(self.y_grid)
        
        # Set left edge to V_left
        for j in range(ny):
            idx = self.solver_2d._index_2d_to_1d(0, j)
            A[idx, :] = 0
            A[idx, idx] = 1.0
            rhs[idx] = lead_potentials[0]
        
        # Set right edge to V_right  
        for j in range(ny):
            idx = self.solver_2d._index_2d_to_1d(nx-1, j)
            A[idx, :] = 0
            A[idx, idx] = 1.0
            rhs[idx] = lead_potentials[1]
        
        # Solve system
        try:
            potential_1d = spla.spsolve(A, rhs)
            potential_grid = potential_1d.reshape((nx, ny))
        except Exception as e:
            warnings.warn(f"Graphene Poisson solve failed: {e}")
            potential_grid = np.zeros((nx, ny))
        
        # Map back to lattice sites
        try:
            potential_sites = self.map_potential_to_sites(potential_grid)
        except Exception as map_error:
            # Fallback to simple 1D approximation for problematic geometries
            warnings.warn(f"2D mapping failed ({map_error}), using 1D approximation")
            potential_sites = self._solve_1d_approximation(charge_density_sites, bias_voltage, lead_potentials)
        
        return potential_sites

    def _solve_on_graph(self, charge_density_sites: np.ndarray,
                        lead_potentials: Tuple[float, float]) -> np.ndarray:
        """Solve Poisson directly on the lattice graph with Dirichlet leads.

        This is meshless and robust. It treats −Δφ = ρ/ε with a graph Laplacian L.
        We enforce Dirichlet BC by fixing nodes near left/right edges.
        """
        n = self.n_sites
        if n == 0:
            return np.array([])

        L = self.graph_L
        x = self.lattice_sites[:, 0]
        x_min, x_max = float(np.min(x)), float(np.max(x))
        width = max(x_max - x_min, 1e-6)

        # Pick boundary nodes: k% nearest to left/right edges
        k_pct = 0.02  # 2% on each side
        k = max(2, int(k_pct * n))
        left_idx = np.argsort(x)[:k]
        right_idx = np.argsort(-x)[:k]
        left_idx = np.unique(left_idx)
        right_idx = np.unique(right_idx)

        # Dirichlet values
        V_left, V_right = lead_potentials
        phi_B = np.zeros(n)
        phi_B[left_idx] = V_left
        phi_B[right_idx] = V_right
        B_mask = np.zeros(n, dtype=bool)
        B_mask[left_idx] = True
        B_mask[right_idx] = True
        I_mask = ~B_mask

        # Assemble reduced system: L_II phi_I = b_I - L_IB phi_B
        L_II = L[I_mask][:, I_mask].tocsr()
        L_IB = L[I_mask][:, B_mask].tocsr()

        # RHS from charge: scale to approximate continuum units
        # Use proper physical scaling instead of arbitrary degree normalization
        # Based on actual lattice spacing and physical constants
        a_cc = 1.42e-10  # Carbon-carbon bond length (m)
        area_per_site = (a_cc ** 2) * np.sqrt(3) / 2  # Hexagonal unit cell area
        eps0 = 8.854e-12  # F/m
        scale_factor = area_per_site / (eps0 * max(self.epsilon, 1e-12))
        b_full = -charge_density_sites * scale_factor
        b_I = b_full[I_mask] - (L_IB @ phi_B[B_mask])

        # Solve the sparse SPD system; try CG first, then spsolve
        phi = np.zeros(n)
        try:
            try:
                phi_I, info = spla.cg(L_II, b_I, maxiter=200, tol=1e-8)
                if info != 0 or not np.all(np.isfinite(phi_I)):
                    raise RuntimeError(f"CG did not converge (info={info})")
            except Exception as cg_err:
                phi_I = spla.spsolve(L_II, b_I)
            phi[I_mask] = phi_I
            phi[B_mask] = phi_B[B_mask]
        except Exception as e:
            # As a last resort, return linear ramp between leads
            warnings.warn(f"Graph solve failed: {e}; returning linear potential")
            phi = np.linspace(V_left, V_right, n)

        # Normalize to zero-mean to avoid drift
        phi -= np.mean(phi)
        # Re-anchor leads exactly
        phi[left_idx] = V_left
        phi[right_idx] = V_right
        return phi
    
    def _solve_on_graph_improved(self, charge_density_sites: np.ndarray,
                                lead_potentials: Tuple[float, float]) -> np.ndarray:
        """Improved graph-Laplacian solver with proper physical scaling."""
        return self._solve_on_graph(charge_density_sites, lead_potentials)
    
    def _solve_1d_approximation(self, charge_density_sites: np.ndarray, 
                               bias_voltage: float, 
                               lead_potentials: Tuple[float, float]) -> np.ndarray:
        """
        Simple 1D Poisson solver fallback for problematic geometries.
        Assumes transport is mainly along x-direction.
        """
        try:
            # Project sites onto x-axis for 1D approximation
            x_coords = self.lattice_sites[:, 0]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            
            if x_max - x_min < 1e-10:  # Degenerate geometry  
                # Return linear potential between leads instead of zeros
                phi_linear = np.linspace(lead_potentials[0], lead_potentials[1], len(charge_density_sites))
                return phi_linear
            
            # Sort by x-coordinate
            sort_indices = np.argsort(x_coords)
            x_sorted = x_coords[sort_indices]
            rho_sorted = charge_density_sites[sort_indices]
            
            # Simple 1D finite difference
            n_points = len(x_sorted)
            if n_points < 3:
                return np.zeros(len(charge_density_sites))
            
            # Create 1D Laplacian matrix
            dx = (x_max - x_min) / (n_points - 1)
            A = np.zeros((n_points, n_points))
            
            # Interior points: d²φ/dx² = -ρ/ε
            for i in range(1, n_points-1):
                A[i, i-1] = 1.0 / (dx**2)
                A[i, i] = -2.0 / (dx**2)
                A[i, i+1] = 1.0 / (dx**2)
            
            # Boundary conditions
            A[0, 0] = 1.0  # Left boundary
            A[-1, -1] = 1.0  # Right boundary
            
            # Right-hand side
            rhs = -rho_sorted / self.epsilon
            rhs[0] = lead_potentials[0]   # Left potential
            rhs[-1] = lead_potentials[1]  # Right potential
            
            # Solve
            phi_sorted = np.linalg.solve(A, rhs)
            
            # Unsort to match original order
            phi_sites = np.zeros_like(charge_density_sites)
            phi_sites[sort_indices] = phi_sorted
            
            return phi_sites
            
        except Exception as e:
            warnings.warn(f"1D Poisson approximation failed: {e}")
            # Return linear potential between leads rather than zeros
            phi_linear = np.linspace(lead_potentials[0], lead_potentials[1], len(charge_density_sites))
            return phi_linear