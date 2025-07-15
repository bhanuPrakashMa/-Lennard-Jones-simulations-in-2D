import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

class LennardJones2D:
    def __init__(self, L, N, rc, dt, m=1.0):
        self.L = L
        self.N = N
        self.rc = rc
        self.rc_sq = rc**2
        self.dt = dt
        self.m = m
        self.U_rc = 4 * (1 / rc**12 - 1 / rc**6) 

        # Particle properties: positions, velocities, forces
        self.r = np.zeros((N, 2))  
        self.v = np.zeros((N, 2))  
        self.a = np.zeros((N, 2))  
        # Cell list parameters
        self.cell_size = rc 
        self.num_cells_x = int(L / self.cell_size)
        self.num_cells_y = int(L / self.cell_size)
        self.cell_lists = defaultdict(list)

        # Energy tracking
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        self.total_energy_history = []
        self.temperature_history = []
        self.momentum_history = []

        # RDF parameters
        self.rdf_bins = np.arange(0, L / 2, 0.05)
        self.rdf_histogram = np.zeros(len(self.rdf_bins) - 1)
        self.rdf_normalization = 0


    def _apply_pbc(self, pos):
       
        return pos % self.L

    def _minimum_image_distance(self, r_ij):
        
        r_ij -= self.L * np.round(r_ij / self.L)
        return r_ij

    def _lennard_jones_potential_and_force(self, r_sq, r_vec):
        if r_sq >= self.rc_sq:
            return 0.0, 0.0, 0.0

        r = np.sqrt(r_sq)
        inv_r_6 = 1.0 / (r_sq**3)
        inv_r_12 = inv_r_6**2

        potential = 4.0 * (inv_r_12 - inv_r_6) - self.U_rc

        force_magnitude_term = 48.0 * (inv_r_12 - 0.5 * inv_r_6) / r_sq
        force_x = r_vec[0] * force_magnitude_term
        force_y = r_vec[1] * force_magnitude_term

        return potential, force_x, force_y

    def _initialize_lattice(self, density=None):

        if density is not None:
            
            approx_side = np.sqrt(self.N / density)
            cells_per_side = int(np.sqrt(self.N))
            self.N = cells_per_side * cells_per_side
            self.L = cells_per_side / np.sqrt(density) 
            print(f"Adjusted N to {self.N} and L to {self.L:.2f} for a nice lattice.")
        else:
            cells_per_side = int(np.sqrt(self.N))
            if cells_per_side * cells_per_side != self.N:
                print(f"Warning: N={self.N} is not a perfect square.")

        spacing = self.L / cells_per_side
        idx = 0
        for i in range(cells_per_side):
            for j in range(cells_per_side):
                if idx < self.N:
                    self.r[idx, 0] = (i + 0.5) * spacing
                    self.r[idx, 1] = (j + 0.5) * spacing
                    idx += 1
        
        self.r += (np.random.rand(self.N, 2) - 0.5) * 0.01 * spacing

    def _initialize_velocities(self, target_T=0.5):
        
        self.v = np.random.randn(self.N, 2) 
        self.v -= np.mean(self.v, axis=0)  


        current_KE = 0.5 * self.m * np.sum(self.v**2)
        current_T = current_KE / (self.N - 1) 

        scale_factor = np.sqrt(target_T / current_T)
        self.v *= scale_factor
        print(f"Initial target temperature: {target_T}, Initial calculated temperature: {current_T * scale_factor**2:.4f}")

    def _build_cell_lists(self):
        self.cell_lists.clear()
        for i in range(self.N):
            cell_x = int(self.r[i, 0] / self.cell_size) % self.num_cells_x
            cell_y = int(self.r[i, 1] / self.cell_size) % self.num_cells_y
            self.cell_lists[(cell_x, cell_y)].append(i)

    def _calculate_forces_and_potential_energy(self):
        
        self.a.fill(0.0) 
        total_potential_energy = 0.0

        for cell_x in range(self.num_cells_x):
            for cell_y in range(self.num_cells_y):
                current_cell_particles = self.cell_lists[(cell_x, cell_y)]

                # Iterate through particles in the current cell
                for i in current_cell_particles:
                    
                    for j in current_cell_particles:
                        if i < j: 
                            r_ij_vec = self.r[i] - self.r[j]
                            r_ij_vec = self._minimum_image_distance(r_ij_vec)
                            r_sq = np.sum(r_ij_vec**2)

                            potential, fx, fy = self._lennard_jones_potential_and_force(r_sq, r_ij_vec)
                            total_potential_energy += potential
                            self.a[i, 0] += fx / self.m
                            self.a[i, 1] += fy / self.m
                            self.a[j, 0] -= fx / self.m
                            self.a[j, 1] -= fy / self.m


                    for dx_cell in [-1, 0, 1]:
                        for dy_cell in [-1, 0, 1]:
                            if dx_cell == 0 and dy_cell == 0:
                                continue 

                            neighbor_cell_x = (cell_x + dx_cell + self.num_cells_x) % self.num_cells_x
                            neighbor_cell_y = (cell_y + dy_cell + self.num_cells_y) % self.num_cells_y
                            neighbor_cell_particles = self.cell_lists[(neighbor_cell_x, neighbor_cell_y)]

                            for i in current_cell_particles:
                                for j in neighbor_cell_particles:                               
                                    pass 

        self.a.fill(0.0)
        total_potential_energy = 0.0

        for i in range(self.N):
            cell_x_i = int(self.r[i, 0] / self.cell_size) % self.num_cells_x
            cell_y_i = int(self.r[i, 1] / self.cell_size) % self.num_cells_y

            for dx_cell in [-1, 0, 1]:
                for dy_cell in [-1, 0, 1]:
                    neighbor_cell_x = (cell_x_i + dx_cell + self.num_cells_x) % self.num_cells_x
                    neighbor_cell_y = (cell_y_i + dy_cell + self.num_cells_y) % self.num_cells_y

                    if (dx_cell > 0) or (dx_cell == 0 and dy_cell >= 0): 
                        
                        if dx_cell == 0 and dy_cell == 0:
                            for j in self.cell_lists[(neighbor_cell_x, neighbor_cell_y)]:
                                if i < j:
                                    r_ij_vec = self.r[i] - self.r[j]
                                    r_ij_vec = self._minimum_image_distance(r_ij_vec)
                                    r_sq = np.sum(r_ij_vec**2)

                                    potential, fx, fy = self._lennard_jones_potential_and_force(r_sq, r_ij_vec)
                                    total_potential_energy += potential
                                    self.a[i, 0] += fx / self.m
                                    self.a[i, 1] += fy / self.m
                                    self.a[j, 0] -= fx / self.m
                                    self.a[j, 1] -= fy / self.m
                        else: 
                            for j in self.cell_lists[(neighbor_cell_x, neighbor_cell_y)]:
                                r_ij_vec = self.r[i] - self.r[j]
                                r_ij_vec = self._minimum_image_distance(r_ij_vec)
                                r_sq = np.sum(r_ij_vec**2)

                                potential, fx, fy = self._lennard_jones_potential_and_force(r_sq, r_ij_vec)
                                total_potential_energy += potential
                                self.a[i, 0] += fx / self.m
                                self.a[i, 1] += fy / self.m
                                self.a[j, 0] -= fx / self.m
                                self.a[j, 1] -= fy / self.m
        return total_potential_energy

    def _calculate_kinetic_energy(self):
        
        return 0.5 * self.m * np.sum(self.v**2)

    def _calculate_temperature(self):
       
        return self._calculate_kinetic_energy() / (self.N - 1)

    def _calculate_total_momentum(self):
        """Calculates total momentum of the system."""
        return np.sum(self.m * self.v, axis=0)

    def _berendsen_thermostat(self, target_T, tau):
       
        current_T = self._calculate_temperature()
        if current_T == 0: 
            return
        gamma = np.sqrt(1 + (self.dt / tau) * (target_T / current_T - 1))
        self.v *= gamma

    def _compute_rdf(self):
    
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r_ij_vec = self.r[i] - self.r[j]
                r_ij_vec = self._minimum_image_distance(r_ij_vec)
                r = np.sqrt(np.sum(r_ij_vec**2))

                
                if r < self.L / 2: 
                    bin_idx = int(r / 0.05)
                    if bin_idx < len(self.rdf_histogram):
                        self.rdf_histogram[bin_idx] += 1

    def _normalize_rdf(self, num_timesteps_for_rdf):
        
        rho = self.N / (self.L**2)
        dr = self.rdf_bins[1] - self.rdf_bins[0] 

        for i in range(len(self.rdf_histogram)):
            r = self.rdf_bins[i] + dr / 2 
            r_inner = self.rdf_bins[i]
            r_outer = self.rdf_bins[i+1]
            annulus_area = np.pi * (r_outer**2 - r_inner**2)
            
            N_ideal = self.N * rho * annulus_area

            if N_ideal > 0:
                
                self.rdf_histogram[i] = (self.rdf_histogram[i] / num_timesteps_for_rdf) / N_ideal
            else:
                self.rdf_histogram[i] = 0

    def run_simulation(self, num_steps, initial_T, ensemble='NVE', target_T=None, tau_dt_ratio=None,
                       equilibrium_steps_for_rdf=0):
    
        if ensemble == 'NVT':
            if target_T is None or tau_dt_ratio is None:
                raise ValueError("target_T and tau_dt_ratio are required for NVT ensemble.")
            tau = self.dt / tau_dt_ratio
            print(f"Running NVT simulation with Berendsen thermostat (tau = {tau:.2f})")

        print(f"Initializing with N={self.N}, L={self.L}, dt={self.dt}, rc={self.rc}")
        self._initialize_lattice()
        self._initialize_velocities(initial_T)

       
        self._build_cell_lists()
        potential_energy = self._calculate_forces_and_potential_energy()
        kinetic_energy = self._calculate_kinetic_energy()

        self.kinetic_energy_history.append(kinetic_energy)
        self.potential_energy_history.append(potential_energy)
        self.total_energy_history.append(kinetic_energy + potential_energy)
        self.temperature_history.append(self._calculate_temperature())
        self.momentum_history.append(self._calculate_total_momentum())

        print(f"Starting simulation for {num_steps} steps...")
        start_time = time.time()

        rdf_accumulation_steps = 0

        for step in range(num_steps):
            
            self.r += self.v * self.dt + 0.5 * self.a * self.dt**2
            self.r = self._apply_pbc(self.r)

            
            self._build_cell_lists()          
            a_old = np.copy(self.a)          
            potential_energy = self._calculate_forces_and_potential_energy()      
            self.v += 0.5 * (a_old + self.a) * self.dt
        
            if ensemble == 'NVT':
                self._berendsen_thermostat(target_T, tau)
          
            kinetic_energy = self._calculate_kinetic_energy()
            current_temperature = self._calculate_temperature()
            current_momentum = self._calculate_total_momentum()

            self.kinetic_energy_history.append(kinetic_energy)
            self.potential_energy_history.append(potential_energy)
            self.total_energy_history.append(kinetic_energy + potential_energy)
            self.temperature_history.append(current_temperature)
            self.momentum_history.append(current_momentum)

            if step >= equilibrium_steps_for_rdf:
                self._compute_rdf()
                rdf_accumulation_steps += 1

            if (step + 1) % 1000 == 0:
                print(f"Step {step + 1}/{num_steps} | KE: {kinetic_energy:.4f} | PE: {potential_energy:.4f} | E_tot: {kinetic_energy + potential_energy:.4f} | T: {current_temperature:.4f} | Momentum: {np.linalg.norm(current_momentum):.2e}")

        end_time = time.time()
        print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

        if rdf_accumulation_steps > 0:
            self._normalize_rdf(rdf_accumulation_steps)
            print(f"RDF accumulated over {rdf_accumulation_steps} steps.")

    def plot_energies(self, title="Energy Evolution"):
        """Plots kinetic, potential, and total energy over time."""
        timesteps = np.arange(len(self.total_energy_history)) * self.dt
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, self.kinetic_energy_history, label='Kinetic Energy')
        plt.plot(timesteps, self.potential_energy_history, label='Potential Energy')
        plt.plot(timesteps, self.total_energy_history, label='Total Energy')
        plt.xlabel('Time')
        plt.ylabel('Energy (reduced units)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_temperature(self, title="Temperature Evolution"):
        """Plots temperature over time."""
        timesteps = np.arange(len(self.temperature_history)) * self.dt
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, self.temperature_history, label='Temperature')
        plt.xlabel('Time')
        plt.ylabel('Temperature (reduced units)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_momentum(self, title="Total Momentum Evolution"):
        """Plots total momentum over time."""
        timesteps = np.arange(len(self.momentum_history)) * self.dt
        momentum_magnitude = np.array([np.linalg.norm(m) for m in self.momentum_history])
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, momentum_magnitude, label='Total Momentum Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Momentum Magnitude (reduced units)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.yscale('log') 
        plt.show()

    def plot_rdf(self, title="Radial Distribution Function g(r)"):
        """Plots the radial distribution function."""
        rdf_centers = self.rdf_bins[:-1] + (self.rdf_bins[1] - self.rdf_bins[0]) / 2
        plt.figure(figsize=(8, 6))
        plt.plot(rdf_centers, self.rdf_histogram)
        plt.xlabel('Distance r')
        plt.ylabel('g(r)')
        plt.title(title)
        plt.grid(True)
        plt.xlim(0, self.L / 2) 
        plt.show()

    def plot_particle_distribution(self, title="Particle Distribution"):
        """Plots the current particle distribution in the simulation box."""
        plt.figure(figsize=(8, 8))
        plt.scatter(self.r[:, 0], self.r[:, 1], s=10, alpha=0.7)
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title(title)
        plt.grid(True)
        plt.show()


# Simulation Runs

L = 30.0
rc = 2.5
dt = 0.01
num_steps_nve = 5000 
num_steps_nvt = 10000 
equilibrium_steps_ratio = 0.5 

print("--- Part a) NVE Simulations ---")

# Case 1: N = 100, NVE
print("\n--- N=100, NVE, dt=0.01 ---")
sim_n100_dt01 = LennardJones2D(L=L, N=100, rc=rc, dt=0.01)
sim_n100_dt01.run_simulation(num_steps=num_steps_nve, initial_T=0.5, ensemble='NVE')
sim_n100_dt01.plot_energies("NVE: N=100, dt=0.01 - Energy Evolution")
sim_n100_dt01.plot_momentum("NVE: N=100, dt=0.01 - Total Momentum")
sim_n100_dt01.plot_temperature("NVE: N=100, dt=0.01 - Temperature Evolution")
sim_n100_dt01.plot_particle_distribution("NVE: N=100, dt=0.01 - Final Particle Distribution")


# Case 2: N = 400 (Example of different N, perfect square), NVE
print("\n--- N=400, NVE, dt=0.01 ---")
sim_n400_dt01 = LennardJones2D(L=L, N=400, rc=rc, dt=0.01)
sim_n400_dt01.run_simulation(num_steps=num_steps_nve, initial_T=0.5, ensemble='NVE')
sim_n400_dt01.plot_energies("NVE: N=400, dt=0.01 - Energy Evolution")
sim_n400_dt01.plot_momentum("NVE: N=400, dt=0.01 - Total Momentum")
sim_n400_dt01.plot_temperature("NVE: N=400, dt=0.01 - Temperature Evolution")


# Case 3: N = 900, NVE
print("\n--- N=900, NVE, dt=0.01 ---")
sim_n900_dt01 = LennardJones2D(L=L, N=900, rc=rc, dt=0.01)
sim_n900_dt01.run_simulation(num_steps=num_steps_nve, initial_T=0.5, ensemble='NVE')
sim_n900_dt01.plot_energies("NVE: N=900, dt=0.01 - Energy Evolution")
sim_n900_dt01.plot_momentum("NVE: N=900, dt=0.01 - Total Momentum")
sim_n900_dt01.plot_temperature("NVE: N=900, dt=0.01 - Temperature Evolution")


# Case 4: N = 100, NVE, different dt (e.g., dt=0.005) - Smaller dt
print("\n--- N=100, NVE, dt=0.005 ---")
sim_n100_dt005 = LennardJones2D(L=L, N=100, rc=rc, dt=0.005)
sim_n100_dt005.run_simulation(num_steps=num_steps_nve, initial_T=0.5, ensemble='NVE')
sim_n100_dt005.plot_energies("NVE: N=100, dt=0.005 - Energy Evolution")
sim_n100_dt005.plot_momentum("NVE: N=100, dt=0.005 - Total Momentum")
sim_n100_dt005.plot_temperature("NVE: N=100, dt=0.005 - Temperature Evolution")

# Case 5: N = 100, NVE, different dt (e.g., dt=0.02)
print("\n--- N=100, NVE, dt=0.02 ---")
sim_n100_dt02 = LennardJones2D(L=L, N=100, rc=rc, dt=0.02)
sim_n100_dt02.run_simulation(num_steps=num_steps_nve, initial_T=0.5, ensemble='NVE')
sim_n100_dt02.plot_energies("NVE: N=100, dt=0.02 - Energy Evolution")
sim_n100_dt02.plot_momentum("NVE: N=100, dt=0.02 - Total Momentum")
sim_n100_dt02.plot_temperature("NVE: N=100, dt=0.02 - Temperature Evolution")


print("\n--- Part b) NVT Simulations with Berendsen Thermostat ---")
tau_dt_ratio = 0.0025 # Given in problem statement

# N = 100, T = 0.1
print("\n--- N=100, T=0.1, NVT ---")
eq_steps_100_01 = int(num_steps_nvt * equilibrium_steps_ratio)
sim_nvt_100_01 = LennardJones2D(L=L, N=100, rc=rc, dt=dt)
sim_nvt_100_01.run_simulation(num_steps=num_steps_nvt, initial_T=0.1, ensemble='NVT',
                              target_T=0.1, tau_dt_ratio=tau_dt_ratio,
                              equilibrium_steps_for_rdf=eq_steps_100_01)
sim_nvt_100_01.plot_temperature("NVT: N=100, T=0.1 - Temperature Evolution")
sim_nvt_100_01.plot_energies("NVT: N=100, T=0.1 - Energy Evolution")
sim_nvt_100_01.plot_rdf("NVT: N=100, T=0.1 - Radial Distribution Function")
sim_nvt_100_01.plot_particle_distribution("NVT: N=100, T=0.1 - Final Particle Distribution")


# N = 100, T = 1.0
print("\n--- N=100, T=1.0, NVT ---")
eq_steps_100_10 = int(num_steps_nvt * equilibrium_steps_ratio)
sim_nvt_100_10 = LennardJones2D(L=L, N=100, rc=rc, dt=dt)
sim_nvt_100_10.run_simulation(num_steps=num_steps_nvt, initial_T=1.0, ensemble='NVT',
                              target_T=1.0, tau_dt_ratio=tau_dt_ratio,
                              equilibrium_steps_for_rdf=eq_steps_100_10)
sim_nvt_100_10.plot_temperature("NVT: N=100, T=1.0 - Temperature Evolution")
sim_nvt_100_10.plot_energies("NVT: N=100, T=1.0 - Energy Evolution")
sim_nvt_100_10.plot_rdf("NVT: N=100, T=1.0 - Radial Distribution Function")
sim_nvt_100_10.plot_particle_distribution("NVT: N=100, T=1.0 - Final Particle Distribution")


# N = 625, T = 1.0 (perfect square)
print("\n--- N=625, T=1.0, NVT ---")
eq_steps_625_10 = int(num_steps_nvt * equilibrium_steps_ratio)
sim_nvt_625_10 = LennardJones2D(L=L, N=625, rc=rc, dt=dt)
sim_nvt_625_10.run_simulation(num_steps=num_steps_nvt, initial_T=1.0, ensemble='NVT',
                              target_T=1.0, tau_dt_ratio=tau_dt_ratio,
                              equilibrium_steps_for_rdf=eq_steps_625_10)
sim_nvt_625_10.plot_temperature("NVT: N=625, T=1.0 - Temperature Evolution")
sim_nvt_625_10.plot_energies("NVT: N=625, T=1.0 - Energy Evolution")
sim_nvt_625_10.plot_rdf("NVT: N=625, T=1.0 - Radial Distribution Function")
sim_nvt_625_10.plot_particle_distribution("NVT: N=625, T=1.0 - Final Particle Distribution")


# N = 900, T = 1.0 (perfect square)
print("\n--- N=900, T=1.0, NVT ---")
eq_steps_900_10 = int(num_steps_nvt * equilibrium_steps_ratio)
sim_nvt_900_10 = LennardJones2D(L=L, N=900, rc=rc, dt=dt)
sim_nvt_900_10.run_simulation(num_steps=num_steps_nvt, initial_T=1.0, ensemble='NVT',
                              target_T=1.0, tau_dt_ratio=tau_dt_ratio,
                              equilibrium_steps_for_rdf=eq_steps_900_10)
sim_nvt_900_10.plot_temperature("NVT: N=900, T=1.0 - Temperature Evolution")
sim_nvt_900_10.plot_energies("NVT: N=900, T=1.0 - Energy Evolution")
sim_nvt_900_10.plot_rdf("NVT: N=900, T=1.0 - Radial Distribution Function")
sim_nvt_900_10.plot_particle_distribution("NVT: N=900, T=1.0 - Final Particle Distribution")


plt.show()
