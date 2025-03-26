from scipy.constants import epsilon_0, e, k
import numpy as np
from scipy.integrate import odeint
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cbernoulli
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
from scipy.optimize import root
from itertools import accumulate
from multiprocessing import Pool
from itertools import repeat
from params import vac,  freq_ct, sigma_pero, nx_htl, nx_pero, nx_etl, consider_cations, t_high, vapp, vlow, points_early_time, start_sim
import sys
import scipy.integrate as integrate
from scipy.ndimage import convolve1d
from lmfit import Parameters, minimize


class Simulation():
    """ This is the simulation class that is used to compute transient simulations and capacitance frequency simulations in the dark at 0V DC bias. 
        The basic functionalities are the simulate capacitance transient, current transient, and capcaitance frequency simulations.
    """

    def __init__(self, device, timesteps=[-1], init_q2=None,  coarse_x=False, grid=None):
        """Initialization of simulation object

        Args:
            device (Device): Takes object of the Device class
            timesteps (array, optional): Array containing the timesteps for transient simulations. For capacitance frequency simulations, [-1] or nothing should be passed. Defaults to [-1].
            init_q2 (_type_, optional): Initial density of accumulated ions at the perokvsite/HTL interface. Defaults to None.
            coarse_x (bool, optional): If True, the coarse_x grid parameters in the params . Defaults to False.
            grid (_type_, optional): _description_. Defaults to None.
        """
        self.device = device  # object of Device class
        self.init_q2 = init_q2  # Initially accumulated charges in region II
        self.timesteps = np.array(timesteps)  # Define timesteps
        self.original_timesteps = np.array(
            timesteps)  # Define original timesteps
        if not -1 in timesteps:  # Add additional early timesteps to correctly solve transients for early times
            self.early_timesteps = np.logspace(
                start_sim, np.log10(self.timesteps[0]), points_early_time)[:-1]
            self.timesteps = np.concatenate(
                (self.early_timesteps, timesteps, ))
        # Define nuber of timesteps
        self.num_timesteps = len(self.timesteps)

        # Initialize time dependent variables
        self.n1_t = None
        self.w1_t = None
        self.n2_t = None
        self.w2_t = None
        self.n4_t = None
        self.w4_t = None
        self.n5_t = None
        self.w5_t = None
        self.e_field = None
        self.pot = None
        self.net_density = None
        self.p_dc = None
        self.n_dc = None
        self.c_dc = None
        self.h = None
        self.position_steps = None
        self.x_resolution = None

        # Use passed grid
        if not grid is None:
            self.position_steps = grid
            self.x_resolution = len(grid)
            self.h = np.diff(grid)
        # Otherwise construct grid
        else:
            if coarse_x:
                self.construct_grid(sigma_pero, coarse_nx_pero,
                                    coarse_nx_htl, coarse_nx_etl)
            else:
                self.construct_grid(sigma_pero, nx_pero, nx_htl, nx_etl)

        # Initialize AC variables
        self.psi_ac = np.zeros(
            (self.num_timesteps, self.x_resolution), dtype=complex)
        self.p_ac = np.zeros(
            (self.num_timesteps, self.x_resolution), dtype=complex)
        self.n_ac = np.zeros(
            (self.num_timesteps, self.x_resolution), dtype=complex)
        self.c_ac = np.zeros(
            (self.num_timesteps, self.x_resolution), dtype=complex)
        self.jn_ac = np.zeros(
            (self.num_timesteps, self.x_resolution), dtype=complex)
        self.jp_ac = np.zeros(
            (self.num_timesteps, self.x_resolution), dtype=complex)
        self.jc_ac = np.zeros(
            (self.num_timesteps, self.x_resolution), dtype=complex)
        self.jdisp_ac = np.zeros(
            (self.num_timesteps, self.x_resolution), dtype=complex)

        # Define indexes of where the different regions start/end
        self.idx_w1 = np.zeros(self.num_timesteps, dtype=int)
        self.idx_interface1_1 = np.where(
            self.position_steps >= self.device.d_htl)[0][0]
        self.idx_interface1_2 = np.where(
            self.position_steps >= self.device.d_htl)[0][1]
        self.idx_w2 = np.zeros(self.num_timesteps, dtype=int)
        self.idx_w4 = np.zeros(self.num_timesteps, dtype=int)
        self.idx_interface2_1 = np.where(
            self.position_steps >= (self.device.d_htl + self.device.d_pero))[0][0]
        self.idx_interface2_2 = np.where(
            self.position_steps >= (self.device.d_htl + self.device.d_pero))[0][1]
        self.idx_w5 = np.zeros(self.num_timesteps, dtype=int)

        # Create dielectric constant array array
        self.arr_epsilon = np.full(
            len(self.position_steps), self.device.epsr_htl, )
        self.arr_epsilon[self.idx_interface1_2:self.idx_interface2_2] = self.device.epsr_pero
        self.arr_epsilon[self.idx_interface2_2:] = self.device.epsr_etl

        # Initialize the conductance, capacitance, currents, impednace
        self.Ct_Gs = np.zeros(self.num_timesteps)
        self.Ct_Cs = np.zeros(self.num_timesteps)
        self.Jt_ion = np.zeros(self.num_timesteps)
        self.Jt_disp = np.zeros(self.num_timesteps)
        self.Jt = np.zeros(self.num_timesteps)
        self.Ct_Zs = np.zeros(self.num_timesteps)

        # Define applied voltage during pulse (vapp), voltage during transient (vlow) and ac voltage (vac)
        self.vapp = vapp
        self.vlow = vlow
        self.vac = vac

        # Normalization parameters
        self.vt = k*self.device.temp/e
        self.x0 = max(self.h)
        self.psi0 = k*self.device.temp/e
        self.c0 = np.max([self.device.n_htl, self.device.n_etl])
        self.d0 = k*self.device.temp * \
            np.max([self.device.mu_n_pero, self.device.mu_p_pero])/e
        self.lambda_htl = np.sqrt(self.psi0*self.device.epsr_htl *
                                  epsilon_0/(e*self.c0*self.x0**2))
        self.lambda_pero = np.sqrt(self.psi0*self.device.epsr_pero *
                                   epsilon_0/(e*self.c0*self.x0**2))
        self.lambda_etl = np.sqrt(self.psi0*self.device.epsr_etl *
                                  epsilon_0/(e*self.c0*self.x0**2))
        self.t0 = self.x0**2/self.d0
        self.consider_cations = consider_cations

    def construct_grid(self, sigma_pero, nx_pero, nx_htl, nx_etl):
        """Function to create the grid for the simulation 

        Args:
            sigma_pero (float): Variable that impacts the resolution of the grid at the perovskite/CTL interfaces
            nx_pero (int): Number of grid points in the perovskite layer
            nx_htl (int): Number of grid points in the HTL layer
            nx_etl (int): Number of grid points in the ETL layer
        """

        def get_x_right_interface(sigma, nx, d, startx=None):
            """Local function to get the x values for the right interface

            Args:
                sigma (float): Variable that impacts the resolution of the grid at the perovskite/CTL interfaces
                nx (int): Number of x points
                d (float): Thickness of the layer
                startx (float, optional): Starting value of the x position. Defaults to None.

            Returns:
                numpy array: Array containing the x values
            """
            x = np.array([0.5*(np.tanh(sigma*(2*idx/(nx-1) - 1))/np.tanh(sigma)+1)
                          for idx in range(nx, 2*nx)])
            x = x - x[0]
            if startx != None:
                return startx + x*d/x[-1]
            else:
                return x*d/x[-1]

        def get_x_left_interface(sigma, nx, d, startx=None):
            """Local function to get the x values for the left interface

            Args:
                sigma (float): Variable that impacts the resolution of the grid at the perovskite/CTL interfaces
                nx (int): Number of x points
                d (float): Thickness of the layer
                startx (float, optional): Starting value of the x position. Defaults to None.

            Returns:
                numpy array: Array containing the x values
            """
            x = np.array([0.5*(np.tanh(sigma*(2*idx/(nx-1) - 1))/np.tanh(sigma)+1)
                          for idx in range(nx, 2*nx)])
            x = np.flip(x)
            x = x - x[0]
            if startx != None:
                return startx + x*d/x[-1]
            else:
                return x*d/x[-1]

        def get_x_sym(sigma, nx, d, startx=None):
            """Local function to get the x values for a layer with left and right interface

            Args:
                sigma (float): Variable that impacts the resolution of the grid at the perovskite/CTL interfaces
                nx (int): Number of x points
                d (float): Thickness of the layer
                startx (float, optional): Starting value of the x position. Defaults to None.

            Returns:
                numpy array: Array containing the x values
            """
            x = np.array([0.5*(np.tanh(sigma*(2*idx/(nx-1) - 1))/np.tanh(sigma)+1)
                          for idx in range(0, nx)])
            if startx != None:
                return startx + x*d/x[-1]
            else:
                return x*d/x[-1]

        def residual_htl(params):
            """Residual for fitting the x values in the HTL so they have the same resolution as in the perovksite

            Args:
                params (Parameters object): Fitting parameters

            Returns:
                float: This difference between the minimum x resolution in the HTL and the minimum x resolution in the perovskite
            """
            x_htl = get_x_right_interface(
                params['sigma_htl'], nx_htl, self.device.d_htl)
            h_htl = np.diff(x_htl)
            return min(h_htl) - min(h_pero)

        def residual_etl(params):
            """Residual for fitting the x values in the ETL so they have the same resolution as in the perovskite

            Args:
                params (Parameters object): Fitting parameters

            Returns:
                float: This difference between the minimum x resolution in the ETL and the minimum x resolution in the perovskite
            """
            x_etl = get_x_left_interface(
                params['sigma_etl'], nx_etl, self.device.d_etl)
            h_etl = np.diff(x_etl)
            return min(h_etl) - min(h_pero)

        # We start by creating the grid for the perovskite layer
        x_pero = get_x_sym(sigma_pero, nx_pero, self.device.d_pero)
        h_pero = np.diff(x_pero)

        # We then fit the x values for the HTL and ETL so they have the same resolution as the perovskite
        params = Parameters()
        params.add('sigma_htl', value=1, min=0.0001, max=5)
        out = minimize(residual_htl, params, )
        sigma_htl = out.params['sigma_htl'].value

        params = Parameters()
        params.add('sigma_etl', value=1, min=0.0001, max=5)
        out = minimize(residual_etl, params, )
        sigma_etl = out.params['sigma_etl'].value

        # Here the grid in the ETL is constructed so it matches the resolution of the perovskite at the interface
        while True:
            x_htl = get_x_right_interface(sigma_htl, nx_htl, self.device.d_htl)

            fits_left_int = round(min(np.diff(x_htl)), 15) == round(
                min(np.diff(x_pero[:10])), 15)

            if fits_left_int == True:
                break
            else:
                nx_htl -= 1
                print('Reducing Points in HTL to {}'.format(nx_htl))
                if nx_htl < 5:
                    print(
                        'Number of points in HTL has fallen below 3. Please change sigma of perovskite')
                    sys.exit()
                params = Parameters()
                params.add('sigma_htl', value=1, min=0.0001, max=5)
                out = minimize(residual_htl, params, )
                sigma_htl = out.params['sigma_htl'].value

        x_pero = get_x_sym(sigma_pero, nx_pero,
                           self.device.d_pero, x_htl[-1])

        # Here the grid in the ETL is constructed so it matches the resolution of the perovskite at the interface
        while True:
            x_etl = get_x_left_interface(
                sigma_etl, nx_etl, self.device.d_etl, x_pero[-1])

            fits_right_int = round(
                min(np.diff(x_etl)), 15) == round(min(np.diff(x_pero[10:])), 15)

            if fits_right_int == True:
                break
            else:
                nx_etl -= 1
                print('Reducing Points in ETL to {}'.format(nx_etl))
                if nx_etl < 5:
                    print(
                        'Number of points in ETL has fallen below 3. Please change sigma of perovskite')
                    sys.exit()

                params = Parameters()
                params.add('sigma_etl', value=1, min=0.0001, max=5)
                out = minimize(residual_etl, params, )
                sigma_etl = out.params['sigma_etl'].value

        # Finally, we get the x positions for the complete device
        self.position_steps = np.concatenate([x_htl, x_pero, x_etl])
        self.x_resolution = len(self.position_steps)
        self.h = np.diff(self.position_steps)

    def ode_const_w2q(self, q2, t=None, vapp=0.0):
        """Function of the differential equation 

        Args:
            q2 (float): charge in region II
            t (float, optional): time. Defaults to None.
            vapp (float, optional): Applied voltage. Defaults to 0.

        Returns:
            _type_: dq2/dt
        """
        e3 = self.get_e3q(q2, vapp)
        n2, w2 = self.get_n2w2(q2)

        if (w2 + n2 * w2 / self.device.n_ion) > self.device.d_pero:
            return -0

        return - e * self.device.mu_ion * self.device.n_ion * e3

    def get_w2(self, n):
        """Returning the debye width of region II in the perovskite

        Args:
            n (float): charge density

        Returns:
            float: width of region II
        """
        return np.sqrt((epsilon_0*self.device.epsr_pero*self.device.vt)/(e*n))

    def get_Ld(self, n, epsr):
        """Returning the debye width

        Args:
            n (float): charge density

        Returns:
            float: debye width
        """
        return np.sqrt((epsilon_0*epsr*self.device.vt)/(e*n))

    def get_e3q(self, q2, vapp=0.0):
        """Function to calculate the bulk electric field

        Args:
            q2 (float): charge in region II
            vapp (float, optional): Applied voltage. Defaults to 0.0

        Returns:
            float: electric field in region III
        """

        # First we calcuate the charge density and width of region II dependent on the charge
        n2, w2 = self.get_n2w2(q2)

        # Case if the HTL and ETL is doped
        if self.device.htl_doping == True and self.device.etl_doping == True:

            # Normal accumulation (positive charges accumulate at the HTL/perovskite interface)
            if q2[0] >= 0:

                # Define a,b,c to solve the quadratic equation
                a = (epsilon_0*self.device.epsr_pero**2/(2*e))*(1/(self.device.n_htl *
                                                                self.device.epsr_htl) + 1/(self.device.n_etl*self.device.epsr_etl))
                b = -1*(w2*n2*self.device.epsr_pero) * (1/(self.device.n_htl *
                                                           self.device.epsr_htl) + 1/(self.device.n_etl*self.device.epsr_etl)) - self.device.d_pero
                c = (n2*w2)**2*e/2 * (1/(self.device.n_htl*epsilon_0*self.device.epsr_htl) + 1/(self.device.n_etl*epsilon_0*self.device.epsr_etl)
                                      ) + e*n2*w2**2/(2*epsilon_0 * self.device.epsr_pero) * (1 + n2/self.device.n_ion) - (self.device.v_bi - vapp)

                # Ignore cases where b**2 < 4*a*c
                if b**2 < 4*a*c:
                    e3 = 10e3
                else:
                    e3 = (-b - np.sqrt(b**2-4*a*c))/(2*a)

                # Calculate the depletion width w1 and w5
                w1 = (w2*n2*e-self.device.epsr_pero*epsilon_0 * e3) / \
                    (self.device.n_htl * e)
                w5 = (w2*n2*e-self.device.epsr_pero*epsilon_0 * e3) / \
                    (self.device.n_etl * e)

                # Recalculate the electric field if the depletion with in the ETL is larger than the width
                if (w5 > self.device.d_etl) and not (w1 > self.device.d_htl):

                    a = (epsilon_0 * self.device.epsr_pero)**2 / \
                        (2*epsilon_0 * self.device.epsr_htl * self.device.n_htl * e)

                    b = -1 * (self.device.epsr_pero * n2 * w2 / (self.device.epsr_htl * self.device.n_htl) +
                              self.device.d_pero + self.device.epsr_pero / self.device.epsr_etl*self.device.d_etl)

                    c = n2**2 * w2**2 * e / (2*epsilon_0 * self.device.epsr_htl*self.device.n_htl) + e*n2*w2**2 / (2*epsilon_0 * self.device.epsr_pero) + e*n2**2*w2**2 / self.device.n_ion / (
                        2*epsilon_0 * self.device.epsr_pero) - e*self.device.n_etl * self.device.d_etl**2 / (2*epsilon_0 * self.device.epsr_etl) + e*n2*w2*self.device.d_etl / (epsilon_0*self.device.epsr_etl) - (self.device.v_bi - vapp)

                    if b**2 < 4*a*c:
                        e3 = 10e3
                    else:
                        e3 = (-b - np.sqrt(b**2-4*a*c))/(2*a)

                    w1 = (w2*n2*e-self.device.epsr_pero*epsilon_0 * e3) / \
                        (self.device.n_htl * e)
                    w5 = (w2*n2*e-self.device.epsr_pero*epsilon_0 * e3) / \
                        (self.device.n_etl * e)

                # Recalculate the electric field if the depletion with in the HTL is larger than the width
                elif not (w5 > self.device.d_etl) and (w1 > self.device.d_htl):

                    a = (epsilon_0 * self.device.epsr_pero)**2 / \
                        (2*epsilon_0 * self.device.epsr_etl * self.device.n_etl * e)

                    b = -1 * (self.device.epsr_pero * n2 * w2 / (self.device.epsr_etl * self.device.n_etl) +
                              self.device.d_pero + self.device.epsr_pero / self.device.epsr_htl*self.device.d_htl)

                    c = n2**2 * w2**2 * e / (2*epsilon_0 * self.device.epsr_etl*self.device.n_etl) + e*n2*w2**2 / (2*epsilon_0 * self.device.epsr_pero) + e*n2**2*w2**2 / self.device.n_ion / (
                        2*epsilon_0 * self.device.epsr_pero) - e*self.device.n_htl * self.device.d_htl**2 / (2*epsilon_0 * self.device.epsr_htl) + e*n2*w2*self.device.d_htl / (epsilon_0*self.device.epsr_htl) - (self.device.v_bi - vapp)

                    if b**2 < 4*a*c:
                        e3 = 10e3
                    else:
                        e3 = (-b - np.sqrt(b**2-4*a*c))/(2*a)

                    w1 = (w2*n2*e-self.device.epsr_pero*epsilon_0 * e3) / \
                        (self.device.n_htl * e)
                    w5 = (w2*n2*e-self.device.epsr_pero*epsilon_0 * e3) / \
                        (self.device.n_etl * e)

                # Recalculate the electric field if the depletion with in the ETL and the HTL are larger than their width
                if (w5 > self.device.d_etl) and (w1 > self.device.d_htl):

                    e3 = (e*n2 * w2**2 / (2*epsilon_0*self.device.epsr_pero) + e*n2**2 * w2**2 / self.device.n_ion / (2*epsilon_0*self.device.epsr_pero) - self.device.n_htl * e * self.device.d_htl **
                          2 / (2*epsilon_0 * self.device.epsr_htl) + e*n2*w2 * self.device.d_htl / (epsilon_0 * self.device.epsr_htl) - self.device.n_etl * e * self.device.d_etl **
                          2 / (2*epsilon_0 * self.device.epsr_etl) + e*n2*w2 * self.device.d_etl / (epsilon_0 * self.device.epsr_etl) - (self.device.v_bi - vapp)) / (self.device.d_pero + self.device.epsr_pero / self.device.epsr_htl * self.device.d_htl + self.device.epsr_pero / self.device.epsr_etl * self.device.d_etl)

            # If the accumulated charge at the HTL/perovskite interface is negative, define the ionic accumulation layer at the perovskite/ETL interface
            else:
                w4 = self.get_Ld(self.device.n_ion, self.device.epsr_pero)
                a = (epsilon_0*self.device.epsr_pero**2/(2*e))*(1/(self.device.n_htl *
                                                                self.device.epsr_htl) + 1/(self.device.n_etl*self.device.epsr_etl))
                b = -1*(w2*n2*self.device.epsr_pero) * (1/(self.device.n_htl *
                                                           self.device.epsr_htl) + 1/(self.device.n_etl*self.device.epsr_etl)) - self.device.d_pero
                c = (n2*w2)**2*e/2 * (1/(self.device.n_htl*epsilon_0*self.device.epsr_htl) + 1/(self.device.n_etl*epsilon_0*self.device.epsr_etl)
                                      ) + e*n2*w2/(2*epsilon_0 * self.device.epsr_pero) * (w2+w4) - (self.device.v_bi - vapp)
                if b**2 < 4*a*c:
                    e3 = 10e3
                else:
                    e3 = (-b - np.sqrt(b**2-4*a*c))/(2*a)

                # If the accumulated charge at perovksite/ETL interace is large enough it gets screened by electric charges that acucmulate in Debye alyers at the interfaces
                if q2[0] < self.device.epsr_pero * epsilon_0 * e3:
                    w1 = self.get_Ld(self.device.n_htl, self.device.epsr_htl)
                    w5 = self.get_Ld(self.device.n_etl, self.device.epsr_etl)
                    e3 = (w1*w2*n2*e/(2*epsilon_0*self.device.epsr_htl) + e*n2*w2**2/(2*epsilon_0*self.device.epsr_pero) + e*w4*w2*n2/(2*epsilon_0*self.device.epsr_pero) + w5*w2*n2*e/(2*epsilon_0 *
                          self.device.epsr_etl) - (self.device.v_bi - vapp)) / (self.device.epsr_pero * w1 / (2*self.device.epsr_htl) + self.device.epsr_pero * w5 / (2*self.device.epsr_etl) + self.device.d_pero)

        # Case if the HTL and ETL are undoped
        elif self.device.htl_doping == False and self.device.etl_doping == False:
            e3 = (-self.device.v_bi + vapp + e * n2 * w2 * self.device.d_htl / (epsilon_0 * self.device.epsr_htl) + e * n2 * w2 * self.device.d_etl / (
                epsilon_0 * self.device.epsr_etl) + e/(2*epsilon_0*self.device.epsr_pero)*(n2*w2**2)*(1+n2/self.device.n_ion)) / (self.device.epsr_pero/self.device.epsr_etl * self.device.d_etl + self.device.d_pero + self.device.epsr_pero/self.device.epsr_htl * self.device.d_htl)

        # Case if the HTL is undoped and the ETL is doped
        elif self.device.htl_doping == False and self.device.etl_doping == True:

            # Normal accumulation (positive charges accumulate at the HTL/perovskite interface)
            if q2[0] >= 0:
                a = self.device.epsr_pero**2 * epsilon_0 / \
                    (2 * self.device.epsr_etl * self.device.n_etl * e)
                b = -1 * (self.device.epsr_pero / self.device.epsr_htl * self.device.d_htl + self.device.d_pero +
                          self.device.epsr_pero * n2 * w2 /
                          (self.device.epsr_etl * self.device.n_etl))
                c = e*n2*w2*(self.device.d_htl/(epsilon_0*self.device.epsr_htl) + w2/(2*epsilon_0 * self.device.epsr_pero) + n2 *
                             w2*(1/(2*epsilon_0*self.device.epsr_pero*self.device.n_ion) + 1/(2*epsilon_0 * self.device.epsr_etl * self.device.n_etl))) - (self.device.v_bi - vapp)

                if b**2 < 4*a*c:
                    e3 = 10e3
                else:
                    e3 = (-b - np.sqrt(b**2-4*a*c))/(2*a)

                # Recalculate the electric field if the depletion with in the ETL is larger than the thickness
                if (w2*n2*e-self.device.epsr_pero*epsilon_0 * e3) / (self.device.n_etl * e) > self.device.d_etl:

                    e3 = (e*n2*w2*self.device.d_htl / (epsilon_0 * self.device.epsr_htl) + e*n2*w2**2 / (2 * epsilon_0 * self.device.epsr_pero) + e*n2**2 * w2**2 / self.device.n_ion / (2*epsilon_0 * self.device.epsr_pero) - e*self.device.n_etl * self.device.d_etl**2 / (2*epsilon_0 *
                          self.device.epsr_etl) + e*n2*w2*self.device.d_etl / (epsilon_0 * self.device.epsr_etl) - (self.device.v_bi - vapp)) / (self.device.epsr_pero / self.device.epsr_htl * self.device.d_htl + self.device.d_pero + self.device.epsr_pero / self.device.epsr_etl * self.device.d_etl)

            # If the accumulated charge at the HTL/perovskite interface is negative, define the ionic accumulation layer at the perovskite/ETL interface
            else:

                w4 = self.get_Ld(self.device.n_ion, self.device.epsr_pero)

                a = self.device.epsr_pero**2 * epsilon_0 / \
                    (2 * self.device.epsr_etl * self.device.n_etl * e)
                b = -1 * (self.device.epsr_pero / self.device.epsr_htl * self.device.d_htl + w2 * n2 *
                          self.device.epsr_pero / (self.device.epsr_etl * self.device.n_etl) + self.device.d_pero)
                c = e * n2 * w2 * self.device.d_htl / (epsilon_0 * self.device.epsr_htl) + e * n2 * w2**2 / (2 * epsilon_0 * self.device.epsr_pero) + e * w4 * n2 * w2 / (
                    2 * epsilon_0 * self.device.epsr_pero) + (n2*w2)**2 * e / (2*epsilon_0 * self.device.epsr_etl * self.device.n_etl) - (self.device.v_bi - vapp)

                if b**2 < 4*a*c:
                    e3 = 10e3
                else:
                    e3 = (-b - np.sqrt(b**2-4*a*c))/(2*a)
                # If the accumulated charge at perovksite/ETL interace is large enough it gets screened by electric charges that acucmulate in Debye alyers at the interfaces
                if q2[0] <= self.device.epsr_pero * epsilon_0 * e3:

                    w5 = self.get_Ld(self.device.n_etl, self.device.epsr_etl)

                    e3 = (e*n2*w2*self.device.d_htl/(epsilon_0*self.device.epsr_htl) + e*n2*w2**2/(2*epsilon_0*self.device.epsr_pero) + e*n2*w2*w4/(2*epsilon_0*self.device.epsr_pero) + e*n2*w2*w5/(2*epsilon_0 *
                                                                                                                                                                                                     self.device.epsr_etl) - (self.device.v_bi - vapp)) / (self.device.epsr_pero/self.device.epsr_htl * self.device.d_htl + self.device.epsr_pero*w5/(2*self.device.epsr_etl) + self.device.d_pero)

        # Case if the HTL is doped and the ETL is undoped
        elif self.device.htl_doping == True and self.device.etl_doping == False:

            # Normal accumulation (positive charges accumulate at the HTL/perovskite interface)
            if q2[0] >= 0:
                a = self.device.epsr_pero**2 * epsilon_0 / \
                    (2*self.device.epsr_htl*e*self.device.n_htl)
                b = -1*(self.device.epsr_pero/self.device.epsr_etl * self.device.d_etl + self.device.d_pero +
                        self.device.epsr_pero * n2*w2/(self.device.epsr_htl * self.device.n_htl))
                c = e*n2*w2*(self.device.d_etl/(epsilon_0*self.device.epsr_etl) + w2/(2*epsilon_0 * self.device.epsr_pero) + n2*w2*(1/(2*epsilon_0 *
                                                                                                                                       self.device.epsr_pero*self.device.n_ion) + 1/(2*epsilon_0*self.device.epsr_htl*self.device.n_htl))) - (self.device.v_bi - vapp)
                e3 = (-b - np.sqrt(b**2-4*a*c))/(2*a)

                if b**2 < 4*a*c:
                    e3 = 10e3
                else:
                    e3 = (-b - np.sqrt(b**2-4*a*c))/(2*a)

                # Recalculate the electric field if the depletion with in the HTL is larger than the thickness
                if (w2*n2*e-self.device.epsr_pero*epsilon_0 * e3) / (self.device.n_htl * e) > self.device.d_htl:
                    e3 = (e*n2*w2*self.device.d_etl / (epsilon_0 * self.device.epsr_etl) + e*n2*w2**2 / (2*epsilon_0 * self.device.epsr_pero) + e * n2**2 * w2**2 / self.device.n_ion / (2*epsilon_0 * self.device.epsr_pero) - self.device.n_htl * e * self.device.d_htl**2 / (2*epsilon_0 *
                          self.device.epsr_htl) + n2 * w2 * e * self.device.d_htl / (epsilon_0 * self.device.epsr_htl) - (self.device.v_bi - vapp)) / (self.device.epsr_pero / self.device.epsr_htl * self.device.d_htl + self.device.d_pero + self.device.epsr_pero / self.device.epsr_etl * self.device.d_etl)

            # If the accumulated charge at the HTL/perovskite interface is negative, define the ionic accumulation layer at the perovskite/ETL interface
            else:
                w4 = self.get_Ld(self.device.n_ion, self.device.epsr_pero)

                a = self.device.epsr_pero**2 * epsilon_0 / \
                    (2 * self.device.epsr_htl * self.device.n_htl * e)
                b = -1 * (self.device.epsr_pero / self.device.epsr_etl * self.device.d_etl + w2 * n2 *
                          self.device.epsr_pero / (self.device.epsr_htl * self.device.n_htl) + self.device.d_pero)
                c = e * n2 * w2 * self.device.d_etl / (epsilon_0 * self.device.epsr_etl) + e * n2 * w2**2 / (2 * epsilon_0 * self.device.epsr_pero) + e * w4 * n2 * w2 / (
                    2 * epsilon_0 * self.device.epsr_pero) + (n2*w2)**2 * e / (2*epsilon_0 * self.device.epsr_htl * self.device.n_htl) - (self.device.v_bi - vapp)

                e3 = (-b - np.sqrt(b**2-4*a*c))/(2*a)
                # If the accumulated charge at perovksite/ETL interace is large enough it gets screened by electric charges that acucmulate in Debye alyers at the interfaces
                if q2[0] <= self.device.epsr_pero * epsilon_0 * e3:
                    w1 = self.get_Ld(self.device.n_htl, self.device.epsr_htl)

                    e3 = (e*n2*w2*self.device.d_etl/(epsilon_0*self.device.epsr_etl) + e*n2*w2**2/(2*epsilon_0*self.device.epsr_pero) + e*n2*w2*w4/(2*epsilon_0*self.device.epsr_pero) + e*n2*w2*w1/(2*epsilon_0 *
                          self.device.epsr_htl) - (self.device.v_bi - vapp)) / (self.device.epsr_pero/self.device.epsr_etl * self.device.d_etl + self.device.epsr_pero*w1/(2*self.device.epsr_htl) + self.device.d_pero)
        return e3

    def get_init_q2(self, vapp):
        """Function to compute the accumulated charge q2 dependent on the applied voltage 

        Args:
            vapp (float): Applied voltage
        """

        # Find accumulated charge, where the electric field is 0
        self.init_q2 = root(self.get_e3q, 0.001, args=(vapp)).x

        # Calculate the density and width n2 and w2
        n2, w2 = self.get_n2w2(self.init_q2)

        # Limit the sum of the widths w2 and w4 to be maximum the perovskite thickness
        if (w2 + n2 * w2 / self.device.n_ion) > self.device.d_pero:
            self.init_q2 = e * (self.device.d_pero - w2) * self.device.n_ion

    def get_n2w2(self, q2_t):
        """Function to get n2 and w2

        Args:
            q2_t (float): charge in region II

        Returns:
            floats: density and width in region II
        """
        w2_t = np.zeros(q2_t.shape)
        n2_t = np.zeros(q2_t.shape)

        neg_mask = q2_t < 0
        pos_mask = q2_t >= 0

        # If the charge in region II is negative
        n2_t[neg_mask] = -1*self.device.n_ion
        w2_t[neg_mask] = q2_t[neg_mask] / (e*n2_t[neg_mask])

        # If the charge in region II is positive
        w2_t[pos_mask] = self.get_w2(self.device.n_ion)
        n2_t[pos_mask] = q2_t[pos_mask] / (w2_t[pos_mask] * e)

        return n2_t, w2_t

    def get_n1w1(self, q2_t, e3_t):
        """Function to get n1 and w1

        Args:
            q2_t (float): charge in region II
            e3_t (float): electric field in region III
        Returns:
            floats: density and width in region I
        """
        w1_t = np.zeros(q2_t.shape)
        n1_t = np.zeros(q2_t.shape)

        neg_mask = q2_t < (e3_t * epsilon_0 * self.device.epsr_pero)
        pos_mask = q2_t >= (e3_t * epsilon_0 * self.device.epsr_pero)

        # If electric charges accumulate at the HTL/perovskite interface
        w1_t[neg_mask] = self.get_Ld(self.device.n_htl, self.device.epsr_htl)
        n1_t[neg_mask] = (q2_t[neg_mask] - e3_t[neg_mask] *
                          epsilon_0 * self.device.epsr_pero)/(w1_t[neg_mask]*e)

        # If electric charges deplete from the HTL
        n1_t[pos_mask] = self.device.n_htl
        w1_t[pos_mask] = (q2_t[pos_mask] - e3_t[pos_mask] *
                          epsilon_0 * self.device.epsr_pero)/(n1_t[pos_mask]*e)

        return n1_t, w1_t

    def get_n4w4(self, q2_t):
        """Function to get n4 and w4

        Args:
            q2_t (float): charge in region IV

        Returns:
            floats: density and width in region IV
        """
        w4_t = np.zeros(q2_t.shape)
        n4_t = np.zeros(q2_t.shape)

        neg_mask = q2_t < 0
        pos_mask = q2_t >= 0

        # If ions accumulate in region IV
        w4_t[neg_mask] = self.get_Ld(
            self.device.n_ion, self.device.epsr_pero)
        n4_t[neg_mask] = q2_t[neg_mask] / (w4_t[neg_mask] * e)

        # If the ions deplete from region IV
        n4_t[pos_mask] = self.device.n_ion
        w4_t[pos_mask] = q2_t[pos_mask] / (e*n4_t[pos_mask])

        return n4_t, w4_t

    def get_n5w5(self, q2_t, e3_t):
        """Function to get n5 and w5

        Args:
            q2_t (float): charge in region II
            e3_t (float): electric field in region III
        Returns:
            floats: density and width in region 5
        """
        w5_t = np.zeros(q2_t.shape)
        n5_t = np.zeros(q2_t.shape)

        neg_mask = q2_t < (e3_t * epsilon_0 * self.device.epsr_pero)
        pos_mask = q2_t >= (e3_t * epsilon_0 * self.device.epsr_pero)

        # If electric charges accumulate at the perovskite/ETL interface
        w5_t[neg_mask] = self.get_Ld(
            self.device.n_etl, self.device.epsr_etl)
        n5_t[neg_mask] = (q2_t[neg_mask] - e3_t[neg_mask] *
                          epsilon_0 * self.device.epsr_pero)/(w5_t[neg_mask]*e)

        # If electric charges deplete from the ETL
        n5_t[pos_mask] = self.device.n_etl
        w5_t[pos_mask] = (q2_t[pos_mask] - e3_t[pos_mask] *
                          epsilon_0 * self.device.epsr_pero)/(n5_t[pos_mask]*e)

        return n5_t, w5_t

    def solve_dc(self,):
        """Function to compute the DC solution fter applying a voltage pulse
        """

        # First check if a time dependent solution should be compute (if -1 in timesteps, only calculate steady state)
        if not -1 in self.timesteps:
            # Compute the initial accumulated charge for a finite voltage pulse
            if self.init_q2 == None and t_high != -1:

                # Solve accumulated charge for vapp=vlow
                self.get_init_q2(vapp=self.vlow)

                # Solve accumulated charge for applied bias and length t_high
                q2_t_voltage_pulse = odeint(self.ode_const_w2q,
                                            self.init_q2, t=np.logspace(-5, np.log10(t_high)), args=(self.vapp, ))  # Args contains the applied voltage

                # This is the initially accumulated charge at the end of the voltage pulse
                self.init_q2 = q2_t_voltage_pulse[-1]

                # If device is not depleted
                n2, w2 = self.get_n2w2(self.init_q2)
                if (w2 + n2 * w2 / self.device.n_ion) > self.device.d_pero:
                    self.get_init_q2(vapp=self.vapp)
            # Compute the initial accumulated charge for steady state
            elif self.init_q2 == None and t_high == -1:
                self.get_init_q2(vapp=self.vapp)

            # Save for the accumulated carrier density after the voltage pulse
            self.q2_t = odeint(self.ode_const_w2q,
                               self.init_q2, self.timesteps, args=(self.vlow, ))  # Args contains the applied voltage

        # If -1 in the timesteps, only save the steady state solution
        else:
            self.get_init_q2(vapp=self.vlow)
            self.q2_t = np.array([self.init_q2])

        # Based on q2, now calculate the electric field
        e3_t = np.array([self.get_e3q(q, vapp=self.vlow) for q in self.q2_t])
        self.e3_t = e3_t

        # Compute the densities and widths depending on the calculated time-dependent charge if htl and etl are doped
        if self.device.htl_doping == True and self.device.etl_doping == True:

            self.n2_t, self.w2_t = self.get_n2w2(self.q2_t)
            self.n1_t, self.w1_t = self.get_n1w1(self.q2_t, e3_t)
            self.n4_t, self.w4_t = self.get_n4w4(self.q2_t)
            self.n5_t, self.w5_t = self.get_n5w5(self.q2_t, e3_t)

            # Compute the idexes of the different widths
            for idx, t in enumerate(self.timesteps):
                self.idx_w1[idx] = np.argmax(self.position_steps >=
                                             self.device.d_htl - self.w1_t[idx])
                self.idx_w2[idx] = np.argmax(self.position_steps >= self.device.d_htl +
                                             self.w2_t[idx])
                self.idx_w4[idx] = np.argmax(self.position_steps >=
                                             self.device.d_htl + self.device.d_pero - self.w4_t[idx])
                self.idx_w5[idx] = np.argmax(self.position_steps >=
                                             self.w5_t[idx] + self.device.d_htl + self.device.d_pero)

        # Compute the densities and widths depending on the calculated time-dependent charge if htl and etl are not doped
        elif self.device.htl_doping == False and self.device.etl_doping == False:
            self.n2_t, self.w2_t = self.get_n2w2(self.q2_t)
            self.n4_t, self.w4_t = self.get_n4w4(self.q2_t)

            # Compute the idexes of the different widths
            for idx, t in enumerate(self.timesteps):
                self.idx_w2[idx] = np.argmax(self.position_steps >= self.device.d_htl +
                                             self.w2_t[idx])
                self.idx_w4[idx] = np.argmax(self.position_steps >=
                                             self.device.d_htl + self.device.d_pero - self.w4_t[idx])

        # Compute the densities and widths depending on the calculated time-dependent charge if htl and etl are not doped
        elif self.device.htl_doping == False and self.device.etl_doping == True:
            self.n2_t, self.w2_t = self.get_n2w2(self.q2_t)
            self.n4_t, self.w4_t = self.get_n4w4(self.q2_t)
            self.n5_t, self.w5_t = self.get_n5w5(self.q2_t, e3_t)

            # Compute the idexes of the different widths
            for idx, t in enumerate(self.timesteps):
                self.idx_w2[idx] = np.argmax(self.position_steps >= self.device.d_htl +
                                             self.w2_t[idx])
                self.idx_w4[idx] = np.argmax(self.position_steps >=
                                             self.device.d_htl + self.device.d_pero - self.w4_t[idx])
                self.idx_w5[idx] = np.argmax(self.position_steps >=
                                             self.w5_t[idx] + self.device.d_htl + self.device.d_pero)

        # Compute the densities and widths depending on the calculated time-dependent charge if htl and etl are not doped
        elif self.device.htl_doping == True and self.device.etl_doping == False:
            self.n2_t, self.w2_t = self.get_n2w2(self.q2_t)
            self.n4_t, self.w4_t = self.get_n4w4(self.q2_t)
            self.n1_t, self.w1_t = self.get_n1w1(self.q2_t, e3_t)

            # Compute the idexes of the different widths
            for idx, t in enumerate(self.timesteps):
                self.idx_w1[idx] = np.argmax(self.position_steps >=
                                             self.device.d_htl - self.w1_t[idx])
                self.idx_w2[idx] = np.argmax(self.position_steps >= self.device.d_htl +
                                             self.w2_t[idx])
                self.idx_w4[idx] = np.argmax(self.position_steps >=
                                             self.device.d_htl + self.device.d_pero - self.w4_t[idx])

    def calc_potential(self, vapp=0.0):
        """Function to calcualte the potential dependent on the applied bias

        Args:
            vapp (float, optional): Applied bias. Defaults to 0.
        """

        # Initilize the potential array
        pot = np.zeros((self.num_timesteps, self.x_resolution))

        # Doped HTL and doped ETL
        if self.device.htl_doping is True and self.device.etl_doping is True:

            for idx, t in enumerate(self.timesteps):

                e3 = self.get_e3q(self.q2_t[idx], vapp)

                # Region 0
                x = self.position_steps[:self.idx_w1[idx]
                                        ]-(self.device.d_htl - self.w1_t[idx])
                pot[idx, :self.idx_w1[idx]] = - (self.device.v_bi - vapp)

                # Region I
                if (self.w2_t[idx]*self.n2_t[idx]*e-self.device.epsr_pero*epsilon_0 * e3) / (self.device.n_htl * e) <= self.device.d_htl:
                    x = (self.position_steps[self.idx_w1[idx]:self.idx_interface1_2] -
                         (self.device.d_htl - self.w1_t[idx]))
                    pot[idx, self.idx_w1[idx]:self.idx_interface1_2] = - (self.device.v_bi - vapp) + 0.5 * e*self.n1_t[idx] / \
                        (epsilon_0*self.device.epsr_htl) * x ** 2
                else:
                    x = self.position_steps[:self.idx_interface1_2]
                    pot[idx, :self.idx_interface1_2] = -self.device.n_htl * e / (epsilon_0 * self.device.epsr_htl) * (
                        self.device.d_htl * x - 0.5 * x**2) - self.device.epsr_pero / self.device.epsr_htl * e3 * x + self.w2_t[idx] * self.n2_t[idx] * e * x / (epsilon_0 * self.device.epsr_htl) - (self.device.v_bi - vapp)

                # Potential at interface between region I and II
                pot_I_II = pot[idx, self.idx_interface1_1]

                # Region II
                x = self.position_steps[self.idx_interface1_2:self.idx_w2[idx]] - \
                    (self.device.d_htl - self.w1_t[idx])
                pot[idx, self.idx_interface1_2:self.idx_w2[idx]] = pot_I_II - e3 * (x-self.w1_t[idx]) + e*self.n2_t[idx]/(epsilon_0*self.device.epsr_pero) * (
                    x*(self.w1_t[idx] + self.w2_t[idx]) - 0.5*x**2 - self.w1_t[idx] * self.w2_t[idx] - 0.5*self.w1_t[idx]**2)

                # Potential at interface between region II and III
                pot_II_III = pot_I_II + 0.5 * e * \
                    self.n2_t[idx] * self.w2_t[idx]**2 / \
                    (epsilon_0*self.device.epsr_pero) - e3*self.w2_t[idx]

                # Region III
                x = self.position_steps[self.idx_w2[idx]:self.idx_w4[idx]] - \
                    (self.device.d_htl - self.w1_t[idx])
                pot[idx, self.idx_w2[idx]:self.idx_w4[idx]] = pot_II_III - \
                    e3 * (x-self.w1_t[idx] - self.w2_t[idx])

                # Potential at interface between region III and IV
                pot_III_IV = pot_II_III - e3 * \
                    (self.device.d_pero - self.w2_t[idx] - self.w4_t[idx])

                # Region IV
                x = self.position_steps[self.idx_w4[idx]:self.idx_interface2_2] - \
                    (self.device.d_htl - self.w1_t[idx])
                pot[idx, self.idx_w4[idx]:self.idx_interface2_2] = pot_III_IV - e3 * \
                    (x-self.w1_t[idx] - (self.device.d_pero - self.w4_t[idx])) + 0.5 * e * self.n4_t[idx] / (
                        epsilon_0*self.device.epsr_pero) * (x-(self.w1_t[idx] + self.device.d_pero - self.w4_t[idx])) ** 2

                pot_IV_V = pot_III_IV - e3 * \
                    self.w4_t[idx] + 0.5 * e * self.n4_t[idx] * \
                    self.w4_t[idx]**2 / (epsilon_0*self.device.epsr_pero)

                # Region V
                if (self.w2_t[idx]*self.n2_t[idx]*e-self.device.epsr_pero*epsilon_0 * e3) / (self.device.n_etl * e) <= self.device.d_etl:

                    x = self.position_steps[self.idx_interface2_2:self.idx_w5[idx]] - \
                        (self.device.d_htl - self.w1_t[idx])
                    pot[idx, self.idx_interface2_2:self.idx_w5[idx]] = pot_IV_V + e*self.n5_t[idx] / (epsilon_0*self.device.epsr_etl) * (
                        (self.w1_t[idx] + self.device.d_pero + self.w5_t[idx]) * x - 0.5*x**2 - self.w5_t[idx] * (self.w1_t[idx] + self.device.d_pero) - 0.5 * (self.w1_t[idx] + self.device.d_pero) ** 2)

                else:

                    x = self.position_steps[self.idx_interface2_2:]
                    pot[idx, self.idx_interface2_2:] = - self.device.n_etl * e / (epsilon_0 * self.device.epsr_etl) * (0.5*x**2 - x * (self.device.d_htl + self.device.d_pero)) - self.device.epsr_pero / self.device.epsr_etl * e3 * x + self.w2_t[idx] * self.n2_t[idx] * e / (epsilon_0 * self.device.epsr_etl) * x + self.device.n_etl * e / (
                        2*epsilon_0 * self.device.epsr_etl) * (self.device.d_etl**2 - (self.device.d_htl + self.device.d_pero)**2) + self.device.epsr_pero / self.device.epsr_etl * e3 * (self.device.d_htl + self.device.d_pero + self.device.d_etl) - self.n2_t[idx] * self.w2_t[idx] * e / (epsilon_0 * self.device.epsr_etl) * (self.device.d_htl + self.device.d_pero + self.device.d_etl)

        # Undoped HTL and undoped ETL
        elif self.device.htl_doping is False and self.device.etl_doping is False:
            for idx, t in enumerate(self.timesteps):

                e3 = (-self.device.v_bi + vapp + e * self.n2_t[idx] * self.w2_t[idx] * self.device.d_htl / (epsilon_0 * self.device.epsr_htl) + e * self.n2_t[idx] * self.w2_t[idx] * self.device.d_etl / (
                    epsilon_0 * self.device.epsr_etl) + e/(2*epsilon_0*self.device.epsr_pero)*(self.n2_t[idx]*self.w2_t[idx]**2)*(1+self.n2_t[idx]/self.device.n_ion)) / (self.device.epsr_pero/self.device.epsr_etl * self.device.d_etl + self.device.d_pero + self.device.epsr_pero/self.device.epsr_htl * self.device.d_htl)

                # Region I (HTL)
                x = (self.position_steps[:self.idx_interface1_2])
                pot[idx, :self.idx_interface1_2] = - self.device.epsr_pero / \
                    self.device.epsr_htl * e3 * x + e * \
                    self.n2_t[idx] / (epsilon_0*self.device.epsr_htl) * \
                    self.w2_t[idx] * x - (self.device.v_bi - vapp)

                # Potential between I and II
                pot_I_II = - self.device.epsr_pero / \
                    self.device.epsr_htl * e3 * self.device.d_htl + e * \
                    self.n2_t[idx] / (epsilon_0*self.device.epsr_htl) * \
                    self.w2_t[idx] * self.device.d_htl - \
                    (self.device.v_bi - vapp)

                # Region II (Ion accumulation layer)
                x = (
                    self.position_steps[self.idx_interface1_2:self.idx_w2[idx]])
                pot[idx, self.idx_interface1_2:self.idx_w2[idx]] = - e*self.n2_t[idx] / (epsilon_0 * self.device.epsr_pero) * (0.5*x**2 - (
                    self.device.d_htl+self.w2_t[idx])*x + 0.5 * self.device.d_htl**2 + self.device.d_htl*self.w2_t[idx]) - e3 * (x-self.device.d_htl) + pot_I_II

                # Region III (Perovskite bulk)
                x = (self.position_steps[self.idx_w2[idx]:self.idx_w4[idx]])

                pot[idx, self.idx_w2[idx]:self.idx_w4[idx]] = - \
                    e3*(x-self.device.d_htl) + e*self.n2_t[idx] * self.w2_t[idx]**2/(
                    2*epsilon_0*self.device.epsr_pero) - self.device.epsr_pero / \
                    self.device.epsr_htl * e3 * self.device.d_htl + e*self.n2_t[idx] / (
                    epsilon_0*self.device.epsr_htl) * self.w2_t[idx] * self.device.d_htl - (self.device.v_bi - vapp)

                # Potential between III and IV
                pot_III_IV = - e3*((self.device.d_htl+self.device.d_pero-self.w4_t[idx])-self.device.d_htl) + e*self.n2_t[idx] * self.w2_t[idx]**2/(
                    2*epsilon_0*self.device.epsr_pero) - self.device.epsr_pero / \
                    self.device.epsr_htl * e3 * self.device.d_htl + e*self.n2_t[idx] / (
                    epsilon_0*self.device.epsr_htl) * self.w2_t[idx] * self.device.d_htl - (self.device.v_bi - vapp)

                # Region IV (Ion depletion layer)
                x = (
                    self.position_steps[self.idx_w4[idx]:self.idx_interface2_2])
                pot[idx, self.idx_w4[idx]:self.idx_interface2_2] = e*self.n4_t[idx]/(2*epsilon_0*self.device.epsr_pero) * (x-(self.device.d_htl+self.device.d_pero-self.w4_t[idx]))**2 - e3*(
                    x-(self.device.d_htl+self.device.d_pero-self.w4_t[idx])) + pot_III_IV

                # Potential between IV and V
                pot_IV_V = e*self.n4_t[idx]/(2*epsilon_0*self.device.epsr_pero) * ((self.device.d_htl+self.device.d_pero)-(self.device.d_htl+self.device.d_pero-self.w4_t[idx]))**2 - e3*(
                    (self.device.d_htl+self.device.d_pero)-(self.device.d_htl+self.device.d_pero-self.w4_t[idx])) + pot_III_IV

                # Region V (ETL)
                x = (self.position_steps[self.idx_interface2_2:])
                pot[idx, self.idx_interface2_2:] = e*self.n4_t[idx]/(epsilon_0*self.device.epsr_etl) * self.w4_t[idx] * (x - (
                    self.device.d_htl+self.device.d_pero)) - e3 * self.device.epsr_pero/self.device.epsr_etl * (x - (self.device.d_htl + self.device.d_pero)) + pot_IV_V

        # Only ETL doping
        elif self.device.htl_doping is False and self.device.etl_doping is True:
            for idx, t in enumerate(self.timesteps):
                e3 = self.get_e3q(self.q2_t[idx], vapp)

                # Region I (HTL)
                x = (self.position_steps[:self.idx_interface1_2])
                pot[idx, :self.idx_interface1_2] = - self.device.epsr_pero / \
                    self.device.epsr_htl * e3 * x + e * \
                    self.n2_t[idx] / (epsilon_0*self.device.epsr_htl) * \
                    self.w2_t[idx] * x - (self.device.v_bi - vapp)

                # Potential at interface between I and II
                pot_I_II = - self.device.epsr_pero / \
                    self.device.epsr_htl * e3 * self.device.d_htl + e * \
                    self.n2_t[idx] / (epsilon_0*self.device.epsr_htl) * \
                    self.w2_t[idx] * self.device.d_htl - \
                    (self.device.v_bi - vapp)

                # Region II (Ion accumulation layer)
                x = (
                    self.position_steps[self.idx_interface1_2:self.idx_w2[idx]])
                pot[idx, self.idx_interface1_2:self.idx_w2[idx]] = - e*self.n2_t[idx] / (epsilon_0 * self.device.epsr_pero) * (0.5*x**2 - (
                    self.device.d_htl+self.w2_t[idx])*x + 0.5 * self.device.d_htl**2 + self.device.d_htl*self.w2_t[idx]) - e3 * (x-self.device.d_htl) + pot_I_II

                # Potential  between II and III
                pot_II_III = - e*self.n2_t[idx] / (epsilon_0 * self.device.epsr_pero) * (0.5*(self.device.d_htl + self.w2_t[idx])**2 - (self.device.d_htl+self.w2_t[idx])*(
                    self.device.d_htl + self.w2_t[idx]) + 0.5 * self.device.d_htl**2 + self.device.d_htl*self.w2_t[idx]) - e3 * ((self.device.d_htl + self.w2_t[idx])-self.device.d_htl) + pot_I_II

                # Region III (Perovskite bulk)
                x = (self.position_steps[self.idx_w2[idx]:self.idx_w4[idx]])
                pot[idx, self.idx_w2[idx]:self.idx_w4[idx]] = -e3 * \
                    (x-(self.device.d_htl +
                     self.w2_t[idx])) + pot_II_III

                # Potential  between III and IV
                pot_III_IV = -e3 * \
                    ((self.device.d_htl+self.device.d_pero-self.w4_t[idx])-(self.device.d_htl +
                     self.w2_t[idx])) + pot_II_III

                # Region IV (Ion depletion layer)
                x = (
                    self.position_steps[self.idx_w4[idx]:self.idx_interface2_2])
                pot[idx, self.idx_w4[idx]:self.idx_interface2_2] = e * \
                    self.n4_t[idx]/(2*epsilon_0*self.device.epsr_pero) * (x-(self.device.d_htl+self.device.d_pero -
                                                                             self.w4_t[idx]))**2 - e3*(x-(self.device.d_htl+self.device.d_pero-self.w4_t[idx])) + pot_III_IV

                # Potential between IV and V
                x = self.device.d_htl + self.device.d_pero
                pot_IV_V = e * \
                    self.n4_t[idx]/(2*epsilon_0*self.device.epsr_pero) * (x-(self.device.d_htl+self.device.d_pero -
                                                                             self.w4_t[idx]))**2 - e3*(x-(self.device.d_htl+self.device.d_pero-self.w4_t[idx])) + pot_III_IV

                # Region V (ETL)

                # If the depletion layer is larger than ETL thickness
                if (self.w2_t[idx]*self.n2_t[idx]*e-self.device.epsr_pero*epsilon_0 * e3) / (self.device.n_etl * e) <= self.device.d_etl:
                    x = (
                        self.position_steps[self.idx_interface2_2:self.idx_w5[idx]])
                    pot[idx, self.idx_interface2_2:self.idx_w5[idx]] = -e*self.n5_t[idx] / (2*epsilon_0*self.device.epsr_etl) * (x-(self.device.d_htl + self.device.d_pero))**2 - (
                        self.device.epsr_pero / self.device.epsr_etl * e3 - self.n4_t[idx] * self.w4_t[idx]*e / (epsilon_0*self.device.epsr_etl)) * (x-(self.device.d_htl + self.device.d_pero)) + pot_IV_V
                    x = self.position_steps[self.idx_w5[idx]:]
                    pot[idx, self.idx_w5[idx]:] = 0

                else:
                    x = self.position_steps[self.idx_interface2_2:]
                    pot[idx, self.idx_interface2_2:] = - self.device.n_etl * e / (epsilon_0 * self.device.epsr_etl) * (0.5*x**2 - x * (self.device.d_htl + self.device.d_pero)) - self.device.epsr_pero / self.device.epsr_etl * e3 * x + self.w2_t[idx] * self.n2_t[idx] * e / (epsilon_0 * self.device.epsr_etl) * x + self.device.n_etl * e / (
                        2*epsilon_0 * self.device.epsr_etl) * (self.device.d_etl**2 - (self.device.d_htl + self.device.d_pero)**2) + self.device.epsr_pero / self.device.epsr_etl * e3 * (self.device.d_htl + self.device.d_pero + self.device.d_etl) - self.n2_t[idx] * self.w2_t[idx] * e / (epsilon_0 * self.device.epsr_etl) * (self.device.d_htl + self.device.d_pero + self.device.d_etl)

        elif self.device.htl_doping is True and self.device.etl_doping is False:
            for idx, t in enumerate(self.timesteps):
                e3 = self.get_e3q(self.q2_t[idx], vapp)

                # Region 0

                # If the depletion layer is smaller than HTL thickness
                if (self.w2_t[idx]*self.n2_t[idx]*e-self.device.epsr_pero*epsilon_0 * e3) / (self.device.n_htl * e) <= self.device.d_htl:
                    x = self.position_steps[:self.idx_w1[idx]]
                    pot[idx, :self.idx_w1[idx]] = - (self.device.v_bi - vapp)

                    # Region I (Doped HTL)
                    x = self.position_steps[self.idx_w1[idx]:self.idx_interface1_2]
                    pot[idx, self.idx_w1[idx]:self.idx_interface1_2] = - (self.device.v_bi - vapp) + 0.5 * e*self.device.n_htl / \
                        (epsilon_0*self.device.epsr_htl) * \
                        (x - (self.device.d_htl - self.w1_t[idx]))**2
                else:
                    x = self.position_steps[:self.idx_interface1_2]
                    pot[idx, :self.idx_interface1_2] = -self.device.n_htl * e / (epsilon_0 * self.device.epsr_htl) * (
                        self.device.d_htl * x - 0.5 * x**2) - self.device.epsr_pero / self.device.epsr_htl * e3 * x + self.w2_t[idx] * self.n2_t[idx] * e * x / (epsilon_0 * self.device.epsr_htl) - (self.device.v_bi - vapp)

                # Potential at interface between region I and II
                pot_I_II = pot[idx, self.idx_interface1_1]

                # Region II (Ion accumulation layer)
                x = (
                    self.position_steps[self.idx_interface1_2:self.idx_w2[idx]])
                pot[idx, self.idx_interface1_2:self.idx_w2[idx]] = - e*self.n2_t[idx] / (epsilon_0 * self.device.epsr_pero) * (0.5*x**2 - (
                    self.device.d_htl+self.w2_t[idx])*x + 0.5 * self.device.d_htl**2 + self.device.d_htl*self.w2_t[idx]) - e3 * (x-self.device.d_htl) + pot_I_II

                # Potential at interface between region II and III
                pot_II_III = - e*self.n2_t[idx] / (epsilon_0 * self.device.epsr_pero) * (0.5*(self.device.d_htl+self.w2_t[idx])**2 - (
                    self.device.d_htl+self.w2_t[idx])*(self.device.d_htl+self.w2_t[idx]) + 0.5 * self.device.d_htl**2 + self.device.d_htl*self.w2_t[idx]) - e3 * (self.device.d_htl+self.w2_t[idx]-self.device.d_htl) + pot_I_II

                # Region III (Perovskite bulk)
                x = (self.position_steps[self.idx_w2[idx]:self.idx_w4[idx]])
                pot[idx, self.idx_w2[idx]:self.idx_w4[idx]] = -e3 * \
                    (x-(self.device.d_htl +
                     self.w2_t[idx])) + pot_II_III

                # Potential between regions III and IV
                pot_III_IV = -e3 * \
                    (self.device.d_htl+self.device.d_pero-self.w4_t[idx]-(self.device.d_htl +
                     self.w2_t[idx])) + pot_II_III

                # Region IV (Ion depletion layer)
                x = (
                    self.position_steps[self.idx_w4[idx]:self.idx_interface2_2])
                pot[idx, self.idx_w4[idx]:self.idx_interface2_2] = e *\
                    self.n4_t[idx]/(2*epsilon_0*self.device.epsr_pero) * (x-(self.device.d_htl+self.device.d_pero-self.w4_t[idx])
                                                                          )**2 - e3*(x-(self.device.d_htl+self.device.d_pero-self.w4_t[idx])) + pot_III_IV

                # Potential between regions IV and V
                pot_IV_V = e * self.n4_t[idx]/(2*epsilon_0*self.device.epsr_pero) * (
                    self.w4_t[idx])**2 - e3*(self.w4_t[idx]) + pot_III_IV

                # Region V (Undoped ETL)
                x = self.position_steps[self.idx_interface2_2:]
                pot[idx, self.idx_interface2_2:] = e*self.n4_t[idx]*self.w4_t[idx]/(epsilon_0*self.device.epsr_etl) * (x - (
                    self.device.d_htl+self.device.d_pero)) - e3 * self.device.epsr_pero/self.device.epsr_etl * (x - (self.device.d_htl+self.device.d_pero)) + pot_IV_V

        # Add offset because of difference between Wf electrodes and CTLs
        pot += self.device.pot_offset

        self.pot = pot

    def calc_Energy(self, vapp=0.0):
        """Function to calculate the energy of the device as a function fo the applied voltage

        Args:
            vapp (float, optional): Applied bias. Defaults to 0.

        Returns:
            numpy arrays: Cunduction and valence band energies of the device
        """
        # First calcualte the potential
        self.calc_potential(vapp=vapp)

        # Iniitialize the energy arrays
        E_cb = np.zeros((self.num_timesteps, self.x_resolution))
        E_vb = np.zeros((self.num_timesteps, self.x_resolution))

        # If ETL is doped
        if self.device.etl_doping == True:
            E_cb_etl = np.log(self.device.n0_cb_etl /
                              self.device.n_etl) * k * self.device.temp / e + self.device.pot_offset
        # If ETL is undoped
        else:
            E_cb_etl = self.device.wf_cathode - self.device.cb_etl

        # If HTL is doped
        if self.device.htl_doping == True:
            E_vb_htl = np.log(self.device.n_htl /
                              self.device.n0_vb_htl) * k * self.device.temp / e + self.device.pot_offset
        # If HTL is undoped
        else:
            E_vb_htl = self.device.wf_anode - self.device.vb_htl

        # Correct at the interfaces dependent on the energy offset between CB and VB
        for idx, _ in enumerate(self.timesteps):
            E_cb[idx, self.idx_interface2_2:] = E_cb_etl - \
                self.pot[idx, self.idx_interface2_2:]
            E_cb[idx, self.idx_interface1_2:self.idx_interface2_2] = E_cb_etl -\
                self.pot[idx, self.idx_interface1_2:self.idx_interface2_2] -\
                (self.device.cb_pero-self.device.cb_etl)
            E_cb[idx, :self.idx_interface1_2] = E_cb_etl -\
                self.pot[idx, :self.idx_interface1_2] -\
                (self.device.cb_htl-self.device.cb_etl)

            E_vb[idx, :self.idx_interface1_2] = E_vb_htl -\
                self.pot[idx, :self.idx_interface1_2] - (self.device.v_bi-vapp)
            E_vb[idx, self.idx_interface1_2:self.idx_interface2_2] = E_vb_htl -\
                self.pot[idx, self.idx_interface1_2:self.idx_interface2_2] -\
                (self.device.vb_pero-self.device.vb_htl) - (self.device.v_bi-vapp)
            E_vb[idx, self.idx_interface2_2:] = E_vb_htl -\
                self.pot[idx, self.idx_interface2_2:] -\
                (self.device.vb_etl-self.device.vb_htl) - (self.device.v_bi-vapp)
        self.E_cb = E_cb
        self.E_vb = E_vb
        return E_cb, E_vb

    def calc_densities(self, ):
        """Function to calculate the net charge density of the device
        """
        net_density = np.zeros((self.num_timesteps, self.x_resolution))
        for idx, t in enumerate(self.timesteps):

            net_density[idx, self.idx_w1[idx]:self.idx_interface1_2] = -self.n1_t[idx]
            net_density[idx, self.idx_interface1_2:self.idx_w2[idx]
                        ] = self.n2_t[idx]
            net_density[idx, self.idx_w4[idx]:self.idx_interface2_2] = - self.n4_t[idx]
            net_density[idx, self.idx_interface2_2:self.idx_w5[idx]
                        ] = self.n5_t[idx]

        self.net_density = net_density

        self.set_parameters_to_timestep_length()

    def calc_electronic_charge_densities(self, vapp=0.0):
        """Function to calculate the electronic charge densities

        Args:
            vapp (float, optional): Applied voltage. Defaults to 0.
        """

        # First calculate the energies of the conduction and valence band
        E_cb, E_vb = self.calc_Energy(vapp=vapp)
        n = np.zeros((self.num_timesteps, self.x_resolution))
        p = np.zeros((self.num_timesteps, self.x_resolution))
        c = np.zeros((self.num_timesteps, self.x_resolution))

        # Calculate the electron and hole densities dependent on the energy, and the effective density of states
        for idx, t in enumerate(self.timesteps):
            p[idx, :self.idx_interface1_2] = self.device.n0_vb_htl * \
                np.exp(E_vb[idx, :self.idx_interface1_2]
                       * e/(k*self.device.temp))
            p[idx, self.idx_interface1_2:self.idx_interface2_2] = self.device.n0_vb_pero * \
                np.exp(E_vb[idx, self.idx_interface1_2:self.idx_interface2_2]
                       * e/(k*self.device.temp))
            p[idx, self.idx_interface2_2:] = self.device.n0_vb_etl * \
                np.exp(E_vb[idx, self.idx_interface2_2:]
                       * e/(k*self.device.temp))

            n[idx, :self.idx_interface1_2] = self.device.n0_cb_htl * \
                np.exp(-E_cb[idx, :self.idx_interface1_2]
                       * e/(k*self.device.temp))
            n[idx, self.idx_interface1_2:self.idx_interface2_2] = self.device.n0_cb_pero * \
                np.exp(-E_cb[idx, self.idx_interface1_2:self.idx_interface2_2]
                       * e/(k*self.device.temp))
            n[idx, self.idx_interface2_2:] = self.device.n0_cb_etl * \
                np.exp(-E_cb[idx, self.idx_interface2_2:]
                       * e/(k*self.device.temp))

            c[idx, self.idx_interface1_2:self.idx_w2[idx]
              ] = self.device.n_ion + self.n2_t[idx]
            c[idx, self.idx_w2[idx]:self.idx_w4[idx]] = self.device.n_ion

        self.p_dc = p
        self.n_dc = n
        self.c_dc = c

    def construct_jacobian(self, psi_dc, D_n, D_p, n_dc, p_dc, h, Ut, D_c=None, c_dc=None):
        """Function to construct the jacobian to solve the ac solution of the device. The equations are shown in the SI of the article.

        Args:
            psi_dc (array): DC potential through the device
            D_n (array): Array containing the electron diffusion coefficient
            D_p (array): Array containing the hole diffusion coefficient
            n_dc (array): Array containing the electron density
            p_dc (array): Array containing the hole density
            h (array): Array containing the step density
            Ut (float): Thermal voltage
            D_c (array, optional): Array containing the cation diffusion coefficient. Defaults to None.
            c_dc (array, optional): Array containing the cation density. Defaults to None.

        Returns:
            array: Jacobian matrix 
        """

        # Define dimension of jacobian
        if self.consider_cations is True:
            jac_dim = 4*self.x_resolution
        else:
            jac_dim = 3*self.x_resolution

        # Initialize jacobian of potential, electrons, and holes
        jacobian_psi = np.zeros(
            (self.x_resolution, jac_dim), dtype=np.cdouble)
        jacobian_n = np.zeros(
            (self.x_resolution, jac_dim),  dtype=np.cdouble)
        jacobian_p = np.zeros(
            (self.x_resolution, jac_dim),  dtype=np.cdouble)

        # Set the indices that account for the boundary conditions to 1. The boundary conditions are then defined in the boundary condition later.
        jacobian_psi[0, 0] = 1
        jacobian_psi[self.x_resolution-1, self.x_resolution-1] = 1
        jacobian_n[0, self.x_resolution] = 1
        jacobian_n[self.x_resolution-1, 2*self.x_resolution-1] = 1
        jacobian_p[0, 2*self.x_resolution] = 1
        jacobian_p[self.x_resolution-1, 3*self.x_resolution-1] = 1

        if self.consider_cations is True:
            jacobian_c = np.zeros(
                (self.x_resolution, jac_dim),  dtype=np.cdouble)
            jacobian_c[0, 3*self.x_resolution] = 1
            jacobian_c[self.x_resolution-1, 4*self.x_resolution-1] = 1

        mat_bernoulli = np.zeros((2, self.x_resolution, self.x_resolution))
        mat_diff_bernoulli = np.zeros(
            (2, self.x_resolution, self.x_resolution))

        # Precompute a matrix that contains factors that are often used
        for i in range(1, self.x_resolution-1):
            if (i == self.idx_interface1_1) or (i == self.idx_interface2_1):
                pass
            elif (i == self.idx_interface1_2) or (i == self.idx_interface2_2):
                pass
            else:
                factor = 2/(h[i]+h[i-1])
                mat_bernoulli[0, i, i-1] = cbernoulli.bernoulli(
                    (psi_dc[i]-psi_dc[i-1])/Ut)*1/(h[i-1])*factor
                mat_bernoulli[0, i, i + 1] = cbernoulli.bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut)*1/(h[i])*factor
                mat_bernoulli[1, i-1, i] = cbernoulli.bernoulli(
                    (psi_dc[i-1]-psi_dc[i])/Ut)*1/(h[i-1])*factor
                mat_bernoulli[1, i+1, i] = cbernoulli.bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut)*1/(h[i])*factor
                mat_diff_bernoulli[0, i, i-1] = cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i-1])/Ut)*1/(Ut*h[i-1])*factor
                mat_diff_bernoulli[0, i, i+1] = cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut)*1/(Ut*h[i])*factor
                mat_diff_bernoulli[1, i-1, i] = cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i])/Ut)*1/(Ut*h[i-1])*factor
                mat_diff_bernoulli[1, i+1, i] = cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut)*1/(Ut*h[i])*factor

        for i in range(1, self.x_resolution-1):
            # Poisson
            # dFPsi/dPsi

            if i < self.idx_interface1_1:
                jacobian_psi[i, i] = self.lambda_htl**2 * \
                    (-1/h[i-1]-1/h[i])*2/(h[i]+h[i-1])
                jacobian_psi[i, i-1] = self.lambda_htl**2 / \
                    h[i-1]*2/(h[i]+h[i-1])
                jacobian_psi[i, i+1] = self.lambda_htl**2 / \
                    h[i]*2/(h[i]+h[i-1])
            elif (i == self.idx_interface1_1):
                jacobian_psi[i, i] = -1*self.lambda_htl**2 / \
                    h[i-1] * 2/(h[i+1]+h[i-1])
                jacobian_psi[i, i-1] = self.lambda_htl**2 / \
                    h[i-1] * 2/(h[i+1]+h[i-1])
                jacobian_psi[i, i+1] = -1*self.lambda_pero**2 / \
                    h[i+1] * 2/(h[i+1]+h[i-1])
                jacobian_psi[i, i+2] = self.lambda_pero**2 / \
                    h[i+1] * 2/(h[i+1]+h[i-1])
            elif (i == self.idx_interface1_2):
                jacobian_psi[i, i] = 1e3 * jacobian_psi[i-1, i-1]
                jacobian_psi[i, i-1] = - 1e3 * jacobian_psi[i-1, i-1]

            elif (i > self.idx_interface1_2) and (i < self.idx_interface2_1):
                jacobian_psi[i, i] = self.lambda_pero**2 * \
                    (-1/h[i-1]-1/h[i])*2/(h[i]+h[i-1])
                jacobian_psi[i, i-1] = self.lambda_pero**2 / \
                    h[i-1]*2/(h[i]+h[i-1])
                jacobian_psi[i, i+1] = self.lambda_pero**2 / \
                    h[i]*2/(h[i]+h[i-1])
            elif (i == self.idx_interface2_1):
                jacobian_psi[i, i] = -1*self.lambda_pero**2 / \
                    h[i-1] * 2/(h[i+1]+h[i-1])
                jacobian_psi[i, i-1] = self.lambda_pero**2 / \
                    h[i-1] * 2/(h[i+1]+h[i-1])
                jacobian_psi[i, i+1] = -1*self.lambda_etl**2 / \
                    h[i+1] * 2/(h[i+1]+h[i-1])
                jacobian_psi[i, i+2] = self.lambda_etl**2 / \
                    h[i+1] * 2/(h[i+1]+h[i-1])
            elif (i == self.idx_interface2_2):
                jacobian_psi[i, i] = 1e3 * jacobian_psi[i-1, i-1]
                jacobian_psi[i, i-1] = - 1e3 * jacobian_psi[i-1, i-1]

            else:
                jacobian_psi[i, i] = self.lambda_etl**2 * \
                    (-1/h[i-1]-1/h[i])*2/(h[i]+h[i-1])
                jacobian_psi[i, i-1] = self.lambda_etl**2 / \
                    h[i-1]*2/(h[i]+h[i-1])
                jacobian_psi[i, i+1] = self.lambda_etl**2 / \
                    h[i]*2/(h[i]+h[i-1])

            # dFPsi/dn
            jacobian_psi[i, self.x_resolution+i] = -1

            # dFPsi/dp
            jacobian_psi[i, 2*self.x_resolution+i] = 1

            # dFPsi/dc\
            if self.consider_cations is True:
                if (i <= self.idx_interface1_2) or (i >= self.idx_interface2_1):
                    jacobian_psi[i, 3*self.x_resolution+i] = 0
                else:
                    jacobian_psi[i, 3*self.x_resolution+i] = 1

            # # Electron current continuity
            # # dFn/dPsi
            if (i == self.idx_interface1_1):
                pass
            elif (i == self.idx_interface2_1):
                jacobian_n[i, i] = -1*D_n[i-1]/(Ut*h[i-1])*2/(h[i+1]+h[i-1]) * (n_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i])/Ut) + n_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i-1])/Ut))

                jacobian_n[i, i-1] = D_n[i-1]/(Ut*h[i-1])*2/(h[i+1]+h[i-1]) * (n_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i])/Ut) + n_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i-1])/Ut))

                jacobian_n[i, i+1] = -1*D_n[i+1]/(Ut*h[i+1])*2/(h[i+1]+h[i-1]) * (n_dc[i+2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+2]-psi_dc[i+1])/Ut) + n_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i+2])/Ut))

                jacobian_n[i, i+2] = D_n[i+1]/(Ut*h[i+1])*2/(h[i+1]+h[i-1]) * (n_dc[i+2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+2]-psi_dc[i+1])/Ut) + n_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i+2])/Ut))
            elif (i == self.idx_interface1_2):
                pass
            elif (i == self.idx_interface2_2):
                jacobian_n[i, i] = -1*D_n[i]/(Ut*h[i])*2/(h[i]+h[i-2]) * (n_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut) + n_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut))

                jacobian_n[i, i+1] = D_n[i]/(Ut*h[i])*2/(h[i]+h[i-2]) * (n_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut) + n_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut))

                jacobian_n[i, i-1] = -1*D_n[i-2]/(Ut*h[i])*2/(h[i]+h[i-2]) * (n_dc[i-2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-2]-psi_dc[i-1])/Ut) + n_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i-2])/Ut))

                jacobian_n[i, i-2] = D_n[i-2]/(Ut*h[i])*2/(h[i]+h[i-2]) * (n_dc[i-2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-2]-psi_dc[i-1])/Ut) + n_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i-2])/Ut))

            else:
                jacobian_n[i, i] = (n_dc[i-1] * D_n[i-1] * -1 * mat_diff_bernoulli[1, i-1, i] -
                                    n_dc[i] * (D_n[i-1]*mat_diff_bernoulli[0, i, i-1] +
                                               D_n[i]*mat_diff_bernoulli[0, i, i+1]) +
                                    n_dc[i+1]*D_n[i]*-1*mat_diff_bernoulli[1, i+1, i])

                jacobian_n[i, i-1] = (D_n[i-1] * (n_dc[i-1]*mat_diff_bernoulli[1, i-1, i]
                                                  + n_dc[i]*mat_diff_bernoulli[0, i, i-1]))

                jacobian_n[i, i+1] = (D_n[i]*(n_dc[i+1]*mat_diff_bernoulli[1, i+1, i]
                                              + n_dc[i]*mat_diff_bernoulli[0, i, i+1]))

            # # dFn/dn
            if (i == self.idx_interface1_1):
                factor = 100 * \
                    int(np.abs(jacobian_n[i-1, self.x_resolution+i-1]))
                jacobian_n[i, self.x_resolution+i+1] = - factor
                jacobian_n[i, self.x_resolution+i] = factor / \
                    self.device.m_cb_int1

            elif (i == self.idx_interface2_1):
                factor = 100 * \
                    int(np.abs(jacobian_n[i-1, self.x_resolution+i-1]))
                jacobian_n[i, self.x_resolution+i+1] = - factor
                jacobian_n[i, self.x_resolution+i] = factor / \
                    self.device.m_cb_int2

            elif (i == self.idx_interface2_2) or (i == self.idx_interface1_2):
                jacobian_n[i, self.x_resolution+i] = -1*D_n[i]*cbernoulli.bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut)*1/(h[i])*2/(h[i]+h[i-2])

                jacobian_n[i, self.x_resolution+i+1] = D_n[i]*cbernoulli.bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut)*1/(h[i])*2/(h[i]+h[i-2])

                jacobian_n[i, self.x_resolution+i-1] = -1*D_n[i-2]*cbernoulli.bernoulli(
                    (psi_dc[i-1]-psi_dc[i-2])/Ut)*1/(h[i-2])*2/(h[i]+h[i-2])

                jacobian_n[i, self.x_resolution+i-2] = D_n[i-2]*cbernoulli.bernoulli(
                    (psi_dc[i-2]-psi_dc[i-1])/Ut)*1/(h[i-2])*2/(h[i]+h[i-2])

            else:
                jacobian_n[i, self.x_resolution+i] = -1*(D_n[i-1]*mat_bernoulli[0, i, i-1]
                                                         + D_n[i]*mat_bernoulli[0, i, i+1])

                jacobian_n[i, self.x_resolution+i -
                           1] = (D_n[i-1]*mat_bernoulli[1, i-1, i])

                jacobian_n[i, self.x_resolution+i +
                           1] = (D_n[i]*mat_bernoulli[1, i+1, i])
            # # dFn/dp
            # # is completely 0

            # # Hole current continuity
            # # dFp/dPsi
            if (i == self.idx_interface1_1):
                jacobian_p[i, i] = D_p[i-1]/(Ut*h[i-1])*2/(h[i+1]+h[i-1]) * (p_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i-1])/Ut) + p_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i])/Ut))

                jacobian_p[i, i-1] = -1*D_p[i-1]/(Ut*h[i-1])*2/(h[i+1]+h[i-1]) * (p_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i-1])/Ut) + p_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i])/Ut))

                jacobian_p[i, i+1] = D_p[i+1]/(Ut*h[i+1])*2/(h[i+1]+h[i-1]) * (p_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+2]-psi_dc[i+1])/Ut) + p_dc[i+2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i+2])/Ut))

                jacobian_p[i, i+2] = -1*D_p[i+1]/(Ut*h[i+1])*2/(h[i+1]+h[i-1]) * (p_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+2]-psi_dc[i+1])/Ut) + p_dc[i+2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i+2])/Ut))
            elif (i == self.idx_interface2_1):
                jacobian_p[i, i] = D_p[i-1]/(Ut*h[i-1])*2/(h[i-1]+h[i+1]) * (p_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i-1])/Ut) + p_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i])/Ut))

                jacobian_p[i, i-1] = -1 * D_p[i-1]/(Ut*h[i-1])*2/(h[i-1]+h[i+1]) * (p_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i-1])/Ut) + p_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i])/Ut))

                jacobian_p[i, i+1] = D_p[i+1]/(Ut*h[i+1])*2/(h[i-1]+h[i+1]) * (p_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+2]-psi_dc[i+1])/Ut) + p_dc[i+2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i+2])/Ut))

                jacobian_p[i, i+2] = -1 * D_p[i+1]/(Ut*h[i+1])*2/(h[i-1]+h[i+1]) * (p_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+2]-psi_dc[i+1])/Ut) + p_dc[i+2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i+2])/Ut))

            elif (i == self.idx_interface1_2):
                jacobian_p[i, i] = D_p[i]/(Ut*h[i])*2/(h[i]+h[i-2]) * (p_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut) + p_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut))

                jacobian_p[i, i+1] = -1*D_p[i]/(Ut*h[i])*2/(h[i]+h[i-2]) * (p_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut) + p_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut))

                jacobian_p[i, i-1] = D_p[i-2]/(Ut*h[i-2])*2/(h[i]+h[i-2]) * (p_dc[i-2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i-2])/Ut) + p_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-2]-psi_dc[i-1])/Ut))

                jacobian_p[i, i-2] = -1*D_p[i-2]/(Ut*h[i-2])*2/(h[i]+h[i-2]) * (p_dc[i-2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i-2])/Ut) + p_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-2]-psi_dc[i-1])/Ut))
            elif (i == self.idx_interface2_2):
                jacobian_p[i, i] = D_p[i]/(Ut*h[i])*2/(h[i]+h[i-2]) * (p_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut) + p_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut))

                jacobian_p[i, i+1] = -1*D_p[i]/(Ut*h[i])*2/(h[i]+h[i-2]) * (p_dc[i] * cbernoulli.diff_bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut) + p_dc[i+1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut))

                jacobian_p[i, i-1] = D_p[i-2]/(Ut*h[i-2])*2/(h[i]+h[i-2]) * (p_dc[i-2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i-2])/Ut) + p_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-2]-psi_dc[i-1])/Ut))

                jacobian_p[i, i-2] = -1*D_p[i-2]/(Ut*h[i-2])*2/(h[i]+h[i-2]) * (p_dc[i-2] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-1]-psi_dc[i-2])/Ut) + p_dc[i-1] * cbernoulli.diff_bernoulli(
                    (psi_dc[i-2]-psi_dc[i-1])/Ut))
            else:
                jacobian_p[i, i] = (p_dc[i-1] * D_p[i-1] * mat_diff_bernoulli[0, i, i-1] -
                                    p_dc[i] * (D_p[i-1]*-1*mat_diff_bernoulli[1, i-1, i] +
                                               D_p[i]*-1*mat_diff_bernoulli[1, i+1, i]) +
                                    p_dc[i+1]*D_p[i]*1*mat_diff_bernoulli[0, i, i+1])

                jacobian_p[i, i-1] = -1*D_p[i-1] * (p_dc[i-1]*mat_diff_bernoulli[0, i, i-1] +
                                                    p_dc[i]*mat_diff_bernoulli[1, i-1, i])

                jacobian_p[i, i+1] = -1*D_p[i]*(p_dc[i]*mat_diff_bernoulli[1, i+1, i] +
                                                p_dc[i+1]*mat_diff_bernoulli[0, i, i+1])
            # dFp/dn
            # Is completely 0

            # # # dFp/dp
            if (i == self.idx_interface1_1):
                factor = 100 * \
                    int(np.abs(jacobian_p[i-1, 2*self.x_resolution+i-1]))
                jacobian_p[i, 2*self.x_resolution+i] = - factor
                jacobian_p[i, 2*self.x_resolution+i +
                           1] = factor / self.device.m_vb_int1

            elif (i == self.idx_interface1_2):

                jacobian_p[i, 2*self.x_resolution+i] = -1*D_p[i]*cbernoulli.bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut)*1/(h[i])*2/(h[i]+h[i-2])

                jacobian_p[i, 2*self.x_resolution+i+1] = D_p[i]*cbernoulli.bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut)*1/(h[i])*2/(h[i]+h[i-2])

                jacobian_p[i, 2*self.x_resolution+i-1] = -1*D_p[i-2]*cbernoulli.bernoulli(
                    (psi_dc[i-2]-psi_dc[i-1])/Ut)*1/(h[i-2])*2/(h[i]+h[i-2])

                jacobian_p[i, 2*self.x_resolution+i-2] = D_p[i-2]*cbernoulli.bernoulli(
                    (psi_dc[i-1]-psi_dc[i-2])/Ut)*1/(h[i-2])*2/(h[i]+h[i-2])
            elif (i == self.idx_interface2_1):

                factor = 100 * \
                    int(np.abs(jacobian_p[i-1, 2*self.x_resolution+i-1]))

                jacobian_p[i, 2*self.x_resolution+i] = - factor
                jacobian_p[i, 2*self.x_resolution+i +
                           1] = factor / self.device.m_vb_int1

            elif (i == self.idx_interface2_2):

                jacobian_p[i, 2*self.x_resolution+i] = -1*D_p[i]*cbernoulli.bernoulli(
                    (psi_dc[i+1]-psi_dc[i])/Ut)*1/(h[i])*2/(h[i]+h[i-2])

                jacobian_p[i, 2*self.x_resolution+i+1] = D_p[i]*cbernoulli.bernoulli(
                    (psi_dc[i]-psi_dc[i+1])/Ut)*1/(h[i])*2/(h[i]+h[i-2])

                jacobian_p[i, 2*self.x_resolution+i-1] = -1*D_p[i-2]*cbernoulli.bernoulli(
                    (psi_dc[i-2]-psi_dc[i-1])/Ut)*1/(h[i-2])*2/(h[i]+h[i-2])

                jacobian_p[i, 2*self.x_resolution+i-2] = D_p[i-2]*cbernoulli.bernoulli(
                    (psi_dc[i-1]-psi_dc[i-2])/Ut)*1/(h[i-2])*2/(h[i]+h[i-2])
            else:
                jacobian_p[i, 2*self.x_resolution+i] = -1*(D_p[i-1]*mat_bernoulli[1, i-1, i] +
                                                           D_p[i]*mat_bernoulli[1, i+1, i])

                jacobian_p[i, 2*self.x_resolution+i -
                           1] = (D_p[i-1]*mat_bernoulli[0, i, i-1])

                jacobian_p[i, 2*self.x_resolution+i +
                           1] = (D_p[i]*mat_bernoulli[0, i, i+1])

            # # Cation current continuity

            if self.consider_cations is True:
                # Fc/dPsi
                if (i < self.idx_interface1_2) or (i > self.idx_interface2_1):
                    jacobian_c[i, i] = 0
                elif (i == self.idx_interface1_2):
                    jacobian_c[i, i] = D_c[i]/(Ut*h[i])*2/(h[i-2]+h[i]) * (c_dc[i] * cbernoulli.diff_bernoulli(
                        (psi_dc[i+1]-psi_dc[i])/Ut) + c_dc[i+1] * cbernoulli.diff_bernoulli(
                        (psi_dc[i]-psi_dc[i+1])/Ut))
                    jacobian_c[i, i+1] = -1*D_c[i]/(Ut*h[i])*2/(h[i-2]+h[i]) * (c_dc[i] * cbernoulli.diff_bernoulli(
                        (psi_dc[i+1]-psi_dc[i])/Ut) + c_dc[i+1] * cbernoulli.diff_bernoulli(
                        (psi_dc[i]-psi_dc[i+1])/Ut))

                elif (i == self.idx_interface2_1):
                    jacobian_c[i, i] = D_c[i-1]/(Ut*h[i-1])*2/(h[i-1]+h[i+1]) * (c_dc[i-1] * cbernoulli.diff_bernoulli(
                        (psi_dc[i]-psi_dc[i-1])/Ut) + c_dc[i] * cbernoulli.diff_bernoulli(
                        (psi_dc[i-1]-psi_dc[i])/Ut))

                    jacobian_c[i, i-1] = -1*D_c[i-1]/(Ut*h[i-1])*2/(h[i-1]+h[i+1]) * (c_dc[i-1] * cbernoulli.diff_bernoulli(
                        (psi_dc[i]-psi_dc[i-1])/Ut) + c_dc[i] * cbernoulli.diff_bernoulli(
                        (psi_dc[i-1]-psi_dc[i])/Ut))

                else:
                    jacobian_c[i, i] = (c_dc[i-1] * D_c[i-1] * mat_diff_bernoulli[0, i, i-1] -
                                        c_dc[i] * (D_c[i-1]*-1*mat_diff_bernoulli[1, i-1, i] +
                                                   D_c[i]*-1*mat_diff_bernoulli[1, i+1, i]) +
                                        c_dc[i+1]*D_c[i]*1*mat_diff_bernoulli[0, i, i+1])

                    jacobian_c[i, i-1] = -1*D_c[i-1] * (c_dc[i-1]*mat_diff_bernoulli[0, i, i-1] +
                                                        c_dc[i]*mat_diff_bernoulli[1, i-1, i])

                    jacobian_c[i, i+1] = -1*D_c[i]*(c_dc[i]*mat_diff_bernoulli[1, i+1, i] +
                                                    c_dc[i+1]*mat_diff_bernoulli[0, i, i+1])

                # dFc/dn
                # Is completely 0

                # # # dFc/dc
                if (i < self.idx_interface1_2) or (i > self.idx_interface2_1):
                    jacobian_c[i, 3*self.x_resolution + i] = 1
                elif (i == self.idx_interface1_2):
                    jacobian_c[i, 3*self.x_resolution+i] = -1*D_c[i]*cbernoulli.bernoulli(
                        (psi_dc[i+1]-psi_dc[i])/Ut)*1/(h[i])*2/(h[i]+h[i-2])

                    jacobian_c[i, 3*self.x_resolution+i+1] = D_c[i]*cbernoulli.bernoulli(
                        (psi_dc[i]-psi_dc[i+1])/Ut)*1/(h[i])*2/(h[i]+h[i-2])

                elif (i == self.idx_interface2_1):
                    jacobian_c[i, 3*self.x_resolution+i] = -1*D_c[i-1]*cbernoulli.bernoulli(
                        (psi_dc[i-1]-psi_dc[i])/Ut)*1/(h[i-1])*2/(h[i+1]+h[i-1])

                    jacobian_c[i, 3*self.x_resolution+i-1] = D_c[i-1]*cbernoulli.bernoulli(
                        (psi_dc[i]-psi_dc[i-1])/Ut)*1/(h[i-1])*2/(h[i+1]+h[i-1])

                else:
                    jacobian_c[i, 3*self.x_resolution+i] = -1*(D_c[i-1]*mat_bernoulli[1, i-1, i] +
                                                               D_c[i]*mat_bernoulli[1, i+1, i])

                    jacobian_c[i, 3*self.x_resolution+i -
                               1] = (D_c[i-1]*mat_bernoulli[0, i, i-1])

                    jacobian_c[i, 3*self.x_resolution+i +
                               1] = (D_c[i]*mat_bernoulli[0, i, i+1])

        self.jacobian_psi = jacobian_psi
        self.jacobian_n = jacobian_n
        self.jacobian_p = jacobian_p
        if self.consider_cations == True:
            self.jacobian_c = jacobian_c

        if self.consider_cations is True:
            return np.concatenate((jacobian_psi, jacobian_n, jacobian_p, jacobian_c))
        else:
            return np.concatenate((jacobian_psi, jacobian_n, jacobian_p))

    def construct_d(self, freq):
        """Construct matrix D that accounts for the perturbation part of the total matrix

        Args:
            freq (float): Perturbation frequency

        Returns:
            array: Matrix D
        """

        # If cations are considered
        if self.consider_cations is True:
            d = np.zeros(
                (4*self.x_resolution, 4*self.x_resolution), dtype=np.cdouble)

            d[self.x_resolution:, self.x_resolution:] = (-2 * np.pi *
                                                         self.t0 * freq * 1j * np.eye(3 * self.x_resolution))
            d[self.x_resolution+self.idx_interface2_2, :] = 0
            d[2*self.x_resolution+self.idx_interface1_2, :] = 0
            d[3*self.x_resolution:3*self.x_resolution+self.idx_interface1_2+1,
                3*self.x_resolution:3*self.x_resolution+self.idx_interface1_2+1] = 0
            d[3*self.x_resolution+self.idx_interface2_2-1:,
                3*self.x_resolution+self.idx_interface2_2-1:] = 0
        # If cations are not considered
        else:
            d = np.zeros(
                (3*self.x_resolution, 3*self.x_resolution), dtype=np.cdouble)
            d[self.x_resolution:, self.x_resolution:] = (-2 * np.pi *
                                                         self.t0 * freq * 1j * np.eye(2 * self.x_resolution))
            d[self.x_resolution+self.idx_interface2_2, :] = 0
            d[2*self.x_resolution+self.idx_interface1_2, :] = 0
        return d

    def normalize(self, ):
        """Function to normalize the different variables

        Returns:
            arrays: The normalized variables
        """

        # DC potential
        psi_dc = self.pot / self.psi0

        # Electronic densities
        n_dc = self.n_dc / self.c0
        p_dc = self.p_dc / self.c0

        # Electron diffusion coefficent
        D_n = np.zeros(self.x_resolution)
        D_n[:self.idx_interface1_2] = k * \
            self.device.temp * self.device.mu_n_htl / e
        D_n[self.idx_interface1_2:self.idx_interface2_2] = k * \
            self.device.temp * self.device.mu_n_pero / e
        D_n[self.idx_interface2_2:] = k * \
            self.device.temp * self.device.mu_n_etl / e

        D_n = (D_n[1:] + D_n[:-1]) / 2
        D_n = np.append(D_n, D_n[-1])

        D_n = D_n / self.d0

        # Hole diffusion coefficent
        D_p = np.zeros(self.x_resolution)
        D_p[:self.idx_interface1_2] = k * \
            self.device.temp * self.device.mu_p_htl / e
        D_p[self.idx_interface1_2:self.idx_interface2_2] = k * \
            self.device.temp * self.device.mu_p_pero / e
        D_p[self.idx_interface2_2:] = k * \
            self.device.temp * self.device.mu_p_etl / e

        D_p = (D_p[1:] + D_p[:-1]) / 2
        D_p = np.append(D_p, D_p[-1])

        D_p = D_p / self.d0

        # Step array
        h = self.h/self.x0

        # Thermal voltage
        Vt = self.device.vt / self.psi0

        return psi_dc, n_dc, p_dc, D_n, D_p, h, Vt

    def calc_impedance_vs_t(self):
        """Function to compute the impedance of the device as a function of time
        """

        # First solve the DC solution, potential, and electronic carriers
        self.solve_dc()
        self.calc_potential(vapp=self.vlow)
        self.calc_electronic_charge_densities(vapp=self.vlow)

        # Calculate the DC ionic and the DC displacement current, the sum of the two is the total DC current
        self.Jt_ion = (e * self.e3_t * self.device.n_ion *
                       self.device.mu_ion).flatten()
        self.Jt_disp = (epsilon_0 * self.device.epsr_pero *
                        np.gradient(self.e3_t.flatten(), self.timesteps)).flatten()
        self.Jt = self.Jt_ion + self.Jt_disp

        # Normalize the DC variables
        psi_dc, n_dc, p_dc, D_n, D_p, h, Vt = self.normalize()

        # Initialize the admittance
        Ys = np.zeros(self.num_timesteps, dtype=complex)

        # Define the bounary conditions. The densities at the interface are all 0, the potential is the applied ac voltage
        boundaries = np.zeros((3*self.x_resolution), dtype=complex)
        boundaries[0] = self.vac / self.psi0

        # Iterate through the timesteps and solve the AC soltution
        for idx, _ in enumerate(self.timesteps):
            # Construct the jacobian
            jacobian = self.construct_jacobian(
                psi_dc[idx, :], D_n, D_p, n_dc[idx, :], p_dc[idx, :], h, Vt)

            # Define the total matrix (jacobian + D)
            A = csc_array(jacobian+self.construct_d(freq_ct))

            # Solve the system of equations
            sol = spsolve(A, boundaries, use_umfpack=True)

            # Extract the complex AC potential, electron density, and hole density
            psi_ac = sol[0*self.x_resolution:1 *
                         self.x_resolution] * self.psi0
            n_ac = sol[1*self.x_resolution:2 *
                       self.x_resolution] * self.c0
            p_ac = sol[2*self.x_resolution:3 *
                       self.x_resolution] * self.c0

            # Compute the AC current and hole density
            jn_ac = np.zeros(self.x_resolution, dtype=complex)
            jp_ac = np.zeros(self.x_resolution, dtype=complex)

            # Set the value of the ac electron density at the interface to the same value
            n_ac_tmp = n_ac.copy()
            n_ac_tmp[self.idx_interface2_2] = n_ac_tmp[self.idx_interface2_1]

            # Calculate AC electron current
            jn_ac[1:] = 1j*2*np.pi*freq_ct*e*np.array(list(accumulate([(n_ac_tmp[idx-1] + n_ac_tmp[idx])/2*(
                self.position_steps[idx]-self.position_steps[idx-1]) for idx in range(1, self.x_resolution)])))

            # Calculate AC hole current
            p_ac_tmp = p_ac.copy()
            # p_ac_tmp[self.idx_interface1_2] = p_ac_tmp[self.idx_interface1_1]
            jp_ac[:-1] = 1j*2*np.pi*freq_ct*e*np.flip(np.array(list(accumulate([(p_ac_tmp[idx] + p_ac_tmp[idx-1])/2*(
                self.position_steps[idx]-self.position_steps[idx-1]) for idx in reversed(range(1, self.x_resolution))]))))

            # Calculate the ac electric field
            Efield_ac = np.zeros(self.x_resolution, dtype=complex)
            Efield_ac[:self.idx_interface1_2] = -1 * np.gradient(np.real(
                psi_ac[:self.idx_interface1_2]), self.position_steps[:self.idx_interface1_2]) - 1j * np.gradient(np.imag(psi_ac[:self.idx_interface1_2]), self.position_steps[:self.idx_interface1_2])
            Efield_ac[self.idx_interface1_2:self.idx_interface2_2] = -1 * np.gradient(np.real(
                psi_ac[self.idx_interface1_2:self.idx_interface2_2]), self.position_steps[self.idx_interface1_2:self.idx_interface2_2]) - 1j * np.gradient(np.imag(psi_ac[self.idx_interface1_2:self.idx_interface2_2]), self.position_steps[self.idx_interface1_2:self.idx_interface2_2])
            Efield_ac[self.idx_interface2_2:] = -1 * np.gradient(np.real(
                psi_ac[self.idx_interface2_2:]), self.position_steps[self.idx_interface2_2:]) - 1j * np.gradient(np.imag(psi_ac[self.idx_interface2_2:]), self.position_steps[self.idx_interface2_2:])

            # Calculate the ac displacement current
            jdisp_ac = 1j*2*np.pi*self.arr_epsilon*epsilon_0*freq_ct*Efield_ac

            # Calculate the total current
            j_tot = np.average(jdisp_ac+jn_ac+jp_ac)

            # Calculate the admittance
            Ys[idx] = j_tot / self.vac
            self.n_ac[idx, :] = n_ac
            self.p_ac[idx, :] = p_ac
            self.psi_ac[idx, :] = psi_ac
            self.jn_ac[idx, :] = jn_ac
            self.jp_ac[idx, :] = jp_ac
            self.jdisp_ac[idx, :] = jdisp_ac

        # Calculate the conductance
        self.Ct_Gs = np.real(Ys)
        # Calculate the capacitance
        self.Ct_Cs = np.imag(Ys)/(2*np.pi*freq_ct)

        # Calculate the impedance
        self.Ct_Zs = (self.device.r_shunt * 1/(Ys)) / \
            (self.device.r_shunt + 1/(Ys))

        # Remove the initial very early timeteps
        self.set_parameters_to_timestep_length()

    def calc_impedance_vs_t_simplified(self):
        """Function to calcualte the capcaitance based on the parallel plate capacitance approximation of the depletion layers
        """

        # Solve the DC solution
        self.solve_dc()

        # Compute the ionic current, displacement current, total current
        self.Jt_ion = (e * self.e3_t * self.device.n_ion *
                       self.device.mu_ion).flatten()
        self.Jt_disp = (epsilon_0 * self.device.epsr_pero *
                        np.gradient(self.e3_t.flatten(), self.timesteps)).flatten()
        self.Jt = self.Jt_ion + self.Jt_disp

        # Compute capacitance of HTL
        if self.device.htl_doping == True:
            c1 = epsilon_0 * self.device.epsr_htl / self.w1_t
        else:
            c1 = epsilon_0 * self.device.epsr_htl / self.device.d_htl

        # Compute capacitance of perovskite
        c2 = epsilon_0 * self.device.epsr_pero / self.device.d_pero

        # Compute capacitance of ETL
        if self.device.etl_doping == True:
            c3 = epsilon_0 * self.device.epsr_etl / self.w5_t
        else:
            c3 = epsilon_0 * self.device.epsr_etl / self.device.d_etl

        # Calculate capacitance
        self.Ct_Cs = np.array((1/c1 + 1/c2 + 1/c3)**-1)
        self.Ct_Cs = np.array([c[0] for c in self.Ct_Cs])
        self.set_parameters_to_timestep_length()

    def calc_impedance_vs_freq(self, freqs):
        """Function to calculate the impedance as a function of frequency

        Args:
            freqs (array): array containing the frequencies

        Returns:
            arrays: Array of impedance and capacitance 
        """

        # Solve DC solution
        self.solve_dc()

        # Solve DC potential
        self.calc_potential(vapp=self.vlow)

        # Solve DC electronic carrier densities
        self.calc_electronic_charge_densities(vapp=self.vlow)

        # Normalize dc variables
        psi_dc, n_dc, p_dc, D_n, D_p, h, Vt = self.normalize()
        c_dc = self.c_dc / self.c0
        D_c = np.full(self.x_resolution, k*self.device.temp *
                      self.device.mu_ion/e) / self.d0
        D_c[:self.idx_interface1_2] = 0
        D_c[self.idx_interface2_2:] = 0

        # Initiailue admittance, impedance, and capacitance
        Ys = np.zeros(len(freqs), dtype=complex)
        Zs = np.zeros(len(freqs), dtype=complex)
        Cs = np.zeros(len(freqs))

        # Construct boundary condition array
        if self.consider_cations is True:
            boundaries = np.zeros((4*self.x_resolution), dtype=complex)
        else:
            boundaries = np.zeros((3*self.x_resolution), dtype=complex)
        boundaries[0] = self.vac / self.psi0

        # Construct jacobian
        idx = -1
        jacobian = self.construct_jacobian(
            psi_dc[idx, :], D_n, D_p,  n_dc[idx, :], p_dc[idx, :], h, Vt, D_c, c_dc[idx, :], )

        # Interate over the frequencies and calculate the capacitance
        for i, freq in enumerate(freqs):

            # Construct complete matrix (jacobian + D)
            A = csc_array(jacobian+self.construct_d(freq))

            # Solve system of equaitons
            sol = spsolve(A, boundaries, use_umfpack=True)

            # Get ac potential and carrier densities
            psi_ac = sol[0*self.x_resolution:1 *
                         self.x_resolution] * self.psi0
            n_ac = sol[1*self.x_resolution:2 *
                       self.x_resolution] * self.c0
            p_ac = sol[2*self.x_resolution:3 *
                       self.x_resolution] * self.c0

            # Initilize AC currents
            jn_ac = np.zeros(self.x_resolution, dtype=complex)
            jp_ac = np.zeros(self.x_resolution, dtype=complex)

            # Compute the AC electron current
            n_ac_tmp = n_ac.copy()
            n_ac_tmp[self.idx_interface2_2] = n_ac_tmp[self.idx_interface2_1]
            jn_ac[1:] = 1j*2*np.pi*freq*e*np.array(list(accumulate([(n_ac_tmp[idx-1] + n_ac_tmp[idx])/2*(
                self.position_steps[idx]-self.position_steps[idx-1]) for idx in range(1, self.x_resolution)])))

            # Compute the AC hole current
            p_ac_tmp = p_ac.copy()
            p_ac_tmp[self.idx_interface1_2] = p_ac_tmp[self.idx_interface1_1]
            jp_ac[:-1] = 1j*2*np.pi*freq*e*np.flip(np.array(list(accumulate([(p_ac_tmp[idx] + p_ac_tmp[idx-1])/2*(
                self.position_steps[idx]-self.position_steps[idx-1]) for idx in reversed(range(1, self.x_resolution))]))))

            # Compute the AC electric field and displacement current
            Efield_ac = np.zeros(self.x_resolution, dtype=complex)
            Efield_ac[:self.idx_interface1_2] = -1 * np.gradient(np.real(
                psi_ac[:self.idx_interface1_2]), self.position_steps[:self.idx_interface1_2]) - 1j * np.gradient(np.imag(psi_ac[:self.idx_interface1_2]), self.position_steps[:self.idx_interface1_2])
            Efield_ac[self.idx_interface1_2:self.idx_interface2_2] = -1 * np.gradient(np.real(
                psi_ac[self.idx_interface1_2:self.idx_interface2_2]), self.position_steps[self.idx_interface1_2:self.idx_interface2_2]) - 1j * np.gradient(np.imag(psi_ac[self.idx_interface1_2:self.idx_interface2_2]), self.position_steps[self.idx_interface1_2:self.idx_interface2_2])
            Efield_ac[self.idx_interface2_2:] = -1 * np.gradient(np.real(
                psi_ac[self.idx_interface2_2:]), self.position_steps[self.idx_interface2_2:]) - 1j * np.gradient(np.imag(psi_ac[self.idx_interface2_2:]), self.position_steps[self.idx_interface2_2:])
            jdisp_ac = 1j*2*np.pi*self.arr_epsilon*epsilon_0*freq*Efield_ac

            self.n_ac[idx, :] = n_ac
            self.p_ac[idx, :] = p_ac
            self.psi_ac[idx, :] = psi_ac
            self.jn_ac[idx, :] = jn_ac
            self.jp_ac[idx, :] = jp_ac
            self.jdisp_ac[idx, :] = jdisp_ac

            # Consider mobile cations, calculate AC cation current
            if self.consider_cations is True:
                c_ac = sol[3*self.x_resolution:4 *
                           self.x_resolution] * self.c0

                c_ac_tmp = c_ac.copy()
                c_ac_tmp[self.idx_interface1_2] = c_ac_tmp[self.idx_interface1_1]
                c_ac_tmp[self.idx_interface2_1] = c_ac_tmp[self.idx_interface2_2]
                jc_ac = np.zeros(self.x_resolution, dtype=complex)
                jc_ac[0:-1] = 1j*2*np.pi*freq*e*np.flip(np.array(list(accumulate([(c_ac_tmp[idx] + c_ac_tmp[idx+1])/2*(
                    self.position_steps[idx+1]-self.position_steps[idx]) for idx in reversed(range(0, self.x_resolution-1))]))))

                # Compute total current (with cations)
                j_tot = np.average(jdisp_ac+jn_ac+jp_ac+jc_ac)

                self.c_ac[idx, :] = c_ac
                self.jc_ac[idx, :] = jc_ac
            else:
                # Compute total current (without cations)
                j_tot = np.average(jdisp_ac+jn_ac+jp_ac)

            # Compute admittance
            Ys[i] = j_tot / self.vac

        # Compute conductance
        self.Cf_Gs = np.real(Ys)

        # Compute impedance
        self.Cf_Zs = (self.device.r_shunt * 1/(Ys)) / \
            (self.device.r_shunt + 1/(Ys))
        # Compute capacitance
        self.Cf_Cs = np.imag(1/self.Cf_Zs)/(2*np.pi*freqs)

        return Zs, Cs

    def set_parameters_to_timestep_length(self, ):
        """ In order to get an accurate solution at early times, we extend add additional time points in the time array. In this function we remove the added time points.
        """
        idx_start = len(self.early_timesteps)

        var_list = ['timesteps', 'Ct_Cs', 'Ct_Gs', 'Jt', 'Ct_Zs', 'n1_t', 'w1_t', 'n2_t', 'w2_t', 'n4_t',
                    'w4_t', 'n5_t', 'w5_t', 'e_field', 'pot', 'net_density', 'p_dc', 'n_dc', 'c_dc', 'psi_ac', 'p_ac', 'n_ac', 'jn_ac', 'jp_ac', 'jc_ac', 'jdisp_ac', 'idx_w1',  'idx_w2', 'idx_w4', 'idx_w5', 'Jt_ion', 'Jt_disp', 'e3_t', 'E_cb', 'E_vb']

        for attr_name in var_list:
            attr = getattr(self, attr_name)
            if attr is not None:
                if (len(attr.shape) > 1) and attr.shape[1] > 1:
                    if attr.shape[0] > len(self.original_timesteps):
                        setattr(self, attr_name, attr[idx_start:, :])
                else:
                    if len(attr) > len(self.original_timesteps):
                        setattr(self, attr_name, attr[idx_start:])
