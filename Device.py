import numpy as np
from scipy.constants import e, k, epsilon_0
from params import d_pero, d_etl, d_htl, n_htl, n_etl, mu_p_htl, mu_p_pero, mu_n_etl, mu_n_pero, mu_n_htl, mu_p_etl, n0_cb_etl, n0_vb_htl, n0_cb_pero, n0_vb_pero, cb_etl, vb_htl, cb_pero, vb_pero
from params import epsr_htl, epsr_etl, epsr_pero, consider_cations, etl_doping, htl_doping, D0_ion, ea_ion_mob, wf_anode, wf_cathode, n0_vb_etl, n0_cb_htl, vb_etl, cb_htl, n0_ion, ea_dn_ion, dn_ion, r_shunt
import pandas as pd


class Device ():

    def __init__(self, temp):
        self._temp = temp

        self.d_pero = d_pero  # [m] Perovskite thickness
        self.d_etl = d_etl  # [m] ETL thickness
        self.d_htl = d_htl  # [m] HTL thickness
        self._n0_ion = n0_ion  # [1/m3] Mobile ion density
        # [eV] Activation energy of defect formation
        self._ea_dn_ion = ea_dn_ion
        self._dn_ion = dn_ion  # [eV] Activation energy of defect formation

        self._n_ion = self._n0_ion + self.dn_ion * \
            np.exp(-self._ea_dn_ion*e/(k*self.temp)
                   )  # [m^-3]Temperature activated ion density

        self._etl_doping = etl_doping  # True if ETL is doped
        self._htl_doping = htl_doping  # True if HTL is doped

        self._n_htl = n_htl  # [1/m3] Doping density HTL
        self._n_etl = n_etl  # [1/m3] Doping density ETL

        self._wf_anode = wf_anode  # [eV]
        self._wf_cathode = wf_cathode  # [eV]

        self.mu_p_htl = mu_p_htl  # [m^2/Vs]
        self.mu_p_pero = mu_p_pero  # [m^2/Vs]
        self.mu_p_etl = mu_p_etl  # [m^2/Vs]

        self.mu_n_etl = mu_n_etl  # [m^2/Vs]
        self.mu_n_pero = mu_n_pero  # [m^2/Vs]
        self.mu_n_htl = mu_n_htl  # [m^2/Vs]

        self.n0_cb_etl = n0_cb_etl  # [1/m3] Effective DOS of CB in ETL
        self.n0_vb_etl = n0_vb_etl  # [1/m3] Effective DOS of VB in ETL

        self.n0_vb_htl = n0_vb_htl  # [1/m3] Effective DOS of VB in HTL
        self.n0_cb_htl = n0_cb_htl  # [1/m3] Effective DOS of CB in HTL

        self.n0_cb_pero = n0_cb_pero  # [1/m3] Effective DOS of CB in Pero
        self.n0_vb_pero = n0_vb_pero  # [1/m3] Effective DOS of VB in Pero

        self.cb_etl = cb_etl  # [eV] Energy level of CB in ETL
        self.vb_etl = vb_etl  # [eV] Energy level of VB in ETL
        self.cb_htl = cb_htl  # [eV] Energy level of CB in HTL

        self.vb_htl = vb_htl  # [eV] Energy level of VB in HTL
        self.cb_pero = cb_pero  # [eV] Energy level of CB in Pero
        self.vb_pero = vb_pero  # [eV] Energy level of CB in Pero

        self.epsr_htl = epsr_htl  # Permittivity HTL
        self.epsr_etl = epsr_etl  # Permittivity ETL
        self.epsr_pero = epsr_pero  # Permittivity perovskite

        self.r_shunt = r_shunt  # [Ohm m^2] Shunt resistance

        self.m_cb_int2 = None  # Factor to compute interfaces accurately
        self.m_vb_int1 = None  # Factor to compute interfaces accurately

        self.pot_offset = 0

        self._D0_ion = D0_ion  # [m2/s] Diffusion coefficient of mobile ions
        # [eV] Activation energy of mobility of ions

        # [eV] Activation energy of diffusion coefficient of ions
        self._ea_ion_mob = ea_ion_mob
        self.D_ion = self._D0_ion * \
            np.exp(-self._ea_ion_mob*e/(k*self._temp)
                   )  # [m^2/s] Diffusion coefficient of mobile ions
        # [m^2/(Vs)] Mobility of ions
        self.mu_ion = e*self.D_ion / (k*self._temp)

        if self.htl_doping == True:  # Calculate anode work function if HTL is doped
            self.wf_anode = np.log(self.n_htl/self.n0_vb_htl) * \
                k*self._temp/e+self.vb_htl

        if self.etl_doping == True:  # Calculate cathode work function if HTL is doped
            self.wf_cathode = self.cb_etl - \
                np.log(self.n_etl/self.n0_cb_etl)*k*self._temp/e
            self.pot_offset = wf_cathode - self.wf_cathode

        # Calculate built in voltage
        self.v_bi = np.abs(self.wf_anode - self.wf_cathode)

        self.vt = k*self._temp/e  # Thermal voltage
        self.max_acc_nion = None

        self.calc_ms()  # Calculate factors to compute interfaces accurately

    def calc_ms(self,):
        """Calculate factors for offsets at interfaces dependent on the energetic offsets and the effective density of states
        """
        self.m_cb_int2 = self.n0_cb_pero / self.n0_cb_etl * \
            np.exp((self.cb_pero - self.cb_etl)*e/(k*self.temp))
        self.m_vb_int1 = self.n0_cb_pero / self.n0_cb_htl * \
            np.exp(-1*(self.vb_pero - self.vb_htl)*e/(k*self.temp))

        self.m_cb_int1 = self.n0_cb_htl / self.n0_cb_pero * \
            np.exp((self.cb_htl - self.cb_pero)*e/(k*self.temp))
        self.m_vb_int2 = self.n0_cb_etl / self.n0_cb_pero * \
            np.exp(-1*(self.vb_etl - self.vb_pero)*e/(k*self.temp))

    def get_w2(self, n):
        """Calculate the width of the depletion region

        Args:
            n (float): Density of bulk ions [m^-3]

        Returns:
            float: Width of the accuulation width [m]
        """
        return np.sqrt((epsilon_0*self.epsr_pero*self.vt)/(e*n))

    @property
    def temp(self):
        """Getter function for temperature

        Returns:
            float: tempearture [K]
        """
        return self._temp

    @temp.setter
    def temp(self, value):
        """Setter function of temperature; recalcualte all device properties that are dependent on the temperature

        Args:
            value (float): Temperature [K]
        """
        self._temp = value
        # Diffusion coefficient of ions
        self.D_ion = self.D0_ion * np.exp(-self.ea_ion_mob*e/(k*self._temp))
        self.mu_ion = e*self.D_ion / (k*self._temp)  # Mobility of ions
        self.vt = k*self._temp/e  # Thermal voltage

        self._n_ion = self._n0_ion + self.dn_ion * \
            np.exp(-self._ea_dn_ion*e/(k*self.temp))  # Mobile ion density

        if self.htl_doping == True:  # HTL doping
            self.wf_anode = np.log(self.n_htl/self.n0_vb_htl) * \
                k*self._temp/e+self.vb_htl

        if self.etl_doping == True:  # ETL doping
            self.wf_cathode = self.cb_etl - \
                np.log(self.n_etl/self.n0_cb_etl)*k*self._temp/e
            self.pot_offset = wf_cathode - self.wf_cathode

    @property
    def D0_ion(self):
        """Getter function for prefactor of diffusion coefficient of ions

        Returns:
            float:  Prefactor of diffusion coefficient of ions [m^2/s] 
        """
        return self._D0_ion

    @D0_ion.setter
    def D0_ion(self, value):
        """Setter function for prefactor of diffusion coefficient of ions; Adapt parameters that are dependent on the diffusion coefficient

        Args:
            value (float): prefactor of diffusion coefficient of ions [m^2/s]
        """
        self._D0_ion = value
        self.D_ion = self._D0_ion * \
            np.exp(-self._ea_ion_mob*e/(k*self._temp))
        self.mu_ion = e*self.D_ion / (k*self._temp)

    @property
    def ea_ion_mob(self):
        """Getter function for activation energy of diffusion coefficient/mobility of ions

        Returns:
            float: [eV]  activation energy of diffusion coefficient/mobility  of ions
        """
        return self._ea_ion_mob

    @ea_ion_mob.setter
    def ea_ion_mob(self, value):
        """Setter function for activation energy  of diffusion coefficient/mobility ions; Adapt parameters that are dependent on the diffusion coefficient

        Args:
            value (float): activation energy  of diffusion coefficient/mobility ions [eV]
        """
        self._ea_ion_mob = value
        self.D_ion = self._D0_ion * \
            np.exp(-self._ea_ion_mob*e/(k*self._temp))
        self.mu_ion = e*self.D_ion / (k*self._temp)

    @property
    def n_ion(self):
        return self._n_ion

    @n_ion.setter
    def n_ion(self, value):
        """Setter function for ion density

        Args:
            value (float): Setter function for ion density [m^-3]
        """
        self._n_ion = value

    @property
    def dn_ion(self):
        return self._dn_ion

    @dn_ion.setter
    def dn_ion(self, value):
        """Setter function for prefactor of temperature activated ion density

        Args:
            value (float): prefactor of temperature activated ion density [m^-3]
        """
        self._dn_ion = value
        self._n_ion = self._n0_ion + self.dn_ion * \
            np.exp(-self._ea_dn_ion*e/(k*self.temp))

    @property
    def n0_ion(self):
        return self._n0_ion

    @n0_ion.setter
    def n0_ion(self, value):
        """Setter function for constant ion density

        Args:
            value (float): ion density [cm^-3]
        """
        self._n0_ion = value
        self._n_ion = self._n0_ion + self.dn_ion * \
            np.exp(-self._ea_dn_ion*e/(k*self.temp))

    @property
    def ea_dn_ion(self):
        return self._ea_dn_ion

    @ea_dn_ion.setter
    def ea_dn_ion(self, value):
        """Setter function for activation energy of temperature activated ion density

        Args:
            value (float): activation energy of temperature activated ion density [eV]
        """
        self._ea_dn_ion = value
        self._n_ion = self._n0_ion + self.dn_ion * \
            np.exp(-self._ea_dn_ion*e/(k*self.temp))

    @property
    def n_htl(self):
        return self._n_htl

    @n_htl.setter
    def n_htl(self, value):
        """Setter function for doping density of HTL

        Args:
            value (float): doping density of HTL [m^-3]
        """
        self._n_htl = value
        if self._htl_doping == True:
            self.wf_anode = np.log(self.n_htl/self.n0_vb_htl) * \
                k*self._temp/e+self.vb_htl

    @property
    def n_etl(self):
        return self._n_etl

    @n_etl.setter
    def n_etl(self, value):
        """Setter function for doping density of ETL

        Args:
            value (float): doping density of ETL [m^-3]
        """
        self._n_etl = value
        if self.etl_doping == True:
            self.wf_cathode = self.cb_etl - \
                np.log(self.n_etl/self.n0_cb_etl)*k*self._temp/e
            self.pot_offset = wf_cathode - self.wf_cathode

    @property
    def wf_anode(self):
        return self._wf_anode

    @wf_anode.setter
    def wf_anode(self, value):
        """Setter function for anode work function 

        Args:
            value (float): anode work function [eV]
        """
        self._wf_anode = value
        self.v_bi = np.abs(self._wf_anode - self._wf_cathode)

    @property
    def wf_cathode(self):
        return self._wf_cathode

    @wf_cathode.setter
    def wf_cathode(self, value):
        """Setter function for cathode work function 

        Args:
            value (float): cathode work function [eV]
        """
        self._wf_cathode = value
        self.v_bi = np.abs(self._wf_anode - self._wf_cathode)

    @property
    def cb_etl(self):
        return self._cb_etl

    @cb_etl.setter
    def cb_etl(self, value):
        """Setter function for conduction band energy of ETL 

        Args:
            value (float): conduction band energy of ETL  [eV]
        """
        self._cb_etl = value
        if self.etl_doping == True:
            self.wf_cathode = self.cb_etl - \
                np.log(self.n_etl/self.n0_cb_etl)*k*self._temp/e
            self.pot_offset = wf_cathode - self.wf_cathode

    @property
    def vb_htl(self):
        return self._vb_htl

    @vb_htl.setter
    def vb_htl(self, value):
        """Setter function for valence band energy of HTL 

        Args:
            value (float): valence band energy of HTL  [eV]
        """
        self._vb_htl = value
        if self.htl_doping == True:
            self.wf_anode = np.log(self.n_htl/self.n0_vb_htl) * \
                k*self._temp/e+self.vb_htl

    @property
    def htl_doping(self):
        return self._htl_doping

    @htl_doping.setter
    def htl_doping(self, value):
        """Setter function for activation of HTL doping and adapt other parameters that are dependent on HTL doping

        Args:
            value (bool): Activate htl doping
        """
        self._htl_doping = value
        if self.htl_doping == True:
            self.wf_anode = np.log(self.n_htl/self.n0_vb_htl) * \
                k*self._temp/e+self.vb_htl

    @property
    def etl_doping(self):
        return self._etl_doping

    @etl_doping.setter
    def etl_doping(self, value):
        """Setter function for activation of ETL doping and adapt other parameters that are dependent on ETL doping

        Args:
            value (bool): Activate ETL doping
        """
        self._etl_doping = value
        if self._etl_doping == True:
            self.wf_cathode = self.cb_etl - \
                np.log(self.n_etl/self.n0_cb_etl)*k*self._temp/e
            self.pot_offset = wf_cathode - self.wf_cathode

    def get_params_dict(self):
        """Get all the parameters as a dictionary

        Returns:
            Dict: Dictionary with all device parameters
        """
        return {
            'd_pero': self.d_pero,  # [m] Perovskite thickness
            'd_etl': self.d_etl,  # [m] ETL thickness
            'd_htl': self.d_htl,  # [m] HTL thickness
            'wf_anode': self.wf_anode,  # [eV] Anode work function
            'wf_cathode': self.wf_cathode,  # [eV] Anode work function
            'n0_ion': self.n0_ion,  # [1/m3] Mobile ion density
            'dn_ion': self.dn_ion,  #
            'r_shunt': self.r_shunt,  # [Ohm m^2] Shunt resistance
            # [1/m3] Activation energy temperature activated ion density
            'ea_dn_ion': self.ea_dn_ion,
            'n_htl': self.n_htl,  # [1/m3] Doping density HTL
            'n_etl': self.n_etl,  # [1/m3] Doping density ETL
            'mu_p_htl': self.mu_p_htl,  # [m^2/Vs]
            'mu_p_pero': self.mu_p_pero,  # [m^2/Vs]
            'mu_p_etl': self.mu_p_etl,  # [m^2/Vs]
            'mu_n_etl': self.mu_n_etl,  # [m^2/Vs]
            'mu_n_pero': self.mu_n_pero,  # [m^2/Vs]
            'mu_n_htl': self.mu_n_htl,  # [m^2/Vs]
            'n0_cb_etl': self.n0_cb_etl,  # [1/m3] Effective DOS of CB in ETL
            'n0_vb_etl': self.n0_vb_etl,  # [1/m3] Effective DOS of CB in ETL
            'n0_cb_htl': self.n0_cb_htl,  # [1/m3] Effective DOS of CB in ETL
            'n0_vb_htl': self.n0_vb_htl,  # [1/m3] Effective DOS of VB in HTL
            # [1/m3] Effective DOS of CB in Pero
            'n0_cb_pero': self.n0_cb_pero,
            # [1/m3] Effective DOS of VB in Pero
            'n0_vb_pero': self.n0_vb_pero,
            'cb_etl': self.cb_etl,  # [eV] Energy level of CB in ETL
            'vb_etl': self.vb_etl,  # [eV] Energy level of CB in ETL
            'vb_htl': self.vb_htl,  # [eV] Energy level of VB in HTL
            'cb_htl': self.cb_htl,  # [eV] Energy level of CB in HTL
            'cb_pero': self.cb_pero,  # [eV] Energy level of CB in Pero
            'vb_pero': self.vb_pero,  # [eV] Energy level of CB in Pero
            'epsr_htl': self.epsr_htl,  # Permittivity HTL
            'epsr_etl': self.epsr_etl,  # Permittivity ETL
            'epsr_pero': self.epsr_pero,  # Permittivity perovskite
            'D0_ion': self.D0_ion,  # [m^2/Vs] Zero mobility of ions
            'ea_ion_mob': self.ea_ion_mob,  # [m^2/Vs] Zero mobility of ions
            'etl_doping': self.etl_doping,  # ETL doping
            'htl_doping': self.htl_doping,  # HTL doping

        }

    def load_params_from_file(self, path):
        """Load parameters from a csv file

        Args:
            path (Path): Load parameter file from path 
        """
        df = pd.read_csv(path)
        dict = df.set_index('Unnamed: 0')['Value'].to_dict()

        self.wf_anode = float(dict['wf_anode'])  # [eV] Work function of anode
        # [eV] Work function of cathode
        self.wf_cathode = float(dict['wf_cathode'])
        self.d_pero = float(dict['d_pero'])  # [m] Perovskite thickness
        self.d_etl = float(dict['d_etl'])  # [m] ETL thickness
        self.d_htl = float(dict['d_htl'])  # [m] HTL thickness
        self.n0_ion = float(dict['n0_ion'])  # [1/m3] Mobile ion density
        self.dn_ion = float(dict['dn_ion'])  # [1/m3] Mobile ion density
        # [1/m3] Activation energy temperature activated ion density
        self.ea_dn_ion = float(dict['ea_dn_ion'])

        self.n_htl = float(dict['n_htl'])  # [1/m3] Doping density HTL
        self.n_etl = float(dict['n_etl'])  # [1/m3] Doping density ETL
        self.mu_p_htl = float(dict['mu_p_htl'])  # [m^2/Vs]
        self.mu_p_pero = float(dict['mu_p_pero'])  # [m^2/Vs]
        self.mu_p_etl = float(dict['mu_p_etl'])  # [m^2/Vs]
        self.r_shunt = float(dict['r_shunt'])  # [Ohm m^2] Shunt resistance
        self.mu_n_etl = float(dict['mu_n_etl'])  # [m^2/Vs]
        self.mu_n_pero = float(dict['mu_n_pero'])  # [m^2/Vs]
        self.mu_n_htl = float(dict['mu_n_htl'])  # [m^2/Vs]

        # [1/m3] Effective DOS of CB in ETL
        self.n0_cb_etl = float(dict['n0_cb_etl'])
        # [1/m3] Effective DOS of VB in ETL
        self.n0_vb_etl = float(dict['n0_vb_etl'])

        # [1/m3] Effective DOS of VB in HTL
        self.n0_vb_htl = float(dict['n0_vb_htl'])
        # [1/m3] Effective DOS of CB in HTL
        self.n0_cb_htl = float(dict['n0_cb_htl'])

        # [1/m3] Effective DOS of CB in Pero
        self.n0_cb_pero = float(dict['n0_cb_pero'])
        # [1/m3] Effective DOS of VB in Pero
        self.n0_vb_pero = float(dict['n0_vb_pero'])

        self.cb_etl = float(dict['cb_etl'])  # [eV] Energy level of CB in ETL
        self.vb_etl = float(dict['vb_etl'])  # [eV] Energy level of VB in ETL

        self.vb_htl = float(dict['vb_htl'])  # [eV] Energy level of VB in HTL
        self.cb_htl = float(dict['cb_htl'])  # [eV] Energy level of CB in HTL

        # [eV] Energy level of CB in Pero
        self.cb_pero = float(dict['cb_pero'])
        # [eV] Energy level of CB in Pero
        self.vb_pero = float(dict['vb_pero'])

        self.epsr_htl = float(dict['epsr_htl'])  # Permittivity HTL
        self.epsr_etl = float(dict['epsr_etl'])  # Permittivity ETL
        self.epsr_pero = float(dict['epsr_pero'])  # Permittivity perovskite

        self.D0_ion = float(dict['D0_ion'])  # [m2/Vs] Mobility of mobile ions

        # [eV] Activation energy of mobility of ions
        self.ea_ion_mob = float(dict['ea_ion_mob'])

        self.etl_doping = eval(dict['etl_doping'])  # ETL doping
        self.htl_doping = eval(dict['htl_doping'])  # HTL doping
