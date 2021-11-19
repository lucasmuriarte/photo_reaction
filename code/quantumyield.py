# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:34:25 2020

@author: lucas martinez uriarte
"""
import numpy as np
import matplotlib.pyplot as plt
import lmfit
# import seaborn as sns
# import os
# from scipy.signal import savgol_filter as SF
from scipy.integrate import odeint


# from prepocessPhotolysisClass import FilterData


__h = 6.626e-34  # Plank´s constant
__na = 6.022e23  # Avogadro´s number
__c = 2.998e8  # Speed of light


def energy_of_single_photon(landa):
    """
    Calculates the energy of a sinlge photon at of the wavelength (landa) given.
    Used the Plank´s constant (h), and the speec of light (c)

    Parameters
    ----------
    landa:
        The wavelenght of the irradiation source
    """
    return (__h * __c) / (landa * 1e-9)


def number_of_photons(energy: float, landa: float):
    """
    Calculates the number of pheotons given the wavelenght of the rediation
    and the energy measured in mW

    Parameters
    ----------
    energy:
        The energy value meassured

    landa:
        The wavelenght of the irradiation source
    """
    energy /= 1000
    return energy/energy_of_single_photon(landa)


def concentration_of_photons(energy, landa, volume):
    n_photons = number_of_photons(energy, landa)
    mol_photons = n_photons / __na
    molar_concentration = mol_photons/(volume/1000)
    return molar_concentration


class QuantumYields:
    def __init__(self,
                 concentrations_profiles,
                 eps_all,
                 I_all,
                 thermal,
                 I_landa=[485, 405],
                 volume=2,
                 initial_time=30,
                 optical_path=1):
        """
        QuantumYields need irradiation intensities and absorption coeficients
        at which the sample was irradiated

        For a reversible fluorescent protein the reaction is supose
        to go On to Off at 488 nm and Off to On at 405 nm
        therefore
            --> epsilons_488=[epsilon_Off_488,epsilon_On_488]
            --> epsilons_405=[epsilon_Off_405,epsilon_On_405]
            --> Eps_all should be a list containing epsilons_488 and
                epsilons_405 in the order of which concentration profiles
                of experiments are load
        """
        index = int(np.mean([(i.Time-initial_time).abs().sort_values().index[0]
                             for i in concentrations_profiles]))
        self.index = index + 1
        self.concentrations_profiles = [i.iloc[:, 1:].values for i in
                                        concentrations_profiles]
        self.times = [i.iloc[:, 0].values for i in concentrations_profiles]
        self.report = []
        self.epsilons = eps_all
        self.irradiation_intensities = [I_all[i]*I_landa[i]*1e-12/(6.626e-34*2.998e8*6.022e23)/(volume*1e-3) for i in range(len(I_all))]
        self.thermal = 1/(3600*thermal)
        self.concentrations = [(i[i.Time < initial_time].iloc[:, 1:]).mean().values for i in concentrations_profiles]
        self.optical_path = optical_path
        self.params = None
        self.result_Global = None
        self.result_single = None

    # PHOTOCHEMICAL REACTION OF DISAPPEAR (OF A) AND FORMATION (OF B)
    # BETWEEN TWO PRODUCTS A AND B
    def photoReaction(self, concentrations, time, Fi_A, Fi_B, epsilons,
                      irr_intensities, K_A):
        """
        photo-reaction of formation of a product B irradiating A
        where A and B absorb at irradiation wavelength
        """
        ca, cb = concentrations[0], concentrations[1]
        ea, eb = epsilons[0], epsilons[1]
        inten_a = ((ea*ca)/(ea*ca+eb*cb)) * irr_intensities * \
                  (1 - 10 ** (-(ea * ca + eb * cb) * self.optical_path))
        inten_b = ((eb*cb)/(ea*ca+eb*cb)) * irr_intensities * \
                  (1 - 10 ** (-(ea * ca + eb * cb) * self.optical_path))

        variation_of_b_over_time = -Fi_B*inten_b + Fi_A*inten_a + K_A*ca
        # everything that disappear from A goes to B
        variation_of_a_over_time = -variation_of_b_over_time

        return [variation_of_a_over_time, variation_of_b_over_time]
    
    def photoReactionDiff(self, concentrations, time, Fi_A, Fi_B, epsilons,
                          irr_intensity, K_A):
        """
        Since the photochemical reacction is a differential equation,
        we need to do the integration of it
        returns the photo_reaction1_formation integration over time
        """
        return odeint(self.photoReaction, concentrations, time,
                      args=(Fi_A, Fi_B, epsilons, irr_intensity, K_A))
    
    def optimizationGlobal(self, params, time_all, data_all):
        """
        Optimization returning residues to be minized for function
        photoReactionDiff, For several experiments.
        Arguments needed ,(except params) are parameters for photoReactionDiff
        as a list  of element, where each element is data for one experiment.
        It returns the total residues for all the  experimets at same time to
        global fit data and optain global quantum yields. Prams should be a
        parmas objet for lmfit library
        """
        residuos_all = []
        for i in range(len(data_all)):
            residuos = np.empty(data_all[i].shape)
            residuos.flatten()
            time = time_all[i]
            concentration = self.concentrations[i]
            epsilons = self.epsilons[i]
            irr_intensity = self.irradiation_intensities[i]
            if self.thermal is 'param':
                thermal = params['k_A']
            else:
                thermal = self.thermal
            fit = self.photoReactionDiff(concentration, time,
                                         params['Fi_A'].value,
                                         params['Fi_B'].value,
                                         epsilons,
                                         irr_intensity,
                                         thermal)

            residuos = data_all[i].flatten()-fit.flatten()
            residuos_all.append(residuos.flatten())
        return np.array([item for sublist in residuos_all for item in sublist])
            
    def paramsInitialization(self, Fi_A=0.5, Fi_B=0.5, thermal_A=None):

        self.params = lmfit.Parameters()
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        self.params.add_many(('Fi_A', Fi_A, True, 0, 1, None),
                             ('Fi_B', Fi_B, True, 0, 1, None))
        
        if thermal_A is not None:
            self.params.add('k_A', thermal_A, True, 0, 1, None)
            self.thermal = 'param'
    
    def optimizationSingle(self, params, time, data, i):
        """
        Optimization fucntion for lmfit of the function
        photo_reaction2_dif FOR A AND B
        """
        concentration = self.concentrations[i]
        if type(self.epsilons[i]) is not list:
            epsilons = self.epsilons
        else:
            epsilons = self.epsilons[i]
        irr_intensity = self.irradiation_intensities[i]
        residuos = np.empty(data.shape)
        a0, a1 = residuos.shape
        if self.thermal is 'param':
            thermal = params['k_A']
        else:
            thermal = self.thermal
        for i in range(a1):
            residuos[:, i] = data[:, i] - \
                             self.photoReactionDiff(concentration,
                                                    time,
                                                    params['Fi_A'].value,
                                                    params['Fi_B'].value,
                                                    epsilons,
                                                    irr_intensity,
                                                    thermal)[:, i]
        return residuos.flatten()
    
    def fitOptimizationSingle(self):
        times = [i[self.index:] for i in self.times]
        concentrations_profiles = [i[self.index:, :]
                                   for i in self.concentrations_profiles]
        self.result_single = []
        for i in range(len(times)):
            result = lmfit.minimize(self.optimizationSingle, self.params,
                                    args=(times[i], concentrations_profiles[i],
                                          i),
                                    nan_policy='propagate')
            self.result_single.append(result)
    
    def fitOptimizationGlobal(self):
        if len(self.times) > 1:
            times = [i[self.index:] for i in self.times]
            concentrations_profiles = [i[self.index:, :]
                                       for i in self.concentrations_profiles]
            self.result_Global = lmfit.minimize(self.optimizationGlobal,
                                                self.params,
                                                args=(times,
                                                      concentrations_profiles),
                                                nan_policy='propagate')
            return lmfit.fit_report(self.result_Global)
        else:
            self.fitOptimizationSingle()
            print('fitOptimizationSingle method has been run as there '
                  'is only one experiment loaded')
        
    def plotConcentrations(self):
        f, ax = plt.subplots(1, len(self.concentrations_profiles),
                             sharey=True, figsize=(11.5, 5))
        colors = ['b', 'k']
        experiment = ['On to Off', 'Off to On']
        for i, ax in enumerate(ax):
            color = colors[i]
            datos = ax.plot(self.times[i],
                            self.concentrations_profiles[i],
                            color=color,
                            label=str('raw data'))

            plt.setp(datos[1:], label="_")
            y_label = True if i == 0 else False
            self._format_figure(ax, experiment[i], y_label)
        f.tight_layout()
        return ax
        
    def plotFit(self, optimization='single'):
        # TODO simplify this function; although is working is to long
        msg = 'indicate the type of optimization single or global'
        assert (optimization == 'single' or optimization == 'global'), msg
        n = len(self.times)
        self.report = []
        colors = ['b', 'k']
        experiment = ['On to Off', 'Off to On']
        if n > 1:
            f, ax = plt.subplots(1, n, sharey=True, figsize=(11.5, 5))
            fits_export = []
            for i, ax in enumerate(ax):
                color = colors[i]
                data = self.concentrations_profiles[i]
                residuos = np.empty(self.concentrations_profiles[i].shape)
                residuos.flatten()
                concentration = self.concentrations[i]
                time = self.times[i]
                epsilons = self.epsilons[i]
                irr_intensity = self.irradiation_intensities[i]
                if optimization == 'single':
                    self.report.append(' ')
                    self.report.append(f'Fittin of single data set'
                                       f' number {i + 1}')
                    result = self.result_single[i]
                    for ii in lmfit.fit_report(result).split('\n'):
                        self.report.append(ii)
                else:
                    if i == 0:
                        self.report.append(' ')
                        msg = 'Fittin of all data sets globally'
                        self.report.append(msg)
                        result = self.result_Global
                        for ii in lmfit.fit_report(result).split('\n'):
                            self.report.append(ii)

                if self.thermal is 'param':
                    thermal = result.params['k_A']
                else:
                    thermal = self.thermal
                fit = self.photoReactionDiff(concentration,
                                             time[self.index:],
                                             result.params['Fi_A'].value,
                                             result.params['Fi_B'].value,
                                             epsilons,
                                             irr_intensity,
                                             thermal)
                datos = ax.plot(time, data, color=color,
                                label=str('raw data'))
                plt.setp(datos[1:], label="_")

                fites = ax.plot(time[self.index:],
                                fit, 'r', label=str('fit'))
                plt.setp(fites[1:], label="_")

                y_label = True if i == 0 else False
                self._format_figure(ax, experiment[i], y_label)
                fits_export.append(fites)
        else:
            f, ax = plt.subplots(1, n, sharey=True, figsize=(7, 5))
            color = colors[0]
            data = self.concentrations_profiles[0]
            residuos = np.empty(self.concentrations_profiles[0].shape)
            residuos.flatten()
            concentration = self.concentrations[0]
            time = self.times[0]
            epsilons = self.epsilons[0]
            irr_intensity = self.irradiation_intensities[0]
            if optimization == 'single':
                self.report.append(' ') 
                self.report.append(f'Fittin of single data set number 1') 
                result = self.result_single[0]
                for ii in lmfit.fit_report(result).split('\n'):
                    self.report.append(ii)
            if self.thermal is 'param':
                thermal = result.params['k_A']
            else:
                thermal = self.thermal
            fit = self.photoReactionDiff(concentration,
                                         time[self.index:],
                                         result.params['Fi_A'].value,
                                         result.params['Fi_B'].value,
                                         epsilons,
                                         irr_intensity,
                                         thermal)

            datos = ax.plot(time, data,
                            color=color,
                            label=str('raw data'))
            plt.setp(datos[1:], label="_")

            fites = ax.plot(time[self.index:], fit,
                            'r', label=str('fit'))
            plt.setp(fites[1:], label="_")

            self._format_figure(ax, experiment[0])
            fits_export = fites
        f.tight_layout()
        for i in self.report:
            print(i) 
        return fits_export

    @staticmethod
    def _format_figure(ax, title, add_y_label=True):
        ax.set_xlabel('Time (s)', size=14)
        ax.set_title(title, size=14)
        ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        ax.legend(loc=7, prop={'size': 14})
        ax.tick_params(axis='both', which='major', labelsize=14)
        if add_y_label:
            ax.set_ylabel('Concentration (M)', size=14)

    @staticmethod
    def get_report(params):
        report = []
        for ii in lmfit.fit_report(params).split('\n'):
            report.append(ii)
        for i in report:
            print(i)
        return report
