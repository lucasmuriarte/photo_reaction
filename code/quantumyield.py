# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:34:25 2020

@author: 79344
"""
import numpy as np
import matplotlib.pyplot as plt
import lmfit
#import seaborn as sns
#import os
#from scipy.signal import savgol_filter as SF
from scipy.integrate import odeint
#import matplotlib as mpl
#import pandas as pd

# from prepocessPhotolysisClass import FilterData


#h = 6.626e-34
#na = 6.022e23
#c = 2.998e8
#
#def energy_of_single_photon(landa):
#    return (h*c)/(landa*1e-9)
#
#def number_of_photons(energy, landa):
#    "energy in mW"
#    energy /= 1000
#    return energy/energy_of_single_photon(landa)
#
#def concentration_of_photons(energy, landa, volume):
#    n_photons = number_of_photons(energy, landa)
#    mol_photons = n_photons/na
#    Molar_concentration = mol_photons/(volume/1000)
#    return Molar_concentration


class QuantumYields():    
    def __init__(self,
                 Concentrations_profiles, 
                 Eps_all, 
                 I_all,
                 thermal,
                 I_landa=[485,405],
                 volume=2,
                 initial_time=30,
                 optical_path=1):
        '''QuantumYields need irradiation intensities and absorption coeficients at which the sample was irradiated
        --> reaction is supose to go On to Off at 488 nm and Off to On at 405 nm
        
        --> epsilons_488=[epsilon_Off_488,epsilon_On_488]
        --> epsilons_405=[epsilon_Off_405,epsilon_On_405]
        --> Eps_all should be a list containing epsilons_488 and epsilons_405 in the order of which concentration profiles of experiments are load '''
        self.index=int(np.mean([(i.Time-initial_time).abs().sort_values().index[0] for i in Concentrations_profiles]))+1
        self.concentrations_profiles=[i.iloc[:,1:].values for i in Concentrations_profiles]
        self.times=[i.iloc[:,0].values for i in Concentrations_profiles]
        
        self.epsilons=Eps_all
        self.irradiation_intensities=[I_all[i]*I_landa[i]*1e-12/(6.626e-34*2.998e8*6.022e23)/(volume*1e-3) for i in range(len(I_all))]
        self.thermal=1/(3600*thermal)
        self.concentrations=[(i[i.Time<initial_time].iloc[:,1:]).mean().values for i in Concentrations_profiles]
        self.optical_path=optical_path
        
    #PHOTOCHEMICAL REACTION OF DISAPER AND FORMATION OF TWO PRODUCTS
    def photoReaction(self,C,time,Fi_A,Fi_B,Eps,I,K_A):
        """photoreaction of formation of a product B irradiating A where A and B absorb at iradiation wavelenght"""
        ca,cb=C[0],C[1]
        ea,eb=Eps[0],Eps[1]
        Inten_A=((ea*ca)/(ea*ca+eb*cb))*I*(1-10**(-(ea*ca+eb*cb)*self.optical_path))
        Inten_B=((eb*cb)/(ea*ca+eb*cb))*I*(1-10**(-(ea*ca+eb*cb)*self.optical_path))
        dBdt=-Fi_B*Inten_B+Fi_A*Inten_A+K_A*ca
        dAdt=-dBdt
        return [dAdt,dBdt]
    
    def photoReactionDiff(self,C,time,Fi_A,Fi_B,Eps,I,K_A):
        """Integration of the function photo_reaction1_formation"""
        return odeint(self.photoReaction,C,time,args=(Fi_A,Fi_B,Eps,I,K_A))
    
    def optimizationGlobal(self,params, time_all, data_all):
        """Optimization returning residues to be minized for function photoReactionDiff, For several experiments.
        Arguments needed ,(except params) are parameters for photoReactionDiff as list 
        of element, where each element is data for one experiment. 
        It returns the total residues for all the  experimets at same time to global fit data and
        optain global quantum yields. Prams should be a parmas objet for lmfit library"""
        residuos_all=[]
        for i in range(len(data_all)):
            residuos=np.empty(data_all[i].shape)
            residuos.flatten()
            time=time_all[i]
            C=self.concentrations[i]
            Eps=self.epsilons[i]
            I=self.irradiation_intensities[i]
            if self.thermal is 'param':
                thermal=params['k_A']
            else:
                thermal=self.thermal
            fit=self.photoReactionDiff(C,time,params['Fi_A'].value,params['Fi_B'].value,Eps,I,thermal)
            residuos=data_all[i].flatten()-fit.flatten()
            residuos_all.append(residuos.flatten())
        return np.array([item for sublist in residuos_all for item in sublist])
            
    def paramsInitialization(self,Fi_A=0.5,Fi_B=0.5,thermal_A=None,thermal_B=None):
        self.params = lmfit.Parameters()
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        self.params.add_many(('Fi_A', Fi_A, True, 0, 1, None),
                             ('Fi_B', Fi_B, True, 0, 1, None))
        
        if thermal_A is not None:
            self.params.add('k_A', thermal_A, True, 0, 1, None)
            self.thermal='param'
    
    def optimizationSingle(self,params, time, data,i):
        """Optimization fucntion for lmfit of the function photo_reaction2_dif FOR A AND B"""
        C=self.concentrations[i]
        if type(self.epsilons[i]) is not list:
            Eps=self.epsilons
        else:
            Eps=self.epsilons[i]
        I=self.irradiation_intensities[i]
        residuos=np.empty(data.shape)
        a0,a1=residuos.shape
        if self.thermal is 'param':
            thermal=params['k_A']
        else:
            thermal=self.thermal
        for i in range(a1):
            residuos[:,i]=data[:,i]-self.photoReactionDiff(C,time,params['Fi_A'].value,params['Fi_B'].value,Eps,I,thermal)[:,i]
        return residuos.flatten()
    
    def fitOptimizationSingle(self):
        times=[i[self.index:] for i in self.times]
        concentrations_profiles=[i[self.index:,:] for i in self.concentrations_profiles]
        self.result_single=[]
        for i in range(len(times)):
            result=lmfit.minimize(self.optimizationSingle, self.params, args=(times[i], concentrations_profiles[i],i),nan_policy='propagate')
            self.result_single.append(result)
    
    def fitOptimizationGlobal(self):
        if len(self.times)>1:
            times=[i[self.index:] for i in self.times]
            concentrations_profiles=[i[self.index:,:] for i in self.concentrations_profiles]
            self.result_Global=lmfit.minimize(self.optimizationGlobal, self.params, args=(times, concentrations_profiles),nan_policy='propagate')
            return lmfit.fit_report(self.result_Global)
        else:
            self.fitOptimizationSingle()
            print('fitOptimizationSingle method has been run as there is only one experiment loaded')
        
    def plotConcentrations(self):
        f, plot = plt.subplots(1, len(self.concentrations_profiles), sharey=True,figsize=(11.5,5))
        colors=['b','k']
        experiment=['On to Off','Off to On']
        for i,plot in enumerate(plot):
                color=colors[i]
                datos=plot.plot(self.times[i],self.concentrations_profiles[i],color=color,label=str('raw data'))
                plt.setp(datos[1:], label="_")
                plot.set_xlabel('Time (s)',size=14)
                plot.set_title(experiment[i],size=14)
                plot.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
                plot.legend(loc=7,prop={'size':14})
                plot.tick_params(axis = 'both', which = 'major', labelsize = 14)
                if i==0:
                    plot.set_ylabel('Concentration (M)',size=14)
        f.tight_layout()
        return plot
        
    def plotFit(self,optimization='single'):
        assert (optimization=='single' or optimization=='global'), 'indicate the type of optimization single or global'
        n=len(self.times)
        self.report=[]
        colors=['b','k']
        experiment=['On to Off','Off to On']
        if n >1:
            f, plot = plt.subplots(1, n, sharey=True,figsize=(11.5,5))
            fits_export = []
            for i,plot in enumerate(plot):
                    color=colors[i]
                    data=self.concentrations_profiles[i]
                    residuos=np.empty(self.concentrations_profiles[i].shape)
                    residuos.flatten()
                    C=self.concentrations[i]
                    time=self.times[i]
                    Eps=self.epsilons[i]
                    I=self.irradiation_intensities[i]
                    if optimization=='single':
                        self.report.append(' ') 
                        self.report.append(f'Fittin of single data set number {i+1}') 
                        result=self.result_single[i]
                        for ii in lmfit.fit_report(result).split('\n'):
                                self.report.append(ii) 
                    else:
                        if i == 0:
                            self.report.append(' ')
                            self.report.append(f'Fittin of all data sets globally')
                            result=self.result_Global
                            for ii in lmfit.fit_report(result).split('\n'):
                                self.report.append(ii) 
                    if self.thermal is 'param':
                        thermal=result.params['k_A']
                    else:
                        thermal=self.thermal
                    fit=self.photoReactionDiff(C,time[self.index:],result.params['Fi_A'].value,result.params['Fi_B'].value,Eps,I,thermal)
                    datos=plot.plot(time,data,color=color,label=str('raw data'))
                    plt.setp(datos[1:], label="_")
                    fites=plot.plot(time[self.index:],fit,'r',label=str('fit'))
                    plt.setp(fites[1:], label="_")
                    plot.set_xlabel('Time (s)',size=14)
                    plot.set_title(experiment[i],size=14)
                    plot.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
                    plot.legend(loc=7,prop={'size':14})
                    plot.tick_params(axis = 'both', which = 'major', labelsize = 14)
                    if i==0:
                        plot.set_ylabel('Concentration (M)',size=14)
                    fits_export.append(fites)
        else:
            f, plot = plt.subplots(1, n, sharey=True,figsize=(7,5))
            color=colors[0]
            data=self.concentrations_profiles[0]
            residuos=np.empty(self.concentrations_profiles[0].shape)
            residuos.flatten()
            C=self.concentrations[0]
            time=self.times[0]
            Eps=self.epsilons[0]
            I=self.irradiation_intensities[0]
            if optimization=='single':
                self.report.append(' ') 
                self.report.append(f'Fittin of single data set number 1') 
                result=self.result_single[0]
                for ii in lmfit.fit_report(result).split('\n'):
                        self.report.append(ii) 
            else:
                if i == 0:
                    self.report.append(' ')
                    self.report.append(f'Fittin of all data sets globally')
                    result=self.result_Global
                    for ii in lmfit.fit_report(result).split('\n'):
                        self.report.append(ii) 
            if self.thermal is 'param':
                thermal=result.params['k_A']
            else:
                thermal=self.thermal
            fit=self.photoReactionDiff(C,time[self.index:],result.params['Fi_A'].value,result.params['Fi_B'].value,Eps,I,thermal)
            datos=plot.plot(time,data,color=color,label=str('raw data'))
            plt.setp(datos[1:], label="_")
            fites=plot.plot(time[self.index:],fit,'r',label=str('fit'))
            plt.setp(fites[1:], label="_")
            plot.set_xlabel('Time (s)',size=14)
            plot.set_title(experiment[0],size=14)
            plot.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
            plot.legend(loc=7,prop={'size':14})
            plot.tick_params(axis = 'both', which = 'major', labelsize = 14)
            plot.set_ylabel('Concentration (M)',size=14)
            fits_export = fites
        f.tight_layout()
        for i in self.report:
            print(i) 
        return fits_export
    
        def report(self,params):
            report=[]
            for ii in lmfit.fit_report(params).split('\n'):
                            report.append(ii) 
            for i in report:
                print(i) 
            return report
    
#    def photo_reaction1_disapear(self,C,time,Fi_B,Eps,I,K_A):
#        """photoreaction of desapear of a product B irradiating B, which form A, and A come back to B with thermal Ka"""
#        cb=C
#        ca=C-cb
#        ea,eb=Eps[0],Eps[1]
#        Inten_B=((eb*cb)/(ea*ca+eb*cb))*I*(1-10**(-(ea*ca+eb*cb)))
#        dAdt=Fi_B*Inten_B-K_A*ca
#        dBdt=-dAdt
#        return dBdt
#    
#    def photo_reaction1_dif_disapear(self,C,time,Fi_B,Eps,I,K_A):
#        """Integration of the function photo_reaction1_disapear"""
#        return odeint(self.photo_reaction1_disapear,C,time,args=(Fi_B,Eps,I,K_A))
#    
#    def optimizationDisapear(self,params, time, data,i):
#        """Optimization fucntion for lmfit of the function photo_reaction1_dif_disapear"""
#        if type(self.epsilons[i]) is not list:
#            Eps=self.epsilons
#        else:
#            Eps=quantums.epsilons[i]
#        I=quantums.irradiation_intensities[i]
#        thermal=params['k_A']
#        C=quantums.concentrations[i]
#        residuos=data-self.photo_reaction1_dif_disapear(C,time,params['Fi_B'].value,Eps,I,thermal)
#        return residuos
#
#    def fitOptimizationDisapear(self,i):
#        times=self.times[i]
#        concentrations_profiles= self.concentrations_profiles[i][quantums.index:,1]
#        params=lmfit.Parameters()
#        params.add_many(('Fi_B', 0.005, True, 0, 1, None, None),
#                        ('k_A', self.thermal, True, 0, 1, None, None))
#        self.resultDissapear=lmfit.minimize(self.optimizationDisapear, params, args=(times[i], concentrations_profiles[i],i),nan_policy='propagate')
#    
#    def report(self,params):
#        report=[]
#        for ii in lmfit.fit_report(params).split('\n'):
#                        report.append(ii) 
#        for i in report:
#            print(i) 
#        return report
#    
#    def plotDisapear(self):
#        result=quantums.resultDissapear
#        plot,ax = plt.subplots(1, 1, sharey=True,figsize=(7,5))
#        data=quantums.concentrations_profiles[0]
#        residuos=np.empty(quantums.concentrations_profiles[0].shape)
#        residuos.flatten()
#        C=quantums.concentrations[0][1]
#        time=quantums.times[0]
#        Eps=quantums.epsilons[0]
#        thermal=result.params['k_A'].value
#        I=quantums.irradiation_intensities[0]
##        quantums.photo_reaction1_disapear(C,time[quantums.index:],result.params['Fi_B'].value,Eps,I,thermal.value)
#        fit=quantums.photo_reaction1_dif_disapear(C,time[quantums.index:],result.params['Fi_B'].value,Eps,I,thermal)
#        datos=ax.plot(time,data,color='b',label=str('raw data'))
#        plt.setp(datos[1:], label="_")
#        fites=ax.plot(time[quantums.index:],fit,'r',label=str('fit'))
#        plt.setp(fites[1:], label="_")
#        ax.set_xlabel('Time (s)',size=14)
#        #plot.set_title(experiment[0],size=14)
#        ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
#        ax.legend(loc=7,prop={'size':14})
#        ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
#        ax.set_ylabel('Concentration (M)',size=14)
        


# I_On=2.62
# I_Off=0.180

# epsilon_On_488=1690
# epsilon_Off_405=787
# epsilon_On_405=0
# epsilon_Off_488=0
# epsilons_488=[epsilon_Off_488,epsilon_On_488]
# epsilons_405=[epsilon_Off_405,epsilon_On_405]
# Eps_all=[epsilons_488,epsilons_405]



# #leucine
# concentration=[cons_cons] 
# Intensities=[I_On]
# leu=QuantumYields(concentration,Eps_all,2.5,Intensities)
# leu.paramsInitialization(thermal_A=leu.thermal)
# result=leu.fitOptimizationGlobal()
# leu.fitOptimizationSingle()
# leu.plotFit()
# leu.plotFit('global')
# [print(leu.result_All.params[i]) for i in  leu.result_All.params.keys()]


# #WT
# epsilon_On_488=1690
# epsilon_Off_405=0
# epsilon_On_405=0
# epsilon_Off_488=0
# epsilons_488=[epsilon_Off_488,epsilon_On_488]
# epsilons_405=[epsilon_Off_405,epsilon_On_405]
# Eps_all=[epsilons_488,epsilons_405]

# cons_On=cons_On[cons_On.Time<400]
# concentration=[cons_On] 
# Intensities=[I_On]
# WT=QuantumYields(concentration,Eps_all,2,Intensities,initial_time=38)
# WT.paramsInitialization(thermal_A=WT.thermal)
# result=WT.fitOptimizationGlobal()
# WT.fitOptimizationSingle()
# WT.plotFit()

# #ala
# epsilon_On_488=65519
# epsilon_Off_405=24640
# epsilon_On_405=6116
# epsilon_Off_488=0
# epsilons_488=[epsilon_Off_488,epsilon_On_488]
# epsilons_405=[epsilon_Off_405,epsilon_On_405]
# Eps_all=[epsilons_488,epsilons_405]

# concentration=[cons_cons] 
# Intensities=[I_On]
# ala=QuantumYields(concentration,Eps_all,48,Intensities)
# ala.paramsInitialization(thermal_A=ala.thermal)
# result=ala.fitOptimizationGlobal()
# ala.fitOptimizationSingle()
# ala.plotFit()

# #206N
# I_On=0.91

# epsilon_On_488=67857.67
# epsilon_Off_405=26262
# epsilon_On_405=7711
# epsilon_Off_488=0
# epsilons_488=[epsilon_Off_488,epsilon_On_488]
# epsilons_405=[epsilon_Off_405,epsilon_On_405]
# Eps_all=[epsilons_488,epsilons_405]

# concentration=[cons_cons] 
# Intensities=[I_On]
# S206N=QuantumYields(concentration,Eps_all,3,Intensities)
# S206N.paramsInitialization(thermal_A=S206N.thermal)
# result=S206N.fitOptimizationGlobal()
# S206N.fitOptimizationSingle()
# S206N.plotFit()

# #S206N_A
# epsilon_On_488=55309
# epsilon_Off_405=24273
# epsilon_On_405=10097
# epsilon_Off_488=0
# epsilons_488=[epsilon_Off_488,epsilon_On_488]
# epsilons_405=[epsilon_Off_405,epsilon_On_405]
# Eps_all=[epsilons_488,epsilons_405]

# concentration=[cons_cons] 
# Intensities=[I_On]
# S206N_A=QuantumYields(concentration,Eps_all,40,Intensities)
# S206N_A.paramsInitialization(thermal_A=S206N_A.thermal)
# result=S206N_A.fitOptimizationGlobal()
# S206N_A.fitOptimizationSingle()
# S206N_A.plotFit()


# Off_ala=FilterData(file_Off,(204.561,684.921))
# Off_ala.cutData(320,620,itself=True)
# #a.cutDataTime(1000,itself=True)
# Off_ala.baselineDrift((580,630))
# Off_ala.filterSpike(480,0.005)
# Off_ala.filterSpike(410,0.005)
# Off_ala.smoothData(49)
# Off_ala.obtainConcentration(pure='final')
# Off_ala.transformToRealConcentration(epsilon_max_pure=55309)
# Off_ala.plotData()
# Off_ala.plotConcentrations()
# Off_ala.data.to_csv('//pclasdocnew.univ-lille.fr/Doc/79344/Documents/donnes/Photolysis cachan/january 2020/14.01.2020/S206N V151A/S206N V151A Off to On threated.csv')


# On_ala=FilterData(file_On,(204.561,684.921))
# On_ala.cutData(320,620,itself=True)
# #a.cutDataTime(1000,itself=True)
# On_ala.baselineDrift((580,630))
# On_ala.filterSpike(480,0.005)
# On_ala.filterSpike(405,0.005)
# On_ala.smoothData(49)
# On_ala.obtainConcentration(pure='initial')
# On_ala.transformToRealConcentration(epsilon_max_pure=55309)
# On_ala.plotConcentrations()
# On_ala.plotData()
# On_ala.data.to_csv('//pclasdocnew.univ-lille.fr/Doc/79344/Documents/donnes/Photolysis cachan/january 2020/14.01.2020/S206N V151A/S206N V151A On to Off threated.csv')


