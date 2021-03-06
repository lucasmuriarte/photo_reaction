# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:24:57 2019

@author: lucas martinez uriarte
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter as SF
import lmfit


class FilterData:
    def __init__(self, data, wavelength):
        self.wavelength = wavelength
        self.data = data
        self.baseline_drift_done = False
        self.original_data = self.data.copy()

    @classmethod
    def load_cachan(cls, path, time, wavelength_range=(204.561, 684.921),
                    pixels=1024, sep='\t', decimal='.'):
        """
        Function to load the data, in specific manner. The data is obtain from
        a spectrometer in the University Paris-Saclay PPSM group

        Parameters
        ----------
        path: str (string path of the file)

        time: float
            time between the collection of two consecutive spectra.

        pixels: int
            number of pixels of the spectrometer. default 1024

        wavelength_range: tuple
            initial and final values of the wavelength detector range

        sep: str
            separator between columns of the file

        decimal: str
            decimal value used in the data
        """
        wavelength = np.linspace(wavelength_range[0],
                                 wavelength_range[1], pixels)
        name = ['Time']+[round(i, 3) for i in wavelength]
        data = pd.read_csv(path, sep=sep, decimal=decimal,
                           header=None, names=name)
        time = [(i-1)*time for i in data.index]
        data.Time = time
        return cls(data, wavelength)

    @classmethod
    def load_general(cls, path,  sep=',', decimal='.'):
        """
        Function for loading general data, time should be in the rows and
        wavelength the columns.

        Parameters
        ----------
        path: str (string path of the file)

        sep: str
            separator between columns of the file

        decimal: str
            decimal value used in the data

        """
        data = pd.read_csv(path, sep=sep, decimal=decimal)
        wavelength = np.array([float(i) for i in data.columns[1:]])
        return cls(data, wavelength)
    
    def cut_data_time(self, time: int, itself=True):
        """ cut the data in time range
         Parameters
        ----------
        time: int
            every time point bigger than this number will be deleted

        itself: bool
            if True will be done in the data it self False returns the new data
        """
        if itself:
            self.data = self.data[self.data['Time'] < time]
        else:
            return self.data[self.data['Time'] < time]
    
    def cut_data_wavelength(self, left=None, right=None, itself=False):
        """ cut the data in wavelength range

        Parameters
        ----------
        left: int
            lower wavelength values of this number will be deleted

        right: int
            higher wavelength values of this number will be deleted

        itself: bool
            if True will the data it self is cut if False returns
            the new data
        """
        rigth_index = pd.Series(self.wavelength-right).abs().sort_values()
        left_index = pd.Series(self.wavelength-left).abs().sort_values()
        rigth_index = rigth_index.index[0]
        left_index = left_index.index[0]

        cut_data = self.data.iloc[:, left_index+1:rigth_index+1]
        wavelength = self.wavelength[left_index:rigth_index]
        if itself:
            time = self.data.Time.values
            self.data = cut_data
            self.data.insert(loc=0, column='Time', value=time)
            self.wavelength = wavelength
        else:
            return cut_data, wavelength
        
    def baseline_drift(self, baseline_range):
        """
        correct the baseline drift

        Parameters
        ----------
        baseline_range: list or tuple of length 2
            first element: first value of the range;
            second element: the last value of the range

        e.g.: baseline_range=[680, 700]; the average value of the spectra
            between 680 and 700 nm is subtracted.

        the range should be an area were the baseline should be zero
        """
        self.baseline_drift_done = True
        rigth_index = pd.Series(self.wavelength-baseline_range[1]).abs()
        left_index = pd.Series(self.wavelength-baseline_range[0]).abs()
        rigth_index = rigth_index.sort_values().index[0]
        left_index = left_index.sort_values().index[0]
        for i in range(len(self.data)):
            value = self.data.iloc[i, left_index:rigth_index].mean()
            self.data.iloc[i, 1:] = self.data.iloc[i, 1:] - value
            print(i, 'out of', len(self.data), 'completed')
               
    def plotData(self, fsize=14, save=None):
        """
        Plot the data
        Parameters
        ----------
        fsize: default 14
            fotn size for axis and string in figure

        save: string
            string including the path were the figure should be save format as
            tiff. Default is None; which will not save the figure
        """
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.data)))
        cnorm = mpl.colors.Normalize(vmin=self.data.Time.iloc[0],
                                     vmax=self.data.Time.iloc[-1])
        cpickmap = mpl.cm.ScalarMappable(norm=cnorm, cmap=plt.cm.rainbow)
        cpickmap.set_array([])
        plt.figure()
        for ii in range(len(self.data)):
            plt.plot(self.wavelength, self.data.iloc[ii, 1:], color=colors[ii])
            plt.xlabel('Wavelenght (nm)', size=fsize)
            plt.ylabel('Absorbance', size=fsize)
        plt.xlim(self.wavelength[0], self.wavelength[-1])
        plt.minorticks_on()
        plt.tick_params(which='both', direction='in', bottom=True, top=True,
                        left=True, right=True, labelsize=fsize)
        plt.colorbar(cpickmap).set_label(label='Time (S)', size=15)
        if save is not None:
            plt.savefig((save+'.tiff'), bbox_inches='tight', dpi=300)
        plt.show()
    
    def differences(self, point: int):
        """
        calculate the differences between spectra at the given wavelength point
        for the entire spectra in the data set

        Parameters
        ----------
        point: int or float
            value were to calculate the differences
        """
        index = pd.Series(self.wavelength-point).abs().sort_values().index[0]
        self_diff = [0]+[abs(self.data.iloc[i, index]
                             - self.data.iloc[ii, index])
                         for i, ii in enumerate(range(1, len(self.data)))]
        return self_diff
    
    def filterSpike(self, wavelength: int, threshold: float, verbose=True):
        """
        filters data using the differences at a wavelength point calculated
        using differences method

        Parameters
        ----------
        wavelength: int or float
            value of wavelength use to filter

        threshold: float
                threshold value to delete spectra, if the difference between a
                spectrum and its previous at the given wavelength is higher
                than threshold the spectrum is deleted.

        verbose: bool default True
            If True, prints the number and percentage of deleted points
        """
        self.data['diff'] = self.differences(wavelength)
        size1 = len(self.data)
        self.data = (self.data[self.data['diff'] < threshold]).drop(['diff'],
                                                                    axis=1)
        cut = size1-len(self.data)
        if verbose:
            print(f'filtered {cut} points from {size1}, '
                  f'{round(cut/size1*100,1)}% of data')
    
    def plotOneWave(self, wavelength_point, plot_diff=False, fsize=14,
                    save=None):
        """
        Plot one trace of the data set at given wavelength

        Parameters
        ----------
        wavelength_point: int or float
            value of wavelength to be plotted

        fsize: int
            font size for axis and string in figure (default 14)

        plot_diff: bool;
           if True instead of the trace, the differences at this
           wavelength_point will be plotted.  (default False)

        save: string
            string including the path were the figure should be save format as
            tiff. Default is None; which will not save the figure
        """
        index = pd.Series(self.wavelength-wavelength_point).abs()
        index = index.sort_values().index[0]
        if plot_diff:
            diff = self.differences(wavelength_point)
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax[1].plot(self.data['Time'], self.data.iloc[:, index],
                       label=(str(wavelength_point) + ' nm'))
            ax[0].plot(self.data['Time'], diff, label='difference')
            ax[0].set_ylabel('difference', size=fsize)
            ax[1].set_ylabel('Absorbance', size=fsize)
            ax[1].legend()
            ax[0].legend()
        else:
            plt.figure()
            plt.plot(self.data['Time'], self.data.iloc[:, index],
                     label=(str(wavelength_point) + ' nm'))
            plt.legend()
            plt.ylabel('Absorbance', size=fsize)
        plt.minorticks_on()
        plt.tick_params(which='both', direction='in', bottom=True, top=True,
                        left=True, right=True, labelsize=fsize)

        plt.xlabel('Time (S)', size=fsize)
        plt.tight_layout()
        if save is not None:
            plt.savefig((save+'.tiff'), bbox_inches='tight', dpi=300)
        plt.show()
        
    def plotSeveralWaves(self, wavelength_points: list, fsize=14, save=None):
        """
        Plot several wavelengths of the data
        Parameters
        ----------
        wavelength_points: list
            values of wavelength to be plotted

        fsize: int
            font size for axis and string in figure (default 14)

        save: string
            string including the path were the figure should be save format as
            tiff. Default is None; which will not save the figure
        """
        fig, ax = plt.subplots(1)
        for i in wavelength_points:
            index=pd.Series(self.wavelength-i).abs().sort_values().index[0]
            ax.plot(self.data['Time'], self.data.iloc[:, index])
        plt.legend([str(i) + ' nm' for i in wavelength_points], loc='best')
        plt.xlabel('Time (S)', size=fsize)
        plt.ylabel('Absorbance', size=fsize)
        plt.minorticks_on()
        plt.tick_params(which='both', direction='in', bottom=True, top=True,
                        left=True, right=True, labelsize=fsize)
        plt.tight_layout()
        if save is not None:
            plt.savefig((save+'.tiff'), bbox_inches='tight',dpi=300)
        plt.show()
        
    def getSeveralWaves(self, wavelength_points: list):
        """
        return time trace of wavelengths of the data

        Parameters
        ----------
        wavelength_points:
            values of wavelength to be retrieved

        Returns
        -------
        Pandas data frame
        """
        index = [pd.Series(self.wavelength-i).abs().sort_values().index[0]
                 for i in wavelength_points]
        index = [0]+index
        return self.data.iloc[:, index]
        
    def plotFirstLast(self, mean=10, fsize=14, save=None):
        """
        Plot mean of the firsts and lasts spectra of the data

        Parameters
        ----------
        mean: int
            number of spectra to be average. (default 10)

        fsize: int
            font size for axis and string in figure (default 14)

        save: string
            string including the path were the figure should be save format as
            tiff. Default is None; which will not save the figure
        """
        plt.figure()
        plt.plot(self.wavelength, self.data.iloc[0:mean, 1:].mean(),
                 label='First spetrum')
        plt.plot(self.wavelength, self.data.iloc[-mean:-1, 1:].mean(),
                 label='Last spetrum')
        plt.xlabel('Wavelenght (nm)', size=fsize)
        plt.ylabel('Absorbance (A.U.)', size=fsize)
        plt.legend(loc='best')
        plt.minorticks_on()
        plt.tick_params(which='both', direction='in', bottom=True, top=True,
                        left=True, right=True, labelsize=fsize)
        if save is not None:
            plt.savefig((save+'.tiff'), bbox_inches='tight', dpi=300)
        plt.show()

    def plotSpec(self, spec: list, fsize=14, save=None):
        """
        Plot One spectrum from the dataset.

        Parameters
        ----------
        spec: list
            A list of int containing the index of the spectra to be plotted

        fsize: int
            font size for axis and string in figure (default 14)

        save: string
            string including the path were the figure should be save format as
            tiff. Default is None; which will not save the figure
        """
        if type(spec) == int or type(spec) == float:
            spec = [spec]
        plt.figure()
        for i in spec:
            plt.plot(self.wavelength, self.data.iloc[i, 1:],
                     label='First spetrum')
        plt.xlabel('Wavelenght (nm)', size=fsize)
        plt.ylabel('Absorbance (A.U.)', size=fsize)
        plt.legend(loc='best')
        plt.minorticks_on()
        plt.tick_params(which='both', direction='in', bottom=True, top=True,
                        left=True, right=True, labelsize=fsize)
        if save is not None:
            plt.savefig((save+'.tiff'), bbox_inches='tight', dpi=300)
        plt.show()    
    
    def getFirstLast(self, mean=10):
        """
        Return the mean of N spectra for the first and lasts entries of the data

        Parameters
        ----------
        mean:
            number of spectra to be average. int (default 10)
        """
        return [(self.wavelength, self.data.iloc[0:mean, 1:].mean()),
                (self.wavelength, self.data.iloc[-mean:-1, 1:].mean())]
    
    def smoothData(self, window_length, polyorder=2, itself=True):
        """
        smooth the data using savitky-golay filter

        Parameters
        ----------
        window_length: int (WARNING: must be un-even)
            number of fitting previous points to be used to predict the next

        polyorder: int
            order of the polynomial to fit data (default 2)

        itself: bool
            if True will the data it self is cut if False returns
            the new data
        """
        smooth_data = pd.DataFrame(np.zeros(self.data.shape),
                                   columns=self.data.columns)
        smooth_data.Time = self.data.Time.values

        for ii in range(len(self.data)):
            smooth_data.iloc[ii, 1:] = SF(self.data.iloc[ii, 1:],
                                          window_length=window_length,
                                          polyorder=polyorder)
        if itself:
            self.data = smooth_data
        else:
            return smooth_data
        
    def leastSquares(self, concentrations, mix_spec, spec_a, spec_b):
        """
        return the residues between the mixture of a spectrum, and the sum
        of and two other spectra spec_a, spec_b multiply by their
        concentrations.

        Parameters
        ----------
        concentrations: lmfit.Paramaeters()
            contain the concentration values of the spec_a and spec_b

        mix_spec: np.array
            xpectra that want to be decomposed

        spec_a: np.array
            Spectral shape of first pure spectrum

        spec_b: np.array
            Spectral shape of second pure spectrum
        """
        residues = mix_spec - (concentrations['Ca'] * spec_a +
                              (1 - concentrations['Ca']) * spec_b)
        return residues
    
    def pureSpectra(self, pure='initial', time=30, plot=True):
        msg = 'define if the pure spectra is at begining(initial) or the end' \
              ' of experiment(final)'
        assert (pure == 'initial' or pure == 'final'), msg
        if pure == 'initial':
            pure = (self.data[self.data.Time < time]).mean().drop('Time')
            mix = self.data[self.data.Time > (self.data.Time.iloc[-1]-time)]
            mix = mix.mean().drop('Time')
        else:
            pure = self.data[self.data.Time > (self.data.Time.iloc[-1]-time)]
            pure = pure.mean().drop('Time')
            mix = (self.data[self.data.Time < time]).mean().drop('Time')
        on = pure
        self.abs_pure_max = pure.max()
        contribution = mix[pure.idxmax()]/pure.max()
        off = mix-pure*contribution
        if plot:
            plt.figure()
            on.plot()
            off.plot()
            mix.plot()
            (on*contribution).plot()
            plt.xlabel('Wavelenght (nm)', size=14)
            plt.ylabel('Absorbance', size=14)
            plt.legend(['Pure On', 'Pure Off', 'Photosteady state',
                        'On contribution'])
            plt.show()
        return on, off
    
    def obtainConcentration(self, on_off=None, pure='initial'):
        conentrations = lmfit.Parameters()
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        conentrations.add_many(('Ca', 0.5, True, 0, 1, None))
        self.cons = pd.DataFrame(columns=['Time', 'Off', 'On'])
        self.cons_molar = pd.DataFrame(columns=['Time', 'Off', 'On'])
        if on_off is None:
            on, off = self.pureSpectra(pure=pure)
        else:
            self.abs_pure_max = on_off[0].max()
            on, off = on_off[0], on_off[1]
        for ii in range(len(self.data)):
            data = self.data.iloc[ii, 1:]
            result_fit = lmfit.minimize(self.leastSquares, conentrations,
                                        args=(data, off, on),
                                        nan_policy='propagate')

            c_a = [result_fit.params[key].value for key in
                   result_fit.params.keys()]

            c_b = [1-result_fit.params[key].value for key in
                   result_fit.params.keys()]

            self.cons.loc[ii] = [self.data.iloc[ii, 0]]+c_a+c_b
            self.cons_molar.loc[ii] = [self.data.iloc[ii, 0]]+c_a+c_b

    def plotConcentrations(self):
        self.cons.plot(x='Time', y=['Off', 'On'])
        plt.xlabel('Time (s)', size=14)
        plt.ylabel('Concentration (M)', size=14)
        plt.show()
            
    def transformToRealConcentration(self, epsilon_max_pure):
        self.cons[['Off', 'On']] = self.cons_molar[['Off', 'On']].apply((lambda x: x*self.abs_pure_max/epsilon_max_pure),
                                                                        axis=1)
        return self.cons[['Off', 'On']]
