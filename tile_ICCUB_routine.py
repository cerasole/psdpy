import os, sys
import numpy as np
from pandas import read_csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from ICCUB_utils import Temp_Correction, disentangle_tile, take_run_number_from_filename, take_date_from_filename, take_gain_from_yaml
from ROOT import TH1F, TCanvas, TF1

default_used_asics = [0, 1]
#default_filename = "/lustre/home/gamma/TestBeam2023/SPS_H4_oct_2023/TriggerTile/ICCUB/Data/BETA_00545_MIXED_20231009_215037_DAQ_20231009_215207.csv"
default_filename = "/lustre/home/gamma/TestBeam2023/SPS_H4_oct_2023/TriggerTile/ICCUB/Data/BETA_00548_MIXED_20231010_085422_DAQ_20231010_085550.csv"

SiPM_types = ["3x3", "1x1"]  ### 1x1 canale 25 e 3x3 canale 8


class ICCUB_tile_run (object):

    def __init__ (self, filename = None):

        self.filename = filename
        if self.filename is None:
            print ("Please set the <ICCUB_run name>.filename with the <ICCUB_run name>.from_file() method")
            self.filename = default_filename

        self.nevt = None
        self.nevt_physics = None
        self.data = None
        self.other = None
        self.used_asics = default_used_asics
        self.boolean_I2C = False
        self.subtracted_pedestal = False
        self.temp_correction_flag = None

        self.date = None
        self.gain = None

        self.z_converted = False
        self.A_Birks, self.fh_Birks, self.kB_Birks, self.offset_Birks = None, None, None, None
        self.Birks_function = None

        self.setup()
        return



    def setup (self):
        print ("Processing file %s" % self.filename)
        data = read_csv(self.filename, sep = "\t")

        data, temp_correction_flag = Temp_Correction(data, [0, 1], [400, 400], [269, 212])
        ############data, temp_correction_flag = Temp_Correction(data, asic = [0, 1], [400, 400], compute_shift = True)
        self.temp_correction_flag = temp_correction_flag

        ### Correggi per temp e vfs shifts prima di disentanglare, perchè sono correzioni ASIC dependent
        def plot_singlechannel_VFS_temp (ch = [7, 8], asics = [0, 1]):
            for asic in asics:
                fig, axs = plt.subplots(1, 3, figsize = (12, 6))
                fig.suptitle("ASIC %d - Ch %d" % (asic, ch[asic]))
                mask = data['ASIC'] == asic
                adc_val = data.loc[mask, 'CH['+str(ch[asic])+']']
                vfs_val = data.loc[mask, "VFS"]
                temp_val = data.loc[mask, "TEMP"]
                axs[0].hist(adc_val, bins = np.linspace(2048, 4096, 1025))
                axs[1].hist(vfs_val)
                axs[2].hist(temp_val)
                axs[0].set_ylabel("Events")
                axs[0].grid(), axs[1].grid(), axs[2].grid()
                axs[0].set_xlabel("ADC"), axs[1].set_xlabel("VFS"), axs[2].set_xlabel("Temp")
                axs[0].set_ylim(bottom = 0.5), axs[1].set_ylim(bottom = 0.5), axs[2].set_ylim(bottom = 0.5)
                axs[0].set_yscale("log"), axs[1].set_yscale("log"), axs[2].set_yscale("log")
            plt.show()
            return

        # for debugging temp and vfs shifts
        #plot_singlechannel_VFS_temp()
        #self.data = data
        #return

        # /lustre/home/gamma/TestBeam2023/SPS_H4_oct_2023/TriggerTile/ICCUB/Data/BETA_00546_MIXED_20231009_223307_DAQ_20231009_223438.csv
        # shift di temp su asic 0

        if np.average(data["#TRG_ID"]) > 1000000000000:
            self.boolean_I2C = True

        ordered_data, triggerID, nevt = disentangle_tile(data, used_asics = [0, 1], boolean_I2C = self.boolean_I2C)
        # I2C_subsystemID della tile è 11
        self.nevt = nevt
        self.data = {"VFS": [ np.array( data["VFS"][data["ASIC"] == asic] )[:nevt] for asic in range(2)], \
                     "Temp": [ np.array( data["TEMP"][data["ASIC"] == asic] )[:nevt] for asic in range(2)], \
                     "ADC": {SiPM_types[i]: ordered_data[i] for i in range(len(SiPM_types))} }
        if self.boolean_I2C is True:
            things = ['I2C_subsystemID', 'I2C_triggertype', "I2C_ID"]
            self.other = { things[i_thing] : np.array(triggerID[i_thing]) for i_thing in range(len(things)) }
        else:
            self.other = { "TriggerID" : np.array(triggerID) }

        if self.boolean_I2C is True:
            self.nevt_physics = np.count_nonzero(self.other["I2C_triggertype"] != 0)
        else:
            self.nevt_physics = self.nevt

        self.run_number = take_run_number_from_filename (detector = "TriggerTile", file = self.filename)
        self.date = take_date_from_filename (detector = "TriggerTile", file = self.filename)
        self.gain = take_gain_from_yaml (detector = "TriggerTile", file = self.filename)

        ##### Correzioni di shift negli ADC correlate shift di TEMP, per i run del BT all'SPS-H4 con numero <= 489, e numero 560
        if self.boolean_I2C == True:
            run_number = int(self.filename.split("/")[-1].split("_")[1])
            if run_number <= 489:
                temp = self.data["Temp"][0]
                adc = self.data["ADC"]["3x3"]
                po = [0.255, -60.8]
                def linear(x, a, b):
                    return a * (x) + b + 2192
                modified_adc = np.where(temp > 295, np.round(adc - linear(temp, *po) + 2192, 0), adc)
                self.data["ADC"]["3x3"] = modified_adc
                print ("This run is the number %d and it needs to be corrected further" % run_number)
                print ("The ASIC 0 events with TEMP above 295 have been corrected with a linear function a * (x) + b + 2192 with a, b = ", po)
            if run_number == 560:
                temp = self.data["Temp"][0]
                adc = self.data["ADC"]["3x3"]
                po = [-0.19966541, 49.95468697]
                def linear(x, a, b):
                    return a * (x) + b + 2390
                modified_adc = np.where(temp < 238, np.round(adc - linear(temp, *po) + 2390, 0), adc)
                self.data["ADC"]["3x3"] = modified_adc
                self.data["ADC"]["3x3"][-1] = self.data["ADC"]["3x3"][0]
                print ("This run is the number %d and it needs to be corrected further" % run_number)
                print ("The ASIC 0 events with TEMP below 238 have been corrected with a linear function a * (x) + b + 2390 with a, b = ", po)

        return


    def compute_pedestal (self, Fit = False, subtract_pedestal = False, \
                          save_plot_if_problem = False, cut_in_sigma = 4, cut_in_1sigma_events = 0.5, \
                          outdir = "./"):
        maskPed = self.other["I2C_triggertype"] == 0
        maskSpill = self.other["I2C_triggertype"] != 0
        meansPed = np.zeros(2)
        sigmaPed = np.zeros(2)
        for i_SiPM_type in range(len(SiPM_types)):
            SiPM_type = SiPM_types[i_SiPM_type]
            dataPed = self.data["ADC"][SiPM_type][maskPed]

            if np.average(dataPed) > 2048:
                xmin = 2048
            else:
                xmin = 0
            mask = ((dataPed > xmin) * (dataPed < xmin + 1000))
            mm = np.average(dataPed[mask])
            mask = ((dataPed > mm-100) * (dataPed < mm + 100))
            #ss = np.std(self.data["ADC"][SiPM_type][maskPed * mask])
            ss = np.std(dataPed[mask])

            if Fit:
                nn = "Pedestal Tile - %s - " % SiPM_types[i_SiPM_type]
                c0 = TCanvas(nn, nn, 0, 0, 800, 600)
                c0.Divide(1,1)
                c0.cd(1)
                fgaus = TF1("fgaus", "gaus", mm - 4 * ss, mm + 4 * ss)
                histo_name = "Pedestal ADC, " + SiPM_type
                print(histo_name)
                h1 = TH1F("h1", histo_name, 512, 2180., 4228.)
                for ii in range(len(dataPed)):
                     h1.Fill(dataPed[ii])
                h1.Fit("fgaus","R")
                mm = fgaus.GetParameter(1)
                ss = fgaus.GetParameter(2)
                if save_plot_if_problem == True:
                    cut1 = ss > cut_in_sigma
                    fraction_of_events_within_1_sigma = \
                        np.count_nonzero( (dataPed > mm-1*ss) * (dataPed < mm+1*ss) ) / len(dataPed)
                    cut2 = fraction_of_events_within_1_sigma < cut_in_1sigma_events
                    cut3 = len(dataPed) > 100
                    if (cut1 + cut2) * (cut3):
                        h1.GetXaxis().SetRangeUser(0.9*mm, 1.07*mm)
                        filename_pedestal_plot = outdir + "/Pedestal_fit_run_%d_SiPM_%s.png" %\
                            (int(self.filename.split("/")[-1].split("_")[1]), SiPM_types[i_SiPM_type])
                        c0.SaveAs(filename_pedestal_plot)
                        #input("Press enter to continue...")
                h1.Delete()

            meansPed[i_SiPM_type] = mm
            sigmaPed[i_SiPM_type] = ss
            if subtract_pedestal is True:
                self.data["ADC"][SiPM_type] = self.data["ADC"][SiPM_type] - mm
                self.subtracted_pedestal = True
        return meansPed, sigmaPed



    def plot_data(self, select_pedestal = False, show = True, figs = None, axss = None, rebin = 1., density = False):
        if figs is None:
            figs, axss = plt.subplots(1, 2, figsize = (12, 6))
        mask = self.other["I2C_triggertype"] != 0.

        if select_pedestal is True:
            mask = self.other["I2C_triggertype"] == 0
            string = "calibration"
        else:
            mask = self.other["I2C_triggertype"] != 0
            string = "physics"

        for i_SiPM_type in range(len(SiPM_types)):
            SiPM_type = SiPM_types[i_SiPM_type]
            values = self.data["ADC"][SiPM_type][mask]
            if np.average(values) > 2048:
                xmin = 2048
            else:
                xmin = 0
            xmax = xmin + 2048
            bins = np.linspace(xmin, xmax, int(1025/rebin))
            axss[i_SiPM_type].hist(values, bins = bins, histtype = "step", density = density)
            axss[i_SiPM_type].set_xlabel("ADC - %s" % SiPM_type)
            axss[i_SiPM_type].set_ylabel("Events (out of %d %s events)" % (len(values), string))
            axss[i_SiPM_type].set_yscale("log")
            axss[i_SiPM_type].set_ylim(bottom = 0.5)
            axss[i_SiPM_type].grid()
        if show is True:
            plt.show()
        return figs, axss


    def plot_single_channel(self, str = "3x3", show_plot = True):
        '''
        Function to plot the pedestal and signal distribution for the str = "3x3" or "1x1" SiPM channel on a unique plot
        '''
        fig, ax = plt.subplots(figsize = (8, 6))
        if self.boolean_I2C:
            mask_calib = self.other["I2C_ID"] == 0
            y_calib = self.data["ADC"][str][mask_calib]
            y_physics = self.data["ADC"][str][np.logical_not(mask_calib)]
            ax.hist(y_calib,   bins = np.arange( np.round(np.min(y_calib), 0) , np.round( np.max(y_calib), 0) + 1), \
                    color = "red", histtype = "step", label = "Calibration")
            ax.hist(y_physics, bins = np.arange(np.round(np.min(y_physics), 0), np.round(np.max(y_physics), 0) + 1), \
                    color = "black", histtype = "step", label = "Physics")
            plt.legend(loc = "best")
        else:
            y = self.data["ADC"][str]
            ax.hist(y, bins = np.arange(np.round(np.min(y), 0), np.round(np.max(y), 0) + 1), color = "black", histtype = "step")
        ax.set_xlabel("ADC - %s" % str)
        ax.set_ylabel("Number of events")
        ax.set_yscale("log")
        ax.set_ylim(bottom = 0.5)
        ax.grid()
        if show_plot:
            plt.show()
        return fig, ax


    def plot_vfs_or_temp_distribution(self, vfs_or_temp_string, show_plot = False, nasics = 2, savefig = False, outdir = "./", prefix = ""):
        '''
        Function to plot the vfs_or_temp_string = "VFS" or "TEMP" distributions
        '''
        try:
            vfs_or_temp = self.data[vfs_or_temp_string]
        except:
            print ("Given name non valid, please insert VFS or Temp as argument of this function")
            return
        if vfs_or_temp_string == "VFS":
            xmin, xmax = 1500, 2500
        elif vfs_or_temp_string == "Temp":
            xmin, xmax = 0, 1000
        fig, axs = plt.subplots(1, nasics, figsize = (12, 6))
        for i in range(nasics):
            axs[i].hist(vfs_or_temp[i], bins = np.arange(np.min(vfs_or_temp[i])-1, np.max(vfs_or_temp[i])+1))
            axs[i].grid()
            if i == 0:
                 axs[i].set_ylabel("Events (out of %d)" % self.nevt)
            axs[i].set_xlabel("%s - ASIC %d" % (vfs_or_temp_string, i))
            axs[i].set_ylim(bottom = 0.5)
            axs[i].set_yscale("log")
        if savefig:
            fig.savefig(outdir + prefix + "plot_%s_distribution.png" % vfs_or_temp_string)
        if show_plot:
            plt.show()
        return fig, axs



    def plot_VFS_temp_distributions (self):
        '''
        Function to plot both the vfs and temp distributions
        '''
        l = len(self.data["Temp"])    # 2 per un file ICCUB tile e 4 per un file ICCUB PSD
        fig, ax = plt.subplots(1, 2, figsize = (14, 6))
        vars = ["Temp", "VFS"]
        for j in range(len(vars)):
            for i in range(l):
                ax[j].hist(self.data[vars[j]][i], \
                           bins = np.arange( min(self.data[vars[j]][i])-3, max(self.data[vars[j]][i])+5, 1), \
                           label = "ASIC %d" % i, histtype = "step")
            ax[j].set_xlabel(vars[j])
            ax[j].set_ylabel("Number of events")
            ax[j].grid()
            ax[j].legend()
        plt.show()



    def check_vfs_equal_to_2047 (self, nasics = 2):
        print ("Checking VFS...")
        vfs = self.data["VFS"]
        vfs_shift_flag = [False] * nasics
        numbers_of_vfs_shifts = np.zeros(nasics)
        for i in range(nasics):
            numbers_of_vfs_shifts[i] = np.count_nonzero(vfs[i] != 2047)
            if numbers_of_vfs_shifts[i] != 0:
                print ("For ASIC %d there have been %d counts with VFS different from 2047." % \
                       (i, numbers_of_vfs_shifts[i]))
                vfs_shift_flag[i] = True
        if vfs_shift_flag[i] is False:
            print ("OK")
        return vfs_shift_flag, numbers_of_vfs_shifts



    def check_if_temp_is_constant (self, nasics = 2, dispersion_threshold = 30):
        print ("Checking Temp...")
        temp = self.data["Temp"]
        temp_shift_flag = [False] * nasics
        dispersions, averages = np.zeros(nasics), np.zeros(nasics)
        for i in range(nasics):
            average = int(np.average(temp[i]))
            ###########dispersion = np.std(temp[i])
            dispersion = int(np.max(temp[i]) - np.min(temp[i]))
            print ("For ASIC %d the Temp values vary within %d around %d" % \
                   (i, dispersion, average))
            if dispersion > dispersion_threshold:
                print ("Warning, the ASIC %d had a shift in the Temp values" % (i))
                temp_shift_flag[i] = True
            dispersions[i], averages[i] = dispersion, average
        if temp_shift_flag[i] is False:
            print ("OK")
        return temp_shift_flag, dispersions, averages


    def plot_Temp_evolution (self, str = "3x3", show_plot = True, savefig = False, outdir = "./", prefix = ""):
        print ("Plotting how the Temp variable of the %s channel evolves during the run..." % str)
        if str == "3x3":
            i = 0
        elif str == "1x1":
            i = 1
        else:
            print ("%s string non allowed, please specify whether 3x3 or 1x1.")
            return
        ###
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.plot(self.data["Temp"][i])
        ax.set_xlabel("Number of event during the run")
        ax.set_ylabel("Temp (%s SiPM)" % str)
        ax.grid()
        if savefig:
            fig.savefig(outdir + prefix + "plot_Temp_evolution.png")
        if show_plot:
            plt.show()
        return fig, ax


    def plot_Temp_and_ADC_evolution_in_calibration_events (self, str = "3x3", show_plot = True, savefig = False, outdir = "./", prefix = ""):
        print ("Plotting how the Temp variable and the ADC values of the %s channel evolves during the calibration events during the run..." % str)
        mask_calib = self.other["I2C_triggertype"] == 0
        fig, axs = plt.subplots(2, 1, figsize = (10, 6), sharex = True)
        if str == "3x3":
            i = 0
        elif str == "1x1":
            i = 1
        else:
            print ("%s string non allowed, please specify whether 3x3 or 1x1.")
            return
        ###
        axs[0].plot(self.data["Temp"][i][mask_calib])
        axs[0].set_xlabel("Number of calibration event during the run")
        axs[0].set_ylabel("Temp (%s SiPM)" % str)
        axs[0].grid()
        ###
        axs[1].plot(self.data["ADC"][str][mask_calib])
        axs[1].set_xlabel("Number of calibration event during the run")
        axs[1].set_ylabel("ADC (%s SiPM, calibration events)" % str)
        axs[1].grid()
        ###
        if savefig:
            fig.savefig(outdir + prefix + "plot_Temp_ADC_evolution_in_calibration_events.png")
        if show_plot:
            plt.show()
        return fig, axs


    def plot_correlation_Temp_and_ADC_in_calibration_events (self, str = "3x3", show_plot = True, savefig = False, outdir = "./", prefix = ""):
        print ("Plotting how the Temp variable and the ADC values of the %s channel correlate during the calibration events during the run..." % str)
        mask_calib = self.other["I2C_triggertype"] == 0
        fig, ax = plt.subplots(figsize = (8, 6))
        if str == "3x3":
            i = 0
        elif str == "1x1":
            i = 1
        else:
            print ("%s string non allowed, please specify whether 3x3 or 1x1.")
            return
        ###
        x, y = self.data["Temp"][i][mask_calib], self.data["ADC"][str][mask_calib]
        im = ax.hist2d(x, y, bins = (np.arange(min(x), max(x)), np.arange(min(y), max(y))), norm = mpl.colors.LogNorm())
        ax.set_xlabel("Temp (%s SiPM, calibration events)" % str)
        ax.set_ylabel("ADC (%s SiPM, calibration events)" % str)
        cbar = plt.colorbar(im[3])
        cbar.set_label('Number of events')
        if savefig:
            fig.savefig(outdir + prefix + "plot_Temp_ADC_%s_SiPM_correlation_in_calibration_events.png" % str)
        if show_plot:
            plt.show()
        return fig, ax


    def convert_to_Z (self):
        SiPM_types = ["3x3", "1x1"]
        birks_dir = "/lustrehome/cerasole/DAQA/HERD/BeamTest2023/ICCUB/ADC_Z_conversion/runs/"
        if (self.boolean_I2C is True):
            path = os.path.join(birks_dir, "Run_%d_%s" % (self.run_number, self.date.isoformat()), "Birks_results_TriggerTile_%s_SiPM.csv")
            paths = [path % SiPM_type for SiPM_type in SiPM_types]
            if os.path.exists(paths[0]) | os.path.exists(paths[1]):
                birks_filenames = paths
            else:
                print ("No existing Z-conversion filenames for this run, exiting the function.")
                return
        else:
            print ("You chose a non-I2C file, exiting the function.")
            return
        print (20*"#" + " CONVERSION TO Z! " + 20*"#")
        self.data["Z"], self.z_converted = {}, {}
        self.A_Birks, self.fh_Birks, self.kB_Birks, self.offset_Birks = {}, {}, {}, {}
        self.Birks_function = {}
        for SiPM_type in SiPM_types:
            try:
                birks_filename = path % SiPM_type
                print ("Opening filename %s..." % birks_filename)
                results = read_csv(birks_filename)
                self.A_Birks[SiPM_type], self.fh_Birks[SiPM_type], self.kB_Birks[SiPM_type], self.offset_Birks[SiPM_type] = \
                    results["A"][0], results["fh"][0], results["kB"][0], results["offset"][0]
                self.z_converted[SiPM_type] = True
                Z_Pb_square = 82.**2
                extFunc0 = TF1("Extended_Birks_law_%s_SiPM" % SiPM_type, \
                    "( (2*[0]*(1-[1])*x) / (1 + 2*[2]*(1-[1])*x) ) + 2*[0]*[1]*x + [3]", \
                    0., Z_Pb_square)
                extFunc0.SetParNames("A", "fh", "kB", "offset")
                extFunc0.SetParameter(0, self.A_Birks[SiPM_type])
                extFunc0.SetParameter(1, self.fh_Birks[SiPM_type])
                extFunc0.SetParameter(2, self.kB_Birks[SiPM_type])
                extFunc0.SetParameter(3, self.offset_Birks[SiPM_type])
                self.Birks_function[SiPM_type] = extFunc0
                self.data["Z"][SiPM_type] = np.zeros(self.nevt)
                for ievt in range(self.nevt):
                    z_value = np.sqrt(extFunc0.GetX(self.data["ADC"][SiPM_type][ievt]))
                    self.data["Z"][SiPM_type][ievt] = z_value
                print ("Conversion to Z values of %s events went fine, results are in self.data['Z']['%s']" % (SiPM_type, SiPM_type) )
            except:
                print ("ERROR while converting data of SiPM %s.\nPlease check." % SiPM_type)
        return

    '''
    def read_Birks_fit_results_file (self, birks_filename = None):
        if birks_filename is None:
            birks_dir = "/lustrehome/cerasole/DAQA/HERD/BeamTest2023/ICCUB/ADC_Z_conversion/runs/"
            if (boolean_I2C is True):
                path = os.path.join(birks_dir, "Run_%d_%s" % (self.run_number, self.date.isoformat()), "Birks_results_TriggerTile_%s_SiPM.csv")
                paths = [path % SiPM_type for SiPM_type in ["3x3", "1x1"]]
                if os.path.exists(paths[0]) | os.path.exists(paths[1])
                    birks_filenames = paths
            else:
                print ("No filename for Birks' law fit result provided and none found in the directory %s for this run" % birks_dir)
                return
        try:
            print ("Opening file %s..." % birks_filename)  # Birks_results_TriggerTile_1x1_SiPM.csv
            results = read_csv(birks_filename)
            SiPM_type = birks_filename.split("/")[-1].split("_")[3]
            self.A_Birks, self.fh_Birks, self.kB_Birks, self.offset_Birks = \
                results["A"][0], results["fh"][0], results["kB"][0], results["offset"][0]
            self.z_converted = True
            Z_Pb_square = 82.**2
            extFunc0 = TF1("Extended_Birks_law_%s_SiPM" % SiPM_type, \
                "( (2*[0]*(1-[1])*x) / (1 + 2*[2]*(1-[1])*x) ) + 2*[0]*[1]*x + [3]", \
                0., Z_Pb_square)
            extFunc0.SetParNames("A", "fh", "kB", "offset")
            extFunc0.SetParameter(0, self.A_Birks)
            extFunc0.SetParameter(1, self.fh_Birks)
            extFunc0.SetParameter(2, self.kB_Birks)
            extFunc0.SetParameter(3, self.offset_Birks)
            print (self.A_Birks, self.fh_Birks, self.kB_Birks, self.offset_Birks)
            self.data["Z"] = {}
            self.data["Z"][SiPM_type] = np.zeros(self.nevt)
            for ievt in range(self.nevt):
                z_value = np.sqrt(extFunc0.GetX(self.data["ADC"][SiPM_type][ievt]))
                self.data["Z"][SiPM_type][ievt] = z_value
            print ("Conversion to Z values went fine, results are in self.data['Z'][%s]" % SiPM_type)
        except:
            print ("Error while processing file %s.\nPlease check" % birks_filename)
        return
    '''

'''
    def plot_data_root(self, SiPM_type, select_pedestal = False):
        # SiPM_type can be "1x1" or "3x3"

        c0 = TCanvas("Canvas - %s" % SiPM_type, "Canvas - %s" % SiPM_type, 0, 0, 800, 600)
        c0.Divide(1, 1)
        c0.cd()

        if select_pedestal is True:
            mask = self.other["I2C_triggertype"] == 0
        else:
            mask = self.other["I2C_triggertype"] != 0
        values = self.data["ADC"][SiPM_type][mask]
        non_calib_nevts = len(values)
        if np.average(values) > 2048:
            xmin = 2048
        else:
            xmin = 0
        xmax = xmin + 2048

        fr0 = gPad.DrawFrame(xmin, 0.5, xmax, non_calib_nevts/2.)
        fr0.GetXaxis().SetTitle("ADC")
        fr0.GetYaxis().SetTitle("Events out of %d" % self.nevt)
        fr0.GetXaxis().CenterTitle()
        fr0.GetYaxis().CenterTitle()
        gPad.SetLogy()
        gPad.SetGrid()

        h0 = TH1F("Histogram - %s" % SiPM_type, "Histogram - %s" % SiPM_type, 1024, xmin, xmax)
        for value in values:
            h0.Fill(value)
        h0.Draw("same")
        input ("Press enter to continue...")
        return h0
'''
