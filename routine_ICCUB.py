import os, sys
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from ICCUB_utils import disentangle, Temp_Correction, \
                        take_run_number_from_filename, take_date_from_filename, take_gain_from_yaml
from ROOT import TH1F, TCanvas, TF1

sys.path.insert(0, "/lustrehome/cerasole/DAQA/HERD/BeamTest2023/ICCUB/PSD_ADC_Z_conversion/")
from utils import get_combination_of_PSD_data

# Indipendentemente da tutto, i 4 SiPM di stesso tipo entro una barra sono sempre ordinati così
default_used_asics = [0, 1, 2, 3]
bars = [[5, 6, 7, 8], [1, 2, 3, 4]]
first_bar_channels = [[0, 4, 8, 12], [12, 8, 4, 0]]
readout_sequence = np.array([2, 1, 4, 3])

SiPM_types = ["High Z", "Low Z"]

default_filename = "/lustre/home/gamma/TestBeam2023/SPS_H4_oct_2023/PSD/ICCUB/Data//BETA_00501_MIXED_20231007_194836_DAQ_20231007_194952.csv"

class ICCUB_run (object):

    def __init__ (self, filename = None):

        self.filename = filename
        if self.filename is None:
            print ("Please set the <ICCUB_run name>.filename with the <ICCUB_run name>.from_file() method")
            self.filename = default_filename

        self.nevt = None
        self.nevt_physics = None
        self.data = None
        self.other = None
        self.sipm_channels = None
        self.used_asics = default_used_asics
        self.used_bars = set(np.array([bars[used_asic % 2] for used_asic in self.used_asics]).flatten())
        self.boolean_I2C = False
        self.subtracted_pedestal = False
        self.temp_correction_flag = None

        self.run_number = None
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

        data, temp_correction_flag = Temp_Correction(data, asic = [0, 1, 2, 3], compute_mask = True, compute_shift = True)
        self.temp_correction_flag = temp_correction_flag
        # Per il PSD, solo il run 560 ha avuto uno shift dell'ordine del centinaio (intorno a 240) in Temp sull'asic 3.
        # Con compute_shift = True e compute_mask = True, pongo la soglia a average(TEMP)+100 e correggo gli ADC
        # Se avessi messo a mano i valori di soglia di Temp e di correzione di ADC, sarebbero stati quelli in questo comando:
        #################data, temp_correction_flag = Temp_Correction(data, [3], [400], [242], compute_shift = False)

        ### Nessun run del PSD ha avuto shift di VFS
        ### (vedi in /lustrehome/cerasole/DAQA/HERD/BeamTest2023/ICCUB/checks/ i file
        ### check_VFS_TEMP_in_all_SPS_runs.py e SPS_VFS_TEMP/PSD/results_VFS_TEMP_check.txt)

        ### Correggi per temp e vfs shifts prima di disentanglare, perchè sono correzioni ASIC dependent
        def plot_singlechannel_VFS_temp (ch = 0, asics = [0, 1, 2, 3]):
            for asic in asics:
                fig, axs = plt.subplots(1, 3, figsize = (12, 6))
                mask = data['ASIC'] == asic
                adc_val = data.loc[mask, 'CH['+str(ch)+']']
                vfs_val = data.loc[mask, "VFS"]
                temp_val = data.loc[mask, "TEMP"]
                axs[0].hist(adc_val, bins = np.linspace(2048, 4096, 100))
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

        if np.average(data["#TRG_ID"]) > 1000000000000:
            self.boolean_I2C = True
        ordered_data, triggerID, nevt, sipm_channels = disentangle( \
            data, used_asics = self.used_asics, bars = bars, first_bar_channels = first_bar_channels, \
            readout_sequence = readout_sequence, boolean_I2C = self.boolean_I2C \
            )
        self.nevt = nevt
        self.data = {"VFS": [ np.array( data["VFS"][data["ASIC"] == asic] )[:nevt] for asic in range(4)], \
                     "Temp": [ np.array( data["TEMP"][data["ASIC"] == asic] )[:nevt] for asic in range(4)], \
                     "ADC": {SiPM_types[i]: {"Bar %d" % j :{"SiPM %d" % k : ordered_data[i][j-1][k-1] \
                            for k in range(1, 5)} for j in self.used_bars} for i in range(2)} }
        if self.boolean_I2C is True:
            things = ['I2C_subsystemID', 'I2C_triggertype', "I2C_ID"]
            self.other = { things[i_thing] : np.array(triggerID[i_thing]) for i_thing in range(len(triggerID)) }
        else:
            self.other = { 'trigger_ID' : np.array(triggerID) }
        self.sipm_channels = {SiPM_types[i]: {"Bar %d" % j :{"SiPM %d" % k : sipm_channels[i][j-1][k-1] \
                            for k in range(1, 5)} for j in self.used_bars} for i in range(2)}

        if self.boolean_I2C is True:
            self.nevt_physics = np.count_nonzero(self.other["I2C_triggertype"] != 0)
        else:
            self.nevt_physics = self.nevt

        self.run_number = take_run_number_from_filename (detector = "PSD", file = self.filename)
        self.date = take_date_from_filename (detector = "PSD", file = self.filename)
        self.gain = take_gain_from_yaml (detector = "PSD", file = self.filename, run_number = self.run_number)

        return


    def compute_pedestal(self, Fit = False, subtract_pedestal = False, \
                         save_plot_if_problem = True, cut_in_sigma = 10, cut_in_1sigma_events = 0.5, outdir = "./"):
        maskPed = self.other["I2C_triggertype"] == 0
        maskSpill = self.other["I2C_triggertype"] != 0
        meansPed = np.zeros((2, 8, 4))
        sigmaPed = np.zeros((2, 8, 4))
        for i_SiPM_type in range(len(SiPM_types)):
            SiPM_type = SiPM_types[i_SiPM_type]
            for i_bar in range(8):
                for i_sipm in range(4):
                    dataPed = self.data["ADC"][SiPM_type]["Bar %d" %(i_bar + 1)]["SiPM %d" %(i_sipm + 1)][maskPed]

                    if np.average(dataPed) > 2048:
                        xmin = 2048
                    else:
                        xmin = 0
                    mask = ((dataPed > xmin) * (dataPed < xmin + 1000))
                    mm = np.average(dataPed[mask])

                    mask = ((dataPed > mm - 100) * (dataPed < mm + 100))
                    ss = np.std(dataPed[mask])
                    #ss = np.std(dataPed[mask])

                    if Fit:
                        fgaus = TF1("fgaus", "gaus", mm - 4 * ss, mm + 4 * ss)
                        histo_name = "Pedestal ADC, "+SiPM_type+", Bar %d" %(i_bar + 1)+", SiPM %d" %(i_sipm + 1)
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
                                filename_pedestal_plot = outdir + "/Pedestal_fit_run_%d_SiPM_%s_%s_%s.png" %\
                                    (int(self.filename.split("/")[-1].split("_")[1]), SiPM_types[i_SiPM_type], \
                                    "Bar%d" % (i_bar + 1), "SiPM%d" % (i_sipm + 1) )
                                c0.SaveAs(filename_pedestal_plot)
                                input("Press enter to continue...")
                        h1.Delete()

                    meansPed[i_SiPM_type][i_bar][i_sipm] = mm
                    sigmaPed[i_SiPM_type][i_bar][i_sipm] = ss
                    if subtract_pedestal is True:
                        self.data["ADC"][SiPM_type]["Bar %d" %(i_bar + 1)]["SiPM %d" %(i_sipm + 1)] = \
                            self.data["ADC"][SiPM_type]["Bar %d" %(i_bar + 1)]["SiPM %d" %(i_sipm + 1)] - mm
                        self.subtracted_pedestal = True
        return meansPed, sigmaPed



    def plot_data (self, yscale = "log", show = True, figs = None, axss = None, density = True, select_pedestal = False):
        if figs is None:
            figs, axss = [], []
            same_plot = False
        else:
            counter = -1
            same_plot = True
        for i_SiPM_type in range(len(SiPM_types)):
            SiPM_type = SiPM_types[i_SiPM_type]
            for i_bar in range(8):
                if i_bar == 0 or i_bar == 4:
                    if same_plot is False:
                        fig, axs = plt.subplots(4, 4, sharex = True, sharey=True, figsize = (12, 9))
                        fig.subplots_adjust(wspace = 0, hspace = 0)
                    else:
                        counter += 1
                        fig, axs = figs[counter], axss[counter]
                for i_sipm in range(4):
                    if i_bar < 4:
                        i_x, i_y = i_bar, 3 - i_sipm
                        xlabel, ylabel = "SiPM %d" % (i_sipm + 1), "Bar %d" % (i_bar + 1)
                    else:
                        i_x, i_y = i_sipm, 3 - i_bar % 4
                        xlabel, ylabel = "Bar %d" % (i_bar + 1), "SiPM %d" % (i_sipm + 1)
                    d = self.data["ADC"][SiPM_type]["Bar %d" % (i_bar + 1)]["SiPM %d" % (i_sipm + 1)]
                    if self.boolean_I2C is True:
                        if select_pedestal is True:
                            mask = self.other["I2C_triggertype"] == 0
                            print ("Note: you're plotting only PEDESTAL events from this I2C run!")
                        elif select_pedestal is False:
                            mask = self.other["I2C_triggertype"] != 0
                            print ("Note: you're plotting only ON-SPILL events from this I2C run!")
                        d = d[mask]
                    if np.average(d) < 2048:
                        bins = np.linspace(0, 2047, 1024)
                    else:
                        bins = np.linspace(2048, 2048 * 2 - 1, 1024)
                    n, bins, patches = axs[i_x][i_y].hist(d, bins = bins, facecolor = "None", histtype = 'step', \
                                                              label = "Bar %d - SiPM %d" % (i_bar + 1, i_sipm + 1), density = density)
                    axs[i_x][i_y].grid(linestyle = "dotted")
                    if i_x == 3:
                        axs[i_x][i_y].set_xlabel(xlabel)
                    if i_y == 0:
                        axs[i_x][i_y].set_ylabel(ylabel)
                if i_bar == 3 or i_bar == 7:
                    axs[0][0].set_yscale(yscale)
                    fig.supxlabel("ADC counts")
                    fig.supylabel("Events (bin size = %.1f ADC)" % (bins[1] - bins[0]))
                    fig.suptitle("%s SiPMs" % SiPM_type)
                    if same_plot is False:
                        figs.append(fig)
                        axss.append(axs)
        if show is True:
            plt.show()
        return figs, axss


    def plot_together_calib_and_non_calib(self, yscale = "log", show = True, figs = None, axss = None, density = True):
        figs, axss = self.plot_data(yscale = "log", show = False, figs = figs, axss = axss, density = density, select_pedestal = False)
        figs, axss = self.plot_data(yscale = "log", show = True, figs = figs, axss = axss, density = density, select_pedestal = True)
        return


    def plot_vfs_or_temp_distribution(self, vfs_or_temp_string, show_plot = False):
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
        bins = np.arange(xmin, xmax, 50)
        fig, axs = plt.subplots(1, 4, figsize = (12, 6))
        for i in range(4):
            axs[i].hist(vfs_or_temp[i])
            axs[i].grid()
            if i == 0:
                 axs[i].set_ylabel("Events (out of %d)" % self.nevt)
            axs[i].set_xlabel("%s - ASIC %d" % (vfs_or_temp_string, i))
            axs[i].set_ylim(bottom = 0.5)
            axs[i].set_yscale("log")
        if show_plot == "True":
            plt.show()
        return fig, axs


    def check_vfs_equal_to_2047 (self):
        print ("Checking VFS...")
        vfs = self.data["VFS"]
        vfs_shift_flag = False
        for i in range(4):
            number_of_vfs_shifts = np.count_nonzero(vfs[i] != 2047)
            if number_of_vfs_shifts != 0:
                print ("For ASIC %d there have been %d counts with VFS different from 2047."\
                       % (i, number_of_vfs_shifts))
                vfs_shift_flag = True
        if vfs_shift_flag is False:
            print ("OK")
        return vfs_shift_flag, number_of_vfs_shifts


    def check_if_temp_is_constant (self, dispersion_threshold = 30):
        print ("Checking Temp...")
        temp = self.data["Temp"]
        temp_shift_flag = False
        dispersions, averages = np.zeros(4), np.zeros(4)
        for i in range(4):
            dispersion = int(np.max(temp[i]) - np.min(temp[i]))
            #dispersion = np.std(temp[i])
            average = int(np.average(temp[i]))
            print ("For ASIC %d the Temp values vary within %d around %d" % \
                   (i, dispersion, average))
            if dispersion > dispersion_threshold:
                print ("Warning, the ASIC %d had a shift in the Temp values" % (i))
                temp_shift_flag = True
            dispersions[i], averages[i] = dispersion, average
        if temp_shift_flag is False:
            print ("OK")
        return temp_shift_flag, dispersions, averages


    def plot_single_channel(self, SiPM_type, bar, SiPM, yscale = "log"):
        bar_string, sipm_string = "Bar %d" % bar, "SiPM %d" % SiPM
        values = self.data["ADC"][SiPM_type][bar_string][sipm_string]
        if np.average(values) > 2048:
            xmin = 2048
        else:
            xmin = 0
        xmax = xmin + 2048
        bins = np.linspace(xmin, xmax, 50)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(values, bins = bins)
        ax.set_xlabel("ADC values")
        ax.set_ylabel("Events out of %d" % self.nevt)
        ax.grid()
        ax.set_yscale(yscale)
        plt.show()
        return


    def convert_to_Z (self):
        SiPM_types = ["High Z", "Low Z"]
        ibars = range(1, 9)
        birks_dir = "/lustre/home/gamma/TestBeam2023/SPS_H4_oct_2023/Analysis/PSD_ADC_Z_conversion/runs/"
        path = os.path.join(birks_dir, "Run_%d" % (self.run_number), "Birks_results_PSD_%s_Z_SiPM_Bar_%d.csv")

        """
        if (self.boolean_I2C is True):
            paths = [ [path % (SiPM_type.split()[0], ibar) for ibar in ibars] for SiPM_type in SiPM_types]
            for SiPM_type in SiPM_types:
                for ibar in ibars:
                    if os.path.exists( path % (SiPM_type.split()[0], ibar) ) is False:
                        print ("No existing Z-conversion filenames for this run for SiPM %s of Bar %d, exiting the function." % (SiPM_type, ibar))
                        return
        else:
            print ("You chose a non-I2C file, exiting the function.")
            return
        """

        print (20*"#" + " CONVERSION TO Z! " + 20*"#")
        self.data["Z"], self.z_converted = {}, {}
        self.A_Birks, self.fh_Birks, self.kB_Birks, self.offset_Birks = {}, {}, {}, {}
        self.Birks_function = {}
        for SiPM_type in SiPM_types:
            self.A_Birks[SiPM_type], self.fh_Birks[SiPM_type], self.kB_Birks[SiPM_type], self.offset_Birks[SiPM_type] = {}, {}, {}, {}
            self.Birks_function[SiPM_type] = {}
            self.data["Z"][SiPM_type] = {}
            for ibar in ibars:
                try:
                    birks_filename = path % (SiPM_type.split()[0], ibar)
                    print ("Opening filename %s..." % birks_filename)
                    results = read_csv(birks_filename)
                    self.A_Birks[SiPM_type]["Bar %d" % ibar], self.fh_Birks[SiPM_type]["Bar %d" % ibar], \
                        self.kB_Birks[SiPM_type]["Bar %d" % ibar], self.offset_Birks[SiPM_type]["Bar %d" % ibar] = \
                            results["A"][0], results["fh"][0], results["kB"][0], results["offset"][0]
                    #self.z_converted[SiPM_type] = True
                    Z_Pb_square = 82.**2
                    extFunc0 = TF1("Extended_Birks_law_%s_Z_SiPM_Bar_%d" % (SiPM_type.split()[0], ibar), \
                        "( (2*[0]*(1-[1])*x) / (1 + 2*[2]*(1-[1])*x) ) + 2*[0]*[1]*x + [3]", \
                            0., Z_Pb_square)
                    extFunc0.SetParNames("A", "fh", "kB", "offset")
                    extFunc0.SetParameter(0, self.A_Birks[SiPM_type]["Bar %d" % ibar])
                    extFunc0.SetParameter(1, self.fh_Birks[SiPM_type]["Bar %d" % ibar])
                    extFunc0.SetParameter(2, self.kB_Birks[SiPM_type]["Bar %d" % ibar])
                    extFunc0.SetParameter(3, self.offset_Birks[SiPM_type]["Bar %d" % ibar])
                    self.Birks_function[SiPM_type]["Bar %d" % ibar] = extFunc0
                    self.data["Z"][SiPM_type]["Bar %d" % ibar] = np.zeros(self.nevt)

                    ### Che dannata combinazione prendere???
                    all_true_vector = np.array([True] * self.nevt)
                    if SiPM_type == "High Z":
                        type_of_combination = "Radice della media dei quadrati"
                        sipms = np.arange(1, 5)
                    else:
                        type_of_combination = "Media aritmetica"
                        sipms = [2, 3]
                        if (ibar == 1) * (self.run_number < 602):
                            sipms = [1, 4]
                        elif (ibar == 2) * (self.run_number >= 602):
                            sipms = [1, 4]
                        elif (ibar == 6) * (self.run_number < 602):
                            sipms = [3, 4]
                        elif (ibar == 6) * (self.run_number >= 602):
                            sipms = [1, 4]
                        elif (ibar >= 5) * (ibar <= 8) * (ibar != 6) * (self.run_number < 602):
                            sipms = [1, 2]     
                        else:
                            sipms = [2, 3]

                    combination_of_psd_adc_data = get_combination_of_PSD_data(
                        run = self, 
                        mask_physics = all_true_vector,
                        SiPM_type = SiPM_type,
                        bar = "Bar %d" % ibar, 
                        type_of_combination = type_of_combination, 
                        sipms = sipms
                    )

                    Z_Pb_square = 82.**2
                    extFunc0 = TF1("Extended_Birks_law","( (2*[0]*(1-[1])*x) / (1 + 2*[2]*(1-[1])*x) ) + 2*[0]*[1]*x + [3]", 0., Z_Pb_square)
                    extFunc0.SetParNames("A", "fh", "kB", "offset")
                    extFunc0.SetParameter(0, self.A_Birks[SiPM_type]["Bar %d" % ibar])
                    extFunc0.SetParameter(1, self.fh_Birks[SiPM_type]["Bar %d" % ibar])
                    extFunc0.SetParameter(2, self.kB_Birks[SiPM_type]["Bar %d" % ibar])
                    extFunc0.SetParameter(3, self.offset_Birks[SiPM_type]["Bar %d" % ibar])

                    self.data["Z"][SiPM_type]["Bar %d" % ibar] = np.zeros(self.nevt)
                    for ievt in range(self.nevt):
                        if combination_of_psd_adc_data[ievt] > 0:
                            z_value = np.sqrt(extFunc0.GetX(combination_of_psd_adc_data[ievt]))
                            if z_value > 50:
                                print ("Maggiore di 50 YEEEEEEEEEEEEEE")
                        else:
                            z_value = 0.
                        self.data["Z"][SiPM_type]["Bar %d" % ibar][ievt] = z_value

                    print ("Conversion to Z values of %s, Bar %d events went fine, results are in self.data['Z']['%s']['Bar %d']" % \
                        (SiPM_type, ibar, SiPM_type, ibar) )
                except:
                    self.data["Z"][SiPM_type]["Bar %d" % ibar] = -999. * np.ones(self.nevt)
                    print ("ERROR while converting data of SiPM %s, bar %d.\nPlease check." % (SiPM_type, ibar))

        return
