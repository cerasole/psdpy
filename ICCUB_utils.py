import numpy as np
import datetime
import yaml

psd_nasics, tile_nasics = 4, 2
str_tile = "asic 0: 3x3, ch8 ; asic 1: 1x1, ch25"
str_psd = "asic 0: High-Z barre 5-6-7-8; \
asic 1: High-Z barre 1-2-3-4; \
asic 2: Low-Z barre 5-6-7-8; \
asic 3: Low-Z barre 1-2-3-4"
detectors_dict = { "TriggerTile":{"nasics":tile_nasics, "info": str_tile}, \
                           "PSD":{"nasics":psd_nasics , "info": str_psd} , }

def disentangle (data, used_asics, bars, first_bar_channels, readout_sequence, boolean_I2C, boolean_channels = False):
    '''
    ordered_data = vettore di 2 elementi:
    - elemento 0 per HIGH Z SiPMs
    - elemento 1 per LOW Z SiPMs
        ordered_data[0] = vettore di 8 elementi
        - elemento 0 per HIGH Z SiPMs della barra 1
        - elemento 1 per HIGH Z SiPMs della barra 2 ...
        ordered_data[0][0] = vettore di 4 elementi
            - elemento 0 per HIGH Z SiPM 1 della barra 1
            - elemento 1 per HIGH Z SiPM 2 della barra 1
                ordered_data[0][0][0] = numpy array con tutti i canali ADC acquisiti nel run dall'HIGH Z SiPM 1 della barra 1.
    '''
    ### Suddivisione dati in base ad ASIC, barra e SiPM

    ordered_data = [[], []]       ### [HIGH Z, LOW Z]
    ordered_data[0] = [[]] * 8    ### [    8 BARRE HIGH Z DALLA 1 ALLA 8   ,     8 BARRE LOW Z DALLA 1 ALLA 8    ]
    ordered_data[1] = [[]] * 8
    ### IMPORTANTE: SE DEFINISCI COSì UNA LISTA, CON [[]] * 8, E FAI L'APPEND A ordered_data[0][0], APPENDERà
    ### CIò CHE STAI APPENDENDO A TUTTI GLI 8 ordered_data[0][i]
    ### SE INVECE FAI UN'ASSEGNAZIONE NUOVA, COME ordered_data[asic_id // 2][bar - 1] = a,
    ### SOLO L'ELEMENTO COINVOLTO, DI INDICE BAR-1, è MODIFICATO, MENTRE GLI ALTRI NO.

    sipm_channels = [[], []]
    sipm_channels[0] = [[]] * 8
    sipm_channels[1] = [[]] * 8

    print ("Collecting data from the asics", used_asics)

    #used_asics = set(np.array(data["ASIC"]))

    # This part of the code deals with the fact that in one acquisition there can be different number of events recorded by the 4 ASICS due to
    # closure of the acquisition during the buffering (es: asics 0 and 1 recorded 4900 evts, while 2 recorded 4982 and 4976).
    # 1 - I take the minimum among the numbers of events taken by the 4 asics, call it min_nevt_asics
    # 2 - I check that the TRIGGER_IDs of the first min_nevt_asics for the 4 asics are the same and, if not, print ERROR
    # 3 - In the real disentangling part, I take only the first min_nevt_asics events
    nevt_asics = [len(data["#TRG_ID"][data["ASIC"] == asic_id]) for asic_id in used_asics]
    min_nevt_asics = np.min(nevt_asics)
    max_nevt_asics = np.max(nevt_asics)
    triggerID = np.array(data[data["ASIC"] == 0]["#TRG_ID"][:min_nevt_asics])

    for asic_id in range(4):
        comparison = np.array(data[data["ASIC"] == asic_id]["#TRG_ID"][:min_nevt_asics])
        number_of_mismatches = np.count_nonzero([triggerID == comparison] == False)
        if number_of_mismatches != 0:
             print ("ERROR")
    if boolean_I2C is True:
        triggerID = decode_trig_id(triggerID)

    nevt = min_nevt_asics

    ######## Correzione dello shift di +1 fra la colonna dei triggerID e la colonna degli ADC
    triggerID = correzione_shift_triggerID_adc(triggerID)

    for asic_id in used_asics:
        # Selezione sull'ASIC
        asic_mask = data["ASIC"] == asic_id
        asic_data = data[asic_mask]
        first_bar_channel = first_bar_channels[asic_id % 2]
        barss = bars[asic_id % 2]
        for bar in barss:
            # Selezione sulla barra
            a = [np.array(asic_data["CH[%d]" % channel_id])[:nevt] \
                for channel_id in first_bar_channel[bar - 1 - 4] + readout_sequence - 1]  #  Selezione sul SiPM
            sipm_channels[asic_id // 2][bar - 1] = first_bar_channel[bar - 1 - 4] + readout_sequence - 1
            if boolean_channels:
                print ("\n\n########## BAR ", bar)
                print ("ASIC %d" % asic_id)
                print (first_bar_channel[bar - 1 - 4] + readout_sequence - 1)
            ordered_data[asic_id // 2][bar - 1] = a

    return ordered_data, triggerID, nevt, sipm_channels


def disentangle_tile (data, used_asics = [0, 1], boolean_I2C = True):

    print ("Collecting data from the asics", used_asics)

    # This part of the code deals with the fact that in one acquisition there can be different number of events recorded by the 4 ASICS due to
    # closure of the acquisition during the buffering (es: asics 0 and 1 recorded 4900 evts, while 2 recorded 4982 and 4976).
    # 1 - I take the minimum among the numbers of events taken by the 4 asics, call it min_nevt_asics
    # 2 - I check that the TRIGGER_IDs of the first min_nevt_asics for the 4 asics are the same and, if not, print ERROR
    # 3 - In the real disentangling part, I take only the first min_nevt_asics events
    nevt_asics = [ len(data["#TRG_ID"][data["ASIC"] == asic_id]) for asic_id in used_asics]
    min_nevt_asics = np.min(nevt_asics)
    max_nevt_asics = np.max(nevt_asics)
    triggerID = np.array(data[data["ASIC"] == 0]["#TRG_ID"][:min_nevt_asics])
    comparison = np.array(data[data["ASIC"] == 1]["#TRG_ID"][:min_nevt_asics])
    number_of_mismatches = np.count_nonzero([triggerID == comparison] == False)
    if number_of_mismatches != 0:
        print ("ERROR")
    if boolean_I2C is True:
        triggerID = decode_trig_id(triggerID)

    nevt = min_nevt_asics

    ######## Correzione dello shift di +1 fra la colonna dei triggerID e la colonna degli ADC
    triggerID = correzione_shift_triggerID_adc(triggerID)

    ordered_data = [[], []]
    asic0_mask = data["ASIC"] == 0
    asic0_data = data[asic0_mask]
    ordered_data[0] = np.array(asic0_data["CH[7]"])[:nevt]
    asic1_mask = data["ASIC"] == 1
    asic1_data = data[asic1_mask]
    ordered_data[1] = np.array(asic1_data["CH[8]"])[:nevt]

    return ordered_data, triggerID, nevt


def decode_trig_id (trig_ids):
    # print(bin(num))
    # D=int('0b'+str(bin(num)[2:6]), base=2)
    # TT=int('0b'+str(bin(num)[6:14]), base=2)
    # TS=int('0b'+str(bin(num)[14:]), base=2)
    nevt = len(trig_ids)
    I2C_IDs, I2C_triggertypes, I2C_subsystemIDs = np.zeros(nevt), np.zeros(nevt), np.zeros(nevt)
    for i_trig_id in range(nevt):
        num = trig_ids[i_trig_id]
        I2C_IDs[i_trig_id] = num & 0xffffffff
        I2C_triggertypes[i_trig_id] = (num >> 32) & 0x00000001
        I2C_subsystemIDs[i_trig_id] = (num >> 40) & 0xff
    return I2C_subsystemIDs, I2C_triggertypes, I2C_IDs


def VFS_Correction(data0, asic = [0, 1, 2, 3], maskRanges = [1550, 1850, 1600, 1600], FilterShift = [0, 0, 0, 0]):
    #maskRanges-> Each asic with a VMon> of this valus need to be shift
    #FIlterShift-> Value of shifting It depend only by the ASIC
    for i in range(len(asic)):
        Cut = data0['ASIC'] == asic[i]
        Cut1 = data0['VFS'] > maskRanges[i]
        for ch in range(16):
            data0.loc[Cut & Cut1,'CH['+str(ch)+']'] = data0.loc[Cut & Cut1, 'CH[' + str(ch) + ']'] - FilterShift[i]
    return data0


def Temp_Correction(
    data0, 
    asic = [0, 1, 2, 3], 
    maskRanges = [400, 400, 400, 400], 
    FilterShift = [0, 0, 0, 0], 
    compute_mask = False, 
    compute_shift = False
):

    """
    data0 = DataFrame di pandas a partire dal csv
    asic = asic sui quali fare la correzione
    maskRanges = il valore di limite fra gli stati low e high della variabile Temp
    FilterShift = il valore dello shift da fare
    compute_mask = boolean che dice se calcolare in modo automatico i valori di maskRanges
    compute_shift = boolean che dice se calcolare in modo automatico i valori di FilterShift
    """

    nasics = len(asic)
    temp_correction_flag = [False] * nasics
    for i in range(nasics):
        Cutt = data0['ASIC'] == asic[i]
        if compute_mask == True:   ### sarà messo per il psd, non per tile
            maskRanges[i] = np.average( data0['TEMP'][Cutt] ) + 100
        Cutt1 = data0['TEMP'] > maskRanges[i]

        if np.count_nonzero( (Cutt & Cutt1) == True):
            temp_correction_flag[i] = True

            if (compute_shift == True):
                t = data0["TEMP"][data0["ASIC"]==asic[i]]
                a = data0["CH[0]"][data0["ASIC"]==asic[i]]
                counts1, bins1 = np.histogram(a[t < maskRanges[i]], bins = 600)
                counts2, bins2 = np.histogram(a[t > maskRanges[i]], bins = 600)
                FilterShift[i] = bins2[np.argmax(counts2)] - bins1[np.argmax(counts1)]
                print ("Data of ASIC %d shifted by %d" % (asic[i], FilterShift[i]))

            print ("Remind that you're shifting backwards the TEMP values above %d of asic %d (%d events out of %d) by %d \
                    \n(and correponding ADC channels by the same amounts)" \
                    % (maskRanges[i], asic[i], np.count_nonzero( (Cutt & Cutt1) == True), np.count_nonzero(Cutt==True), FilterShift[i]) )
        data0.loc[Cutt & Cutt1, 'TEMP'] = data0.loc[Cutt & Cutt1, 'TEMP'] - FilterShift[i]
        for ch in range(16):
            data0.loc[Cutt & Cutt1,'CH['+str(ch)+']'] = data0.loc[Cutt & Cutt1, 'CH[' + str(ch) + ']'] - FilterShift[i]

    return data0, temp_correction_flag


######## Correzione dello shift di +1 fra la colonna dei triggerID e la colonna degli ADC
def correzione_shift_triggerID_adc(triggerID):
    try:
        length_triggerID = len(triggerID)
        triggerID = np.array(triggerID)
        for i in range(len(triggerID)):
            triggerID[i] = np.concatenate( (triggerID[i][1:], [triggerID[i][0]]) )
    except:
        triggerID = np.concatenate( (triggerID[1:], [triggerID[0]]) )
    return triggerID

def check_filename (file = None):
    if file is None:
        raise ValueError ("No filename was given, please provide one.")
    return

def check_detector (detector = "TriggerTile"):
    if "PSD" in detector:
        detector = "PSD"
    elif "Tile" in detector:
        detector = "TriggerTile"
    else:
        raise ValueError ("No correct detector name was given.")
    return detector

def take_run_number_from_filename (detector = "TriggerTile", file = None):
    check_filename(file)
    detector = check_detector(detector)
    run_filename = file.split("/")[-1]
    single_fields = run_filename.split("_")
    try:
        run_number = int(single_fields[1])
    except:
        run_number = None
    return run_number


def take_date_from_filename (detector = "TriggerTile", file = None, i_prefix = 0):
    check_filename(file)
    detector = check_detector(detector)
    run_filename = file.split("/")[-1]
    single_fields = run_filename.split("_")
    ### Date
    yy, mm, dd = int(single_fields[i_prefix+3][0:4]), int(single_fields[i_prefix+3][4:6]), int(single_fields[i_prefix+3][6:8])
    hh, mi, ss = int(single_fields[i_prefix+4][0:2]), int(single_fields[i_prefix+4][2:4]), int(single_fields[i_prefix+4][4:6])
    date = datetime.datetime(yy, mm, dd, hh, mi, ss)
    if detector == "PSD":
        date = date + datetime.timedelta(hours = 2)
    return date


def take_gain_from_yaml (detector = "TriggerTile", file = None, run_number = None):
    check_filename(file)
    detector = check_detector(detector)
    nasics = detectors_dict[detector]["nasics"]
    asic_gain_lh  = np.zeros( nasics )     # gain dell'asic: low (1) or high (0)
    asic_gain_val = np.zeros( nasics )     # valore del gain, da 0 a 15

    yfile = file.replace(".csv", ".yaml")
    with open(yfile, "r") as f:
        data = yaml.safe_load(f)
        for asic in range(nasics):
            asic_gain_lh[asic]  = data["asic%d" % asic]["mux_path"]
            asic_gain_val[asic] = data["asic%d" % asic]["gain"][0]

    if run_number is not None:
        if (detector == "PSD") * (run_number < 563):
            asic_gain_lh[0] = -999
            asic_gain_lh[2] = -999
            asic_gain_val[0] = data["asic%d" % 0][0] = -999
            asic_gain_val[2] = data["asic%d" % 2][0] = -999    

    gain = {"Gain lh" : asic_gain_lh,   \
            "Gain val" : asic_gain_val, \
            "info" : detectors_dict[detector]["info"]}

    return gain
