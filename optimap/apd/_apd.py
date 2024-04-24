import numpy as np
def normalize_trace(y, y_min: float = None, y_max: float = None):
    
    #find the maximum intensity point of the beat
    if y_max is None:
     y_max = np.max(y)

    #find the minimum value 
    if y_min is None:
        y_min = np.min(y)

    #make all values between 0 and 1 (min=0, max=1); subtract the repolarization percentage make the max value = 1 - repol
    return (np.array(y) - y_min) / (y_max - y_min)

def detect_apd(trace, repol, y_rmin: float = None, y_rmax: float = None , fps: float=500, pulse_min_width: float=5, pulse_max_width: float=200):

    if not 0 < repol < 100:
        raise ValueError("The Repolarization value is not in bounds (between 0 and 100)")

    time = (np.arange(len(trace))/fps) * 1000

    #make a copy of original array 
    y_array = np.array(trace)
    t_array = np.array(time)

    #check if each y value has a corresponding t value
    if len(y_array) != len(t_array):
        raise ValueError("The number of trace and time elements do not correspond")

    #find the maximum intensity point of the beat
    if y_rmax is None:
        y_rmax = np.max(y_array)

    #find the minimum value 
    if y_rmin is None:
        y_rmin = np.min(y_array)
    
    normalize_y_values = normalize_trace(y_array, y_rmin, y_rmax)
    repolarized_y_values = (normalize_y_values - (1 - repol/100))

    return apd_beat_map(repolarized_y_values, t_array, pulse_min_width, pulse_max_width)


def apd_beat_map(repolarized_y_values, time, pulse_min_width: float=5, pulse_max_width: float=200):

    apd_values = []
    (sign_change_positions, sign_y) = sign_change(repolarized_y_values)

    if not (len(sign_change_positions) == 0 or len(sign_change_positions) == 1):
        #sort the times and subtract the two from one another; print final result
        first_pt_found = False
        for i in sign_change_positions :
            #finds the upstroke (neg-->pos) ; split upstroke 
            if first_pt_found == False:
                if i+1 < len(sign_y) and sign_y[i] < sign_y[i+1]:
                    #positive 
                    apd_pt_1 = i+1
                    first_pt_found = True
            #finds the downstroke ONLY if upstroke is detected 
            else:
                if i+1 < len(sign_y) and sign_y[i] > sign_y[i+1]:
                    #negative
                    apd_pt_2 = i+1
                    time1 = time[apd_pt_1]
                    time2 = time[apd_pt_2]
                    time1index = apd_pt_1
                    time2index = apd_pt_2 
                    apd_value = time2-time1
                    first_pt_found = False
                    if pulse_min_width <= apd_value <= pulse_max_width:
                        apd_values.append([apd_value, time1, time2, time1index, time2index])
    return apd_values


def sign_change(y_array):
    #look for a change in sign (neg to pos/ pos to neg) --> these are going to be the two time points were comparing
    sign_y = np.sign(y_array)
    num_zeros = len(sign_y[sign_y == 0])
    if num_zeros > 1:
        sign_change_pos = np.where(sign_y == 0)[0]
    else:
        sign_change_pos = np.where(np.diff(np.sign(y_array)))[0]

    return sign_change_pos, sign_y

