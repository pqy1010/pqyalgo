import numpy as np

def findpeaks(data):
    ''' find signal peaks and return index and val'''
    data = data.ravel()
    diffdata = np.diff(np.sign(np.diff(data)))
    maxloc_tuple = np.where(diffdata < 0)
    maxloc = maxloc_tuple[0] + 1
    maxval = np.take(data, maxloc)
    return maxval, maxloc-1

def smooth(data,num):
    '''smooth data like matlab function smooth 
    input: data and smooth num(odd)'''
    if num%2==0:
        print('num must be odd')
        return []

    res=np.zeros_like(data)
    datalen=len(data)
    halfnum=int((num-1)/2)
    for i in range(datalen):
        if i<halfnum:
            b=0
            e=i+1
        elif i>datalen-halfnum:
            e=datalen+1
            b=datalen-i
        else:
            b=i-halfnum
            e=i+halfnum+1
        res[i]=sum(data[b:e])/(e-b)
    return res

def xcorrfig(data,sec):
    '''time-xcorr 2Darr
    input data,sec'''
    fs=500
    datalen=len(data)
    seclen=int(np.floor(datalen/fs))
    seglen=sec*fs
    seghalf=int(seglen/2)
    corrstart=seghalf+136
    corrend=seghalf+545
    xcorrlist=[]
    for i in range(sec*fs,seclen*fs,fs):
        tempdata=data[i-sec*fs:i]
        tempdata2=tempdata[list(range(-1,-len(tempdata),-1))]
        scorr=np.convolve(tempdata,tempdata2,'same')
        scorr=scorr/scorr[seghalf]
        valiscorr=scorr[corrstart:corrend]
           
        xcorrlist.append(valiscorr)
        
    xcorrarr=np.mat(xcorrlist)
    return(xcorrarr)

def centerclip(envdata, fs, freflag):
    '''binaryzation data by peaks
    input: data,fs,freflag'''
    first_non_zero_index = 0
    for each in envdata:
        if (each == 0):
            first_non_zero_index += 1
        else:
            break;
    if (first_non_zero_index == len(envdata)):
        return envdata.ravel()
    max_pn = 60
    thr = 100
    peak_num = 0
    clip = 0
    idb = 0
    cnt = 0
    if (freflag == 2):
        max_pn = 26
    peakval, peakloc = findpeaks(envdata)  # 寻找峰值
    peak_num = np.size(peakval)
    peaksort = -np.sort(-peakval, axis=0, kind='quicksort')
    if (peak_num >= max_pn):
        clip = np.median(peaksort[0:max_pn])
    elif (peak_num == 1):
        clip = peakval
    else:
        clip = np.median(peaksort)
    outdata = envdata.ravel()
    outdata = np.where(outdata > clip, 1, 0)  # 二值化
    for i in range(np.size(outdata)):
        if (cnt == 0):
            if (outdata[i] == 1):
                cnt = 1
                idb = i
        else:
            if (outdata[i] == 0):
                if (cnt > thr):
                    j = 0
                    while (j < cnt):
                        outdata[idb + j] = 0
                        j = j + 1
                cnt = 0
            else:
                cnt = cnt + 1
    return outdata

def hilbertenv(data):
    '''calculate envdata by hilbert transmition
    input: data'''
    datanomean=data-np.mean(data)
    datalen=len(data)
    datahalf=int(datalen/2)
    datafft=np.fft.fft(datanomean)
    
    hdata=np.zeros_like(datafft)
    hdata[0]=datafft[0]
    hdata[1:datahalf]=2*datafft[1:datahalf]
    
    idata=np.abs(np.fft.ifft(hdata))
    return idata

def teager(data):
    '''calculate teager energy
    input:data'''
    res=np.zeros(len(data))
    for i in range(1,len(data)-1):
        res[i]=data[i]**2-data[i-1]*data[i+1]
    res[0]=data[0]
    res[-1]=data[-1]
    return res

def xcorr(data):
    '''calculate xcorr
    input : data'''
    res=np.zeros_like(data)
    res=res.astype(np.float)
    for i in range(len(data)):
        res[i]=np.matmul(data[i:],data[0:len(data)-i])
    return res
