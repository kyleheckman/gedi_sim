#!/usr/bin/env python3

DESCRIPTION='''
Read LVIS binary files generated from 1998 to 2009.

Formats:
    .lce
    .lge
    .lgw

    Data versions 1.00 to 1.04

LVIS Operations Team
Sarah Story
2020/7/17
'''

import numpy as np

def get_datatype(filetype,version):
    
    if filetype=='lce':
        if version=='1.00':
            dt=np.dtype([('tlon','>f8'),('tlat','>f8'),('zt','>f4')])
        elif version=='1.01':
            dt=np.dtype([('lfid','>u4'),('shotnumber','>u4'),('tlon','>f8'),('tlat','>f8'),('zt','>f4')])
        elif version=='1.02':
            dt=np.dtype([('lfid','>u4'),('shotnumber','>u4'),('time','>f8'),('tlon','>f8'),('tlat','>f8'),('zt','>f4')])
        elif version=='1.03':
            dt=np.dtype([('lfid','>u4'),('shotnumber','>u4'),('azimuth','>f4'),('incidentangle','>f4'),('range','>f4'),('time','>f8'),('tlon','>f8'),('tlat','>f8'),('zt','>f4')])
        elif version=='1.04':
            dt=np.dtype([('lfid','<u4'),('shotnumber','<u4'),('azimuth','<f4'),('incidentangle','<f4'),('range','<f4'),('time','<f8'),('tlon','<f8'),('tlat','<f8'),('zt','<f4')])
    
    
    elif filetype=='lge':
        if version=='1.00':
            dt=np.dtype([('glon','>f8'),('glat','>f8'),('zg','>f4'),('rh25','>f4'),('rh50','>f4'),('rh75','>f4'),('rh100','>f4')])
        elif version=='1.01':
            dt=np.dtype([('lfid','>u4'),('shotnumber','>u4'),('glon','>f8'),('glat','>f8'),('zg','>f4'),('rh25','>f4'),('rh50','>f4'),('rh75','>f4'),('rh100','>f4')])
        elif version=='1.02':
            dt=np.dtype([('lfid','>u4'),('shotnumber','>u4'),('time','>f8'),('glon','>f8'),('glat','>f8'),('zg','>f4'),('rh25','>f4'),('rh50','>f4'),('rh75','>f4'),('rh100','>f4')])
        elif version=='1.03':
            dt=np.dtype([('lfid','>u4'),('shotnumber','>u4'),('azimuth','>f4'),('incidentangle','>f4'),('range','>f4'),('time','>f8'),('glon','>f8'),('glat','>f8'),('zg','>f4'),('rh25','>f4'),('rh50','>f4'),('rh75','>f4'),('rh100','>f4')])
        elif version=='1.04':
            dt=np.dtype([('lfid','<u4'),('shotnumber','<u4'),('azimuth','<f4'),('incidentangle','<f4'),('range','<f4'),('time','<f8'),('glon','<f8'),('glat','<f8'),('zg','<f4'),('rh25','<f4'),('rh50','<f4'),('rh75','<f4'),('rh100','<f4')])
    
    
    elif filetype=='lgw':
        if version=='1.00':
            dt=np.dtype([('lon0','>f8'),('lat0','>f8'),('z0','>f4'),('lon431','>f8'),('lat431','>f8'),('z431','>f4'),('sigmean','>f4'),('wave','>u1',432)])
        elif version=='1.01':
            dt=np.dtype([('lfid','>u4'),('shotnumber','>u4'),('lon0','>f8'),('lat0','>f8'),('z0','>f4'),('lon431','>f8'),('lat431','>f8'),('z431','>f4'),('sigmean','>f4'),('wave','>u1',432)])
        elif version=='1.02':
            dt=np.dtype([('lfid','>u4'),('shotnumber','>u4'),('time','>f8'),('lon0','>f8'),('lat0','>f8'),('z0','>f4'),('lon431','>f8'),('lat431','>f8'),('z431','>f4'),('sigmean','>f4'),('wave','>u1',432)])
        elif version=='1.03':
            dt=np.dtype([('lfid','>u4'),('shotnumber','>u4'),('azimuth','>f4'),('incidentangle','>f4'),('range','>f4'),('time','>f8'),('lon0','>f8'),('lat0','>f8'),('z0','>f4'),('lon431','>f8'),('lat431','>f8'),('z431','>f4'),('sigmean','>f4'),('txwave','>u1',80),('rxwave','>u1',432)])
        elif version=='1.04':
            dt=np.dtype([('lfid','<u4'),('shotnumber','<u4'),('azimuth','<f4'),('incidentangle','<f4'),('range','<f4'),('time','<f8'),('lon0','<f8'),('lat0','<f8'),('z0','<f4'),('lon431','<f8'),('lat431','<f8'),('z431','<f4'),('sigmean','<f4'),('txwave','<u2',120),('rxwave','<u2',528)])
    
    return dt

def read_legacy_lvis(input_file,version):

    if input_file.find('lce')!=-1:
        filetype='lce'
    elif input_file.find('lge')!=-1:
        filetype='lge'
    elif input_file.find('lgw')!=-1:
        filetype='lgw'
    else:
        print('Unrecognized file type.')
        return

    if not version in ['1.00','1.01','1.02','1.03','1.04']:
        print('Unrecognized data version.')
        return

    try:
        dt=get_datatype(filetype,version)
        x=np.fromfile(input_file,dt)

        # Sanity check lat/lon - catch version number or byte order problems
        if filetype=='lce':
            l=x['tlat'][np.where(x['tlat'] != 0)]
            if max(np.log10(np.abs(l))) > 2 or min(np.log10(np.abs(l))) < -12:
                # Absurd value in latitude. Try flipping the byte order. If it's still bad,
                # likely that file version number was wrong.
                if max(np.log10(np.abs(l.newbyteorder()))) > 2 or min(np.log10(np.abs(l.newbyteorder()))) < -12:
                    print('Warning: Bad latitude values. Check LVIS data version!')
                else:
                    z=x.newbyteorder()
                    return z
            else:
                pass
        elif filetype=='lge':
            l=x['glat'][np.where(x['glat'] != 0)]
            if max(np.log10(abs(l))) > 2 or min(np.log10(abs(l))) < -12:
                if max(np.log10(np.abs(l.newbyteorder()))) > 2 or min(np.log10(np.abs(l.newbyteorder()))) < -12:
                    print('Warning: Bad latitude values. Check LVIS data version!')
                else:
                    z=x.newbyteorder()
                    return z
            else:
                pass
        elif filetype=='lgw':
            l=x['lat0'][np.where(x['lat0'] != 0)]
            if max(np.log10(abs(l))) > 2 or min(np.log10(abs(l))) < -12:
                if max(np.log10(np.abs(l.newbyteorder()))) > 2 or min(np.log10(np.abs(l.newbyteorder()))) < -12:
                    print('Warning: Bad latitude values. Check LVIS data version!')
                else:
                    z=x.newbyteorder()
                    return z
            else:
                pass

        return x
    except IOError:
        print('Legacy LVIS file does not exist or is not readable.')
        return