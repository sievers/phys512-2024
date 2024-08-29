import numpy as np
import ephem
import time
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from urllib import request

#def read_latest_tles(url='https://celestrak.com/NORAD/elements/orbcomm.txt'):
def read_latest_tles(url='http://celestrak.org/NORAD/elements/gp.php?GROUP=orbcomm&FORMAT=tle'):
    raw=request.urlopen(url)
    lines=[line.decode('utf-8').strip() for line in raw.readlines()]
    nsat=len(lines)//3
    tles=[None]*nsat
    for i in range(nsat):
        tles[i]=lines[3*i:3*(i+1)]
    return tles

def ctime2mjd(tt=None,type='Dublin'):
    if tt is None:
        tt=time.time()    
    jd=tt/86400+2440587.5
    return jd-2415020

tles=read_latest_tles('http://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle')
#tles=read_latest_tles('http://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle')
#tles=read_latest_tles()

lat=np.empty(len(tles))
lon=np.empty(len(tles))
observer=ephem.Observer()

plt.clf()
ax=plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
for iter in range(100):
    tt=time.time()
    djd=ctime2mjd(tt)
    observer.date=djd
    t1=time.time()
    plt.ion()
    for i,tle in enumerate(tles):
        tle_rec=ephem.readtle(*tle)
        tle_rec.compute(observer)
        lat[i]=tle_rec.sublat
        lon[i]=tle_rec.sublong
    t2=time.time()
    if iter==0:
        plot_data,=ax.plot(lon*180/np.pi,lat*180/np.pi,'r.')
    else:
        plot_data.set_data(lon*180/np.pi,lat*180/np.pi)
    plt.pause(0.001)
    time.sleep(1)

