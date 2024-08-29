import sounddevice as sd
import time,glob,numpy,random,curses
from curses import wrapper
import subprocess

def get_key(mystr,key):
    i1=mystr.find(key)
    if i1<0:
        return 'Not found'
    i2=mystr.find('\n',i1+2)
    return mystr[i1+len(key):i2].strip()

def play_song(fname):
    faad='faad'
    to_exec=faad + ' -i '+ fname  #first, read header so we can report song info
    aa=subprocess.check_output(to_exec,stderr=subprocess.STDOUT,shell=True)
    aa=aa.decode('ascii',errors='ignore')    #python3 requires explicitly converting to a string
    nn=get_key(aa,'\nTille        :')#we'll get name of song
    writer=get_key(aa,'\nComposer     :')  #and composer
    album=get_key(aa,'\nAlbum        :')
    myinfo=get_key(aa,'LC AAC') #we'll get some info about the file    
    tags=myinfo.split()
    rate=tags[-2]
    nchan=tags[-4]
    length=tags[0]
    if tags[1][-1]==',':
        tags[1]=tags[1][:-1]
    mystr=('playing ' + nn + ' from ' + album + ' by ' + writer + ' for ' + length + ' ' + tags[1])
    
    to_exec=faad+ ' -f2 -w -b4 ' + fname  #this magic command will have faad read data into float arrays
    ff=open('/dev/null','w')
    dd=subprocess.check_output(to_exec,stderr=ff,shell=True) #this calls faad to do the heavy lifting 
    ff.close()
    dat=numpy.frombuffer(dd,dtype='float32') #convert string output from faad into numpy array
    nchan=int(nchan)
    dat=numpy.reshape(dat,[len(dat)//nchan,nchan])
    fs=int(rate) #get the sample rate the speakers are expecting
    sd.play(dat,fs,blocking=False) #play through speakers using sounddevice
    return float(length),mystr #return length of song.  

def main(stdscr,dt=0.3):
    dr='/Users/sievers/Music/old/ipod_touch/'
    fnames=glob.glob(dr + '/*/*.m4a') #get filenames from a directory
    random.shuffle(fnames) #randomly reorder the file names
    stdscr.nodelay(True)
    fnum=0
    while fnum<len(fnames): #loop over filenames in music library
            fname=fnames[fnum]
            mylen,mystr=play_song(fname)  #play the song, and learn how long it is
            stdscr.clear()
            stdscr.addstr(0,0,mystr)
            stdscr.refresh()
            t0=time.time()
            #for ii in range(nn):
            ii=0
            while (time.time()-t0)<mylen:
                ii=ii+1
                stdscr.addstr(2,0,repr(int(ii*dt)).rjust(4))
                stdscr.refresh()
                c=stdscr.getch()
                if c==ord('q'):
                    return
                elif c==ord('n'):
                    break
                elif c==ord('p'): #previous file
                    if fnum>0:
                        fnum=fnum-2
                        break
                elif c==ord('r'): #rewind file
                    fnum=fnum-1
                    break
                elif c==ord('f'):
                    stdscr.addstr(3,0,fname)
                elif c==ord('h'):
                    stdscr.addstr(4,0,'shuffle commands:')
                    stdscr.addstr(5,0,'q - quit')
                    stdscr.addstr(6,0,'n - next song')
                    stdscr.addstr(7,0,'p - previous song')
                    stdscr.addstr(8,0,'r - rewind song')
                    stdscr.addstr(9,0,'f - print file path on disk')
                time.sleep(dt)
            fnum=fnum+1
if __name__=='__main__':
    stdscr = curses.initscr()
    wrapper(main)

