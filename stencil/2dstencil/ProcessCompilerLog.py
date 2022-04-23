import re


def parseptx(contents, funcname): 
    i=0
    while((i<len(contents))):
        # print(contents[i])
        m=re.search("function '(.*?)' for (.*?)",contents[i].decode('UTF-8'))
        arch=re.search("sm_\d+", contents[i].decode('UTF-8'))
        if m and arch:
            if(re.search(funcname,m.group(1))):
                templateinterger=re.findall(r"i(\d+)E",m.group(1))
                templatebool=re.findall(r"b(\d)E",m.group(1))
                templatetype=re.findall(r"I(\w)L",m.group(1))

                rtiley=templateinterger[0]
                halo=templateinterger[1]
                regfolder=templateinterger[2]
                
                UseSM=templatebool[0]
                Type=templatetype[0]

                spillinfos=re.findall(r"\d+",contents[i+2].decode('UTF-8'))
                loadspill=spillinfos[2]
                storespill=spillinfos[1]

                meminfos=re.findall(r"\d+",contents[i+3].decode('UTF-8'))
                regusage=meminfos[0]
                smusage=meminfos[1]

                return arch.group(0),Type, UseSM, regfolder, regusage,smusage,storespill,loadspill
            i+=4
        else:
            i+=1
        # i+=1




import subprocess

def compile(archstring, halostring, regfolderstring, realstring,useSM,asyncSM,box,btype,isSmall,TILE_Y):
    basicstring="nvcc -std=c++14 --cubin -gencode {0} -Xptxas \"-v\" -DCONFIGURE,HALO={1},TYPE={2},RTILE_Y={9},RTILE_X=256,REG_FOLDER_Y={3}{4}{5}{6}{8} -DBLOCKTYPE={7} ./jacobi-general.cu"

    generated_string=basicstring.format(archstring,halostring,realstring,regfolderstring,useSM,asyncSM,box,btype,isSmall,TILE_Y)

    sub = subprocess.Popen(generated_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    return sub.stdout.read()

import sys


def singletask(filename,halostring,realstring,useSM,box,archstring,asyncSM,btype,isSmall,TILE_Y):
    regfolder=0
    file = open(filename, 'a')
    original_out=sys.stdout
    while (True):
    # for regfolder in range(3):
        subprocess_return = compile(archstring,halostring,regfolder,realstring,useSM,asyncSM,box,btype,isSmall,TILE_Y)
        print(subprocess_return)
        returnlist =parseptx(subprocess_return.splitlines(), "general")
        spillnum=int(returnlist[6])+int(returnlist[7])
        # file.write("kernel_general",halostring,returnlist[0],returnlist[1],returnlist[2],returnlist[3],returnlist[4],returnlist[5],returnlist[6],returnlist[7])
        sys.stdout = file 
        print("kernel_general",TILE_Y,halostring,returnlist[0],returnlist[1],returnlist[2],returnlist[3],returnlist[4],returnlist[5],returnlist[6],returnlist[7])
        sys.stdout = original_out

        regfolder+=1
        # break
        if spillnum>=128:
            break

def HuseSM(filename,halostring,realstring,box,archstring,asyncSM,btype,isSmall,TILE_Y):
        singletask(filename,halostring,realstring,"",box,archstring,asyncSM,btype,isSmall,TILE_Y)
        singletask(filename,halostring,realstring,",USESM",box,archstring,asyncSM,btype,isSmall,TILE_Y)

def HuseSMreal(filename,halostring,box,archstring,asyncSM,btype,isSmall):
    HuseSM(filename,halostring,"float",box,archstring,asyncSM,btype,isSmall,8)
    HuseSM(filename,halostring,"float",box,archstring,asyncSM,btype,isSmall,16)
    HuseSM(filename,halostring,"double",box,archstring,asyncSM,btype,isSmall,8)

def Hrange(filename,halostart,haloend,box,archstring,asyncSM,btype,isSmall):
    for i in range(halostart,haloend+1):
        HuseSMreal(filename,i,box,archstring,asyncSM,btype,isSmall)

from multiprocessing import Process
import os

# archstring="arch=compute_80,code=sm_80"
# halostring=1

# Hrange("star_80_128.log",1,6,"","arch=compute_80,code=sm_80","",2,"")
# Hrange("star_70_128.log",1,6,"","arch=compute_70,code=sm_70","",2,"")

# Hrange("box_80_128.log",1,2,",BOX","arch=compute_80,code=sm_80","",2,"")
# Hrange("box_70_128.log",1,2,",BOX","arch=compute_70,code=sm_70","",2,"")


# Hrange("star_80_256.log",1,6,"","arch=compute_80,code=sm_80","",1,"")
# Hrange("star_70_256.log",1,6,"","arch=compute_70,code=sm_70","",1,"")

# Hrange("box_80_256.log",1,2,",BOX","arch=compute_80,code=sm_80","",1,"")
# Hrange("box_70_256.log",1,2,",BOX","arch=compute_70,code=sm_70","",1,"")

p1 = Process(target=Hrange, args=("star_80_128.log",1,6,"","arch=compute_80,code=sm_80","",2,""))
p2 = Process(target=Hrange, args=("star_70_128.log",1,6,"","arch=compute_70,code=sm_70","",2,""))

p3 = Process(target=Hrange, args=("box_80_128.log",1,2,",BOX","arch=compute_80,code=sm_80","",2,""))
p4 = Process(target=Hrange, args=("box_70_128.log",1,2,",BOX","arch=compute_70,code=sm_70","",2,""))

p5 = Process(target=Hrange ,args=("star_80_256.log",1,6,"","arch=compute_80,code=sm_80","",1,""))
p6 = Process(target=Hrange ,args=("star_70_256.log",1,6,"","arch=compute_70,code=sm_70","",1,""))

p7 = Process(target=Hrange ,args=("box_80_256.log",1,2,",BOX","arch=compute_80,code=sm_80","",1,""))
p8 = Process(target=Hrange ,args=("box_70_256.log",1,2,",BOX","arch=compute_70,code=sm_70","",1,""))


p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p6.start()
p7.start()
p8.start()

p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()
p7.join()
p8.join()




