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

                # rtiley=templateinterger[0]
                halo=templateinterger[0]
                itempthread=templateinterger[1]
                regfolder=templateinterger[3]
                
                UseSM=templatebool[0]
                Type=templatetype[0]

                spillinfos=re.findall(r"\d+",contents[i+2].decode('UTF-8'))
                loadspill=spillinfos[2]
                storespill=spillinfos[1]

                meminfos=re.findall(r"\d+",contents[i+3].decode('UTF-8'))
                regusage=meminfos[0]
                smusage=meminfos[1]
                # print("kernel_general",arch.group(0),
                #     Type, UseSM, regfolder, 
                #     regusage,smusage,storespill,loadspill)
                # spillnum=int(loadspill)+int(storespill)
                # return spillnum
                return arch.group(0),Type, UseSM, regfolder, regusage,smusage,storespill,loadspill
            i+=4
        else:
            i+=1
        # i+=1




import subprocess

def compile(archstring,halostring,realstring,regfolderstring,btype,useSM,box,type0,poisson,bdim,ipt):
    basicstring="nvcc -std=c++14 --cubin -gencode {0} -Xptxas \"-v\" -DCONFIGURE,HALO={1},TYPE={2},REG_FOLDER_Z={3},BDIM={9},BLOCKTYPE={4}{5}{6}{7}{8} -DITERMPT={10} ./j3d-general.cu"
    generated_string=basicstring.format(archstring,halostring,realstring,regfolderstring,btype,useSM,box,type0,poisson,bdim,ipt)

    sub = subprocess.Popen(generated_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    return sub.stdout.read()

import sys


def singletask(filename,archstring,halostring,realstring,btype,useSM,box,type0,poisson,bdim,ipt):
    regfolder=0
    file = open(filename, 'a')
    original_out=sys.stdout
    while (True):
    # for regfolder in range(3):
        subprocess_return = compile(archstring,halostring,realstring,regfolder,btype,useSM,box,type0,poisson,bdim,ipt)
        print(subprocess_return)
        returnlist =parseptx(subprocess_return.splitlines(), "general")
        spillnum=int(returnlist[6])+int(returnlist[7])
        sys.stdout = file 
        print("kernel_general",halostring,bdim,ipt,returnlist[0],returnlist[1],returnlist[2],returnlist[3],returnlist[4],returnlist[5],returnlist[6],returnlist[7])
        sys.stdout = original_out

        regfolder+=1
        # break
        if spillnum>=128:
            break

def HuseSM(filename,archstring,halostring,realstring,btype,box,type0,poisson,bdim,ipt):
        singletask(filename,archstring,halostring,realstring,btype,"",box,type0,poisson,bdim,ipt)
        singletask(filename,archstring,halostring,realstring,btype,",USESM",box,type0,poisson,bdim,ipt)

def HrangeDIM(filename,archstring,halostring,realstring,btype,box,type0,poisson,ipt):
    HuseSM(filename,archstring,halostring,realstring,btype,box,type0,poisson,128,ipt)
    HuseSM(filename,archstring,halostring,realstring,btype,box,type0,poisson,256,ipt)

def HuseSMreal(filename,archstring,halostring,btype,box,type0,poisson):
    HrangeDIM(filename,archstring,halostring,"float",btype,box,type0,poisson,8)
    HrangeDIM(filename,archstring,halostring,"float",btype,box,type0,poisson,16)
    HrangeDIM(filename,archstring,halostring,"double",btype,box,type0,poisson,8)

def HrangeSTAR(filename,archstring,btype):
    HuseSMreal(filename,archstring,1,btype,"","","")
    HuseSMreal(filename,archstring,2,btype,"","","")


def HrangeBOX(filename,archstring,btype):
    HuseSMreal(filename,archstring,1,btype,",BOX","","")

def HrangeTYPE0(filename,archstring,btype):
    HuseSMreal(filename,archstring,1,btype,",BOX",",TYPE0","")

def HrangePOISSON(filename,archstring,btype):
    HuseSMreal(filename,archstring,1,btype,",BOX",",TYPE0",",POISSON")


# archstring="arch=compute_80,code=sm_80"
# halostring=1
from multiprocessing import Process
import os

# HrangeSTAR      ("star_80_128.log","arch=compute_80,code=sm_80",2)
# HrangeBOX       ("box_80_128.log","arch=compute_80,code=sm_80",2)
# HrangeTYPE0     ("type0_80_128.log","arch=compute_80,code=sm_80",2)
# HrangePOISSON   ("poisson_80_128.log","arch=compute_80,code=sm_80",2)

# HrangeSTAR      ("star_70_128.log","arch=compute_70,code=sm_70",2)
# HrangeBOX       ("box_70_128.log","arch=compute_70,code=sm_70",2)
# HrangeTYPE0     ("type0_70_128.log","arch=compute_70,code=sm_70",2)
# HrangePOISSON   ("poisson_70_128.log","arch=compute_70,code=sm_70",2)

# HrangeSTAR      ("star_80_256.log","arch=compute_80,code=sm_80",1)
# HrangeBOX       ("box_80_256.log","arch=compute_80,code=sm_80",1)
# HrangeTYPE0     ("type0_80_256.log","arch=compute_80,code=sm_80",1)
# HrangePOISSON   ("poisson_80_256.log","arch=compute_80,code=sm_80",1)

# HrangeSTAR      ("star_70_256.log","arch=compute_70,code=sm_70",1)
# HrangeBOX       ("box_70_256.log","arch=compute_70,code=sm_70",1)
# HrangeTYPE0     ("type0_70_256.log","arch=compute_70,code=sm_70",1)
# HrangePOISSON   ("poisson_70_256.log","arch=compute_70,code=sm_70",1)


p1 = Process(target=HrangeSTAR, args=("star_80_128.log","arch=compute_80,code=sm_80",2))
p2 = Process(target=HrangeBOX, args=("box_80_128.log","arch=compute_80,code=sm_80",2))
p3 = Process(target=HrangeTYPE0     ,args=("type0_80_128.log","arch=compute_80,code=sm_80",2))
p4 = Process(target=HrangePOISSON   ,args=("poisson_80_128.log","arch=compute_80,code=sm_80",2))

p5 = Process(target=HrangeSTAR      ,args=("star_70_128.log","arch=compute_70,code=sm_70",2))
p6 = Process(target=HrangeBOX       ,args=("box_70_128.log","arch=compute_70,code=sm_70",2))
p7 = Process(target=HrangeTYPE0     ,args=("type0_70_128.log","arch=compute_70,code=sm_70",2))
p8 = Process(target=HrangePOISSON   ,args=("poisson_70_128.log","arch=compute_70,code=sm_70",2))

p9 = Process(target=HrangeSTAR      ,args=("star_80_256.log","arch=compute_80,code=sm_80",1))
p10 = Process(target=HrangeBOX       ,args=("box_80_256.log","arch=compute_80,code=sm_80",1))
p11 = Process(target=HrangeTYPE0     ,args=("type0_80_256.log","arch=compute_80,code=sm_80",1))
p12 = Process(target=HrangePOISSON   ,args=("poisson_80_256.log","arch=compute_80,code=sm_80",1))

p13 = Process(target=HrangeSTAR      ,args=("star_70_256.log","arch=compute_70,code=sm_70",1))
p14 = Process(target=HrangeBOX       ,args=("box_70_256.log","arch=compute_70,code=sm_70",1))
p15 = Process(target=HrangeTYPE0     ,args=("type0_70_256.log","arch=compute_70,code=sm_70",1))
p16 = Process(target=HrangePOISSON   ,args=("poisson_70_256.log","arch=compute_70,code=sm_70",1))
p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p6.start()
p7.start()
p8.start()
p9.start()
p10.start()
p11.start()
p12.start()
p13.start()
p14.start()
p15.start()
p16.start()

p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()
p7.join()
p8.join()
p9.join()
p10.join()
p11.join()
p12.join()
p13.join()
p14.join()
p15.join()
p16.join()


