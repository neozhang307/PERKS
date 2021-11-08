import re

#sample code for ptx analysis

# with open("./log") as f:
#     contents=f.readlines()
#     print("funcname","sm","type","halo","rf","reg","sm","sps","spl")

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

def compile(archstring, halostring, regfolderstring, realstring,useSM,asyncSM,box):
    basicstring="nvcc -std=c++11 --cubin -gencode {0} -Xptxas \"-v\" -DCONFIGURE,HALO={1},TYPE={2},RTILE_Y=8,RTILE_X=256,REG_FOLDER_Y={3}{4}{5}{6} ./jacobi-general.cu"

    generated_string=basicstring.format(archstring,halostring,realstring,regfolderstring,useSM,asyncSM,box)
    # print(generated_string)

    sub = subprocess.Popen(generated_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    return sub.stdout.read()

import sys


def singletask(filename,halostring,realstring,useSM,box,archstring,asyncSM):
    regfolder=0
    file = open(filename, 'a')
    original_out=sys.stdout
    while (True):
    # for regfolder in range(3):
        subprocess_return = compile(archstring,halostring,regfolder,realstring,useSM,asyncSM,box)
        print(subprocess_return)
        returnlist =parseptx(subprocess_return.splitlines(), "general")
        spillnum=int(returnlist[6])+int(returnlist[7])
        # file.write("kernel_general",halostring,returnlist[0],returnlist[1],returnlist[2],returnlist[3],returnlist[4],returnlist[5],returnlist[6],returnlist[7])
        sys.stdout = file 
        print("kernel_general",halostring,returnlist[0],returnlist[1],returnlist[2],returnlist[3],returnlist[4],returnlist[5],returnlist[6],returnlist[7])
        sys.stdout = original_out

        regfolder+=1
        # break
        if spillnum!=0:
            break

def HuseSM(filename,halostring,realstring,box,archstring,asyncSM):
        singletask(filename,halostring,realstring,"",box,archstring,asyncSM)
        singletask(filename,halostring,realstring,",USESM",box,archstring,asyncSM)

def HuseSMreal(filename,halostring,box,archstring,asyncSM):
    HuseSM(filename,halostring,"float",box,archstring,asyncSM)
    HuseSM(filename,halostring,"double",box,archstring,asyncSM)

def Hrange(filename,halostart,haloend,box,archstring,asyncSM):
    for i in range(halostart,haloend+1):
        HuseSMreal(filename,i,box,archstring,asyncSM)

def HuseSMrealbox(halostring,archstring,asyncSM):
    HuseSMreal(halostring,"",archstring,asyncSM)
    HuseSMreal(halostring,",BOX",archstring,asyncSM)

def HuseSMrealboxArch(halostring):
    HuseSMrealbox(halostring,"arch=compute_80,code=sm_80","")
    HuseSMrealbox(halostring,"arch=compute_80,code=sm_80",",ASYNCSM")
    HuseSMrealbox(halostring,"arch=compute_70,code=sm_70","")


archstring="arch=compute_80,code=sm_80"
halostring=1
# regfolderstring=16
# realstring="float"
# useSM=",USESM"
# useSM=""
# asyncSM=",ASYNCSM"
# asyncSM=""
# box=",BOX"
# box=""
# HuseSMrealboxArch(1)
# print("star")

Hrange("star_80.log",1,6,"","arch=compute_80,code=sm_80","")
Hrange("star_80_async.log",1,6,"","arch=compute_80,code=sm_80",",ASYNCSM")
Hrange("star_70.log",1,6,"","arch=compute_70,code=sm_70","")

Hrange("box_80.log",1,2,",BOX","arch=compute_80,code=sm_80","")
Hrange("box_80_async.log",1,2,",BOX","arch=compute_80,code=sm_80",",ASYNCSM")
Hrange("box_70.log",1,2,",BOX","arch=compute_70,code=sm_70","")
