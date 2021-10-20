import re

#sample code for ptx analysis

# with open("./log") as f:
#     contents=f.readlines()
#     i=0
#     print("funcname","sm","type","halo","rf","reg","sm","sps","spl")
#     while((i<len(contents))):
#         m=re.search("function '(.*?)' for (.*?)",contents[i])
#         arch=re.search("sm_\d+", contents[i])
#         if m and arch:
#             if(re.search("general",m.group(1))):
#                 templateinterger=re.findall(r"i(\d+)E",m.group(1))
#                 templatebool=re.findall(r"b(\d)E",m.group(1))
#                 templatetype=re.findall(r"I(\w)L",m.group(1))

#                 rtiley=templateinterger[0]
#                 halo=templateinterger[1]
#                 regfolder=templateinterger[2]
                
#                 UseSM=templatebool[0]
#                 Type=templatetype[0]

#                 spillinfos=re.findall(r"\d+",contents[i+2])
#                 loadspill=spillinfos[2]
#                 storespill=spillinfos[1]

#                 meminfos=re.findall(r"\d+",contents[i+3])
#                 regusage=meminfos[0]
#                 smusage=meminfos[1]
#                 print("kernel_general",arch.group(0),
#                     Type, UseSM, regfolder, 
#                     regusage,smusage,storespill,loadspill)
#                 spillnum=int(loadspill)+int(storespill)
#                 print(spillnum!=0)
#                 # print(templateinterger)
#                 # print(templatebool)
#                 # print(templatetype)
#             i+=4
#         else:
#             i+=1    




import subprocess
# process = subprocess.Popen(['nvcc', ' -gencode arch=compute_80,code=sm_80 -Xptxas "-v -dlcm=cg" -DGEN,HALO=1 ./jacobi-star.cu'],
#                      stdout=subprocess.PIPE, 
#                      stderr=subprocess.PIPE)
# stdout, stderr = process.communicate()
# # process.run()
# stdout, stderr
# subprocess.run(["ls", "-l"])
# import os
# stream = os.popen(' nvcc -gencode arch=compute_80,code=sm_80 -Xptxas "-v -dlcm=cg" -DGEN,HALO=1 -dlink ./jacobi-star.cu')
# output = stream.read()
# output

archstring="arch=compute_80,code=sm_80"
halostring=1
regfolderstring=16
realstring="float"
useSM=",USESM"

basicstring="nvcc --cubin -gencode {0} -Xptxas \"-v -dlcm=cg\" -I./js2d5pt ./jacobi-general.cu"

generated_string=basicstring.format(archstring,halostring,regfolderstring,realstring,useSM)
print(generated_string)

subprocess = subprocess.Popen(generated_string, shell=True, stdout=subprocess.PIPE)
subprocess_return = subprocess.stdout.read()
print(subprocess_return)