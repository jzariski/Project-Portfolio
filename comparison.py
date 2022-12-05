import os
import numpy as np
import sys
import time
from pathlib import Path
#import matplotlib.pyplot as plt
import statistics as s
#import matplotlib.colors as colors
import numpy.linalg as lg
sys.path.append('../shocktubecalc')
import sod

delt = 0.0001

class comparison:
    
    def __init__ (self):
    
        self.xdiskPath = "/xdisk/kkratter/jzariski/sodstuff/"
        self.parPath = "/home/u5/jzariski/Sod_Stuff/simple_shock_tube_calculator/shocktubecalc/comparisons/setups/sod1d/sod1d.par"
        self.parPath2 = "/home/u5/jzariski/Sod_Stuff/simple_shock_tube_calculator/shocktubecalc/comparisons/setups/sod1d/sod1d2.par"
        self.outputPath = "/home/u5/jzariski/Sod_Stuff/simple_shock_tube_calculator/shocktubecalc/comparisons/sodoutput.out"
        self.jobPath = "/home/u5/jzariski/public/"
        self.analytic = []
        
        
    def runJob(self):
        os.system('sbatch job.sh')


    def changePar(self,nz):
        og = open(self.parPath,'r')
        out = open(self.parPath2,'w')
        done1 = True
        for line in og:
            if line[0]=='N' and line[1]=='z' and done1:
                out.write('Nz                      '+str(nz)+'\n')
                done1 = False
            else:
                out.write(line)
        
        out.close()
        og.close()
        os.system("cp " + self.parPath2 + " " + self.parPath)
        os.system("rm " + self.parPath2)
        
        
    def getCalculated(self,size):
        nz = self.sizeToInt(size)
        positions, regions, values = sod.solve(left_state=(1, 1, 0), right_state=(0.125, 0.1, 0.),
                                           geometry=(-1., 1., 0), t=0.5, gamma=1.4, npts=nz)
                                           
        self.analytic = values['rho']                     
        self.changePar(nz)
        self.runJob()
        filepath = Path(self.xdiskPath+'gasdens2.dat')
        while not filepath.is_file():
            time.sleep(5)
            filepath = Path(self.xdiskPath+'gasdens2.dat')
            print('waiting')
        gasses = np.fromfile(self.xdiskPath+"gasdens2.dat")
        os.system('rm ' +self.xdiskPath+'gasdens2.dat')
        timeSpent = self.outputTime()
        return gasses, timeSpent
        
    def costFunc(self,numeric):
        return (1/len(numeric))*lg.norm(self.analytic-numeric)
        
    # Should be paired directly after getCaclulated
    # May combine them later
    def outputTime(self):
        time = 0
        filepath = Path(self.xdiskPath+'gasdens5.dat') # Making sure the whole simulation is done
        while not filepath.is_file():
            time.sleep(5)
            filepath = Path(self.xdiskPath+'gasdens5.dat')
        og = open(self.outputPath,'r')
        for line in og:
            if line[0]=='M' and line[1]=='A':
                time += float(line.split()[2])
        og.close()
        os.system('rm ' +self.xdiskPath+'gasdens5.dat')
        return time
    
    def intToSize(self, nz):
        return 1/nz
    
    def sizeToInt(self, size):
        return int(round(1/size))
        


def optFunc(cTool, size):
    values = cTool.getCalculated(size)
    mainNorm = cTool.costFunc(values[0])
    return mainNorm, values[1]
    
def df(cTool, f, size):
    fplus = f(cTool, size + delt)[0]
    fminus = f(cTool, size - delt)[0]
    return (fplus - fminus)/(2*delt)
    
    
def gradDesc(x0,f,gradf,tol, cTool):
    xk = np.copy(x0)
    gradVal = gradf(cTool, f,xk)
    regF = f(cTool, xk)
    currSize = int(round(1/xk))
    while lg.norm(gradVal) >tol and lg.norm(1-regF[1]) > tol:
        print('Size', int(round(1/xk)))
        deltax = -1*gradVal
        t = 0.5
        print('Old x', xk)
        print('New x', xk+t*deltax)
        while (xk+t*deltax) < 0:
            t = t / 10
            print('New x', xk+t*deltax)
        newReg = f(cTool, xk+t*deltax)
        while newReg[0]>=regF[0]:
            t = t*0.5
            newReg = f(cTool, xk+t*deltax)
        while newReg[1] > 1:
            t = t*0.5
            newReg = f(cTool, xk+t*deltax)
            print('making time smaller')
            print('time', newReg[1])
        xk = xk +t*deltax
        regF = newReg
        compSize = int(round(1/xk))
        print('currSize', currSize)
        print('compSize', compSize)
        if lg.norm(compSize - currSize) <= 2:
            return newReg
        currSize = compSize
        print('newReg', newReg[0])
        print('currTime', newReg[1])
        
    return newReg

compTool = comparison()

final = gradDesc(0.005,optFunc, df,1e-4,compTool)
print(int(round(1/final[0])), final[1])
    
    












