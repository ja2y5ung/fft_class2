#a

import numpy as np
from numpy import pi, sin, zeros, array
from numpy.fft import fft
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings(action='ignore')

Frequency   = 12800
VRT         = 8000
HRZ         = 900


class back:
  oFile = 0
  oLngth  = 0
  oMean = 0
  oAmp  = 0
  oPhs  = 0

  Fr  = 0
  Fs  = 0

  slctNum  = 0

  data = 0
  lngth = 0
  mean = 0
  amp = 0
  phs = 0

  intrvl  = 0
  intrvlData  = 0
  intrvlDataMean  = 0

    

  def __init__(self):
      pass


  def loadFile(self,  \
               _path = '',  \
               _skipRows = 0, \
               _maxRows = 2500):
    print('데이터 불러오는 중..')
    self.oFile = np.loadtxt(_path,  \
                            skiprows = _skipRows,  \
                            max_rows = _maxRows).T
    print('데이터 불러오기 완료')


  def initData(self, maxLngth = 144000):
  #변수 블럭
    lngthData  = self.oFile.shape[1]
    lngthRst = lngthData % maxLngth
    cnt = lngthData // maxLngth
    cntData = self.oFile.shape[0]

  #작업 블럭
    if lngthData >= maxLngth:
      resData   = zeros((cntData, maxLngth))
      tmp1      = self.oFile[:,0:cnt*maxLngth].reshape(cntData, cnt, maxLngth).sum(axis = 1, keepdims = True).reshape(cntData,maxLngth)
      tmp2      = self.oFile[:,cnt*maxLngth:].reshape(cntData, 1, lngthRst).sum(axis = 1, keepdims = True).reshape(cntData,lngthRst)
      
      resData[:,:]          = resData[:,:] + tmp1
      resData[:,:lngthRst]  = resData[:,:lngthRst] + tmp2
      resData[:,:]          = resData[:,:] / (cnt + 1)

      resLngth  = maxLngth
      resMean   = resData.mean(axis = 1, keepdims = True).reshape(cntData)

      tmpFFT    = fft(resData - resMean.reshape(cntData, 1)) / maxLngth
      resAmp    = 2*abs(tmpFFT)[:,:maxLngth//2]
      resPhs    = np.angle(tmpFFT, deg = False)[:,:maxLngth]

    else:
      resData   = zeros((cntData, lngthData))
      resData[:,:]  = self.oFile
      
      resLngth  = lngthData
      resMean   = resData.mean(axis = 1, keepdims = True).reshape(cntData)

      tmpFFT    = fft(resData - resMean.reshape(cntData, 1)) / resLngth
      resAmp    = 2*abs(tmpFFT)[:,:resLngth//2]
      resPhs    = np.angle(tmpFFT, deg = False)[:,:resLngth//2]


  #결과 블럭
    self.oData  = resData
    self.oLngth = resLngth
    self.oMean  = resMean
    self.oAmp   = resAmp
    self.oPhs   = resPhs
    self.Fs     = Frequency
    self.Fr     = 1/self.oLngth























    
  def scltData(self, _num = [0]):
  #변수 블럭
    num = _num[0]

  #작업 블럭
    resData   = self.oData[num]
    resMean   = self.oMean[num]
    resLngth  = self.oLngth
    resAmp    = self.oAmp[num]
    resPhs    = self.oPhs[num]
      
    if num != 2:
      resPhs    = self.oPhs[num]
    else:
      resPhs    = self.oPhs[1]
    
  #결과 블럭
    self.data   = resData - resMean
    self.lngth  = resLngth
    self.mean   = resMean
    self.amp    = resAmp
    self.phs    = resPhs

    self.copyData = np.copy(self.data)
    self.copyAmp  = np.copy(self.amp)
    self.copyPhs  = np.copy(self.phs)

    self.slctNum  = num




















  def showData(self, original = False):

  #변수블럭
    self.fig1 = plt.figure('원 데이터와 선택된 구간')
    end       = self.lngth
    x         = np.linspace(0, end, end, endpoint = False)
    y         = self.data + self.mean

  #작업 블럭
    p = self.fig1.add_subplot(1, 1, 1)
    p.plot(x,y)
    plt.grid(True)
    
  #결과 블럭
    self.fig1.tight_layout()
    self.fig1.show()
























  def slctIntrvl(self, _intrvl = [0,1], _scale = [1]):
  #변수 블럭
    self.fig1 = plt.figure('원 데이터와 선택된 구간')
    cntIntrvl = len(_intrvl) // 2
    y = []
    x = []
    m = []
      
    for i in range(cntIntrvl):
      srt = _intrvl[2*i]
      end = _intrvl[2*i + 1]
      num = end - srt
        
  #작업 블럭
      x.append(np.linspace(srt, end, num, endpoint = False))
      y.append((self.copyData[srt:end])*_scale[i])
      m.append(x[i].mean())


  #결과 블럭
      self.data[srt:end]  = self.copyData[srt:end]*_scale[i]

    self.intrvl = np.array(_intrvl)
    self.intrvlData = np.array(y)
    self.intrvlMean = np.array(m)

    p = self.fig1.add_subplot(1, 1, 1)
    plt.cla()
    plt.plot(self.data + self.mean)
    for i in range(cntIntrvl):
      p.plot(x[i], y[i] + self.mean)
      plt.grid(True)

    self.fig1.tight_layout()
    self.fig1.show()

    self.clcFft()




















  def clcFft(self):
  #변수 블럭
    self.fig2 = plt.figure('선택된 구간의 FFT')
    cntIntrvl = len(self.intrvl) // 2
    resAmp = []
    resPhs = []


    for i in range(cntIntrvl):
      half  = len(self.intrvlData[i]) // 2
      useData = self.intrvlData[i]

      srt = self.intrvl[2*i]
      end = self.intrvl[2*i + 1]
      num = end - srt

  #작업 블럭
      tmpFFT  = fft(useData) / (half*2)
      tmpAmp  = 2*abs(tmpFFT)[:half]
      tmpPhs  = np.angle(tmpFFT, deg = False)[:half]

      resAmp.append(tmpAmp)
      resPhs.append(tmpPhs)

  #결과 블럭
    self.ampLst = np.array(resAmp)
    self.phsLst = np.array(resPhs)
    self.copyAmp  = np.copy(self.ampLst)
    self.copyPhs  = np.copy(self.phsLst)


    for i in range(cntIntrvl):
      p = self.fig2.add_subplot(2, cntIntrvl, 1 + i)
      plt.cla()
      p.plot(self.intrvlData[i] + self.mean)
      plt.grid(True)

      p = self.fig2.add_subplot(2, cntIntrvl, 1 + i + cntIntrvl)
      plt.cla()
      p.stem(self.ampLst[i], markerfmt = 'none')
      plt.grid(True)

    self.fig2.tight_layout()
    self.fig2.show()
    




















      

  def slctFft(self, _intrvl = [0,1], _scale = [1]):
  #변수 블럭
    self.fig2 = plt.figure('선택된 구간의 FFT')
    cntIntrvl = len(self.intrvl) // 2
    x, x1 = [], []
    y, y1 = [], []

    for i in range(cntIntrvl):
      cntFFT = len(_intrvl[i]) // 2
      for j in range(cntFFT):
        srt = _intrvl[i][j]
        end = _intrvl[i][j+1]
        num = end - srt
        
  #작업 블럭
        res = self.copyAmp[i][srt:end]*_scale[i][j]

  #결과 블럭
    self.ampLst[i][srt:end] = res

        
    for i in range(cntIntrvl):
      cntFFT = len(_intrvl[i]) // 2
      for j in range(cntFFT):
        p = self.fig2.add_subplot(2, cntIntrvl, 1 + i + cntIntrvl)
        plt.cla()
        p.stem(self.ampLst[i], markerfmt = 'none')
        plt.grid(True)

    for i in range(cntIntrvl):
      cntFFT = len(_intrvl[i]) // 2
      for j in range(cntFFT):
        srt = _intrvl[i][j]
        end = _intrvl[i][j+1]
        num = end - srt
        x   = np.linspace(srt, end, num, endpoint = False)
        p = self.fig2.add_subplot(2, cntIntrvl, 1 + i + cntIntrvl)
        p.stem(x,self.ampLst[i][srt:end], linefmt = 'orange', markerfmt = 'none')
        plt.grid(True)

    self.fig2.tight_layout()
    self.fig2.show()

















  def slctPhs(self, _inptPhs = 0):
  #변수 블럭
    self.fig1 = plt.figure('원 데이터와 선택된 구간')
    _inptPhs  = np.deg2rad(_inptPhs)
    lngth = self.lngth//2
    amp = self.copyAmp
    phs = self.copyPhs + _inptPhs
    

    f   = 1 /self.lngth
    t     = np.arange(0, self.lngth, 1)
    n     = np.arange(0, lngth, 1).reshape(lngth, 1)

  #작업 블럭

    vrt = VRT
    hrz = HRZ
    vCnt  = lngth // vrt
    hCnt  = self.lngth // hrz

    tmp   = np.zeros((vrt, self.lngth))

    #가로 크기
    for ii in range(vCnt+1):
      # 세로 계산
      if ii != vCnt:

        # 가로 크기 hCnt+1
        for jj in range(hCnt+1):
          # 가로 계산
          if jj != hCnt:
            A   = amp.reshape(lngth, 1)[ii*vrt:ii*vrt+vrt]
            W   = 2*pi*f*n[ii*vrt:ii*vrt+vrt]
            P   = phs.reshape(lngth, 1)[ii*vrt:ii*vrt+vrt] + (pi/2)
            tmp[:, jj*hrz:jj*hrz+hrz] = tmp[:, jj*hrz:jj*hrz+hrz] + A*sin(W*t[jj*hrz:jj*hrz+hrz] + P)

          # 가로 나머지 계산
          else:
            A   = amp.reshape(lngth, 1)[ii*vrt:ii*vrt+vrt]
            W   = 2*pi*f*n[ii*vrt:ii*vrt+vrt]
            P   = phs.reshape(lngth, 1)[ii*vrt:ii*vrt+vrt] + (pi/2)
            tmp[:, jj*hrz:] = tmp[:, jj*hrz:] + A*sin(W*t[jj*hrz:] + P)

                  
      # 세로 나머지 계산
      else:

        # 가로 크기 hCnt+1
        for jj in range(hCnt+1):
          # 가로 계산
          if jj != hCnt:
            A       = amp.reshape(lngth, 1)[ii*vrt:]
            W       = 2*pi*f*n[ii*vrt:]
            P       = phs.reshape(lngth, 1)[ii*vrt:] + (pi/2)
            tmp[:len(A),jj*hrz:jj*hrz+hrz] = tmp[:len(A),jj*hrz:jj*hrz+hrz] + A*sin(W*t[jj*hrz:jj*hrz+hrz] + P)

          # 가로 나머지 계산
          else:
            A       = amp.reshape(lngth, 1)[ii*vrt:]
            W       = 2*pi*f*n[ii*vrt:]
            P       = phs.reshape(lngth, 1)[ii*vrt:] + (pi/2)
            tmp[:len(A),jj*hrz:jj*hrz+hrz] = tmp[:len(A),jj*hrz:jj*hrz+hrz] + A*sin(W*t[jj*hrz:jj*hrz+hrz] + P)

  
  #결과 블럭
    resY  = tmp.sum(axis = 0) + self.mean
    self.data   = resY

    p = self.fig1.add_subplot(1, 1, 1)
    plt.cla()
    p.plot(t,resY)
    plt.grid(True)
    self.fig1.tight_layout()
    self.fig1.show()
    






















  def genSgnl(self, _cntGenSmpl, _inptDC = 0):
    print('신호 생성중..')
  #변수 블럭
    self.fig3 = plt.figure('생성한 신호')
    resY = []
    cntIntrvl = len(self.intrvl)//2

    for i in range(cntIntrvl):
      lngth = len(self.ampLst[i])
      amp   = self.ampLst[i]
      phs   = self.phsLst[i]

      f     = 1/(lngth*2)
      t     = np.arange(0, _cntGenSmpl, 1)
      n     = np.arange(0, lngth, 1).reshape(lngth, 1)

  

      vrt = VRT
      hrz = HRZ
      vCnt  = lngth // vrt
      hCnt  = _cntGenSmpl // hrz

      tmp   = np.zeros((vrt, _cntGenSmpl))
  #작업 블럭
      #가로 크기
      for ii in range(vCnt+1):
        print('신호 생성중...' + str(round(ii/(vCnt+1)*100,1)) + '%')
        # 세로 계산
        if ii != vCnt:

          # 가로 크기 hCnt+1
          for jj in range(hCnt+1):
            # 가로 계산
            if jj != hCnt:
              A   = amp.reshape(lngth, 1)[ii*vrt:ii*vrt+vrt]
              W   = 2*pi*f*n[ii*vrt:ii*vrt+vrt]
              P   = phs.reshape(lngth, 1)[ii*vrt:ii*vrt+vrt] + (pi/2)
              tmp[:, jj*hrz:jj*hrz+hrz] = tmp[:, jj*hrz:jj*hrz+hrz] + A*sin(W*t[jj*hrz:jj*hrz+hrz] + P)

            # 가로 나머지 계산
            else:
              A   = amp.reshape(lngth, 1)[ii*vrt:ii*vrt+vrt]
              W   = 2*pi*f*n[ii*vrt:ii*vrt+vrt]
              P   = phs.reshape(lngth, 1)[ii*vrt:ii*vrt+vrt] + (pi/2)
              tmp[:, jj*hrz:] = tmp[:, jj*hrz:] + A*sin(W*t[jj*hrz:] + P)

                    
        # 세로 나머지 계산
        else:

          # 가로 크기 hCnt+1
          for jj in range(hCnt+1):
            # 가로 계산
            if jj != hCnt:
              A       = amp.reshape(lngth, 1)[ii*vrt:]
              W       = 2*pi*f*n[ii*vrt:]
              P       = phs.reshape(lngth, 1)[ii*vrt:] + (pi/2)
              tmp[:len(A),jj*hrz:jj*hrz+hrz] = tmp[:len(A),jj*hrz:jj*hrz+hrz] + A*sin(W*t[jj*hrz:jj*hrz+hrz] + P)

            # 가로 나머지 계산
            else:
              A       = amp.reshape(lngth, 1)[ii*vrt:]
              W       = 2*pi*f*n[ii*vrt:]
              P       = phs.reshape(lngth, 1)[ii*vrt:] + (pi/2)
              tmp[:len(A),jj*hrz:jj*hrz+hrz] = tmp[:len(A),jj*hrz:jj*hrz+hrz] + A*sin(W*t[jj*hrz:jj*hrz+hrz] + P)


  #결과 블럭
    resY  = tmp.sum(axis = 0) + _inptDC

    self.Y  = resY

    p = self.fig3.add_subplot(1, 1, 1)
    p.plot(self.Y)
    plt.grid(True)
    self.fig3.tight_layout()
    self.fig3.show()

    self.genSgnlRst(_cntGenSmpl)














  def genSgnlRst(self, _cntGenSmpl = 0):
    print('나머지 신호 생성중')
  #변수 블럭
    lngth = _cntGenSmpl // 2
    cntData = self.oFile.shape[0]
    amp   = self.oAmp
    phs   = self.oPhs


    f     = 1/(lngth*2)
    t     = np.arange(0, _cntGenSmpl, 1).reshape(1,_cntGenSmpl)
    n     = np.arange(0, lngth, 1).reshape(lngth, 1)

  
    vrt = VRT // cntData
    hrz = HRZ // cntData
    vCnt  = lngth // vrt
    hCnt  = _cntGenSmpl // hrz

    tmp   = np.zeros((cntData,vrt, _cntGenSmpl))


  #작업 블럭
    #가로 크기
    for ii in range(vCnt+1):
      print('나머지 신호 생성중..' + str(round(ii/(vCnt+1)*100,2)) + '%')
      # 세로 계산
      if ii != vCnt:

        # 가로 크기 hCnt+1
        for jj in range(hCnt+1):
          # 가로 계산
          if jj != hCnt:
            A   = amp.reshape(cntData, lngth, 1)[:,ii*vrt:ii*vrt+vrt]
            W   = 2*pi*f*n[ii*vrt:ii*vrt+vrt]
            P   = phs.reshape(cntData, lngth, 1)[:,ii*vrt:ii*vrt+vrt] + (pi/2)
            tmp[:,:, jj*hrz:jj*hrz+hrz] = tmp[:,:, jj*hrz:jj*hrz+hrz] + A*sin(W*t[:,jj*hrz:jj*hrz+hrz] + P)

          # 가로 나머지 계산
          else:
            
            A   = amp.reshape(cntData, lngth, 1)[:,ii*vrt:ii*vrt+vrt]
            W   = 2*pi*f*n[ii*vrt:ii*vrt+vrt]
            P   = phs.reshape(cntData, lngth, 1)[:,ii*vrt:ii*vrt+vrt] + (pi/2)
            tmp[:,:, jj*hrz:] = tmp[:,:, jj*hrz:] + A*sin(W*t[:,jj*hrz:] + P)

                  
      # 세로 나머지 계산
      else:

        # 가로 크기 hCnt+1
        for jj in range(hCnt+1):
          # 가로 계산
          if jj != hCnt:
            A       = amp.reshape(cntData, lngth, 1)[:,ii*vrt:]
            W       = 2*pi*f*n[ii*vrt:]
            P       = phs.reshape(cntData ,lngth, 1)[:,ii*vrt:] + (pi/2)
            tmp[:,:A.shape[1],jj*hrz:jj*hrz+hrz] = tmp[:,:A.shape[1],jj*hrz:jj*hrz+hrz] + A*sin(W*t[:,jj*hrz:jj*hrz+hrz] + P)

          # 가로 나머지 계산
          else:
            A       = amp.reshape(cntData, lngth, 1)[:,ii*vrt:]
            W       = 2*pi*f*n[ii*vrt:]
            P       = phs.reshape(cntData, lngth, 1)[:,ii*vrt:] + (pi/2)
            tmp[:,:A.shape[1],jj*hrz:jj*hrz+hrz] = tmp[:,:A.shape[1],jj*hrz:jj*hrz+hrz] + A*sin(W*t[:,jj*hrz:jj*hrz+hrz] + P)      



    resY  = tmp.sum( axis = 1) + self.oMean.reshape(cntData,1)

    resY[self.slctNum,:] = self.Y

    for i in range(cntData):
      self.fig4 = plt.figure('최종 결과')
      p = self.fig4.add_subplot(cntData,1,1+i)
      p.plot(resY[i])
      plt.grid(True)

    self.fig4.tight_layout()    
    self.fig4.show()
















                
    

if __name__ == '__main__':
  size = 2500
  back = back()
  back.loadFile('data.txt', 0, size)
  back.initData()
  back.scltData([2])
  back.showData()

##  back.slctIntrvl([0,size], [1])
##  back.slctFft([[0,size//2]], [[1]])
##  back.genSgnl(size, 10)

  back.slctPhs(0)
  back.slctIntrvl([0,100,2000,2200], [1, 1])
  back.slctFft([[0,10], [0,40]], [[1],[1]])
  back.genSgnl(size, 10)

  

















  
