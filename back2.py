# 2021 05 07 18:29 새로운 일의 시작

import numpy as np
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings(action='ignore')




class FuckMePlz:

    file            = 0     # file
    data            = []    # data
    lngth           = 0     # lngth of data
    dc              = 0     # dc of data
    Fs              = 0     # sampling frq

    fft             = 0     # fft of data
    amp             = 0     # amp of data
    phs             = 0     # phs of data

    ampLst          = []    # amp list of section
    phsLst          = []    # phs list of section
    intrvlDataLst   = []

    fig1            = 0     # show orignal data
    fig2            = 0     # show selected section
    fig3            = 0     # show fft of selected section
    fig4            = 0 

    

    def __init__(self):
        pass

    # 파일 불러오기 inpt[ path ] -> outpt[ file ]
    def loadFile(self, _path = 'fucking_data.txt' ):
        self.file = np.loadtxt( _path )


        
    # 데이터 선택하기 inpt[ col of data ] -> outpt[ data ]
    def slctData( self, _num = [1, 2] ):
        cnt     = len( _num )
        data    = []
        
        for i in range( cnt ):
            idx = _num[i]
            
            data.append( self.file[:, idx] )
            
        self.data = data



    # 변수 초기화 inpt[ none ] -> outpt[ Data properties ]
    def initData(self):
        self.lngth  = len( self.data[0] )
        self.Fs     = 12800
        self.dc     = self.data[0].mean()

        half        = self.lngth // 2

        self.fft    = np.fft.fft( self.data[0], axis = 0 )
        self.amp    = 2 * abs( self.fft[0:half] )
        self.phs    = np.angle( self.fft[0:half], deg = False)



    # 데이터 출력 inpt[ none ] -> outpt[ show graph ]
    def showData(self):
        self.fig1   = plt.figure("원본 데이터")
        p           = self.fig1.add_subplot(1,1,1)
        t           = np.linspace(0, self.lngth, self.lngth, endpoint = False)
        
        p.plot(t, self.data[0] )
        p.set_xlabel('Number of samples')
        p.set_ylabel('x(N)')
        p.set_title('Original')
        
        plt.grid()
        self.fig1.show()



    # 데이터 범위 선택하기 inpt[ intrvl value(samples) ] -> outpt[ cut data, show graph ]
    def getIntrvl(self, _intrvl = [30000, 100000, 1700000, 2200000], _mult = [1, 1]):
        self.fig2   = plt.figure('원 데이터에서 선택한 구간')
        plt .cla()
        
        p           = self.fig2.add_subplot(1,1,1)
        t           = np.linspace(0, self.lngth, self.lngth, endpoint = False)
        
        p.plot(t, self.data[0] )
        p.set_xlabel('Number of sample')
        p.set_ylabel('x(N)')
        plt.grid()

        pltLgn          = ['Original']
        intrvlDataLst    = []
        cntIntrvl       = len( _intrvl ) // 2
        data            = self.data[0]

        # 섹션 수 만큼 반복
        for i in range( cntIntrvl ):
            srt     = _intrvl[2*i]
            end     = _intrvl[2*i + 1]

            # 입력 범위에 문제가 생기면
            if ( srt > end or srt > self.lngth or end > self.lngth ):
                return -1


            num     = int( end - srt )
            cutNum  = np.linspace(srt, end, num, endpoint = False)

            intrvlDataLst.append( data[srt:end] * _mult[i] )


            p.plot(cutNum, intrvlDataLst[i])
            pltLgn.append('Section ' + chr(65+i))

            
        # outpt[ cutting data ]
        self.intrvlDataLst = intrvlDataLst


        plt.grid()
        plt.legend(pltLgn)
        self.fig2.tight_layout()
        self.fig2.show()

        # next prossesing[ FFT ]
        self.getFft(_intrvl)



    # 잘라낸 구간의 FFT 구하기 inpt[ intrvl data list ] -> outpt[ fft of cutting data, graph ]
    def getFft(self, _intrvl):
        self.fig3   = plt.figure('선택한 구간의 FFT')
        cntIntrvl   = len( _intrvl ) // 2
        data        = self.intrvlDataLst

        fftLst      = []
        ampLst      = []
        phsLst      = []

        # 섹션 갯수 만큼 반복
        for i in range( cntIntrvl ):
            half    = len( data[i] ) // 2
            
            fft     = np.fft.fft(data[i], axis = 0)
            amp     = 2 * abs( fft[0:half] )
            phs     = np.angle( fft[0:half], deg = False)

            srt     = _intrvl[2*i]
            end     = _intrvl[2*i+1]
            num     = int( end - srt )

            cut     = np.linspace(srt, end, num, endpoint = False)

            fftLst.append( fft )
            ampLst.append( amp )
            phsLst.append( phs )

            p = self.fig3.add_subplot(2, cntIntrvl, 1 + 2*i - i )
            p.plot(cut, data[i])
            p.set_title('Section ' + chr(65+i))
            p.set_xlabel('Interval samples')
            p.set_ylabel('x(N)')
            plt.grid()

            p = self.fig3.add_subplot(2, cntIntrvl, 1 + 2*i - i + cntIntrvl)
            p.stem(amp, markerfmt = 'none')
            p.set_xlabel('Point[Hz]')
            p.set_ylabel('∣X(P)∣')
            plt.grid()

        self.fig3.tight_layout()
        self.fig3.show()

        # outpy[ List of FFT ]
        self.ampLst = ampLst
        self.phsLst = phsLst


    # 섹션에서 진폭 범위 선택하기 inpt[ Hz intervl list, multple value list] -> outpt[ clc amp data, graph] 
    def slctFft(self, _intrvl = [2000, 3000, 10000, 15000], _mult = [0.9, 1.1]):
        fig3        = plt.figure('선택한 구간의 FFT')

        amp         = self.ampLst
        phs         = self.phsLst
        
        cntIntrvl   = len( amp )
        cntMuti     = len( _mult )

        # 섹션 갯수 만큼 반복
        for i in range( cntIntrvl ):
            p       = self.fig3.add_subplot(2, cntIntrvl, 1 + 2*i  - i + cntIntrvl)
            lngth   = len( amp[i] ) 
            Hz      = np.linspace(0, lngth, lngth, endpoint = False)
        
                
if __name__ == '__main__':
    fuck = FuckMePlz()
    # 데이터 준비
    fuck.loadFile()
    fuck.slctData()
    fuck.initData()
    fuck.showData()
    
    # 데이터 작업
    fuck.getIntrvl()
        
                
        
        
    
