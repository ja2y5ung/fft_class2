# 2021 05 07 18:29 새로운 일의 시작
# 2021 05 08 01:39 그래프 시각적으로 표시하는데 까지 했고, 신호 생성, 에러 출력, 데이터 저장만 하면 끝날 수도 있다.
# 2021 05 09 00:19 그래프를 그리는데 반복문을 230만번 반복하는데에서 문제가 생겻다. 넘파이로 빠르게 해결 할수 있기를
# 2021 05 09 09:21 먼가 느린 부분이 있는데 어찌 수정했으면 좋겟음..
# 2021 05 10 04:01 sicpy FFT가 더 빠르다고 함. 데이터 오프셋을 사용한는게 좋을거 같음. 잘라낸 구간 안에도 오프셋을 넣어 보자
# 2021 05 10 04:35 그래프 합성하다가 잠, 문제는 그래프가 점점 작아지고 합성결과가 정확한지 확인해야 할
# 2021 05 10 16:28 반복문을 사용한 코드에서 넘파이 매트릭스 연산으로 바꿈, 조카튼 딥카피를 햇는데도 갑이 바뀐다 
# 2021 05 11 12:50 일단 밤새기 전날 코드 백업해
import numpy as np
from matplotlib import pyplot as plt
import warnings
from numpy import sin, pi
from scipy.fftpack import fft as sfft, fftfreq, ifft


warnings.filterwarnings(action='ignore')




class FuckMePlz:

    oFile           = 0     # file
    oData           = []    # data
    oLngth          = 0     # lngth of data
    oDc             = 0     # dc of data
    Fs              = 0     # sampling frq

    oFft            = 0     # fft of data
    oAmp            = 0     # amp of data
    oPhs            = 0     # phs of data

    ampLst          = []    # amp list of section
    phsLst          = []    # phs list of section
    intrvlDataLst   = []    # cutting data
    intrvLst        = []    # valuse of interval
    resAmpLst       = []    # res for save
    maxLst          = []

    fig1            = 0     # show orignal data
    fig2            = 0     # show selected section
    fig3            = 0     # show fft of selected section
    fig4            = 0     # show new signal
    fig5            = 0     # show error

    

    def __init__(self):
        pass



    # 파일 불러오기 inpt[ path ] -> outpt[ file ]
    def loadFile(self, _path = 'smallData.txt' ):
        print('파일 불러오는 중..')
        self.oFile = np.loadtxt( _path )
        print('불러오기 완료')


        
    # 데이터 선택하기 inpt[ col of data ] -> outpt[ data ]
    def slctData( self, _num = [2,3] ):
        cntDtCol    = len( _num )
        res         = []

        
        for i in range( cntDtCol ):
            idx = _num[i]
            res.append( self.oFile[:, idx] )


        self.oData = res



    # 변수 초기화 inpt[ none ] -> outpt[ Data properties ]
    def initData(self):
        print("변수 초기화 시작")
        self.oLngth = len( self.oData[0] )
        self.Fs     = 12800
        self.oDc    = self.oData[0].mean()
        self.oData[0] = self.oData[0] - self.oDc
        half        = self.oLngth // 2

        self.oFft   = sfft( self.oData[0], axis = 0 ) / self.oLngth
        self.oAmp   = 2 * abs( self.oFft[0:half] )
        self.oPhs   = np.angle( self.oFft[0:half], deg = False)
        print('변수 초기화 완료')



    # 데이터 출력 inpt[ none ] -> outpt[ show graph ]
    def showData(self):
        self.fig1 = plt.figure("원본 데이터")

        
        end = self.oLngth
        num = end
        t   = np.linspace(0, end, num, endpoint = False)


        p   = self.fig1.add_subplot(1,1,1)
        p   .plot(t, self.oData[0] )
        p   .set_xlabel('Number of samples')
        p   .set_ylabel('x(N)')
        p   .set_title('Original')

        plt.grid(True)
        self.fig1.show()



    # 데이터 범위 선택하기 inpt[ intrvl value(samples) ] -> outpt[ cut data, show graph ]
    def getIntrvl(self, _intrvl = [0,10], _mult = [1]):
        self.fig2   = plt.figure('원 데이터에서 선택한 구간')
        
        end = self.oLngth
        num = end
        t   = np.linspace(0, end, num, endpoint = False)
        
        p   = self.fig2.add_subplot(1,1,1)
        plt.cla()
        p   .plot(t, self.oData[0] )
        p   .set_xlabel('Number of sample')
        p   .set_ylabel('x(N) - dc')
        

        pltLgn          = ['Original']
        intrvlDataLst   = []
        cntIntrvl       = len( _intrvl ) // 2
        data            = self.oData[0]
        self.maxLst     = []
        
        # 섹션 수 만큼 반복
        for i in range( cntIntrvl ):
            srt     = _intrvl[2*i]
            end     = _intrvl[2*i + 1]
            self.maxLst.append(end-srt)
            # 입력 범위에 문제가 생기면
            if ( srt > end or srt > self.oLngth or end > self.oLngth ):
                return -1


            num     = int( end - srt )
            cutNum  = np.linspace(srt, end, num, endpoint = False)

            intrvlDataLst.append( data[srt:end] * _mult[i] )
            
            p.plot(cutNum, intrvlDataLst[i])
            pltLgn.append('Section ' + chr(65+i))

            
        # outpt[ cutting data ]
        self.intrvlDataLst  = intrvlDataLst
        self.intrvLst       = _intrvl



        plt.grid(True)
        plt.legend(pltLgn)
        self.fig2.tight_layout()
        self.fig2.show()

        # next prossesing[ FFT ]
        self.clcFft(_intrvl)



    # 잘라낸 구간의 FFT 구하기 inpt[ intrvl data list ] -> outpt[ fft of cutting data, graph ]
    def clcFft(self, _intrvl):
        self.fig3   = plt.figure('선택한 구간의 FFT')
        print('범위 선택한 구간 fft 시작')
        plt.clf()
        cntIntrvl   = len( _intrvl ) // 2
        data        = self.intrvlDataLst

        fftLst      = []
        ampLst      = []
        phsLst      = []

        # 섹션 갯수 만큼 반복
        for i in range( cntIntrvl ):
            half    = len( data[i] ) // 2

            offSet  = data[i].mean()

            fft     = sfft(data[i] - offSet, axis = 0) / ( half*2 )
            amp     = 2 * abs( fft[0:half] )
            phs     = np.angle( fft[0:half], deg = False)

            srt     = _intrvl[2*i]
            end     = _intrvl[2*i+1]
            num     = int( end - srt )

            cut     = np.linspace(srt, end, num, endpoint = False)


            fftLst.append( fft )
            ampLst.append( np.copy(amp) )
            phsLst.append( phs )

            p = self.fig3.add_subplot(2, cntIntrvl, 1 + 2*i - i )
            plt.cla()
            p.plot(cut, data[i] - offSet)
            p.set_title('Section ' + chr(65+i))
            p.set_xlabel('Number of samples')
            p.set_ylabel('x(N) - dc')
            plt.grid(True)

            p = self.fig3.add_subplot(2, cntIntrvl, 1 + 2*i - i + cntIntrvl)
            plt.cla()
            p.stem(amp, markerfmt = 'none')
            p.set_xlabel('Point[Hz]')
            p.set_ylabel('∣X(P)∣')
            plt.grid(True)

            
        self.fig3.tight_layout()
        self.fig3.show()

        # outpy[ List of FFT ]
        self.ampLst = ampLst
        self.phsLst = phsLst

        print('범위 선택한 구간 fft 완료')


    # 섹션에서 진폭 범위 선택하기 inpt[ Hz intervl list, multple value list] -> outpt[ clc amp data, graph] 
    def getFft(self, _intrvl = [0,20], _mult = [1]):
        self.clcFft(self.intrvLst)
        print('선별 시작')
        self.fig3        = plt.figure('선택한 구간의 FFT')
        amp         = self.ampLst.copy()
        phs         = self.phsLst



        cntIntrvl   = len( amp ) 
        cntMult     = len( _mult ) // 2

        # 섹션 갯수 만큼 반복
        for i in range( cntIntrvl ):
            p       = self.fig3.add_subplot(2, cntIntrvl, 1 + 2*i  - i + cntIntrvl)
            plt.cla()
            lngth   = len( amp[i] ) 
            Hz      = np.linspace(0, lngth, lngth, endpoint = False)

            # 섹션에서 선택된 범위 갯수 만큼 반복
            for j in range( cntMult ):
                srt             = _intrvl[2*i]
                end             = _intrvl[2*i+1]
                amp[i][srt:end] = self.ampLst[i][srt:end] * _mult[ i*cntMult + j]

                p.stem(Hz, amp[i], markerfmt = 'none')
                plt.grid(True)

                # 여러 구간에 색 칠한 그래프 그릴려고 다시 반복
                for k in range( cntMult ):
                    srt     = _intrvl[2*i+k*cntMult] 
                    end     = _intrvl[2*i+k*cntMult+1]
                    cutHz   = np.linspace(srt, end, end - srt, endpoint = False )
                    p.stem(cutHz, self.ampLst[i][srt:end], linefmt = 'orange', markerfmt = 'none' )
                    plt.grid(True)
                    
            # 섹션에서 선택된 범위가 한개인 경우
            if( cntMult == 0 ):
                srt             = _intrvl[i*2]
                end             = _intrvl[i*2 + 1]
                cutHz           = np.linspace(srt, end, end - srt, endpoint = False)
                amp[i][srt:end] = self.ampLst[i][srt:end] * _mult[i]
                
                p.stem(Hz, amp[i], markerfmt = 'none')
                p.stem(cutHz, amp[i][srt:end], linefmt = 'orange', markerfmt = 'none')
                p.set_xlabel('Point[Hz]')
                p.set_ylabel('|∠X(P)|')
                plt.grid(True)

                
        self.fig3.show()

        #outpt[ res for A*Sin( wt +q ) ]
        self.resAmpLst = amp
        print('선별 완료')


    # 신호 생성 inpt[ cnt samples ] -> outpt[ generate signal ]
    def genSgnl(self, _cntSmpl = 2500 ):
        self.fig4   = plt.figure('합성 결과')
        plt.clf()

        cntIntrvl   = len( self.intrvlDataLst )

        Y           = 0
        eY          = 0

        # 시계열에서  선택된 구간의 갯수 만큼 반복 
        for i in range( cntIntrvl ):
            
            data    = self.intrvlDataLst[i]
            lngth   = len( data ) // 2

            amp     = self.resAmpLst[i]
            phs     = self.phsLst[i]


            f       = 1 / self.oLngth
            t       = np.arange(0, _cntSmpl, 1)
            et      = np.arange(0, self.oLngth)
            n = np.arange(0, lngth, 1)
            tn = n.reshape(lngth,1)
            


            A = amp.reshape(lngth,1)
            w = 2 * pi * f * tn
            q = phs.reshape(lngth,1)+ (pi/2)
            Y = Y + A * sin( w*t + q )
            
            eY = eY + A*sin(w*et + q )
            Y = Y.sum(axis = 0)
            eY = eY.sum(axis = 0 )




        self.Y  = Y + self.oDc
        self.eY = eY + self.oDc


        p       = self.fig4.add_subplot(1,1,1)
        #p.plot(self.oData[0] + self.oDc)
        p.plot(self.eY)
        
        p.set_title('Generate signal')
        p.set_xlabel('Number of samples')
        p.set_ylabel('x(N)')
        plt.grid(True)



        e       = ( (self.oData[0] + self.oDc) - (self.eY.T))**2
        e       = np.sqrt( e.T.mean() )
        self.e  = e


        self.fig4.show()

    # show error graph inpt[none] -> outpt[ error graph]
    def showError(self):
        self.fig5   = plt.figure('에러 그래프')
        plt.clf()
        p           = self.fig5.add_subplot(1,1,1)
        t           = np.linspace(0,self.oLngth, self.oLngth, endpoint = False )
        
        breakpoint()
        p.plot(t, self.oData[0] + self.oDc)
        p.plot(t, self.eY, 'r')
        p.set_xlabel('Number of sampls')
        p.set_ylabel('x(N)')
        p.set_title('error')
        p.legend(['Oiginal', 'Generated'])

        plt.grid(True)

        self.fig5.show()


                
               
if __name__ == '__main__':
    fuck = FuckMePlz()
    # 데이터 준비
    fuck.loadFile()
    fuck.slctData()
    fuck.initData()
    fuck.showData()
    
    # 데이터 작업
    fuck.getIntrvl([0, 1000], [1])
    fuck.getFft([20, 50], [2])
    fuck.genSgnl(2500)
        
                
        
        
    
