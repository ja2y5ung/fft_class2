# 2021 05 07 18:29 새로운 일의 시작
# 2021 05 08 01:39 그래프 시각적으로 표시하는데 까지 했고, 신호 생성, 에러 출력, 데이터 저장만 하면 끝날 수도 있다.
# 2021 05 09 00:19 그래프를 그리는데 반복문을 230만번 반복하는데에서 문제가 생겻다. 넘파이로 빠르게 해결 할수 있기를
# 2021 05 09 09:21 먼가 느린 부분이 있는데 어찌 수정했으면 좋겟음..
# 2021 05 10 04:01 sicpy FFT가 더 빠르다고 함. 데이터 오프셋을 사용한는게 좋을거 같음. 잘라낸 구간 안에도 오프셋을 넣어 보자
# 2021 05 10 04:35 그래프 합성하다가 잠, 문제는 그래프가 점점 작아지고 합성결과가 정확한지 확인해야 할
# 2021 05 10 16:28 반복문을 사용한 코드에서 넘파이 매트릭스 연산으로 바꿈, 조카튼 딥카피를 햇는데도 갑이 바뀐다
# 2021 05 11 02:49 자고싶다. 에러 그래프 그리고 데이터 저장만 어떻게 하면 될거 같다..
# 2021 05 11 03:57 오.. 시바.. 데이터 저장까지 어찌 될거 같다.. 잘 수 있ㄷ.. 04:10 파일 저장 잘됫고.. 이제 잠.
import numpy as np
from matplotlib import pyplot as plt
import warnings
from numpy import sin, pi
from scipy.fftpack import fft as sfft, fftfreq, ifft


warnings.filterwarnings(action='ignore')




class FuckMePlz:

    oFile           = 0     # file
    oData           = []    # dataㄴ
    oLngth          = 0     # lngth of data
    oDc             = 0     # dc of data
    Fs              = 0     # sampling frq

    oDc2            = 0

    oFft            = 0     # fft of data
    oAmp            = 0     # amp of data
    oPhs            = 0     # phs of data

    ampLst          = []    # amp list of section
    phsLst          = []    # phs list of section
    intrvlDataLst   = []    # cutting data
    intrvLst        = []    # valuse of interval
    resAmpLst       = []    # res for save
    maxLst          = []

    intrvlDataLst2  = []
    ampLst2         = []
    phsLst2         = []
    resAmpLst2      = []

    mData           = 0    # 데이터를 16등분하고 평균을 구한 값
    mData2          = 0

    fig1            = 0     # show orignal data
    fig2            = 0     # show selected section
    fig3            = 0     # show fft of selected section
    fig4            = 0     # show new signal
    fig5            = 0     # show error

    e               = 0
    Y               = 0
    eY              = 0

    cntSmpl         = 0

    

    def __init__(self):
        pass



    # 파일 불러오기 inpt[ 데이터 경로 ] -> outpt[ 데이터 프레임 ]
    def loadFile(self, _path = 'smallData.txt' ):
        print('파일 불러오는 중..')
        self.oFile = np.loadtxt( _path, dtype = np.float16  )
        print('파일 불러오기 완료')


        
    # 데이터 선택하기 inpt[ 데이터의 번호 ] -> outpt[ 선택한 데이터 ]
    def slctData(self, _num = [2,3] ):
        cntDtCol    = len( _num )
        res         = []

        for i in range( cntDtCol ):
            idx = _num[i]
            res.append( self.oFile[:, idx] )


        self.oData = res



    # 변수 초기화 inpt[ none ] -> outpt[ Data properties ]
    def initData(self, _a = 0):
        step = 144000
        fft = np.zeros((1,step), dtype = np.float16)
        dat = np.zeros((1,step), dtype = np.float16)
        dat2 = np.zeros((1,step), dtype = np.float16)
        temp = np.zeros((step))
        cnt = len(self.oData[0])//step
        
        for i in range(cnt+1):
            if i != cnt:
                srt     = i*step
                end     = srt + step
                fft[0:step] = fft[0:step] + sfft(self.oData[0][srt:end], axis = 0 ) / step
                dat[0:step] = dat[0:step] + self.oData[0][srt:end]
                dat2[0:step] = dat2[0:step] + self.oData[1][srt:end]
            else:
                srt     = i*step
                end     = len(self.oData[0])-1
                fft[0:step] = fft[0:step] + (sfft(self.oData[0][srt:],n=step, axis = 0 ) / len(self.oData[0][srt:]) ).reshape(1,step)
                temp[:(end-srt+1)] = self.oData[0][srt:]
                dat[0:step] = dat[0:step] + temp
                temp[:(end-srt+1)] = self.oData[1][srt:]
                dat2[0:step] = dat2[0:step] + temp
        fft = fft / (cnt+1)
        
       
        print("변수 초기화 시작")
        #self.oLngth     = len( self.oData[0] )
        self.oLngth      = 144000
        self.Fs         = 12800
        self.oDc        = dat.mean()
        self.oDc2       = self.oData[1].mean()
        self.oData[0]   = self.oData[0] - self.oDc
        half            = self.oLngth // 2

        self.mData  = dat
        self.mData2  = dat2

        self.oFft       = sfft( self.oData[0], axis = 0 ) / self.oLngth
        self.oAmp       = 2 * abs( self.oFft[0:half] )
        self.oPhs       = np.angle( self.oFft[0:half], deg = False)
        print('변수 초기화 완료')



    # 데이터 출력 inpt[ none ] -> outpt[ show graph ]
    def showData(self):
        self.fig1 = plt.figure("원본 데이터")

        
        end = self.oLngth
        num = end
        t   = np.linspace(0, end, num, endpoint = False)


        p   = self.fig1.add_subplot(1,1,1)
        p   .plot(t, self.mData[0] )
        p   .set_xlabel('Number of samples')
        p   .set_ylabel('x(N) - dc')
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
        p   .plot(t, self.mData[0] )
        p   .set_xlabel('Number of sample')
        p   .set_ylabel('x(N) - dc')
        

        pltLgn          = ['Original']
        intrvlDataLst   = []
        intrvlDataLst2  = []
        cntIntrvl       = len( _intrvl ) // 2
        data            = self.mData[0]
        data2           = self.mData2[0]
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
            intrvlDataLst2.append( data2[srt:end] * _mult[i] )
            
            p.plot(cutNum, intrvlDataLst[i])
            pltLgn.append('Section ' + chr(65+i))

            
        # outpt[ cutting data ]
        self.intrvlDataLst  = intrvlDataLst
        self.intrvlDataLst2 = intrvlDataLst2
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
        data2       = self.intrvlDataLst2

        fftLst      = []
        ampLst      = []
        phsLst      = []

        fftLst2     = []
        ampLst2     = []
        phsLst2     = []
        
        # 섹션 갯수 만큼 반복
        for i in range( cntIntrvl ):
            half    = len( data[i] ) // 2

            offSet  = data[i].mean()

            fft     = sfft(data[i] - offSet, axis = 0) / ( half*2 )
            amp     = 2 * abs( fft[0:half] )
            phs     = np.angle( fft[0:half], deg = False)

            fft2    = sfft(data2[i] - self.oDc2, axis = 0 ) / (half*2)
            amp2    = 2 * abs( fft2[0:half] )
            phs2    = np.angle( fft2[0:half] )
             

            srt     = _intrvl[2*i]
            end     = _intrvl[2*i+1]
            num     = int( end - srt )

            cut     = np.linspace(srt, end, num, endpoint = False)


            fftLst.append( fft )
            ampLst.append( np.copy(amp) )
            phsLst.append( phs )

            fftLst2.append( fft2 )
            ampLst2.append( amp2 )
            phsLst2.append( phs2 )

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

        self.ampLst2 = ampLst2
        self.phsLst2 = phsLst2

        print('범위 선택한 구간 fft 완료')


    # 섹션에서 진폭 범위 선택하기 inpt[ Hz intervl list, multple value list] -> outpt[ clc amp data, graph] 
    def getFft(self, _intrvl = [0,20], _mult = [1]):
        self.clcFft(self.intrvLst)
        print('선별 시작')
        self.fig3   = plt.figure('선택한 구간의 FFT')
        amp         = self.ampLst.copy()
        phs         = self.phsLst

        amp2        = self.ampLst2
        phs2        = self.phsLst2


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
                amp2[i][srt:end] = self.ampLst2[i][srt:end] * _mult[ i*cntMult + j]
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
        self.resAmpLst2 = amp2
        print('선별 완료')


    # 신호 생성 inpt[ cnt samples ] -> outpt[ generate signal ]
    def genSgnl(self, _cntSmpl = 2500, _dc = 0 ):
        self.fig4   = plt.figure('합성 결과')
        plt.clf()
        self.cntSmpl = _cntSmpl
        print("신호 생성중..")
        cntIntrvl   = len( self.intrvlDataLst )

        Y           = 0
        Y2          = 0
        eY          = 0
        resY        = 0

        # 시계열에서  선택된 구간의 갯수 만큼 반복 
        for i in range( cntIntrvl ):
            
            data    = self.intrvlDataLst[i]
            data2   = self.intrvlDataLst2[i]
            lngth   = len( data ) // 2
            amp     = self.resAmpLst[i]
            phs     = self.phsLst[i]

            amp2    = self.resAmpLst2[i]
            phs2     = self.phsLst2[i]


            f       = 1 / self.oLngth
            t       = np.arange(0, _cntSmpl, 1)
            et      = np.arange(0, self.oLngth)
            n       = np.arange(0, lngth, 1)
            tn      = n.reshape(lngth,1)



            #메모리 크기 조절 하는 부분
            v = 1000
            mcnt = _cntSmpl // v
            
            for i in range(50):
                print(  str(round(i/50,2)*100) + '%'   )
                A = amp.reshape(lngth,1)[i*v:i*v+v]
                w = 2 * pi * f * tn[i*v:i*v+v]
                q = phs.reshape(lngth,1)[i*v:i*v+v]+ (pi/2)
                Y = Y + A * sin( w*t + q )

                A2  = amp2.reshape(lngth,1)[i*v:i*v+v]
                w2 = 2 * pi * f * tn[i*v:i*v+v]
                q2 = phs2.reshape(lngth,1)[i*v:i*v+v]+ (pi/2)
                Y2 = Y2 + A2 * sin( w2*t + q2 )


            Y = Y.sum(axis = 0)
            Y2 = Y2.sum(axis = 0 )

        self.Y  = Y + _dc
        self.Y2 = Y + _dc

        self.inptDc = _dc

        p       = self.fig4.add_subplot(1,1,1)
        p.plot(self.Y)
        p.set_title('Generate signal')
        p.set_xlabel('Number of samples')
        p.set_ylabel('x(N) + input dc')
        plt.grid(True)


        self.fig4.show()
        print("신호 생성 완료")

    # show error graph inpt[none] -> outpt[ error graph]
    def showError(self):
        self.fig5   = plt.figure('에러 그래프')
        plt.clf()
        p           = self.fig5.add_subplot(1,1,1)
        t           = np.linspace(0,self.oLngth, self.oLngth, endpoint = False )
        f           = 1 / self.oLngth
        cntIntrvl   = len( self.intrvlDataLst )
        Y = 0
        print('에러 계산중...')
        for i in range( cntIntrvl ):
            
            data    = self.intrvlDataLst[i]
            lngth   = len( data ) // 2

            amp     = self.resAmpLst[i]
            phs     = self.phsLst[i]
            n       = np.arange(0, lngth, 1)
            tn      = n.reshape(lngth,1)
            et      = np.arange(0, self.oLngth)

            v = 1000
            for i in range(50):
                print(  str(round(i/50,2)*100) + '%'   )
                A = amp.reshape(lngth,1)[i*v:i*v+v]
                w = 2 * pi * f * tn[i*v:i*v+v]
                q = phs.reshape(lngth,1)[i*v:i*v+v]+ (pi/2)
                Y = Y + A * sin( w*et + q )

            Y = Y.sum(axis = 0) + self.inptDc
            
        
        p.plot(t, self.mData[0] + self.oDc)
        p.plot(t, Y, 'r')
        p.set_xlabel('Number of sampls')
        p.set_ylabel('x(N)')
        p.set_title('error')
        p.legend(['Oiginal', 'Generated + dc'])

        plt.grid(True)

        e = (self.mData[0] - Y)**2
        e = np.sqrt(e.mean())
        self.e = e
        self.fig5.show()
        print('에러 계산 완료')


    # 파일 저장
    def saveFile(self):
        self.oFile = self.oFile[0:self.cntSmpl]
        self.oFile[0:self.cntSmpl,1:2] = self.Y.reshape( self.cntSmpl,1)
        self.oFile[0:self.cntSmpl,2:3] = self.Y2.reshape( self.cntSmpl,1)
        np.savetxt('saveFile.txt', self.oFile, fmt = '%1.5f     ')
                
               
if __name__ == '__main__':
    fuck = FuckMePlz()
    # 데이터 준비
    fuck.loadFile()
    fuck.slctData()
    fuck.initData()
    fuck.showData()
    
    # 데이터 작업
    fuck.getIntrvl([0, 100000], [1])
    fuck.getFft([0, 100000//2],[1])
    fuck.genSgnl(100000,0)
    fuck.showError()
        
                
        
        
    
