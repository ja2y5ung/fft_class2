# 최선을 다해서 수정함.. 메모리 사용률에 따른 다이나믹한 연산을 하고 싶었지만.. 못했다
# 파일 저장 기능을 추가 해야
import numpy as np
from numpy.fft import fft
from numpy import zeros, array, pi, sin
from matplotlib import pyplot as plt
import warnings
#import psutil
from sys import getsizeof

warnings.filterwarnings(action='ignore')
#PSU = psutil.Process()


Frequency   = 12800
VRT         = 10000
HRZ         = 900

class fuckMe:

    oFile           = 0
    oData           = 0
    oLngth          = 0
    oMean           = 0

    data            = 0
    lngth           = 0
    mean            = 0

    Fs              = 0
    Fr              = 0


    intrvl          = 0
    intrvlData      = 0 #전체 구간이 0번째에 포함됨
    intrvlDataMean  = 0

    ampLst          = [] #전체 구간에 대한 amp 없음
    phsLst          = []

    copyAmp         = []

    Y               = 0
    e               = 0


    
    def __init__(self):
        pass


    
    def loadFile(self, _path = 'fucking_data.txt'):
        print('데이터 불러오는 중..')
        self.oFile = np.loadtxt(_path)#
        print('데이터 불러오기 완료')


        
    def slctData(self, _num = [1,2]):
        lngthData   = len(self.oFile)
        cntData     = len(_num)
        
        res1        = zeros((cntData, lngthData))
        res2        = zeros(cntData)
        
        for i in range(cntData):
            idx     = _num[i]
            res1[i] = self.oFile[:,idx] - res2[i]
            res2[i] = self.oFile[:,idx].mean()


        self.oMean = res2#
        self.oData = res1#
        self.oLngth= len(self.oData[0])


        
    def draw(self, _fig = 0, _x = [], _y = []):
        if _fig == 1:
            self.fig1 = plt.figure('원본 데이터와 선택된 구간')
            
            cnt = len(_y)
            
            self.fig1   = plt.figure('원본 데이터와 선택된 구간',figsize = (16,4))
            self.fig1.set_size_inches(16,4)

            for i in range(cnt):
                p   = self.fig1.add_subplot(1, 2, 1 + i)
                plt.cla()
                p   .set_xlabel('Number of samples')
                p   .set_ylabel('x(N)')
                p   .set_title('Original '+ chr(65+i))
                p   .plot(_x[i], _y[i])
                plt .grid(True)
            
            self.fig1.tight_layout()
            self.fig1.show()

            


                
        elif _fig == 2:
            #self.fig2   = plt.figure('원본 데이터와 서택된 구간', figsize = (16,4))
            #self.fig2.set_size_inches(16,4)
            
            
            cntData = len(self.oData)
            # 데이터 갯수
            for i in range(cntData):
                p    = self.fig1.add_subplot(1, 2, 1+i)
                pltLgn      = ['Original        Avg:' + str(self.oMean[i])]
                plt.cla()
                # 그려질 그래프 개수
                cnt = len(_y[i])
                for j in range(cnt):
                    p    .set_xlabel('Number of samples')
                    p    .set_ylabel('x(N)')
                    p    .set_title('Data'+str(i+1))
                    pltLgn.append('Section ' + chr(65 + j))
                    p    .plot(_x[i][j], _y[i][j])
                    plt.grid(True)
                    
                plt.legend(pltLgn)
                
            self.fig1.show()




            
        elif _fig == 3:         
            self.fig3   = plt.figure('선택된 구간의 FFT')
            plt.clf()
            bigOne = 0
            cntData = len(self.data)
            # 데이터 갯수
            for i in range(cntData):
                
                cntIntrvl = len(self.intrvl[i]) // 2
                # 시계열 선택된 구간 갯수
                for j in range(cntIntrvl):
                    if cntIntrvl>bigOne:
                        bigOne = cntIntrvl
                    
                    self.fig3.set_size_inches(8*bigOne,5*cntData)
                    p   = self.fig3.add_subplot(4, cntIntrvl, 1+j + i*cntData*cntIntrvl)
                    p   .set_title('Data' + str(i+1) + ' section' + chr(65+j))
                    p   .set_xlabel('Interval number of samples')
                    if j == 0: 
                        p   .set_ylabel('x(N)')
                    p   .plot(_x[i][j], self.intrvlData[i][j+1])
                    plt .grid(True)



                    p   = self.fig3.add_subplot(4, cntIntrvl, 1+j + i*cntData*cntIntrvl + cntIntrvl)
                    p   .set_xlabel('Point(HZ)')
                    if j == 0: 
                        p   .set_ylabel('|∠X(P)|')

                    p   .stem(self.ampLst[i][j], markerfmt = 'none')
                    plt .grid(True)



            self.fig3.tight_layout()
            plt.subplots_adjust( top = 0.96, bottom = 0.05, wspace=0.133, hspace=0.512)
            self.fig3.show()
            

            
        elif _fig == 4:
            self.fig3   = plt.figure('선택된 구간의 FFT')
            bigOne = 0
            


            # 데이터 갯수
            cntData = len(self.data)
            for i in range(cntData):
                # 데이터에서 선택된 시계열 구간 갯수
                cntIntrvl = len(self.ampLst[i])
                
                if cntIntrvl>bigOne:
                        bigOne = cntIntrvl
                        
                self.fig3.set_size_inches(8*bigOne,5*cntData)
                
                for j in range(cntIntrvl):

                    p = self.fig3.add_subplot(4, cntIntrvl,1+j + i*cntData*cntIntrvl + cntIntrvl)
                    plt.cla()
                    if j == 0:
                        p   .set_ylabel('|∠X(P)|')
                    p .stem(self.ampLst[i][j], markerfmt = 'none')

    
                    # 데이터에서 선택된 시계열 구간에서 선택된 주파수 계열에서 선택된 갯수
                    cntFft = len(_y[i][j])
                    for k in range(cntFft):
                        p   .set_xlabel('Point[Hz]')
                        p .stem(_x[i][j][k],_y[i][j][k], linefmt = 'orange', markerfmt = 'none')
                        plt.grid(True)
            self.fig3.tight_layout()
            self.fig3.show()



        elif _fig == 5:
            self.fig4   = plt.figure('생성한 신호', figsize =(16,4))
            
            
            cntData     = len(self.data)
            for i in range(cntData):
                pltLgn      = []
                p = self.fig4.add_subplot(1,cntData,1+i)
                p.set_title('Generated signal')
                p.set_xlabel('Number of samples')
                if i == 0:
                    p.set_ylabel('x(N)')
                pltLgn.append('Generated data' + str(i+1) + ' + inputDC')
                p.plot(_x[i], _y[i])
                plt.legend(pltLgn)
                plt.grid(True)
            self.fig4.tight_layout()
            
            self.fig4.show()
                
            



            


        elif _fig == 6:
            cnt = len(_y)
            self.fig5 = plt.figure('에러률',figsize = (10,4))

            for i in range(cnt):
                p = self.fig5.add_subplot(1,1,1)
                pltLgn = ['Orignal', 'Generated + input dc']
                p.set_title('Error')
                p.set_xlabel('Number of samples')
                p.set_ylabel('x(N)')
                p.plot(_x[i], self.data[0] + self.mean[0])
                p.plot(_x[i],_y[i], 'r')
                plt.legend(pltLgn)
                plt.grid(True)
            self.fig5.tight_layout()
            self.fig5.show()


            
    def initData(self, maxLngth = 144000):
        lngthData   = len(self.oData[0]) #데이터 길이
        lngthRst    = lngthData % maxLngth #데이터의 나머지 길이 
        cnt         = lngthData // maxLngth #데이터 반복 횟수
        cntData     = len(self.oData) # 데이터 갯수

        if lngthData >= maxLngth:
            res                 = zeros((len(self.oData), maxLngth))
            tmp                 = self.oData[:,0:cnt*maxLngth].reshape(cntData, cnt, maxLngth).sum(axis = 1, keepdims = True).reshape(cntData,maxLngth)
            tmpRst              = self.oData[:,cnt*maxLngth:].reshape(cntData, 1, lngthRst).sum(axis = 1, keepdims = True).reshape(cntData,lngthRst)

            res[:,:]            = res[:,:] + tmp
            res[:,:lngthRst]    = res[:,:lngthRst] + tmpRst
            res[:,:]            = res[:,:] / (cnt + 1)
            
            self.mean           = res.mean(axis = 1, keepdims = True)#
            #self.data           = res - self.mean#
            self.data           = res
            self.lngth          = maxLngth#
        else:
            self.mean           = self.oData.mean(axis = 1, keepdims = True)#
            #self.data           = self.oData + self.mean#
            self.data           = self.oData
            self.lngth          = len(self.oData[0])#

        self.Fs                 = Frequency
        self.Fr                 = 1 / self.lngth


        
    def showData(self, original = False):
        if not original:
            end = self.lngth
            t   = np.linspace(0, end, end, endpoint = False)
            self.draw(1, [t,t], [self.data[0], self.data[1]])
            
        else:
            end = self.oLngth
            t   = np.linspace(0, end, end, endpoint = False)
            self.draw(1, [t,t], [self.data[0] , self.data[1]])

            
    def slctIntrvl(self, _intrvl = [41234, 75643], _scale = [1]):   
        tmpTime     = []
        tmpData     = []
        tmpMean     = []
        
        tmpD        = []
        tmpM        = []
        tmpT        = []

        
        cntData     = len(self.data)# 데이터 갯수
        for i in range(cntData):
            tmpTime     = [np.linspace(0, self.lngth, self.lngth, endpoint = False)]
            tmpData     = [self.data[i]]
            tmpMean     = [self.data[i].mean()]
            
            cntIntrvl   = len(_intrvl[i]) // 2 # 한 데이터에서 선택된 구간 갯수
            for j in range(cntIntrvl):
                srt     = _intrvl[i][2*j]
                end     = _intrvl[i][2*j + 1]
                
                # 범위가 잘못 입력된 경우
                if(srt > end or srt > self.lngth or end > self.lngth):
                    return -1

                num     = end - srt
                tmpTime .append(np.linspace(srt, end, num, endpoint = False))
                tmpData .append(self.data[i][srt:end]*_scale[i][j])
                tmpMean .append(tmpData[i].mean())
            
            tmpD.append(tmpData[:])
            tmpM.append(tmpMean[:])
            tmpT.append(tmpTime[:])

        # result
        self.intrvl         = _intrvl#시계열 선택된 범위
        self.intrvlData     = array(tmpD)#선택된 범위 안에 있는 데이터
        self.intrvlDataMean = array(tmpM)
        
        self.draw(2, tmpT, tmpD)
        
        # 다음 실행될 메서드
        self.clcFft()




        
    def clcFft(self):
        
        
        resA        = []
        resP        = []
        resT        = []
        resD        = []
        

        # 데이터 갯수
        cntData     = len(self.data)
        for i in range(cntData):
            tmpA      = []
            tmpP      = []
            tmpC      = []
            tmpD      = []
            # 시계열에서 선택된 구간 갯수
            cntIntrvl   = len(self.intrvl[i]) // 2
            for j in range(cntIntrvl):
                half    = len(self.intrvlData[i][j+1]) // 2
                tmpFft  = fft(self.intrvlData[i][j+1]) / len(self.intrvlData[i][j+1])           
                tmpAmp  = abs(tmpFft[:half])*2
                tmpPhs  = np.angle(tmpFft[:half], deg = False)
                tmpA    .append(tmpAmp)
                tmpP    .append(tmpPhs)
                
                srt     = self.intrvl[i][2*j]
                end     = self.intrvl[i][2*j+1]
                num     = end - srt
                cut     = np.linspace(srt, end, num, endpoint = False)
                tmpC  .append(cut)
                tmpD  .append(self.intrvlData[i][j])

                
                
            resA.append(tmpA)
            resP.append(tmpP)
            resT.append(tmpC)
            resD.append(tmpD)
        
            
        # result
        self.ampLst = resA#시계열 선택된 각 구간의 amp들 전체 구간 없음
        self.phsLst = resP#시계열 선택된 각 구간의 phs들 전체 구간 없음
        self.copyAmp = np.copy(resA)
        self.draw(3, resT, resD)
        


        
    def slctFft(self, _intrvl = [ [[0,10], [30,40,60,80] ], [ [0,70], [100, 120, 200, 240, 300,320], [400,450,500,600] ] ], _scale = [ [[1], [1,1]], [[1],[1,1,1],[1,1]]   ] ):
        #self.clcFft()#원본 유지를 위해 실행함

        resAmp          = []
        resPhs          = []
        resCut          = []

        # 데이터 갯수
        cntData = len(self.data)
        for i in range(cntData):
            tmpCut1 = []
            tmpAmp1 = []
            tmpPhs1 = []
            
            # 시계열에서 선택된 구간 갯수
            cntIntrvl   = len(self.ampLst[i])
            for j in range(cntIntrvl):
                tmpCut2 = []
                tmpAmp2 = []
                tmpPhs2 = []

                # fft에서 선택된 갯수
                cntFft  = len(_intrvl[i][j]) // 2
                for k in range(cntFft):
                    srt = _intrvl[i][j][2*k]
                    end = _intrvl[i][j][2*k+1]


                    tmpCut2.append(np.linspace(srt, end, end-srt, endpoint = False))
                    tmpAmp2.append(self.copyAmp[i][j][srt:end]*_scale[i][j][k])
                    tmpPhs2.append(self.phsLst[i][j][srt:end])

                tmpCut1.append(tmpCut2)
                tmpAmp1.append(tmpAmp2)
                tmpPhs1.append(tmpPhs2)
                
            resCut.append(tmpCut1)
            resAmp.append(tmpAmp1)
            resPhs.append(tmpPhs1)
            
        # result
        #self.ampLst[:][:][:] = self.copyAmp[i][j][srt:end]*_scale[i*cntIntrvlFft+j]#주파수 범위에서 선택된 amp들만 증폭
        self.ampLst[:][:][:] = resAmp[:][:][:]
        self.draw(4, resCut, resAmp)


        
    def genSgnl(self, _cntGenSmpl = 10000):
        print('신호 생성중..')
        tmpY = []
        tmpT = []

        # 데이터 갯수
        cntData = len(self.data)
        for i in range(cntData):
            # 데이터에서 선택된 시계열 갯수
            cntIntrvl = len(self.intrvl[i]) // 2
            for j in range(cntIntrvl):
                lngth   = len(self.ampLst[i][j])
                amp     = self.ampLst[i][j]
                phs     = self.phsLst[i][j]

                f       = 1 / lngth
                t       = np.arange(0, _cntGenSmpl, 1)
                n       = np.arange(0, lngth, 1).reshape(lngth, 1)


                # processing >>
                vrt = VRT
                hrz = HRZ
                vCnt = lngth // vrt
                hCnt = _cntGenSmpl // hrz

                tmp = zeros((vrt, _cntGenSmpl))
                print('신호 생성중..')
                # 세로 크기 vCnt+1
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
            tmpY.append(tmp.sum(axis = 0 ))
            tmpT.append(t)
        self.Y = tmpY#
        self.draw(5, tmpT, self.Y)
        
        print('신호 생성 완료')




    def getError(self):
        print('에러 계산중..')

        tmpY = []
        tmpT = []

        # 데이터 갯수
        cntData = len(self.data)
        for i in range(cntData):
            # 데이터에서 선택된 시계열 갯수
            cntIntrvl = len(self.intrvl[i]) // 2
            for j in range(cntIntrvl):
                lngth   = len(self.ampLst[i][j])
                amp     = self.ampLst[i][j]
                phs     = self.phsLst[i][j]

                f       = 1 / lngth
                t       = np.arange(0, self.lngth, 1)
                n       = np.arange(0, lngth, 1).reshape(lngth, 1)


                # processing >>
                vrt = VRT
                hrz = HRZ
                vCnt = lngth // vrt
                hCnt = _cntGenSmpl // hrz

                tmp = zeros((vrt, _cntGenSmpl))
                print('신호 생성중..')
                # 세로 크기 vCnt+1
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
            tmpY.append(tmp.sum(axis = 0 ))
            tmpT.append(t)
        self.Y = tmpY#
        self.draw(5, tmpT, self.Y)
        
        print('신호 생성 완료')

























        
        cntIntrvl = len(self.intrvlData)
        Y = 0
        resY = 0
        
        #시계열에서 선택된 갯수
        for k in range(cntIntrvl):
            lngth   = len(self.intrvlData[k]) // 2
            amp     = self.ampLst[k]
            phs     = self.phsLst[k]


            f       = 1 / len(self.intrvlData[k])
            t       = np.arange(0, self.lngth, 1)
            n       = np.arange(0, lngth, 1).reshape(lngth, 1)


            # processing >>
            vrt = VRT
            hrz = HRZ
            vCnt = lngth // vrt
            hCnt = self.lngth // hrz

            tmp = zeros((vrt, self.lngth))
            print('에러 계산중..')
            # 세로 크기 vCnt+1
            for i in range(vCnt+1):

                # 세로 계산
                if i != vCnt:
                    # 가로 크기 hCnt+1
                    for j in range(hCnt+1):
                        # 가로 계산
                        if j != hCnt:
                            A   = amp.reshape(lngth, 1)[i*vrt:i*vrt+vrt]
                            W   = 2*pi*f*n[i*vrt:i*vrt+vrt]
                            P   = phs.reshape(lngth, 1)[i*vrt:i*vrt+vrt] + (pi/2)
                            tmp[:, j*hrz:j*hrz+hrz] = tmp[:, j*hrz:j*hrz+hrz] + A*sin(W*t[j*hrz:j*hrz+hrz] + P)

                        # 가로 나머지 계산
                        else:
                            A   = amp.reshape(lngth, 1)[i*vrt:i*vrt+vrt]
                            W   = 2*pi*f*n[i*vrt:i*vrt+vrt]
                            P   = phs.reshape(lngth, 1)[i*vrt:i*vrt+vrt] + (pi/2)
                            tmp[:, j*hrz:] = tmp[:, j*hrz:] + A*sin(W*t[j*hrz:] + P)

                            
                # 세로 나머지 계산
                else:

                    # 가로 크기 hCnt+1
                    for j in range(hCnt+1):
                        # 가로 계산
                        if j != hCnt:
                            A       = amp.reshape(lngth, 1)[i*vrt:]
                            W       = 2*pi*f*n[i*vrt:]
                            P       = phs.reshape(lngth, 1)[i*vrt:] + (pi/2)
                            tmp[:len(A),j*hrz:j*hrz+hrz] = tmp[:len(A),j*hrz:j*hrz+hrz] + A*sin(W*t[j*hrz:j*hrz+hrz] + P)

                        # 가로 나머지 계산
                        else:
                            A       = amp.reshape(lngth, 1)[i*vrt:]
                            W       = 2*pi*f*n[i*vrt:]
                            P       = phs.reshape(lngth, 1)[i*vrt:] + (pi/2)
                            tmp[:len(A),j*hrz:j*hrz+hrz] = tmp[:len(A),j*hrz:j*hrz+hrz] + A*sin(W*t[j*hrz:j*hrz+hrz] + P)
        


                    
        

        #Result
        eY      = tmp.sum(axis = 0) + self.mean[0]
        e       = (self.data[0] - eY)**2
        self.e  = np.sqrt(e.sum())


        self.draw(6,[t], [eY])

        print('에러 계산 완료')

        
         



if __name__ == '__main__':

    fuck = fuckMe()
    fuck.loadFile()
    fuck.slctData()
    fuck.initData()
    fuck.showData()

    #fuck.slctIntrvl([0,5000, 10000,20000],[1,1])
    #fuck.slctFft([0,100,1000,2000,0,200,1000,4000], [1,1,1,1])
    
    #fuck.slctIntrvl([0,5000,  10000,20000,  6000,7000],[1,1,1])
    #fuck.slctFft([0,100,1000,2000,  0,200,1000,4000,  0,100,300,500], [1,1,  1,1,  1,1])

    #fuck.slctIntrvl([0,12800], [1])
    #fuck.slctFft([0,12800//2], [1])
    #fuck.genSgnl(12800)

    fuck.slctIntrvl([[0,1000, 10000,13000], [50000,55000,70000,90000,100000,125000]],[[1, 1], [1, 1, 1]])
    fuck.slctFft()
    fuck.genSgnl(14400)

    
    #fuck.getError()
