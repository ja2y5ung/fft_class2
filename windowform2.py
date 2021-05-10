import tkinter as tk
from tkinter import filedialog
from ttkwidgets.frames import ScrolledFrame
from back2 import FuckMePlz

class windowform2():

    def __init__(self):
        self.work2 = FuckMePlz()

        self.window = tk.Tk()
        self.window.title('control')
        self.window.geometry("600x900+50+50")
        self.window.resizable(True,True)
        self.mainMenu = tk.Menu(self.window)

        self.window.config(menu = self.mainMenu)
        self.fileMenu = tk.Menu(self.mainMenu, tearoff=0)
        self.mainMenu.add_cascade(label = "파일", menu = self.fileMenu)
        self.fileMenu.add_command(label = "열기", command = self.open_file)
        self.fileMenu.add_command(label = "저장 하기", command = self.save_file, state = "disable")
        self.fileMenu.add_command(label = "끝내기", command = self.exit_file)        

        self.fileMenu2 = tk.Menu(self.mainMenu, tearoff=0)
        self.mainMenu.add_cascade(label = "기능", menu = self.fileMenu2)
        self.fileMenu2.add_command(label = "예측 신호 생성", command = self.range_select,state = "disable")
        
        #filename frame
        self.fileframe = tk.Frame(self.window, width=300, height = 100, relief="solid", bd=1)
        self.fileframe.pack(side="top",fill="both")
        #tool_frame
        self.tool_frame = ScrolledFrame(self.window, compound=tk.RIGHT)
        self.tool_frame.pack(side="left",fill='both')
        #value_frame
        self.value_frame = ScrolledFrame(self.window, compound=tk.RIGHT)
        self.value_frame.pack(side="right",fill='both')
        #number of range frame
        self.range_frame = tk.Frame(self.tool_frame.interior, width=300, height = 150)
        #choice range frame
        self.nb_range_frame = tk.Frame(self.tool_frame.interior, width=300, height = 150)
        #filename_label
        self.filename_text = tk.StringVar(self.fileframe)
        self.filename_text.set(" file = 파일을 열어주세요.")

        self.text_label_input(self.fileframe,self.filename_text,'left')
        self.window.mainloop()

    def open_file(self):
        
        self.filename = filedialog.askopenfilenames(initialdir = "E:/Images", title = "파일선택",
                                               filetypes = (("txt files", "*.txt"), ("all files", "*.*")))
        self.filename_text.set(" file = " + str(self.filename[0]))
        self.fileMenu2.entryconfig(0,state = "normal")
        self.work2.loadFile(self.filename[0])
        self.work2.slctData()
        self.work2.initData()
        self.work2.showData()
        
    def save_file(self):
        self.Y = self.work.saveSgnl()

        self.YFrame = pd.DataFrame(self.Y, columns = ['genSgnl'])
        self.filename2 = filedialog.asksaveasfilename(initialdir = "E:/Images", title = "경로 선택",
                                               filetypes = (("csv files", "*.csv"), ("all files", "*.*")))
        self.YFrame.to_csv(str(self.filename2),index=False)

    def exit_file(self):
        self.window.quit()
        self.window.destroy()

    def widget_clear(self, widget):
        widget.pack_forget()

    def text_label_input(self,window,text,loc = 'top'):
        label = tk.Label(window, textvariable = text)
        label.pack(side = loc)

    def label_input(self,window,string,loc = "top"):
        label = tk.Label(window, text = string)
        label.pack(side = loc)

    def text_input(self, window, string,wid,loc1,loc2):
        self.label_input(self.range_frame,"  ")
        self.label_input(window, string,loc2)
        text_box = tk.Entry(window, width = wid)
        text_box.pack(side = loc1)
        return text_box

    def text_input2(self, window,window2, string,wid,loc1,loc2):
        self.label_input(window, string,loc2)
        text_box = tk.Entry(window2, width = wid)
        text_box.pack(side = loc1)
        return text_box

    def button_input(self,window,string,cmnd,loc,wid=10):
        button = tk.Button(window, width = wid,text = string,command = cmnd)
        button.pack(side=loc)

    def range_select(self):
        self.widget_clear(self.nb_range_frame)
        self.widget_clear(self.range_frame)
        
        self.count = 0
        self.range_frame=tk.Frame(self.tool_frame.interior, width=300, height = 150)
        self.range_frame.pack(side="top",expand = True)
        self.range_text_box = self.text_input(self.range_frame,"  ● 시계열 범위 갯수 입력 (1이상 정수만 입력)",10,"top","top")
        self.rf_buttonframe=tk.Frame(self.range_frame, width=300, height = 350)
        self.rf_buttonframe.pack(side="bottom")        
        self.button_input(self.rf_buttonframe,"입   력",self.number_range,"left")

    def number_range(self):
        self.widget_clear(self.rf_buttonframe)
        
        a = self.range_text_box.get()
        if self.range_text_box.get() != '' and self.count == 0 and int(self.range_text_box.get()) > 0:
            self.list1,self.list2,self.list3 = [], [], []
            self.nb_range_frame=tk.Frame(self.tool_frame.interior, width=300, height = 350)
            self.nb_range_frame.pack(side="top",fill = 'x')
            self.label_input(self.nb_range_frame,"  ","top")
            self.label_input(self.nb_range_frame," ● 시계열 범위 입력","top")
            for i in range(int(self.range_text_box.get())):
                rng_exp_frame=tk.Frame(self.nb_range_frame, width=300, height = 350)
                self.list2.append(rng_exp_frame)
            for j in range(int(self.range_text_box.get())):
                self.label_input(self.nb_range_frame,"- " + chr(j+65) + " section - ","top")
                rng_box = self.text_input2(self.list2[j], self.list2[j]," 범위 : ",10,"left","left")
                exp_box = self.text_input2(self.list2[j], self.list2[j],"   확대 비율 : ",10,"left","left")
                self.label_input(self.list2[j]," ","left")
                self.list2[j].pack(side="top",fill = 'x')
                self.list1.append(rng_box)
                
            self.label_input(self.nb_range_frame,"○ 100,200처럼 범위 사이를\n 쉼표로 구분 해주세요.","top")
            self.label_input(self.nb_range_frame,"○ 0~" + str(self.work.lngthData) +" 사이로 입력해주세요.","top")
            self.nrf_buttonframe=tk.Frame(self.nb_range_frame, width=300, height = 350)
            self.nrf_buttonframe.pack(side="bottom")  
            self.button_input(self.nrf_buttonframe,"입   력",self.hz_range_num,"left")             
            self.rf_buttonframe=tk.Frame(self.range_frame, width=300, height = 350)
            self.rf_buttonframe.pack(side="bottom")              
            self.button_input(self.rf_buttonframe,"갯수 리셋",self.range_select,"left")
        else:
            pass

    def hz_range_num(self):
        pass




    
window2 = windowform2()

























