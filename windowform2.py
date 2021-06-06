import tkinter as tk
from tkinter import filedialog
from ttkwidgets.frames import ScrolledFrame
from ff2_210601 import back

class windowform2():

    def __init__(self):
        self.work2 = back()

        self.window = tk.Tk()
        self.window.title('control')
        self.window.geometry("500x900+50+50")
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
        self.fileMenu2.add_command(label = "예측 신호 생성", command = self.data_choice, state = "disable")
        
        #filename frame
        self.fileframe = tk.Frame(self.window, width=300, height = 100, relief="solid", bd=1)
        self.fileframe.pack(side="top",fill="both")
        
        self.file_name_frame = tk.Frame(self.fileframe, width=300, height = 85)
        self.file_name_frame.pack(side="left",fill="both")
        
        self.file_tool_frame = tk.Frame(self.fileframe, width=300, height = 85)
        self.file_tool_frame.pack(side="right",fill="both")

        
        #tool_frame
        self.tool_frame = ScrolledFrame(self.window, compound=tk.RIGHT)
        self.tool_frame.pack(side="left",fill='both')
        #value_frame
        self.value_frame = ScrolledFrame(self.window, compound=tk.RIGHT)
        self.value_frame.pack(side="right",fill='both')
        #phase select frame
        self.phase_frame = tk.Frame(self.tool_frame.interior, width=300, height = 150)
        #number of range frame
        self.range_frame = tk.Frame(self.tool_frame.interior, width=300, height = 150)
        #choice range frame
        self.nb_range_frame = tk.Frame(self.tool_frame.interior, width=300, height = 150)
        #number of hz range frame
        self.hzrangeSlctframe = tk.Frame(self.tool_frame.interior, width=300, height = 150)
        #choice hz range frame
        self.hz_range_frame = tk.Frame(self.tool_frame.interior, width=300, height = 150)
        #sample choice frame
        self.samplechoiceframe = tk.Frame(self.tool_frame.interior, width=300, height = 150)
        #error graph frame
        self.errorGraframe = tk.Frame(self.tool_frame.interior, width=300, height = 350)
        #filename_label
        self.filename_text = tk.StringVar(self.file_name_frame)
        self.filename_text.set(" - file : 파일을 열어주세요.")
        #value_label
        self.dc = tk.StringVar(self.value_frame);self.dc.set("0")
        self.sr = tk.StringVar(self.value_frame);self.sr.set("0")
        self.er = tk.StringVar(self.value_frame);self.er.set("None")
        self.sp = tk.StringVar(self.value_frame);self.sp.set("0")
        self.fr = tk.StringVar(self.value_frame);self.fr.set("0")
        self.ipdc = tk.StringVar(self.value_frame);self.ipdc.set("0")
        

        self.text_label_input(self.file_name_frame, self.filename_text,'top')
        self.window.mainloop()

    def open_file(self):
        
        self.filename = filedialog.askopenfilenames(initialdir = "E:/Images", title = "파일선택",
                                               filetypes = (("txt files", "*.txt"), ("all files", "*.*")))
        self.filename_text.set(" - file : " + str(self.filename[0]))
        
        self.start_file_num = tk.StringVar(self.file_name_frame)
        self.start_file_num.set(" - Skip Header Line : None")
        self.text_label_input(self.file_name_frame,self.start_file_num,'top','nw')

        self.label_input(self.value_frame.interior,"< DC value >")
        self.text_label_input(self.value_frame.interior, str(self.dc))

        self.label_input(self.value_frame.interior,"  ")
        self.label_input(self.value_frame.interior,"< sampling rate >")
        self.text_label_input(self.value_frame.interior, str(self.sr))

        self.label_input(self.value_frame.interior,"  ")
        self.label_input(self.value_frame.interior,"< One point of Hz >")
        self.text_label_input(self.value_frame.interior, str(self.sp))

        self.label_input(self.value_frame.interior,"  ")
        self.label_input(self.value_frame.interior,"< Input DC Value >")
        self.text_label_input(self.value_frame.interior, str(self.ipdc))

        self.label_input(self.value_frame.interior,"  ")
        self.label_input(self.value_frame.interior,"< error >")
        self.text_label_input(self.value_frame.interior, str(self.er))

        self.start_data()
        self.fileMenu.entryconfig(0,state = "disable")
        
    def start_data(self):
        self.start_data_box = self.text_input(self.file_tool_frame," < skip header(정수 입력) > ",5,"top","top")
        self.button_input(self.file_tool_frame,"입력",self.slct_data,"top")
        self.label_input(self.file_tool_frame, " ","top")
        self.CheckVar1 = tk.IntVar()
        self.c1 = tk.Checkbutton(self.file_name_frame, text = "If you are viewing generated data, please cheak here",variable = self.CheckVar1)
        self.c1.pack()
        
    def slct_data(self):
        self.widget_clear(self.file_tool_frame)
        self.file_tool_frame = tk.Frame(self.fileframe, width=300, height = 85)
        self.file_tool_frame.pack(side="right",fill="both")

        self.c1.config(state="disabled")
        print(self.CheckVar1.get())
        
        self.start_file_num.set(" - Skip Header Line : " + str(self.start_data_box.get()))
        self.work2.loadFile(self.filename[0],int(self.start_data_box.get()))
        self.combobox(self.file_tool_frame)        

    def save_file(self):
        self.filename2 = filedialog.asksaveasfilename(initialdir = "E:/Images", title = "경로 선택",
                                               filetypes = (("csv files", "*.csv"), ("all files", "*.*")))
        
        self.work2.saveFile(self.filename2)
        
    def exit_file(self):
        self.window.quit()
        self.window.destroy()

    def widget_clear(self, widget):
        widget.pack_forget()

    def text_label_input(self,window,text,loc = 'top',loc2 = 'center'):
        label = tk.Label(window, textvariable = text)
        label.pack(side = loc, anchor = loc2)

    def label_input(self,window,string,loc = "top"):
        label = tk.Label(window, text = string)
        label.pack(side = loc)

    def text_input(self, window, string,wid,loc1,loc2):
        self.label_input(window,"  ")
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

    def combobox(self, window):
        values = ['1. 1A-Keyphasor','2. 1A-DisX','3. 1A-DisY','4. 1A-AccX','5. 1A-AccY','6. 1A-AccZ']
        self.combobox = tk.ttk.Combobox(window, height=15, values = values)
        self.label_input(window,"< Data Select >")
        self.combobox.set("데이터를 선택 해주세요.")
        self.combobox.pack()
        self.combobox.bind("<<ComboboxSelected>>", self.callbackFunc)
        self.label_input(window,"   ")

    def callbackFunc(self,event):
        self.widget_clear(self.tool_frame)
        self.fileMenu2.entryconfig(0,state = "normal")
        self.tool_frame = ScrolledFrame(self.window, compound=tk.RIGHT)
        self.tool_frame.pack(side="left",fill='both')
        
        self.num = [int(self.combobox.get().split('. ')[0]) - 1]
        self.work2.slctData(self.num)
        self.dc.set(self.work2.mean)
        self.sr.set(self.work2.Fs)
        self.sp.set(self.work2.Fr)
        self.work2.showData()

    def data_choice(self):
        if self.num[0] == 2:
            self.phase_select()
        else:
            self.range_select()

    def phase_select(self):
        self.widget_clear(self.phase_frame)
        self.phase_frame=tk.Frame(self.tool_frame.interior, width=300, height = 150)
        self.phase_frame.pack(side="top",expand = True)
        self.phase_text_box = self.text_input(self.phase_frame,"  ● 페이즈 값 입력(degree값)",10,"top","top")
        self.ph_buttonframe=tk.Frame(self.phase_frame, width=300, height = 350)
        self.ph_buttonframe.pack(side="bottom")        
        self.button_input(self.ph_buttonframe,"입   력",self.range_select,"left")        
        
    
    def range_select(self):
        self.widget_clear(self.nb_range_frame)
        self.widget_clear(self.range_frame)
        self.widget_clear(self.hzrangeSlctframe)
        self.widget_clear(self.hz_range_frame)
        self.widget_clear(self.samplechoiceframe)

        if self.num[0] == 2:
            self.work2.slctPhs(float(self.phase_text_box.get()))
        
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
                self.list3.append(exp_box)
                
            self.label_input(self.nb_range_frame,"○ 100,200처럼 범위 사이를\n 쉼표로 구분 해주세요.","top")
            self.label_input(self.nb_range_frame,"○ 0~" + str(self.work2.oLngth) +" 사이로 입력해주세요.","top")
            self.nrf_buttonframe=tk.Frame(self.nb_range_frame, width=300, height = 350)
            self.nrf_buttonframe.pack(side="bottom")  
            self.button_input(self.nrf_buttonframe,"입   력",self.hz_range_num,"left")             
            self.rf_buttonframe=tk.Frame(self.range_frame, width=300, height = 350)
            self.rf_buttonframe.pack(side="bottom")              
            self.button_input(self.rf_buttonframe,"갯수 리셋",self.range_select,"left")
        else:
            pass

    def hz_range_num(self):
        self.widget_clear(self.hzrangeSlctframe)
        self.widget_clear(self.hz_range_frame)
        
        start_end_list,exp_list = [], []
        for st_en_box in self.list1:
            a = list(map(int, st_en_box.get().split(',')))
            for i in a:
                start_end_list.append(i)
        for e_box in self.list3:
            exp_list.append(float(e_box.get()))
            
        self.work2.slctIntrvl(start_end_list,exp_list)
        
        self.hzrangeSlctframe = tk.Frame(self.tool_frame.interior, width=300, height = 150)
        self.hzrangeSlctframe.pack(side="top",fill = 'x')
        self.hz_range_text_box = self.text_input(self.hzrangeSlctframe,"  ● 주파수 범위 갯수 입력 (1이상 정수만 입력)   ",10,"top","top")
        self.hzrbuttonframe = tk.Frame(self.hzrangeSlctframe, width=300, height = 350)
        self.hzrbuttonframe.pack(side="bottom")        
        self.button_input(self.hzrbuttonframe,"입   력",self.hz_range,"left")

    def hz_range(self):
        self.widget_clear(self.hzrbuttonframe)
        self.chrcount = 0;self.chlistcount = 0
        
        b = self.hz_range_text_box.get()
        if self.hz_range_text_box.get() != '' and self.count == 0 and int(self.hz_range_text_box.get()) > 0:
            self.hzlist1, self.hzlist2, self.hzlist3 = [], [], []
            self.hz_range_frame = tk.Frame(self.tool_frame.interior, width=300, height = 350)
            self.hz_range_frame.pack(side="top",fill = 'x')
            self.label_input(self.hz_range_frame,"  ","top")
            self.label_input(self.hz_range_frame," ● 주파수 범위 입력","top")
            for i in range( int(self.range_text_box.get()) * int(self.hz_range_text_box.get())):
                rng_exp_frame2 = tk.Frame(self.hz_range_frame, width=300, height = 350)
                self.hzlist2.append(rng_exp_frame2)
            for j in range( int(self.range_text_box.get())):
                for k in range(int(self.hz_range_text_box.get())):
                    self.label_input(self.hz_range_frame,"- " + chr(self.chrcount+65) + " section - ","top")
                    rng_box2 = self.text_input2(self.hzlist2[self.chlistcount], self.hzlist2[self.chlistcount]," 범위 : ",10,"left","left")
                    exp_box2 = self.text_input2(self.hzlist2[self.chlistcount], self.hzlist2[self.chlistcount],"   확대 비율 : ",10,"left","left")
                    self.label_input(self.hzlist2[self.chlistcount]," ","left")
                    self.hzlist3.append(exp_box2)
                    self.hzlist2[self.chlistcount].pack(side="top",fill = 'x')
                    self.hzlist1.append(rng_box2)
                    self.chlistcount += 1
                self.chrcount+=1
                
            self.label_input(self.hz_range_frame,"○ 100,200처럼 범위 사이를\n 쉼표로 구분 해주세요.","top")
            for i in range(len(self.work2.maxFft)):
                self.label_input(self.hz_range_frame,"○ 0~" + str(self.work2.maxFft[i]) +" 사이로 입력해주세요.","top")

            self.nrf_buttonframe=tk.Frame(self.hz_range_frame, width=300, height = 350)
            self.nrf_buttonframe.pack(side="bottom")  
            self.button_input(self.nrf_buttonframe,"입   력",self.sample_choice,"left")             
            self.hzrbuttonframe = tk.Frame(self.hzrangeSlctframe, width=300, height = 350)
            self.hzrbuttonframe.pack(side="bottom")              
            self.button_input(self.hzrbuttonframe,"갯수 리셋",self.hz_range_num,"left")
        else:
            pass


    def sample_choice(self):
        self.widget_clear(self.samplechoiceframe)
        self.widget_clear(self.errorGraframe) 
        start_end_list2,exp_list2,count_list,count_list2 = [], [], [], []
        count,count2 = 0, 0 
        for st_en_box2 in self.hzlist1:
            if count == int(self.hz_range_text_box.get()):
                start_end_list2.append(count_list)
                count_list = []
                count = 0
            a = list(map(int, st_en_box2.get().split(',')))
            for j in a:
                count_list.append(j)
            count += 1
        start_end_list2.append(count_list)
        print(start_end_list2)

        for e_box in self.hzlist3:
            if count2 == int(self.hz_range_text_box.get()):
                exp_list2.append(count_list2)
                count_list2 = []
                count2 = 0
            count_list2.append(float(e_box.get()))
            count2 += 1
        exp_list2.append(count_list2)
        print(exp_list2)

        self.work2.slctFft(start_end_list2,exp_list2)        
        self.samplechoiceframe = tk.Frame(self.tool_frame.interior, width=300, height = 150)
        self.samplechoiceframe.pack(side="top",fill = 'x')
        self.sample_text_box = self.text_input(self.samplechoiceframe,"  ● 샘플 갯수 입력  ",10,"top","top")
        self.ipdc_text_box = self.text_input(self.samplechoiceframe,"  ● Input DC 값 입력  ",10,"top","top")
        
        self.samplebuttonframe = tk.Frame(self.samplechoiceframe, width=300, height = 350)
        self.samplebuttonframe.pack(side="bottom")        
        self.button_input(self.samplebuttonframe,"입   력",self.confirm,"left")
        self.errorGraframe = tk.Frame(self.hzrangeSlctframe, width=300, height = 350)

    def confirm(self):
        self.widget_clear(self.errorGraframe)   
        self.work2.genSgnl(int(self.sample_text_box.get()),float(self.ipdc_text_box.get()))
        self.ipdc.set(float(self.ipdc_text_box.get()))
        self.errorGraframe = tk.Frame(self.samplebuttonframe, width=300, height = 350)
        self.errorGraframe.pack(side="bottom")
        self.button_input(self.errorGraframe,"에러 그래프",self.error_show,"left")        

    def error_show(self):
        self.work2.error()
        self.er.set(self.work2.e)
        self.fileMenu.entryconfig(1,state = "normal")

        
window2 = windowform2()

























