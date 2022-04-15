# -*- coding: utf-8 -*-
"""
POLS 視野等表示 ver.9
Stand-alone

"""
# Import and setting ===========================================================================================
import numpy as np
from control import matlab
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
import pickle
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import messagebox
import tkinter.simpledialog as simpledialog
from mpl_toolkits.mplot3d import Axes3D
from chardet.universaldetector import UniversalDetector# 文字コード判定
import scipy.signal as sign

fp = FontProperties(fname="c:\\Windows\\Fonts\\YuGothM.ttc")#日本語フォント位置指定

deffont=('Yu Gothic', 20)
# file_name = ""


class ShowViewPointapp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default="k-lab_logo.ico")
        tk.Tk.wm_title(self, "EOG Gaze Track 3D")

        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        for F in (StartPage, Page3D, Page2D):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        global sf
        style = ttk.Style()
        style.configure('TButton', font=deffont)
        ttk.Frame.__init__(self, parent)

        label = ttk.Label(self, text="CSVファイルを選択してください", font=deffont)
        label.pack(pady=10)

        button0 = ttk.Button(self, text='ファイル選択', style='TButton', command=lambda: self.load_file())
        button0.pack(pady=10, ipadx=10)

        label2 = ttk.Label(self, text="サンプリング間隔", font=deffont)
        label2.pack(pady=10)

        sf = ttk.Spinbox(self, from_=1, to=20, width=5, increment=1, font=deffont)
        sf.pack(pady=10)

        label3 = ttk.Label(self, text="(msec)", font=deffont)
        label3.pack(pady=10)


        button2 = ttk.Button(self, text="3D", command=lambda: controller.show_frame(Page3D))
        button2.pack(pady=10, ipadx=10)
        
        button3 = ttk.Button(self, text="2D", command=lambda: controller.show_frame(Page2D))
        button3.pack(pady=10, ipadx=10)

    def load_file(self):
        global file_name,encode

        file_name = filedialog.askopenfilename(filetypes=[("CSV Files", ".csv")])

        # 文字コード判定------------------------------------
        detector = UniversalDetector()
        fen = open(file_name, mode='rb')
        for binary in fen:
            detector.feed(binary)
            if detector.done:
                break
        detector.close()
        encode=detector.result['encoding']

        self.text = tk.StringVar()#file nameの更新
        self.text.set("%s" % file_name)

        label_path = ttk.Label(self, textvariable=self.text,font=("Yu Gothic", 17))
        label_path.pack(pady=10)


    def sampling_frequency(self):
        sf_values = int(sf.get())
        freq=1000/sf_values
        return freq

    def Data_analysis(self):
        global file_name,pos, move_point,xylim,AnalysisData,signal

        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)


        # Pols data analysis =======================================================================================

        signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding=encode)
        # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",")
        # 　csv ファイルを行列式として読み込む（utf-8形式）



class Page3D(tk.Frame):

    def __init__(self, parent, controller):
        global  sf, b1, b2, bv1, bv2
        ttk.Frame.__init__(self, parent)

        label = ttk.Label(self, text="位置か速度を選択してください", font=deffont)
        label.grid(pady=10, padx=10, row=0, column=0, columnspan=3)
        
        bv1=tk.StringVar()
        b1 = ttk.Entry(self,width=10, textvariable=bv1)
        b1.insert(tk.END,"0")
        b1.grid(pady=10, padx=10, row=1, column=0)
        
        label1 = ttk.Label(self, text="~", font=deffont)
        label1.grid(row=1, column=1)
        
        bv2=tk.StringVar()
        b2 = ttk.Entry(self,width=10, textvariable=bv2)
        b2.insert(tk.END,"0")
        b2.grid(pady=10, padx=10, row=1, column=2)

        button1 = ttk.Button(self, text='位置', style='TButton', command=lambda: self.gaze_3d())
        button1.grid(pady=10, padx=10, row=3, column=0, columnspan=3)
        
        button3 = ttk.Button(self, text='速度', style='TButton', command=lambda: self.gaze_v())
        button3.grid(pady=10, padx=10, row=4, column=0, columnspan=3)

        button2 = ttk.Button(self, text="戻る", command=lambda: controller.show_frame(StartPage))
        button2.grid(pady=10, padx=10, row=5, column=0, columnspan=3)
    
    def spin_values(self):
        b1_values = float(bv1.get())
        b2_values = float(bv2.get())
        print(bv1.get)
        print(bv2.get)
        print(b1_values)
        print(b2_values)
        return b1_values, b2_values


    def gaze_3d(self):

        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)

        else:
            signal = np.loadtxt("%s" % file_name, skiprows=7, usecols=(0,1), delimiter=",", encoding=encode)
            # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding="utf-8")
            # signal = np.loadtxt("%s" %file_name, skiprows=1, delimiter=",")
            # 　csv ファイルを行列式として読み込む（utf-8形式）
            vsignal=signal*(10-(-10))/65536+(-10)
            #　読み込んだデータを分解能⇒電位[v]に変換

            """
            Memo -------------------------------------------------------------------------------------------------
            signal:
            0=No,1=Idx_Raw,2=テーブル番号,3=絶対位置X,4=絶対位置Y,5=表示状態,6=状態連番,
            7=信号Ch0(水平信号(左眼反転)),8=信号(Ch1垂直信号(mV)),9=信号Ch2(水平眼位(左眼反転)),10=信号Ch3(垂直眼位)
            ------------------------------------------------------------------------------------------------------
            """
            (t, s) = signal.shape  # 行列数表示: t=行数(時間データ),s=列数(計測項目データ)
            
            samplingfreq =StartPage.sampling_frequency(self)
            stime = np.arange(t) / samplingfreq  # 検査時間(sec)

            # bandstop --------------------------------------------------------------------
            fp=np.array([58,61])
            fs=np.array([59,60])
            gpass=3
            gstop=40
            verx=Page2D.bandstop(self,(vsignal[:, 0]), samplingfreq, fp, fs, gpass, gstop)
            very=Page2D.bandstop(self,(vsignal[:, 1]), samplingfreq, fp, fs, gpass, gstop)
            posx = np.cumsum(verx)
            posy = np.cumsum(very)
            
            box1, box2= self.spin_values()
            cbox1=int(box1*samplingfreq)
            cbox2=int(box2*samplingfreq)
           
            # zoom--------------------------------------------------------------------
            if (0<=cbox1) and (cbox1 < t):
             startidx=cbox1
            else:
             startidx=0
            
            if (box2 <= box1):
             endidx=t
            elif (cbox2>0) and (cbox2<=t):
             endidx=cbox2
            else:
             endidx=t
             
            # plot--------------------------------------------------------------------

            plt.style.use("seaborn")
            fig = plt.figure(figsize=(10, 9))
            ax = Axes3D(fig)
            ax.plot(stime[startidx:endidx], posx[startidx:endidx], posy[startidx:endidx], ".-", 
                    label="Eye Movement")
            ax.set_xlabel("Time(sec)")
            ax.set_ylabel("Horizontal Signal")
            ax.set_zlabel("Vertical Signal")
            plt.grid()
            plt.show()
            res = tk.messagebox.askquestion("csvファイルの保存","位相データを.csvファイルとして保存しますか?")
            if res =="yes":
                inputdata = simpledialog.askstring("ファイル名","ファイル名を入力")
                np.savetxt(str(inputdata)+".csv", pos, delimiter=",")

            
    def gaze_v(self):

        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)

        else:
            signal = np.loadtxt("%s" % file_name, skiprows=7, usecols=(0,1), delimiter=",", encoding=encode)
            # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding="utf-8")
            # signal = np.loadtxt("%s" %file_name, skiprows=1, delimiter=",")
            # 　csv ファイルを行列式として読み込む（utf-8形式）
            vsignal=signal*(10-(-10))/65536+(-10)

            """
            Memo -------------------------------------------------------------------------------------------------
            signal:
            0=No,1=Idx_Raw,2=テーブル番号,3=絶対位置X,4=絶対位置Y,5=表示状態,6=状態連番,
            7=信号Ch0(水平信号(左眼反転)),8=信号(Ch1垂直信号(mV)),9=信号Ch2(水平眼位(左眼反転)),10=信号Ch3(垂直眼位)
            ------------------------------------------------------------------------------------------------------
            """
            (t, s) = signal.shape  # 行列数表示: t=行数(時間データ),s=列数(計測項目データ)
            
            samplingfreq =StartPage.sampling_frequency(self)
            stime = np.arange(t) / samplingfreq  # 検査時間(sec)

            # bandstop --------------------------------------------------------------------
            fp=np.array([58,61])
            fs=np.array([59,60])
            gpass=3
            gstop=40
            posx=Page2D.bandstop(self,(vsignal[:, 0]), samplingfreq, fp, fs, gpass, gstop)
            posy=Page2D.bandstop(self,(vsignal[:, 1]), samplingfreq, fp, fs, gpass, gstop)
            box1, box2= self.spin_values()
            cbox1=int(box1*samplingfreq)
            cbox2=int(box2*samplingfreq)
            
            # zoom--------------------------------------------------------------------
            if (0<=cbox1) and (cbox1 < t):
             startidx=cbox1
            else:
             startidx=0
            
            if (box2 <= box1):
             endidx=t
            elif (cbox2>0) and (cbox2<=t):
             endidx=cbox2
            else:
             endidx=t


            plt.style.use("seaborn")
            fig = plt.figure(figsize=(10, 9))
            ax = Axes3D(fig)
            ax.plot(stime[startidx:endidx], posx[startidx:endidx], posy[startidx:endidx], ".-", 
                    label="Eye Movement")
            ax.set_xlabel("Time(sec)")
            ax.set_ylabel("Horizontal Signal")
            ax.set_zlabel("Vertical Signal")
            plt.grid()
            plt.show()
            res = tk.messagebox.askquestion("csvファイルの保存","位相データを.csvファイルとして保存しますか?")
            if res =="yes":
                inputdata = simpledialog.askstring("ファイル名","ファイル名を入力")
                np.savetxt(str(inputdata)+".csv", pos, delimiter=",")


class Page2D(tk.Frame):

    def __init__(self, parent, controller):
        global  sf, b1, b2, bv3, bv4
        ttk.Frame.__init__(self, parent)

        label = ttk.Label(self, text="水平垂直を選択してください", font=deffont)
        label.grid(pady=10, padx=10, row=0, column=0, columnspan=3)
        
        bv3=tk.StringVar()
        b1=ttk.Entry(self, width=10, textvariable=bv3)
        b1.insert(tk.END,"0")
        b1.grid(pady=10, padx=10, row=1, column=0)

        label1 = ttk.Label(self, text="~", font=deffont)
        label1.grid(row=1, column=1)
        
        bv4=tk.StringVar()
        b2=ttk.Entry(self, width=10, textvariable=bv4)
        b2.insert(tk.END,"0")
        b2.grid(pady=10, padx=10, row=1, column=2)

        button1 = ttk.Button(self, text='水平fft', style='TButton', command=lambda: self.gaze_2dpha())
        button1.grid(pady=10, padx=10, row=3, column=0, columnspan=3)
        
        button3 = ttk.Button(self, text='垂直fft', style='TButton', command=lambda: self.gaze_2dver())
        button3.grid(pady=10, padx=10, row=4, column=0, columnspan=3)

        button2 = ttk.Button(self, text="戻る", command=lambda: controller.show_frame(StartPage))
        button2.grid(pady=10, padx=10, row=5, column=0, columnspan=3)
     
    def spin_values(self):
        b1_values = float(bv3.get())
        b2_values = float(bv4.get())
        print(bv3.get)
        print(bv4.get)
        print(b1_values)
        print(b2_values)
        return b1_values, b2_values
        
    def bandstop(self,ssignal,smprate,fp,fs,gpass,gstop):
        fn=smprate/2
        wp=fp/fn
        ws=fs/fn
        N,Wn=sign.buttord(wp,ws,gpass,gstop)
        b,a=sign.butter(N,Wn,"bandstop")
        y=sign.filtfilt(b,a,ssignal)
        return y

    def gaze_2dpha(self):
        
        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)

        else:
            signal = np.loadtxt("%s" % file_name, skiprows=7, usecols=(0,1), delimiter=",", encoding=encode)
            # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding="utf-8")
            # signal = np.loadtxt("%s" %file_name, skiprows=1, delimiter=",")
            # 　csv ファイルを行列式として読み込む（utf-8形式）
            vsignal=signal*(10-(-10))/65536+(-10)
            
            (t, s) = signal.shape  # 行列数表示: t=行数(時間データ),s=列数(計測項目データ)
            
            samplingfreq =StartPage.sampling_frequency(self)
            
            box1, box2= self.spin_values()
            cbox1=int(box1*samplingfreq)
            cbox2=int(box2*samplingfreq)
           
            # zoom--------------------------------------------------------------------
            if (0<=cbox1) and (cbox1 < t):
             startidx=cbox1
            else:
             startidx=0
            
            if (box2 <= box1):
             endidx=t
            elif (cbox2>0) and (cbox2<=t):
             endidx=cbox2
            else:
             endidx=t
             
            ssignal=vsignal[startidx:endidx]
            (t2,s2)=ssignal.shape
            
            """
            Memo -------------------------------------------------------------------------------------------------
            signal:
            0=No,1=Idx_Raw,2=テーブル番号,3=絶対位置X,4=絶対位置Y,5=表示状態,6=状態連番,
            7=信号Ch0(水平信号(左眼反転)),8=信号(Ch1垂直信号(mV)),9=信号Ch2(水平眼位(左眼反転)),10=信号Ch3(垂直眼位)
            ------------------------------------------------------------------------------------------------------
            """
            stime = np.arange(t2) / samplingfreq  # 検査時間(sec)
            
             # FFT --------------------------------------------------------------------
            fft_sig = np.fft.fft(ssignal[:, 0], axis=0)  # X軸Y軸それぞれに対してFFT
            freq = np.fft.fftfreq(t2, d=(1 / samplingfreq))
            ifft_sig = np.real(np.fft.ifft(fft_sig, axis=0))
            sig_ampx = (np.abs(fft_sig/(t2/2)))  # fft結果を絶対値に変換
            posx=sig_ampx
            print(sig_ampx)
            
            # bandstopfilter
            smprate=samplingfreq
            print(smprate)
            fp=np.array([58,61])
            fs=np.array([59,60])
            gpass=3
            gstop=40
            filt_wave=self.bandstop((ssignal[:, 0]), smprate,fp,fs,gpass,gstop)
            f_fft_sig=np.fft.fft(filt_wave,axis=0)
            flt_ampx=np.abs(f_fft_sig)
            flt_ampx=flt_ampx/(t2/2)
           
            
        
            #plot--------------------------------------------------------------------
            fig=plt.figure()
            ax1=fig.add_subplot(121)
            ax1.plot(freq[:int(t2/2)], posx[:int(t2/2)], label='Eye Movement')
            ax1.set_xlabel("frequency [Hz]")
            ax1.set_ylabel("amplitude")
            ax2=fig.add_subplot(122)
            ax2.plot(freq[:int(t2/2)], flt_ampx[:int(t2/2)], label='Filterd Eye Movement')
            #ax.set_xlim(0,10)
            plt.legend(loc="best")
            plt.show()
            """
            res = tk.messagebox.askquestion("csvファイルの保存","位相データを.csvファイルとして保存しますか?")
            if res =="yes":
                inputdata = simpledialog.askstring("ファイル名","ファイル名を入力")
                np.savetxt(str(inputdata)+".csv", pos, delimiter=",")
            """
            
    def gaze_2dver(self):
        
        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)

        else:
            signal = np.loadtxt("%s" % file_name, skiprows=2, usecols=(1), delimiter=",", encoding=encode)
            stime = np.linspace(0, 5, 101)
            Np=[0,1]
            Dp=[1, 2, 170]
            sys=matlab.tf(Np, Dp)
            print(sys)
            x0=[0,0]
            (response, T, x) =matlab.lsim(sys, signal, stime, x0)
            a=np.convolve(signal,sys,mode='full')
            np.savetxt("kadai2.csv", response, delimiter=",")
            
            
            
        
           # plot--------------------------------------------------------------------
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(T,a)
            #ax.plot(T,signal)
            ax.set_xlabel("Time[s]")
            ax.set_ylabel("amplitude")
            #ax.set_xlim(0,10)
            plt.legend(loc="best")
            plt.show()     
            """
            res = tk.messagebox.askquestion("csvファイルの保存","位相データを.csvファイルとして保存しますか?")
            if res =="yes":
                inputdata = simpledialog.askstring("ファイル名","ファイル名を入力")
                np.savetxt(str(inputdata)+".csv", pos, delimiter=",")
            """
    
     
            
           



app = ShowViewPointapp()
app.mainloop()
