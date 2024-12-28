try:
    import tkinter as tk
    from tkinter import messagebox
    import cv2
    import os
    import csv
    import pandas as pd
    from PIL import Image
    import numpy as np
    from tkinter import ttk
    import customtkinter as ctk
    from datetime import datetime
except Exception as e:
    print('Install tkinter Package')
    messagebox.showwarning(title='Install Packages',message='Install these packages : opencv-python, opencv-contrib-python, opencv-python-headless, opencv-contrib-python-headless, os, Pillow, customtkinter')

window=tk.Tk()
window.title("Face recognition system")
# window.config(bg='whitesmoke')
# window.attributes('-alpha',0.5)

# bg_image=ctk.CTkImage(Image.open('cover.jpg'),size=(window.winfo_screenwidth(),window.winfo_screenheight()))
# bg_label=ctk.CTkLabel(window,image=bg_image)
# bg_label.place(x=0,y=0)

fr=tk.Frame(window)
fr.pack()
# fr.attributes('-alpha',0.5)

uif=tk.LabelFrame(fr,text='Add User',bg=window['bg'])
uif.grid(column=0,row=0,padx=5,pady=3)

l1=tk.Label(uif,text="Name :",font=("Algerian",15))
l1.grid(column=0,row=0)
t1=tk.Entry(uif,width=25,bd=5)
t1.grid(column=1,row=0)

l2=tk.Label(uif,text="Age :",font=("Algerian",15))
l2.grid(column=0,row=1)
t2=tk.Spinbox(uif,width=20,bd=5,from_=5,to=100)
t2.grid(column=1,row=1)

l3=tk.Label(uif,text="Address :",font=("Algerian",15))
l3.grid(column=0,row=2)
t3=tk.Entry(uif,width=30,bd=2)
t3.grid(column=1,row=2)

for w in uif.winfo_children():
    w.grid_configure(padx=10,pady=5)



def train_classifier():
    data_dir="data"
    if len(os.listdir(data_dir))<2:
        messagebox.showwarning('Warning','There is no data of any user\n<<<First Add Users>>>')
    else :
        path=[os.path.join(data_dir,f) for f in os.listdir(data_dir)]
        path.pop(0)
        # print(path)
        faces=[]
        ids=[]
        for image in path:
            # print(image)
            img=Image.open(image).convert('L')
            imageNp=np.array(img,'uint8')
            id= int(os.path.split(image)[1].split(".")[1])
            # print(id)
            faces.append(imageNp)
            ids.append(id)
        ids=np.array(ids)
        # Train the classifier and save
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        clf.write("classifier.xml")
        # messagebox.showinfo('Result','Training Dataset Completed!!!')

b1=tk.Button(fr,text="Train Dataset",font=("Algerian",11),bg='grey',fg='white',command=train_classifier)
b1.grid(column=0,row=2,sticky='we',padx=5,pady=1)

def detect_face():
    train_classifier()
    att_list=[]
    def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,clf):
        gray_image= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        features=classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)
        coords=[]
        for(x,y,w,h) in features:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            id,pred=clf.predict(gray_image[y:y+h,x:x+w])
            confidence=int(100*(1-pred/300))

            file=open('database/st_data.csv','r')
            re=csv.reader(file)
            next(re)
            da=list(list(i) for i in re if i)
            id_list=list(i[0] for i in da)
            file.close()
            
            if confidence>76:
                if str(id) in id_list:
                    if str(id) not in att_list:
                        att_list.append(str(id))
                    name=da[id_list.index(str(id))][1]+'_'+da[id_list.index(str(id))][2]
                else:
                    name="UNKNOWN"
            else:
                name="UNKNOWN"
            cv2.putText(img,name,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,color if name != "UNKNOWN" else (0,0,255),2,cv2.LINE_AA)
            coords.append((x,y,w,h))
        return img
    
    # def recognize(img,clf,faceCascade):
    #     draw_boundary(img,faceCascade,1.2,8,(255,255,255),clf)  
    #     return img
    
    faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    
    video_capture= cv2.VideoCapture(0)
    
    while True:
        ret,img= video_capture.read()
        img=draw_boundary(img,faceCascade,1.15,8,(255,255,255),clf)
        if not ret:
            break
        cv2.imshow("Face Detection",img)
        
        if cv2.waitKey(1)==13:
            break;
    
    video_capture.release()
    cv2.destroyAllWindows()
    return att_list
    
b2=tk.Button(fr,text="Detect Face",font=("Algerian",11),bg='grey',fg='white',command=detect_face)
b2.grid(column=0,row=3,sticky='we',padx=5,pady=1)

def generate_dataset():
    if (t1.get()=="" or t2.get()=="" or t3.get()==""):
        messagebox.showwarning(title='Error...',message='Fill your complete Details')
    else :
        file=open('database/st_data.csv','r')
        re=csv.reader(file)
        next(re)
        da=list(list(i) for i in re if i)
        file.close()
        
        file=open('database/st_data.csv','a',newline='')
        wr=csv.writer(file)
        id=len(da)+1
        val=[id,t1.get(),t2.get(),t3.get()]

        att=open("database/Attendance_record.csv",'a',newline='')
        att_wr=csv.writer(att)
        
        face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        def face_cropped(img):
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_classifier.detectMultiScale(gray,1.3,5)
            if faces is ():
                return None
            for(x,y,w,h) in faces:
                cropped_face=img[y:y+h,x:x+w]
            return cropped_face
        cap = cv2.VideoCapture(0)
        img_id=0
        while True:
            ret,frame=cap.read()
            if face_cropped(frame) is not None:
                img_id+=1
                face=cv2.resize(face_cropped(frame),(200,200))
                face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                file_name_path="data/user."+str(id)+"."+str(img_id)+".jpg"
                cv2.imwrite(file_name_path,face)
                cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow("Cropped face",face)
                if cv2.waitKey(1)==13 or int(img_id)==200:
                    break
        cap.release()
        cv2.destroyAllWindows()
        
        wr.writerow(val)
        file.close()
        
        att_wr.writerow([id,t1.get(),0,0,0])
        att.close()
        
        train_classifier()
        
        messagebox.showinfo('Result','User Added Successfully.....')
        t1.delete(0,'end')
        t2.delete(0,'end')
        t3.delete(0,'end')
        file_opener('database/st_data.csv')

b3=tk.Button(fr,text="Add User",font=("Algerian",11),bg='grey',fg='white',command=generate_dataset)
b3.grid(column=0,row=1,sticky='we',padx=5,pady=1)

b5=tk.Button(fr,text="Registered User",font=("Algerian",11),bg='grey',fg='white',command=lambda : file_opener('database/st_data.csv'))
b5.grid(column=0,row=4,sticky='we',padx=5,pady=1)

b6=tk.Button(fr,text="Check Attendance",font=("Algerian",11),bg='grey',fg='white',command=lambda : file_opener('database/Attendance_record.csv'))
b6.grid(column=0,row=5,sticky='we',padx=5,pady=1)

def take_attendance():
    att_list=detect_face()
    # print(att_list)

    file=open('database/st_data.csv','r')
    re=csv.reader(file)
    next(re)
    da=list(list(i) for i in re if i)
    id_list=list(i[0] for i in da)
    file.close()
    
    ## Creating attendace record file 
    record_file_name=datetime.now()
    att_file=open("database/Attendance Record/"+record_file_name.strftime("%Y-%m-%d %H-%M")+".csv",'w',newline='')
    
    wr=csv.writer(att_file)
    wr.writerow(['Id','Name'])
    for i in att_list:
        name=da[id_list.index(i)][1]
        wr.writerow([i,name])

    wr.writerow([])
    wr.writerow(['Date : ',record_file_name.strftime("%d %b %Y")])
    wr.writerow(['Time : ',record_file_name.strftime("%I:%M %p")])
    att_file.close()
    attendance_History()

    ## Marking attendance in attendance file
    up_att=pd.read_csv('database/Attendance_record.csv')
    for i in range(len(da)):
        # print(i)
        up_att.loc[i,'Total']=up_att['Total'][i]+1
        # print(up_att['Id'][i] in att_list)
        if str(up_att['Id'][i]) in att_list:
            # print(i,'Present')
            up_att.loc[i,'Present']=up_att['Present'][i]+1
        else :
            # print(i,'Absent')
            up_att.loc[i,'Absent']=up_att['Absent'][i]+1
        up_att.loc[i,'Percentage']=round((up_att['Present'][i]/up_att['Total'][i])*100,2)
    up_att.to_csv('database/Attendance_record.csv',index=False)
    file_opener('database/Attendance_record.csv')
                   

b4=tk.Button(fr,text="Take Attendance",font=("Algerian",15),width=15,bg='red',fg='white',command=take_attendance)
b4.grid(column=0,row=6,sticky='we',padx=5,pady=5,columnspan=2)

fsearch=tk.LabelFrame(fr,text='Attendance Record')
fsearch.grid(row=0,column=1,rowspan=6,padx=5,pady=10,sticky='news')

atr=ctk.CTkScrollableFrame(fsearch,label_text='Attendance Record',fg_color='white')
atr.grid(column=0,row=3,sticky='news',padx=5,pady=5,columnspan=3)


tk.Label(fsearch,text='Date').grid(row=0,column=0)
tk.Label(fsearch,text='Month').grid(row=0,column=1)
tk.Label(fsearch,text='Year').grid(row=0,column=2)
date=tk.Spinbox(fsearch,from_=0,to=31,width=9)
date.grid(row=1,column=0,padx=4,pady=4)
month=tk.Spinbox(fsearch,from_=0,to=12,width=9)
month.grid(row=1,column=1,padx=4,pady=4)
year=ttk.Combobox(fsearch,value=['2024','2023','2022'],width=9)
year.grid(row=1,column=2,padx=4,pady=4)

def create_table_frame():
    att_record_frame=tk.LabelFrame(fr)
    att_record_frame.grid(column=2,row=0,rowspan=5,sticky='swe',padx=5,pady=10)
    return att_record_frame

def close_table(frame_wid):
    frame_wid.destroy()
    # for wd in frame_wid.winfo_children():
        # wd.destroy()

def file_opener(record_file):
    re=csv.reader(open(record_file,'r'))
    
    table_frame=create_table_frame()
    table_content=list(list(i) for i in re if i)
    # close(table_frame)
    close=tk.Button(table_frame,text='X',command=lambda : close_table(table_frame))
    close.grid(column=1,row=0,pady=2,sticky='e')
    
    scroll_table=ctk.CTkScrollableFrame(table_frame,fg_color='white',width=len(table_content[0])*100)
    scroll_table.grid(column=0,row=1,columnspan=2,pady=2,sticky='swe')
    # scroll_table.grid_rowconfigure(0,weight=1)
    scroll_table.grid_columnconfigure(0,weight=1)
    
        
    table_tree=ttk.Treeview(scroll_table,columns=table_content[0],show='headings')
    
    for heading in table_content[0]:
        table_tree.heading(heading,text=heading)
    
    table_content.pop(0)
    # print(table_content)
    for con in table_content:
        table_tree.insert('',tk.END,value=con)
    table_tree.grid(row=0,column=0)
    

    
def attendance_History():
    for wd in atr.winfo_children():
        wd.destroy()
        
    name=year.get()+'-'+month.get()+'-'+date.get()
    
    if date.get()=='0' or date.get()=='':
        name=year.get()+'-'+month.get()
        if month.get()=='0' or month.get()=='':
            name=year.get()

    # print(name)
    
    csv_files="database/Attendance Record"
    
    for f in os.listdir(csv_files):
        if name in f:
            button_text=f.split('.')[0]
            file_path=os.path.join(csv_files,f)
            btn=tk.Button(atr,text=button_text,command=lambda record_path=file_path: file_opener(record_path)).pack(pady=4)
    
    year.delete(0,'end')
    month.delete(0,'end')
    date.delete(0,'end')

attendance_History()
tk.Button(fsearch,text='Search',fg='black',bg='red',command=attendance_History).grid(row=2,column=0,sticky='news',columnspan=3,padx=5,pady=5)



# window.attributes("-fullscreen",True)
window.geometry(f'{str(window.winfo_screenwidth())}x{str(window.winfo_screenheight())}+0+0')
window.mainloop()