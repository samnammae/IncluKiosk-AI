import tkinter as tk
from tkinter import messagebox

# Tk 객체 생성
root = tk.Tk()
root.withdraw()  # 메인 윈도우 숨기기

# 팝업창 띄우기
messagebox.showinfo("알림", "WebSocket으로 실행된 test.py입니다!")

# 종료
root.destroy()

