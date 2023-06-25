import tkinter as tk
import subprocess

def run_main():
    subprocess.call(["python", "main.py"])

root = tk.Tk()
root.title("Yazlab2.3")
root.geometry("400x200")

button = tk.Button(root, text="Dosya Yüklemek İçin Tıklayın", command=run_main)
button.pack(pady=20)

root.mainloop()