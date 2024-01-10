import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tkinter import StringVar
import cv2
from matplotlib import pyplot as plt

window = tk.Tk()
window.title('Yapay Zeka Proje - Tuğba Aktürk - 2112721014 - 3')
window.geometry('1100x1000')

image_path = 'kuzeyIsiklari.jpg'
cv2_image = cv2.imread(image_path)
resized_image = cv2.resize(cv2_image, (600,400))
image_converted = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image_converted)
photo_image = ImageTk.PhotoImage(image=image)

def open_file():
    global image_path
    return filedialog.askopenfilename(title="Resim Seç", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

def convert_image(cv2_image):
    image_converted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_converted)
    photo_image = ImageTk.PhotoImage(image=image)
    return photo_image

def convert_image_second_variation(image):
    converted_image = np.uint8(image)
    converted_image = Image.fromarray(converted_image)
    converted_image = ImageTk.PhotoImage(converted_image)
    return converted_image


def kamera_adaptive_threshold():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

        cv2.imshow('Adaptive Threshold', adaptive)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

    video.release()
    cv2.destroyAllWindows()

def adaptive_threshold_buton():
    adaptive_threshold_window = tk.Toplevel(window)
    adaptive_threshold_window.title("Adaptive Threshold")
    adaptive_threshold_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def adaptive_threhold_function():
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (600, 400))
        max_value = int(max_value_entry_var.get())
        block_size = int(block_size_entry_var.get())
        c = float(c_entry_var.get())
        adaptive_threshold = cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        photo_adaptive_threshold = convert_image(adaptive_threshold)
        image_label.config(image=photo_adaptive_threshold)
        image_label.photo = photo_adaptive_threshold

    image_label = tk.Label(adaptive_threshold_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(adaptive_threshold_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(adaptive_threshold_window, text="Kamerayı Aç", command=kamera_adaptive_threshold)
    button.grid(row=1, sticky='ns', pady=10)

    max_value_label = tk.Label(adaptive_threshold_window, text="Max Value :")
    max_value_label.grid(row=2, sticky='w', padx=350, pady=10)

    max_value_entry_var = tk.StringVar(value="")
    max_value_entry = tk.Entry(adaptive_threshold_window, bg='#dddddd', textvariable=max_value_entry_var)
    max_value_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    block_size_label = tk.Label(adaptive_threshold_window, text="Block Size :")
    block_size_label.grid(row=3, sticky='w', padx=350, pady=10)

    block_size_entry_var = tk.StringVar(value="")
    block_size_entry = tk.Entry(adaptive_threshold_window, bg='#dddddd', textvariable=block_size_entry_var)
    block_size_entry.grid(row=3, sticky='ns', padx=80, pady=10)

    c_label = tk.Label(adaptive_threshold_window, text="C :")
    c_label.grid(row=4, sticky='w', padx=350, pady=10)

    c_entry_var = tk.StringVar(value="")
    c_entry = tk.Entry(adaptive_threshold_window, bg='#dddddd', textvariable=c_entry_var)
    c_entry.grid(row=4,sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(adaptive_threshold_window, text="Uygula", command=adaptive_threhold_function, bg='gray')
    uygula_buton.grid(row=5, padx=250, pady=10)

adaptive_threshold_anasayfa=tk.Button(window,text="Adaptive Threshold", fg='black', bg='#8b668b', command=adaptive_threshold_buton, relief=tk.SOLID, width=20, height=10)
adaptive_threshold_anasayfa.place(x=30, y=60)


def kamera_otsu_threshold():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray,0,255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        cv2.imshow('Otsu Threshold', otsu)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

    video.release()
    cv2.destroyAllWindows()

def otsu_threshold_buton():
    otsu_threshold_window = tk.Toplevel(window)
    otsu_threshold_window.title("Otsu Threshold")
    otsu_threshold_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def otsu_threhold_function():
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (600, 400))
        thresh = float(thresh_entry_var.get())
        max_value = float(max_value_entry_var.get())
        _, otsu_threshold = cv2.threshold(image, thresh, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        photo_otsu_threshold = convert_image(otsu_threshold)
        image_label.config(image=photo_otsu_threshold)
        image_label.photo = photo_otsu_threshold

    image_label = tk.Label(otsu_threshold_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(otsu_threshold_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(otsu_threshold_window, text="Kamerayı Aç", command=kamera_otsu_threshold)
    button.grid(row=1, sticky='ns', pady=10)

    thresh_label = tk.Label(otsu_threshold_window, text="Thresh :")
    thresh_label.grid(row=2, sticky='w', padx=350, pady=10)

    thresh_entry_var = tk.StringVar(value="")
    thresh_entry = tk.Entry(otsu_threshold_window, bg='#dddddd', textvariable=thresh_entry_var)
    thresh_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    max_value_label = tk.Label(otsu_threshold_window, text="Max Value :")
    max_value_label.grid(row=3, sticky='w', padx=350, pady=10)

    max_value_entry_var = tk.StringVar(value="")
    max_value_entry = tk.Entry(otsu_threshold_window, bg='#dddddd', textvariable=max_value_entry_var)
    max_value_entry.grid(row=3, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(otsu_threshold_window, text="Uygula", command=otsu_threhold_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

otsu_threshold_anasayfa=tk.Button(window, text="Otsu Threshold", fg='black', bg='#8b668b', command=otsu_threshold_buton, relief=tk.SOLID, width=20, height=10)
otsu_threshold_anasayfa.place(x=30, y=280)


def kamera_kenarlik_ekle():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        image_with_border = cv2.copyMakeBorder(frame,10,10, 10,10,borderType=cv2.BORDER_CONSTANT,value=[120,12,240])

        cv2.imshow('Kenarlık Ekle', image_with_border)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

    video.release()
    cv2.destroyAllWindows()

def kenarlik_ekle_buton():
    kenarlik_ekle_window = tk.Toplevel(window)
    kenarlik_ekle_window.title("Kenarlık Ekle")
    kenarlik_ekle_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def kenarlik_ekle_function():
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 400))
        border_width = int(border_width_entry_var.get())
        image_with_border = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, borderType=cv2.BORDER_REPLICATE, value=[120,12,240])
        photo_with_border = convert_image(image_with_border)
        image_label.config(image=photo_with_border)
        image_label.photo = photo_with_border

    image_label = tk.Label(kenarlik_ekle_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(kenarlik_ekle_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(kenarlik_ekle_window, text="Kamerayı Aç", command=kamera_kenarlik_ekle)
    button.grid(row=1, sticky='ns', pady=10)

    border_width_label = tk.Label(kenarlik_ekle_window, text="Border Width :")
    border_width_label.grid(row=2, sticky='w', padx=350, pady=10)

    border_width_entry_var = tk.StringVar(value="")
    border_width_entry = tk.Entry(kenarlik_ekle_window, bg='#dddddd', textvariable=border_width_entry_var)
    border_width_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(kenarlik_ekle_window, text="Uygula", command=kenarlik_ekle_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

kenarlik_ekle_anasayfa=tk.Button(window, text="Kenarlık Ekle", fg='black', bg='#8b668b', command=kenarlik_ekle_buton, relief=tk.SOLID, width=20, height=10)
kenarlik_ekle_anasayfa.place(x=30, y=500)


def kamera_bulaniklastir():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        blurred_image = cv2.blur(frame, (20, 20))

        cv2.imshow('Bulanıklaştır', blurred_image)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

    video.release()
    cv2.destroyAllWindows()

def bulaniklastir_buton():
    bulaniklastir_window = tk.Toplevel(window)
    bulaniklastir_window.title("Bulanıklaştır")
    bulaniklastir_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def bulaniklastir_function():
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 400))
        kernel_size = int(kernel_size_entry_var.get())
        blurred_image = cv2.blur(image, (kernel_size, kernel_size))
        photo_blurred_image = convert_image(blurred_image)
        image_label.config(image=photo_blurred_image)
        image_label.photo = photo_blurred_image

    image_label = tk.Label(bulaniklastir_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(bulaniklastir_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(bulaniklastir_window, text="Kamerayı Aç", command=kamera_bulaniklastir)
    button.grid(row=1, sticky='ns', pady=10)

    kernel_size_label = tk.Label(bulaniklastir_window, text="Kernel Size :")
    kernel_size_label.grid(row=2, sticky='w', padx=350, pady=10)

    kernel_size_entry_var = tk.StringVar(value="")
    kernel_size_entry = tk.Entry(bulaniklastir_window, bg='#dddddd', textvariable=kernel_size_entry_var)
    kernel_size_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(bulaniklastir_window, text="Uygula", command=bulaniklastir_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

bulaniklastir_anasayfa=tk.Button(window, text="Bulanıklaştır", fg='black', bg='#8b668b', command=bulaniklastir_buton, relief=tk.SOLID, width=20, height=10)
bulaniklastir_anasayfa.place(x=250, y=60)


def kamera_keskinlestir():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        keskinlestir = cv2.filter2D(frame, -1, np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]]))

        cv2.imshow('Keskinleştir', keskinlestir)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

    video.release()
    cv2.destroyAllWindows()

def keskinlestir_buton():
    keskinlestir_window = tk.Toplevel(window)
    keskinlestir_window.title("Keskinleştir")
    keskinlestir_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def keskinlestir_function():
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 400))
        kernel = int(kernel_entry_var.get())
        kernel = np.array([[0, -1, 0], [-1, kernel, -1], [0, -1, 0]])
        sharped_image = cv2.filter2D(image, -1, kernel)
        photo_sharped_image = convert_image(sharped_image)
        image_label.config(image=photo_sharped_image)
        image_label.photo = photo_sharped_image

    image_label = tk.Label(keskinlestir_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(keskinlestir_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(keskinlestir_window, text="Kamerayı Aç", command=kamera_keskinlestir)
    button.grid(row=1, sticky='ns', pady=10)

    kernel_label = tk.Label(keskinlestir_window, text="Kernel :")
    kernel_label.grid(row=2, sticky='w', padx=350, pady=10)

    kernel_entry_var = tk.StringVar(value="")
    kernel_entry = tk.Entry(keskinlestir_window, bg='#dddddd', textvariable=kernel_entry_var)
    kernel_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(keskinlestir_window, text="Uygula", command=keskinlestir_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

keskinlestir_anasayfa=tk.Button(window, text="Keskinleştir", fg='black',bg='#8b668b', command=keskinlestir_buton, relief=tk.SOLID, width=20, height=10)
keskinlestir_anasayfa.place(x=250, y=280)


def kamera_gamma_filtrele():
    video = cv2.VideoCapture(0)

    def apply_gamma_correction(frame, gamma=1.0):
        image_normalized = frame / 255.0

        gamma_corrected = np.power(image_normalized, gamma)
        gamma_corrected = np.uint8(gamma_corrected * 255)

        return gamma_corrected

    while (True):
        ret, frame = video.read()

        gamma_corrected_image = apply_gamma_correction(frame, gamma = 5.5)

        cv2.imshow('Gamma Filtrele', gamma_corrected_image)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

    video.release()
    cv2.destroyAllWindows()

def gamma_filtrele_buton():
    gamma_filtrele_window = tk.Toplevel(window)
    gamma_filtrele_window.title("Gamma Filtrele")
    gamma_filtrele_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def gamma_filtrele_function():
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 400))
        gamma = float(gamma_entry_var.get())
        max_value = float(max_value_entry_var.get())
        image_normalized = image / 255.0
        gamma_corrected = np.power(image_normalized, gamma)
        gamma_corrected = np.uint8(gamma_corrected * max_value)
        photo_gamma_corrected = convert_image(gamma_corrected)
        image_label.config(image=photo_gamma_corrected)
        image_label.photo = photo_gamma_corrected

    image_label = tk.Label(gamma_filtrele_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(gamma_filtrele_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(gamma_filtrele_window, text="Kamerayı Aç", command=kamera_gamma_filtrele)
    button.grid(row=1, sticky='ns', pady=10)

    gamma_label = tk.Label(gamma_filtrele_window, text="Gamma :")
    gamma_label.grid(row=2, sticky='w', padx=350, pady=10)

    gamma_entry_var = tk.StringVar(value="")
    gamma_entry = tk.Entry(gamma_filtrele_window, bg='#dddddd', textvariable=gamma_entry_var)
    gamma_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    max_value_label = tk.Label(gamma_filtrele_window, text="Max Value :")
    max_value_label.grid(row=3, sticky='w', padx=350, pady=10)

    max_value_entry_var = tk.StringVar(value="")
    max_value_entry = tk.Entry(gamma_filtrele_window, bg='#dddddd', textvariable=max_value_entry_var)
    max_value_entry.grid(row=3, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(gamma_filtrele_window, text="Uygula", command=gamma_filtrele_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

gammaFiltreleButon=tk.Button(window, text="Gamma Filtrele", fg='black', bg='#8b668b', command=gamma_filtrele_buton, relief=tk.SOLID, width=20, height=10)
gammaFiltreleButon.place(x=250, y=500)


def kamera_sobel_kenar_tespit():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        sobelx = np.abs(sobelx)
        sobely = np.abs(sobely)

        edges = cv2.bitwise_or(sobelx, sobely)

        cv2.imshow('Sobel Kenar Tespiti', edges)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

def sobel_kenar_tespit_buton():

    sobel_kenar_tespit_window = tk.Toplevel(window)
    sobel_kenar_tespit_window.title("Sobel Kenar Tespiti")
    sobel_kenar_tespit_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def sobel_kenar_tespit_function():
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (600, 400))
        kernel_size = int(kernel_size_entry_var.get())
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        sobelx = np.abs(sobelx)
        sobely = np.abs(sobely)
        edges = cv2.bitwise_or(sobelx, sobely)
        photo_image_sobel = convert_image_second_variation(edges)
        image_label.config(image=photo_image_sobel)
        image_label.photo = photo_image_sobel

    image_label = tk.Label(sobel_kenar_tespit_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(sobel_kenar_tespit_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(sobel_kenar_tespit_window, text="Kamerayı Aç", command=kamera_sobel_kenar_tespit)
    button.grid(row=1, sticky='ns', pady=10)

    kernel_size_label = tk.Label(sobel_kenar_tespit_window, text="Kernel Size :")
    kernel_size_label.grid(row=2, sticky='w', padx=350, pady=10)

    kernel_size_entry_var = tk.StringVar(value="")
    kernel_size_entry = tk.Entry(sobel_kenar_tespit_window, bg='#dddddd', textvariable=kernel_size_entry_var)
    kernel_size_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(sobel_kenar_tespit_window, text="Uygula", command=sobel_kenar_tespit_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

sobel_kenar_tespit_anasayfa=tk.Button(window, text="Sobel Kenar Tespiti", fg='black', bg='#8b668b', command=sobel_kenar_tespit_buton, relief=tk.SOLID, width=20, height=10)
sobel_kenar_tespit_anasayfa.place(x=470, y=60)


def kamera_laplacian_kenar_tespit():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        float_gray = cv2.Laplacian(gray, cv2.CV_64F)
        float_gray = np.uint8(np.absolute(float_gray))

        imgBlured = cv2.GaussianBlur(gray, (3, 3), 0)
        sonuc = cv2.Laplacian(imgBlured, ddepth=-1, ksize=3)

        cv2.imshow('Laplacian Kenar Tespiti', sonuc)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

def laplacian_kenar_tespit_buton():
    laplacian_kenar_tespit_window = tk.Toplevel(window)
    laplacian_kenar_tespit_window.title("Laplacian Kenar Tespiti")
    laplacian_kenar_tespit_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def laplacian_kenar_tespit_function():
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (600, 400))
        kernel_size = int(kernel_size_entry_var.get())
        imgBlured = cv2.GaussianBlur(image, (3, 3), 0)
        sonuc = cv2.Laplacian(imgBlured, ddepth=-1, ksize=kernel_size)
        photo_image_laplacian = convert_image_second_variation(sonuc)
        image_label.config(image=photo_image_laplacian)
        image_label.photo = photo_image_laplacian

    image_label = tk.Label(laplacian_kenar_tespit_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(laplacian_kenar_tespit_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(laplacian_kenar_tespit_window, text="Kamerayı Aç", command=kamera_laplacian_kenar_tespit)
    button.grid(row=1, sticky='ns', pady=10)

    kernel_size_label = tk.Label(laplacian_kenar_tespit_window, text="Kernel Size :")
    kernel_size_label.grid(row=2, sticky='w', padx=350, pady=10)

    kernel_size_entry_var = tk.StringVar(value="")
    kernel_size_entry = tk.Entry(laplacian_kenar_tespit_window, bg='#dddddd', textvariable=kernel_size_entry_var)
    kernel_size_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(laplacian_kenar_tespit_window, text="Uygula", command=laplacian_kenar_tespit_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

laplacian_kenar_tespit_anasayfa=tk.Button(window, text="Laplacian Kenar Tespiti", fg='black', bg='#8b668b', command=laplacian_kenar_tespit_buton, relief=tk.SOLID, width=20, height=10)
laplacian_kenar_tespit_anasayfa.place(x=470, y=280)


def kamera_canny_kenar_tespit():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sonuc = cv2.Canny(gray, 50, 150, L2gradient=True)

        float_gray = cv2.Laplacian(gray, cv2.CV_64F)
        float_gray = np.uint8(np.absolute(float_gray))

        cv2.imshow('Canny Kenar Tespiti', sonuc)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

def canny_kenar_tespit_buton():
    canny_kenar_tespit_window = tk.Toplevel(window)
    canny_kenar_tespit_window.title("Canny Kenar Tespiti")
    canny_kenar_tespit_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def canny_kenar_tespit_function():
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (600, 400))
        low_threshold = int(low_threshold_entry_var.get())
        high_threshold = int(high_threshold_entry_var.get())
        sonuc = cv2.Canny(image, low_threshold, high_threshold, L2gradient=True)
        photo_image_canny = convert_image_second_variation(sonuc)
        image_label.config(image=photo_image_canny)
        image_label.photo = photo_image_canny

    image_label = tk.Label(canny_kenar_tespit_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(canny_kenar_tespit_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(canny_kenar_tespit_window, text="Kamerayı Aç", command=kamera_canny_kenar_tespit)
    button.grid(row=1, sticky='ns', pady=10)

    low_threshold_label = tk.Label(canny_kenar_tespit_window, text="Low Threshold :")
    low_threshold_label.grid(row=2, sticky='w', padx=330, pady=10)

    low_threshold_entry_var = tk.StringVar(value="")
    low_threshold_entry = tk.Entry(canny_kenar_tespit_window, bg='#dddddd', textvariable=low_threshold_entry_var)
    low_threshold_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    high_threshold_label = tk.Label(canny_kenar_tespit_window, text="High Threshold :")
    high_threshold_label.grid(row=3, sticky='w', padx=330, pady=10)

    high_threshold_entry_var = tk.StringVar(value="")
    high_threshold_entry = tk.Entry(canny_kenar_tespit_window, bg='#dddddd', textvariable=high_threshold_entry_var)
    high_threshold_entry.grid(row=3, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(canny_kenar_tespit_window, text="Uygula", command=canny_kenar_tespit_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

canny_kenar_tespit_anasayfa=tk.Button(window, text="Canny Kenar Tespiti", fg='black', bg='#8b668b', command=canny_kenar_tespit_buton, relief=tk.SOLID, width=20, height=10)
canny_kenar_tespit_anasayfa.place(x=470, y=500)


def kamera_deriche_kenar_tespit():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        frame = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
        kx, ky = cv2.getDerivKernels(1,1,3,normalize=True)
        kx, ky = cv2.getDerivKernels(1, 1, 3, normalize=True)
        deriche_kernel_x = 0.5 * kx
        deriche_kernel_y = 0.5 * ky
        deriche_x = cv2.filter2D(image, -1, deriche_kernel_x)
        deriche_y = cv2.filter2D(image, -1, deriche_kernel_y)
        edges = np.sqrt(deriche_x ** 2 + deriche_y ** 2)

        cv2.imshow('Deriche Kenar Tespiti', edges)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

def deriche_kenar_tespit_buton():
    deriche_kenar_tespit_window = tk.Toplevel(window)
    deriche_kenar_tespit_window.title("Deriche Kenar Tespiti")
    deriche_kenar_tespit_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def deriche_kenar_tespit_function():
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (600, 400))
        alpha = float(alpha_entry_var.get())
        kernel_size = int(kernel_size_entry_var.get())
        kx, ky = cv2.getDerivKernels(1, 1, kernel_size, normalize=True)
        deriche_kernel_x = alpha * kx
        deriche_kernel_y = alpha * ky
        deriche_x = cv2.filter2D(image, -1, deriche_kernel_x)
        deriche_y = cv2.filter2D(image, -1, deriche_kernel_y)
        edges = np.sqrt(deriche_x ** 2 + deriche_y ** 2)
        photo_image_deriche = convert_image_second_variation(edges)
        image_label.config(image=photo_image_deriche)
        image_label.photo = photo_image_deriche

    image_label = tk.Label(deriche_kenar_tespit_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(deriche_kenar_tespit_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=350, pady=10)

    button = tk.Button(deriche_kenar_tespit_window, text="Kamerayı Aç", command=kamera_deriche_kenar_tespit)
    button.grid(row=1, sticky='ns', pady=10)

    alpha_label = tk.Label(deriche_kenar_tespit_window, text="Alpha: ")
    alpha_label.grid(row=2, sticky='w', padx=350, pady=10)

    alpha_entry_var = tk.StringVar(value="")
    alpha_entry = tk.Entry(deriche_kenar_tespit_window, bg='#dddddd', textvariable=alpha_entry_var)
    alpha_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    kernel_size_label = tk.Label(deriche_kenar_tespit_window, text="Kernel Size :")
    kernel_size_label.grid(row=3, sticky='w', padx=350, pady=10)

    kernel_size_entry_var = tk.StringVar(value="")
    kernel_size_entry = tk.Entry(deriche_kenar_tespit_window, bg='#dddddd', textvariable=kernel_size_entry_var)
    kernel_size_entry.grid(row=3, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(deriche_kenar_tespit_window, text="Uygula", command=deriche_kenar_tespit_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

deriche_kenar_tespit_anasayfa=tk.Button(window, text="Deriche Kenar Tespiti", fg='black', bg='#8b668b', command=deriche_kenar_tespit_buton, relief=tk.SOLID, width=20, height=10)
deriche_kenar_tespit_anasayfa.place(x=690, y=60)


def kamera_harris_kose_tespit():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerHarris(gray, 3, 3, 0.04)
        corners = cv2.dilate(corners, None)
        frame[corners > 0.01 * corners.max()] = [0, 0, 255]

        cv2.imshow('Harris Köşe Tespiti', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

def harris_kose_tespit_buton():
    harris_kose_tespit_window = tk.Toplevel(window)
    harris_kose_tespit_window.title("Harris Köşe Tespiti")
    harris_kose_tespit_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def harris_kose_tespit_function():
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 400))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        block_size = int(block_size_entry_var.get())
        corner_quality = float(corner_quality_entry_var.get())
        corners = cv2.cornerHarris(gray, block_size,3, corner_quality)
        corners = cv2.dilate(corners, None)
        image[corners > 0.01 * corners.max()] = [0, 0, 255]
        photo_image_harris = convert_image(image)
        image_label.config(image=photo_image_harris)
        image_label.photo = photo_image_harris

    image_label = tk.Label(harris_kose_tespit_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(harris_kose_tespit_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=350, pady=10)

    button = tk.Button(harris_kose_tespit_window, text="Kamerayı Aç", command=kamera_harris_kose_tespit)
    button.grid(row=1, sticky='ns', pady=10)

    block_size_label = tk.Label(harris_kose_tespit_window, text="Block Size :")
    block_size_label.grid(row=2, sticky='w', padx=350, pady=10)

    block_size_entry_var = tk.StringVar(value="")
    block_size_entry = tk.Entry(harris_kose_tespit_window, bg='#dddddd', textvariable=block_size_entry_var)
    block_size_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    corner_quality_label = tk.Label(harris_kose_tespit_window, text="Corner Quality :")
    corner_quality_label.grid(row=3, sticky='w', padx=350, pady=10)

    corner_quality_entry_var = tk.StringVar(value="")
    corner_quality_entry = tk.Entry(harris_kose_tespit_window, bg='#dddddd', textvariable=corner_quality_entry_var)
    corner_quality_entry.grid(row=3, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(harris_kose_tespit_window, text="Uygula", command=harris_kose_tespit_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

harris_kose_tespit_anasayfa=tk.Button(window, text="Harris Köşe Tespiti", fg='black', bg='#8b668b', command=harris_kose_tespit_buton, relief=tk.SOLID, width=20, height=10)
harris_kose_tespit_anasayfa.place(x=690, y=280)


def kamera_contur_tespit():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gray, contours, -1, (0, 255, 0), 2)

        cv2.imshow('Contur Tespiti', gray)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

def contur_tespit_buton():
    contur_tespit_window = tk.Toplevel(window)
    contur_tespit_window.title("Contur Tespit")
    contur_tespit_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def contur_tespit_function():
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 400))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        low_threshold = int(low_threshold_entry_var.get())
        high_threshold = int(high_threshold_entry_var.get())
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        photo_image_contur = convert_image_second_variation(image)
        image_label.config(image=photo_image_contur)
        image_label.photo = photo_image_contur

    image_label = tk.Label(contur_tespit_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(contur_tespit_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(contur_tespit_window, text="Kamerayı Aç", command=kamera_contur_tespit)
    button.grid(row=1, sticky='ns', pady=10)

    low_threshold_label = tk.Label(contur_tespit_window, text="Low Threshold :")
    low_threshold_label.grid(row=2, sticky='w', padx=330, pady=10)

    low_threshold_entry_var = tk.StringVar(value="")
    low_threshold_entry = tk.Entry(contur_tespit_window, bg='#dddddd', textvariable=low_threshold_entry_var)
    low_threshold_entry.grid(row=2, sticky='ns', padx=80, pady=10)

    high_threshold_label = tk.Label(contur_tespit_window, text="High Threshold :")
    high_threshold_label.grid(row=3, sticky='w', padx=330, pady=10)

    high_threshold_entry_var = tk.StringVar(value="")
    high_threshold_entry = tk.Entry(contur_tespit_window, bg='#dddddd', textvariable=high_threshold_entry_var)
    high_threshold_entry.grid(row=3, sticky='ns', padx=80, pady=10)

    uygula_buton = tk.Button(contur_tespit_window, text="Uygula", command=contur_tespit_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

contur_tespit_anasayfa=tk.Button(window, text="Contur Tespiti", fg='black', bg='#8b668b', command=contur_tespit_buton, relief=tk.SOLID, width=20, height=10)
contur_tespit_anasayfa.place(x=690, y=500)


def kamera_watershed():
    video = cv2.VideoCapture(0)

    while (True):
        ret, frame = video.read()

        image_gray = cv2.cvtColor(cv2.medianBlur(frame, 31), cv2.COLOR_BGR2GRAY)
        ret, imgTH = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        imgOPN = cv2.morphologyEx(imgTH, cv2.MORPH_OPEN, kernel, iterations=7)
        sureBG = cv2.dilate(imgOPN, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(imgOPN, cv2.DIST_L2, 5)
        ret, sureFG = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sureFG = np.uint8(sureFG)
        unknown = cv2.subtract(sureBG, sureFG)
        ret, markers = cv2.connectedComponents(sureFG, labels=5)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers)
        contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        image_copy = image.copy()
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(image_copy, contours, i, (255, 0, 0), 5)

        cv2.imshow('Watershed', image_copy)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print('bye')
            break

def watershed_buton():
    watershed_window = tk.Toplevel(window)
    watershed_window.title("Watershed")
    watershed_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def watershed_function():
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 400))
        image_blurred = cv2.medianBlur(image, 31)
        image_gray = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2GRAY)
        ret, imgTH = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        imgOPN = cv2.morphologyEx(imgTH, cv2.MORPH_OPEN, kernel, iterations=7)
        sureBG = cv2.dilate(imgOPN, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(imgOPN, cv2.DIST_L2, 5)
        ret, sureFG = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sureFG = np.uint8(sureFG)
        unknown = cv2.subtract(sureBG, sureFG)
        ret, markers = cv2.connectedComponents(sureFG, labels=5)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers)
        contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        image_copy = image.copy()
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(image_copy, contours, i, (255, 0, 0), 5)
        photo_image_watershed = convert_image_second_variation(image_copy)
        image_label.config(image=photo_image_watershed)
        image_label.photo = photo_image_watershed

    image_label = tk.Label(watershed_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(watershed_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(watershed_window, text="Kamerayı Aç", command=kamera_watershed)
    button.grid(row=1, sticky='ns', pady=10)

    uygula_buton = tk.Button(watershed_window, text="Uygula", command=watershed_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

watershed_anasayfa=tk.Button(window,text="Watershed", fg='black', bg='#8b668b', command=watershed_buton, relief=tk.SOLID, width=20, height=10)
watershed_anasayfa.place(x=910, y=60)


def kamera_face_cascade():
    video = cv2.VideoCapture(0)
    yuz_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while (True):
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yuzler = yuz_cascade.detectMultiScale(gray, 1.3, 5)

        for(x, y, w, h) in yuzler:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (85, 255, 0), 3)

        print(yuzler)

        cv2.imshow('Face Cascade', frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    video.release()
    cv2.destroyAllWindows()

def face_cascade_buton():
    face_cascade_window = tk.Toplevel(window)
    face_cascade_window.title("Face Cascade")
    face_cascade_window.geometry('1100x1000')

    def resim_sec():
        global image_path
        image_path = open_file()
        cv2_image = cv2.imread(image_path)
        resized_image = cv2.resize(cv2_image, (600, 400))
        photo_image = convert_image(resized_image)
        image_label.config(image=photo_image)
        image_label.photo = photo_image

    def face_cascade_function():
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 400))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,4)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        photo_image_cascade = convert_image(image)
        image_label.config(image=photo_image_cascade)
        image_label.photo = photo_image_cascade
        cv2.waitKey(0)

    image_label = tk.Label(face_cascade_window, image=photo_image)
    image_label.grid(padx=200)

    button = tk.Button(face_cascade_window, text="Resim Seç", command=resim_sec)
    button.grid(row=1, sticky='w', padx=370, pady=10)

    button = tk.Button(face_cascade_window, text="Kamerayı Aç", command=kamera_face_cascade)
    button.grid(row=1, sticky='ns', pady=10)

    uygula_buton = tk.Button(face_cascade_window, text="Uygula", command=face_cascade_function, bg='gray')
    uygula_buton.grid(row=4, padx=250, pady=10)

yuz_tanima_anasayfa=tk.Button(window, text="Face Cascade", fg='black', bg='#8b668b', command=face_cascade_buton, relief=tk.SOLID, width=20, height=10)
yuz_tanima_anasayfa.place(x=910, y=280)



window.mainloop()