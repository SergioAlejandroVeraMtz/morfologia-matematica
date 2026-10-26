import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class SistemaMorfologiaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Morfología Matemática - ESCOM IPN")
        self.root.geometry("1400x800")
        
        # Variables de control
        self.imagen_original = None
        self.imagen_procesada = None
        self.imagen_original_pil = None  # Para mantener la imagen PIL
        self.imagen_procesada_pil = None  # Para mantener la imagen PIL procesada
        self.tipo_imagen = tk.StringVar(value="grises")
        self.operacion = tk.StringVar(value="erosion")
        self.tam_kernel = tk.IntVar(value=5)
        self.forma_kernel = tk.StringVar(value="rectangular")
        self.iteraciones = tk.IntVar(value=1)
        
        self.configurar_interfaz()
        
    def configurar_interfaz(self):
        # Frame superior para controles
        frame_controles = tk.Frame(self.root, bg="#f0f0f0", relief=tk.RAISED, bd=2)
        frame_controles.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Título
        tk.Label(frame_controles, text="SISTEMA DE MORFOLOGÍA MATEMÁTICA", 
                font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)
        
        # Panel de carga de imágenes
        frame_carga = tk.LabelFrame(frame_controles, text="Carga de Imagen", 
                                   font=("Arial", 10, "bold"), bg="#f0f0f0")
        frame_carga.pack(side=tk.LEFT, padx=10, pady=5)
        
        tk.Button(frame_carga, text="Cargar Imagen", command=self.cargar_imagen,
                 bg="#4CAF50", fg="white", font=("Arial", 10)).pack(pady=5)
        
        tk.Label(frame_carga, text="Tipo de Imagen:", bg="#f0f0f0").pack()
        ttk.Combobox(frame_carga, textvariable=self.tipo_imagen, 
                    values=["binaria", "grises"], width=15).pack(pady=5)
        
        # Panel de parámetros
        frame_param = tk.LabelFrame(frame_controles, text="Parámetros", 
                                   font=("Arial", 10, "bold"), bg="#f0f0f0")
        frame_param.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Operaciones
        tk.Label(frame_param, text="Operación:", bg="#f0f0f0").pack()
        operaciones = [
            "erosion", "dilatacion", "apertura_trad", "apertura_cv",
            "cierre_trad", "cierre_cv", "gradiente", "tophat", "blackhat"
        ]
        self.combo_operacion = ttk.Combobox(frame_param, textvariable=self.operacion, 
                    values=operaciones, width=15, state="readonly")
        self.combo_operacion.pack(pady=5)
        
        # Tamaño del kernel
        tk.Label(frame_param, text="Tamaño Kernel:", bg="#f0f0f0").pack()
        ttk.Scale(frame_param, from_=3, to=31, variable=self.tam_kernel, 
                 orient=tk.HORIZONTAL, length=150, command=self.actualizar_tam_kernel).pack()
        self.label_tam_kernel = tk.Label(frame_param, textvariable=self.tam_kernel, bg="#f0f0f0")
        self.label_tam_kernel.pack()
        
        # Forma del kernel
        tk.Label(frame_param, text="Forma Kernel:", bg="#f0f0f0").pack()
        self.combo_forma = ttk.Combobox(frame_param, textvariable=self.forma_kernel,
                    values=["rectangular", "cruz", "elipse"], width=15, state="readonly")
        self.combo_forma.pack(pady=5)
        
        # Iteraciones
        tk.Label(frame_param, text="Iteraciones:", bg="#f0f0f0").pack()
        self.spin_iteraciones = ttk.Spinbox(frame_param, from_=1, to=20, textvariable=self.iteraciones, 
                   width=10)
        self.spin_iteraciones.pack(pady=5)
        
        # Botón de procesamiento
        frame_botones = tk.Frame(frame_controles, bg="#f0f0f0")
        frame_botones.pack(side=tk.LEFT, padx=10, pady=5)
        
        tk.Button(frame_botones, text="Aplicar Operación", command=self.aplicar_operacion,
                 bg="#2196F3", fg="white", font=("Arial", 11, "bold"),
                 width=15, height=2).pack(pady=10)
        
        tk.Button(frame_botones, text="Guardar Resultado", command=self.guardar_imagen,
                 bg="#FF9800", fg="white", font=("Arial", 10), width=15).pack(pady=5)
        
        tk.Button(frame_botones, text="Mostrar Métricas", command=self.mostrar_metricas,
                 bg="#9C27B0", fg="white", font=("Arial", 10), width=15).pack(pady=5)
        
        tk.Button(frame_botones, text="Restablecer", command=self.restablecer,
                 bg="#F44336", fg="white", font=("Arial", 10), width=15).pack(pady=5)
        
        # Frame principal para imágenes
        frame_principal = tk.Frame(self.root)
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para imagen original
        frame_original = tk.LabelFrame(frame_principal, text="Imagen Original",
                                      font=("Arial", 11, "bold"))
        frame_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.canvas_original = tk.Canvas(frame_original, bg="lightgray", width=400, height=400)
        self.canvas_original.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.texto_original = self.canvas_original.create_text(200, 200, 
                                                              text="Cargue una imagen...",
                                                              font=("Arial", 12))
        
        # Frame para imagen procesada
        frame_procesada = tk.LabelFrame(frame_principal, text="Imagen Procesada",
                                       font=("Arial", 11, "bold"))
        frame_procesada.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.canvas_procesada = tk.Canvas(frame_procesada, bg="lightgray", width=400, height=400)
        self.canvas_procesada.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.texto_procesada = self.canvas_procesada.create_text(200, 200,
                                                                text="Resultado aparecerá aquí...",
                                                                font=("Arial", 12))
        
        # Barra de estado
        self.status_bar = tk.Label(self.root, text="Listo", bd=1, relief=tk.SUNKEN,
                                  anchor=tk.W, bg="white")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Vincular eventos de redimensionamiento
        self.root.bind("<Configure>", self.redimensionar_imagenes)
        
    def actualizar_tam_kernel(self, event=None):
        # Asegurar que el tamaño del kernel sea impar
        valor = self.tam_kernel.get()
        if valor % 2 == 0:
            valor += 1
            self.tam_kernel.set(valor)
        
    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                      ("Todos los archivos", "*.*")]
        )
        
        if ruta:
            try:
                # Cargar imagen
                img_pil = Image.open(ruta)
                
                # Convertir a RGB si es RGBA
                if img_pil.mode == 'RGBA':
                    img_pil = img_pil.convert('RGB')
                
                self.imagen_original_pil = img_pil.copy()
                
                # Convertir a OpenCV
                if self.tipo_imagen.get() == "binaria":
                    # Convertir a escala de grises primero
                    img_gris = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
                    # Aplicar umbral
                    _, self.imagen_original = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY)
                else:
                    # Convertir a escala de grises
                    img_pil_gris = img_pil.convert('L')
                    self.imagen_original = np.array(img_pil_gris)
                
                # Mostrar imagen
                self.mostrar_imagen_original()
                self.status_bar.config(text=f"Imagen cargada: {os.path.basename(ruta)} - Tamaño: {img_pil.size}")
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
                print(f"Error detallado: {e}")
    
    def crear_kernel(self):
        tam = self.tam_kernel.get()
        # Asegurar que el tamaño sea impar
        if tam % 2 == 0:
            tam += 1
            self.tam_kernel.set(tam)
        
        forma = self.forma_kernel.get()
        
        if forma == "rectangular":
            return np.ones((tam, tam), np.uint8)
        elif forma == "cruz":
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (tam, tam))
        elif forma == "elipse":
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam, tam))
        return np.ones((tam, tam), np.uint8)
    
    def aplicar_operacion(self):
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero cargue una imagen")
            return
        
        # Asegurar tamaño impar del kernel
        self.actualizar_tam_kernel()
        
        kernel = self.crear_kernel()
        iteraciones = self.iteraciones.get()
        operacion = self.operacion.get()
        
        try:
            if operacion == "erosion":
                self.imagen_procesada = cv2.erode(self.imagen_original, kernel, iterations=iteraciones)
            elif operacion == "dilatacion":
                self.imagen_procesada = cv2.dilate(self.imagen_original, kernel, iterations=iteraciones)
            elif operacion == "apertura_trad":
                erosionada = cv2.erode(self.imagen_original, kernel, iterations=iteraciones)
                self.imagen_procesada = cv2.dilate(erosionada, kernel, iterations=iteraciones)
            elif operacion == "apertura_cv":
                self.imagen_procesada = cv2.morphologyEx(self.imagen_original, cv2.MORPH_OPEN, kernel, iterations=iteraciones)
            elif operacion == "cierre_trad":
                dilatada = cv2.dilate(self.imagen_original, kernel, iterations=iteraciones)
                self.imagen_procesada = cv2.erode(dilatada, kernel, iterations=iteraciones)
            elif operacion == "cierre_cv":
                self.imagen_procesada = cv2.morphologyEx(self.imagen_original, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)
            elif operacion == "gradiente":
                self.imagen_procesada = cv2.morphologyEx(self.imagen_original, cv2.MORPH_GRADIENT, kernel, iterations=iteraciones)
            elif operacion == "tophat":
                self.imagen_procesada = cv2.morphologyEx(self.imagen_original, cv2.MORPH_TOPHAT, kernel, iterations=iteraciones)
            elif operacion == "blackhat":
                self.imagen_procesada = cv2.morphologyEx(self.imagen_original, cv2.MORPH_BLACKHAT, kernel, iterations=iteraciones)
            
            # Convertir a PIL para mostrar
            if len(self.imagen_procesada.shape) == 2:
                self.imagen_procesada_pil = Image.fromarray(self.imagen_procesada)
            else:
                self.imagen_procesada_pil = Image.fromarray(cv2.cvtColor(self.imagen_procesada, cv2.COLOR_BGR2RGB))
            
            self.mostrar_imagen_procesada()
            self.status_bar.config(text=f"Operación aplicada: {operacion} - Kernel: {self.forma_kernel.get()} {self.tam_kernel.get()}x{self.tam_kernel.get()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar: {str(e)}")
            print(f"Error detallado: {e}")
    
    def mostrar_imagen_original(self):
        if self.imagen_original_pil is None:
            return
            
        self.canvas_original.delete("all")
        tamaño = self.obtener_tamaño_ajustado(self.canvas_original, self.imagen_original_pil)
        
        img_redimensionada = self.imagen_original_pil.resize(tamaño, Image.Resampling.LANCZOS)
        self.img_tk_original = ImageTk.PhotoImage(img_redimensionada)
        
        # Centrar la imagen en el canvas
        x = (self.canvas_original.winfo_width() - tamaño[0]) // 2
        y = (self.canvas_original.winfo_height() - tamaño[1]) // 2
        self.canvas_original.create_image(x + tamaño[0]//2, y + tamaño[1]//2, 
                                         image=self.img_tk_original, anchor=tk.CENTER)
    
    def mostrar_imagen_procesada(self):
        if self.imagen_procesada_pil is None:
            return
            
        self.canvas_procesada.delete("all")
        tamaño = self.obtener_tamaño_ajustado(self.canvas_procesada, self.imagen_procesada_pil)
        
        img_redimensionada = self.imagen_procesada_pil.resize(tamaño, Image.Resampling.LANCZOS)
        self.img_tk_procesada = ImageTk.PhotoImage(img_redimensionada)
        
        # Centrar la imagen en el canvas
        x = (self.canvas_procesada.winfo_width() - tamaño[0]) // 2
        y = (self.canvas_procesada.winfo_height() - tamaño[1]) // 2
        self.canvas_procesada.create_image(x + tamaño[0]//2, y + tamaño[1]//2, 
                                          image=self.img_tk_procesada, anchor=tk.CENTER)
    
    def obtener_tamaño_ajustado(self, canvas, imagen_pil):
        ancho_canvas = canvas.winfo_width()
        alto_canvas = canvas.winfo_height()
        
        # Si el canvas no tiene tamaño aún, usar valores por defecto
        if ancho_canvas < 10 or alto_canvas < 10:
            ancho_canvas = 400
            alto_canvas = 400
        
        ancho_img, alto_img = imagen_pil.size
        
        # Calcular ratio de aspecto
        ratio_img = ancho_img / alto_img
        ratio_canvas = ancho_canvas / alto_canvas
        
        if ratio_img > ratio_canvas:
            # La imagen es más ancha que el canvas
            nuevo_ancho = ancho_canvas - 20  # Margen
            nuevo_alto = int(nuevo_ancho / ratio_img)
        else:
            # La imagen es más alta que el canvas
            nuevo_alto = alto_canvas - 20  # Margen
            nuevo_ancho = int(nuevo_alto * ratio_img)
        
        return (max(10, nuevo_ancho), max(10, nuevo_alto))
    
    def redimensionar_imagenes(self, event=None):
        # Redimensionar imágenes cuando se cambia el tamaño de la ventana
        if self.imagen_original_pil is not None:
            self.mostrar_imagen_original()
        if self.imagen_procesada_pil is not None:
            self.mostrar_imagen_procesada()
    
    def guardar_imagen(self):
        if self.imagen_procesada is None:
            messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar")
            return
        
        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif"), ("Todos", "*.*")]
        )
        
        if ruta:
            try:
                cv2.imwrite(ruta, self.imagen_procesada)
                messagebox.showinfo("Éxito", f"Imagen guardada en:\n{ruta}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la imagen: {str(e)}")
    
    def mostrar_metricas(self):
        if self.imagen_original is None or self.imagen_procesada is None:
            messagebox.showwarning("Advertencia", "Cargue y procese una imagen primero")
            return
        
        try:
            # Asegurar que ambas imágenes tengan el mismo tamaño
            if self.imagen_original.shape != self.imagen_procesada.shape:
                messagebox.showwarning("Advertencia", "Las imágenes tienen diferentes dimensiones")
                return
            
            # Calcular métricas
            mse = np.mean((self.imagen_original.astype("float") - self.imagen_procesada.astype("float")) ** 2)
            
            if mse == 0:
                psnr = float('inf')
            else:
                max_pixel = 255.0
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            # Calcular SSIM (asegurar que las imágenes sean del mismo tipo)
            if self.imagen_original.dtype != self.imagen_procesada.dtype:
                img1 = self.imagen_original.astype(np.float32)
                img2 = self.imagen_procesada.astype(np.float32)
            else:
                img1 = self.imagen_original.astype(np.float32)
                img2 = self.imagen_procesada.astype(np.float32)
            
            ssim_val = ssim(img1, img2, data_range=img1.max() - img1.min())
            
            # Crear ventana de métricas
            ventana_metricas = tk.Toplevel(self.root)
            ventana_metricas.title("Métricas de Calidad")
            ventana_metricas.geometry("400x400")
            ventana_metricas.resizable(False, False)
            
            tk.Label(ventana_metricas, text="MÉTRICAS DE CALIDAD", 
                    font=("Arial", 14, "bold")).pack(pady=10)
            
            frame_metricas = tk.Frame(ventana_metricas)
            frame_metricas.pack(pady=20, padx=20)
            
            metricas = [
                ("MSE (Error Cuadrático Medio):", f"{mse:.4f}"),
                ("PSNR (Relación Señal-Ruido):", f"{psnr:.2f} dB" if psnr != float('inf') else "Infinito"),
                ("SSIM (Similitud Estructural):", f"{ssim_val:.4f}"),
                ("Dimensión Original:", f"{self.imagen_original.shape}"),
                ("Dimensión Procesada:", f"{self.imagen_procesada.shape}"),
                ("Media Original:", f"{np.mean(self.imagen_original):.2f}"),
                ("Media Procesada:", f"{np.mean(self.imagen_procesada):.2f}"),
                ("Desviación Original:", f"{np.std(self.imagen_original):.2f}"),
                ("Desviación Procesada:", f"{np.std(self.imagen_procesada):.2f}")
            ]
            
            for i, (nombre, valor) in enumerate(metricas):
                tk.Label(frame_metricas, text=nombre, anchor="w", 
                        width=25, font=("Arial", 9)).grid(row=i, column=0, sticky="w", pady=2)
                tk.Label(frame_metricas, text=valor, font=("Arial", 9, "bold"),
                        width=15, anchor="w").grid(row=i, column=1, sticky="w", pady=2)
            
            # Botón para cerrar
            tk.Button(ventana_metricas, text="Cerrar", command=ventana_metricas.destroy,
                     bg="#2196F3", fg="white", width=15).pack(pady=20)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular métricas: {str(e)}")
    
    def restablecer(self):
        self.imagen_procesada = None
        self.imagen_procesada_pil = None
        self.canvas_procesada.delete("all")
        self.texto_procesada = self.canvas_procesada.create_text(
            self.canvas_procesada.winfo_width()//2,
            self.canvas_procesada.winfo_height()//2,
            text="Resultado aparecerá aquí...",
            font=("Arial", 12)
        )
        self.status_bar.config(text="Restablecido - Listo")

def main():
    try:
        root = tk.Tk()
        app = SistemaMorfologiaGUI(root)
        
        # Centrar ventana
        root.update_idletasks()
        ancho = root.winfo_width()
        alto = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (ancho // 2)
        y = (root.winfo_screenheight() // 2) - (alto // 2)
        root.geometry(f'{ancho}x{alto}+{x}+{y}')
        
        root.mainloop()
    except Exception as e:
        print(f"Error fatal: {e}")
        messagebox.showerror("Error Fatal", f"Ha ocurrido un error fatal: {str(e)}")

if __name__ == "__main__":
    main()