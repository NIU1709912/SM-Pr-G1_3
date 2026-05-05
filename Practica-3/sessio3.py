import cv2
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
import numpy as np
import metrikz
import eines_sessio3
import time

quantization_matrix =  [[16.,11.,10.,16.,24.,40.,51.,61.],
                        [12.,12.,14.,19.,26.,58.,60.,55.],
                        [14.,13.,16.,24.,40.,57.,69.,56.],
                        [14.,17.,22.,29.,51.,87.,80.,62.],
                        [18.,22.,37.,56.,68.,109.,103.,77.],
                        [24.,35.,55.,64.,81.,104.,113.,92.],
                        [49.,64.,78.,87.,103.,121.,120.,101.],
                        [72.,92.,95.,98.,112.,100.,103.,99.]]


#transformada DCT de matrius
def dct2(block):
    return dct(dct(block, axis=0, norm = 'ortho'), axis=1, norm = 'ortho'); 


#inversa de la transformada DCT de matrius
def idct2(block):
    return np.round(idct(idct(block, axis=1, norm = 'ortho'), axis=0, norm = 'ortho'));

    
#funcio de quantitzacio
def quantit(block):
    quant = np.round(block / quantization_matrix);
    return quant


# funcio inversa de de quantitzacio
def iquantit(block):
    inverse = (block * quantization_matrix);
    return inverse


def block_matching(frame1, frame2, block_size=8, search_window=None):
    h, w = frame1.shape
    actual_pos = []
    motion_vec = []
    errors_predict = []

    start_time = time.time()

    # Recorremos el frame actual (frame2) en bloques de 8x8
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            # Extraemos el bloque actual (float para precision en error y DCT)
            block2 = frame2[i:i+block_size, j:j+block_size].astype(np.float32)

            min_mse = float('inf')
            best_pos = (i, j) # Por defecto, la misma pos (no movement)
            best_error_block = np.zeros((block_size, block_size), dtype=np.float32)

            # Definir limites de busqueda segun V1 o V2
            if search_window is not None:
                # V2: Busqueda restringida alrededor del bloque (cuidando bordes)
                u_min = max(0, i - search_window // 2)
                u_max = min(h - block_size + 1, i + search_window // 2 + 1)
                v_min = max(0, j - search_window // 2)
                v_max = min(w - block_size + 1, j + search_window // 2 + 1)
            else:
                # V1: Busqueda exhaustiva
                u_min, u_max = 0, h - block_size + 1
                v_min, v_max = 0, w - block_size + 1

            # Buscar el bloque mas parecido en el frame1
            for u in range(u_min, u_max):
                for v in range(v_min, v_max):
                    block1 = frame1[u:u+block_size, v:v+block_size].astype(np.float32)

                    # Calculamos MSE entre bloques
                    mse_val = np.mean((block2 - block1) ** 2)

                    # Si encontramos un error menor, actualizamos el mejor bloque
                    if mse_val < min_mse:
                        min_mse = mse_val
                        best_pos = (u, v)
                        best_error_block = block2 - block1

            # Guardamos las coordenadas (X, Y) para dibujar con OpenCV
            actual_pos.append((j, i))
            motion_vec.append((best_pos[1], best_pos[0]))

            # Transformar, cuantizar y aplicar zigzag al error
            dct_block = dct2(best_error_block)
            quantit_block = quantit(dct_block)
            zz_error = eines_sessio3.zigzag(quantit_block)
            errors_predict.append(zz_error)

    exec_time = time.time() - start_time
    print(f"Block Matching completat en {exec_time:.2f} segons.")
    
    return actual_pos, motion_vec, errors_predict




if __name__ == '__main__':

    #frame anterior
    img1 = cv2.imread("frame0_1.png")

    #frame actual
    img2 = cv2.imread("frame0_2.png")

    # convertim a gris per simplicitat
    frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #mostra les dimensions de la imatge
    print("dimensions de la imatge= ")
    print(frame1.shape)
    dim=frame1.shape

    #mostra els dos frames per separat en mida real, es pot canviar per matplotlib per veure mes gran.
    cv2.imshow('frame 1',frame1)
    cv2.imshow('frame 2', frame2)


    #vectors finals
    actual_pos=[]
    motion_vec=[]
    errors_predict=[]

    # matriu per generar els blocs 
    bk=np.zeros((8, 8))


    # GENERAR AQUI EL CODI PER FER EL MOTION VECTORS
    # ######################
    print("Executant Versio 2 (Cerca restringida a 24 pixels)...")
    # search_window=None para V1
    actual_pos, motion_vec, errors_predict = block_matching(frame1, frame2, block_size=8, search_window=24)


    # GENERAR PER ULTIM EL CODI DE VISUALITZACIO 
    # ######################
    
    #### Fent servir CV2 #########
    # Crear una copia en color per dibuixar la línia
    img_gris_color = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)

    # Bucle per dibuixar nomes els vectors que tenen moviment
    moviments_detectats = 0
    for pos_actual, pos_moviment in zip(actual_pos, motion_vec):
        if pos_actual != pos_moviment:
            cv2.line(img_gris_color, pos_actual, pos_moviment, (0, 0, 255), 1)
            moviments_detectats += 1
            
    print(f"S'han detectat {moviments_detectats} blocs amb moviment.")

    # Mostra la imatge amb cv2.imshow amb les linies vermelles
    cv2.imshow('Imatge amb moviments marcats', img_gris_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()