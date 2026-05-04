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
    """
    Realiza el block matching entre dos frames.
    - Si search_window es None, hace la V1 (Búsqueda Exhaustiva).
    - Si search_window es un entero (ej: 24), hace la V2 (Búsqueda en región).
    """
    h, w = frame1.shape
    actual_position = []
    motion_vector = []
    errors_prediction = []

    start_time = time.time()

    # 1. Recorremos el frame actual (frame2) en bloques de 8x8
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            # Extraemos el bloque actual (necesitamos float para operaciones precisas de error y DCT)
            block2 = frame2[i:i+block_size, j:j+block_size].astype(np.float32)

            min_mse = float('inf')
            best_match_pos = (i, j) # Por defecto, la misma posición (sin movimiento)
            best_error_block = np.zeros((block_size, block_size), dtype=np.float32)

            # 2. Definir límites de búsqueda según la versión (V1 o V2)
            if search_window is not None:
                # V2: Búsqueda restringida alrededor del bloque (cuidando los bordes de la imagen)
                u_min = max(0, i - search_window // 2)
                u_max = min(h - block_size + 1, i + search_window // 2 + 1)
                v_min = max(0, j - search_window // 2)
                v_max = min(w - block_size + 1, j + search_window // 2 + 1)
            else:
                # V1: Búsqueda exhaustiva en toda la imagen
                u_min, u_max = 0, h - block_size + 1
                v_min, v_max = 0, w - block_size + 1

            # 3. Buscar el bloque más parecido en el frame1
            for u in range(u_min, u_max):
                for v in range(v_min, v_max):
                    block1 = frame1[u:u+block_size, v:v+block_size].astype(np.float32)

                    # Calculamos el error cuadrático medio (MSE) entre bloques
                    mse_val = np.mean((block2 - block1) ** 2)

                    # Si encontramos un error menor, actualizamos el mejor bloque
                    if mse_val < min_mse:
                        min_mse = mse_val
                        best_match_pos = (u, v)
                        best_error_block = block2 - block1

            # Guardamos las coordenadas en formato (X, Y) que es el que usa OpenCV para dibujar
            actual_position.append((j, i))
            motion_vector.append((best_match_pos[1], best_match_pos[0]))

            # 4. Transformar, cuantizar y aplicar zigzag al error
            dct_block = dct2(best_error_block)
            quantized_block = quantit(dct_block)
            
            # Usamos la función zigzag del archivo proporcionado
            zz_error = eines_sessio3.zigzag(quantized_block)
            errors_prediction.append(zz_error)

    execution_time = time.time() - start_time
    print(f"-> Block Matching completado en {execution_time:.2f} segons.")
    
    return actual_position, motion_vector, errors_prediction

def dibuixa_vectors(imatge_fons, positions, motions):
    # Convertim a color per veure el vermell
    moviments_detectats = 0
    img_out = cv2.cvtColor(imatge_fons, cv2.COLOR_GRAY2BGR)
    for p_act, p_mov in zip(positions, motions):
        if p_act != p_mov:
            # Dibuixem línia vermella (BGR: 0, 0, 255)
            cv2.line(img_out, p_act, p_mov, (0, 0, 255), 1)
            moviments_detectats += 1
            
    print(f"S'han detectat {moviments_detectats} blocs amb moviment.")
    return img_out


if __name__ == '__main__':


    img1 = cv2.imread("frame0_1.png")     #frame anterior
    img2 = cv2.imread("frame0_2.png")     #frame actual

    # convertim a gris per simplicitat
    frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #mostra les dimensions de la imatge
    print("dimensions de la imatge= ")
    print(frame1.shape)
    dim=frame1.shape

    bk=np.zeros((8, 8)) # matriu per generar els blocs 

    # MOSTRAR IMATGES ORIGINALS
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(frame1, cmap='gray')
    plt.title('Frame 1 (Anterior)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(frame2, cmap='gray')
    plt.title('Frame 2 (Actual)')
    plt.axis('off')
    
    plt.suptitle("Imatges Originals")
    plt.show(block=False) # No bloqueja l'execució del codi següent
    plt.pause(2)

    # GENERAR AQUI EL CODI PER FER EL MOTION VECTORS
    # ######################
    print("Executant Versió 1 (Exhaustiva")
    pos1, mov1, errors1 = block_matching(frame1, frame2, block_size=8, search_window=None)
    
    print("Executant Versió 2 (Cerca restringida a 24 píxels)")
    pos2, mov2, errors2 = block_matching(frame1, frame2, block_size=8, search_window=24)


    # GENERAR PER ULTIM EL CODI DE VISUALITZACIO 
    # ######################
    vis_v1 = dibuixa_vectors(frame1, pos1, mov1)
    vis_v2 = dibuixa_vectors(frame1, pos2, mov2)

    plt.figure(figsize=(15, 7))

    # Subplot per a la V1
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(vis_v1, cv2.COLOR_BGR2RGB))
    plt.title('V1: Cerca Exhaustiva')
    plt.axis('off')

    # Subplot per a la V2
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(vis_v2, cv2.COLOR_BGR2RGB))
    plt.title('V2: Cerca Restringida (24px)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()