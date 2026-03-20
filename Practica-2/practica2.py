import os
import cv2
import metrikz
import utility


#----- H261 -----#
def comprimir_y_analizar_h261(video_input_file, q_scale, ffmpeg_path='./ffmpeg.exe'):

    # Comprimeix un vídeo a H.261, extreu els frames i calcula mètriques de qualitat.
    # Configuració de rutes
    video_output_file = f'./Practica-2/video_comprimido.avi'
    frames_dir = f'./Practica-2/frames'
    
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Compressioooo amb fixes per H.261
    print(f"\n>>> Processant Q-SCALE {q_scale}...")
    command_comp = [
        ffmpeg_path, '-y', 
        '-i', video_input_file, 
        '-an', 
        '-vcodec', 'h261', 
        '-s', 'cif',           # H.261 requereix CIF o QCIF
        '-pix_fmt', 'yuv420p', # H.261 requereix YUV420
        '-r', '29.97',         # Frame rate estàndard compatible
        '-q:v', str(q_scale), 
        video_output_file
    ]
    utility.execute_command(command_comp)

    # extracció de framesss (Original i Codificat)
    # Extraiem l'original redimensionat a CIF per poder comparar-lo (si no, les mides no quadrarien)
    utility.execute_command([
        ffmpeg_path, '-y', '-i', video_input_file, 
        '-s', 'cif', f'{frames_dir}/original%d.png'
    ])
    
    utility.execute_command([
        ffmpeg_path, '-y', '-i', video_output_file, 
        f'{frames_dir}/encoded%d.png'
    ])

    # métriquess
    frame_idx = 1
    total_ssim, total_mse, total_snr = 0, 0, 0
    
    while True:
        source = cv2.imread(f'{frames_dir}/original{frame_idx}.png')
        target = cv2.imread(f'{frames_dir}/encoded{frame_idx}.png')

        if source is None or target is None:
            break

        total_ssim += metrikz.ssim(source, target)
        total_mse += metrikz.mse(source, target)
        total_snr += metrikz.snr(source, target)
        frame_idx += 1

    num_frames = frame_idx - 1

    # -- Resultatss
    if num_frames > 0:
        print("-" * 30)
        print(f"RESULTATS FINALS Q={q_scale}")
        print(f"Frames: {num_frames}")
        print(f"MSE Mitjà:  {total_mse / num_frames:.4f}")
        print(f"SNR Mitjà:  {total_snr / num_frames:.4f} dB")
        print(f"SSIM Mitjà: {total_ssim / num_frames:.4f}")
        print("-" * 30)
    else:
        print(f"Error en processar el vídeo amb Q={q_scale}")

#----- MPEG-2 -----#

def comprimir_y_analizar_mpeg2(video_input_file, q_scale, ffmpeg_path='./ffmpeg.exe'):
    # Comprimeix un vídeo a MPEG-2, extreu els frames i calcula mètriques de qualitat.
    # Configuració de rutes
    video_output_file = f'./Practica-2/video_comprimido_mpeg2.avi'
    frames_dir = f'./Practica-2/frames'
    
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Compressió amb MPEG-2
    print(f"\n>>> Processant MPEG-2 amb Q-SCALE {q_scale}...")
    command_comp = [
        ffmpeg_path, '-y', 
        '-i', video_input_file, 
        '-an', 
        '-vcodec', 'mpeg2video', 
        '-q:v', str(q_scale), 
        video_output_file
    ]
    utility.execute_command(command_comp)

    # Extracció de frames (Original i Codificat)
    # Reescalem a CIF per coherència en la comparació de mètriques
    utility.execute_command([
        ffmpeg_path, '-y', '-i', video_input_file, 
        '-s', 'cif', f'{frames_dir}/original%d.png'
    ])
    
    utility.execute_command([
        ffmpeg_path, '-y', '-i', video_output_file, 
        '-s', 'cif', f'{frames_dir}/encoded%d.png'
    ])

    # Càlcul de mètriques
    frame_idx = 1
    total_ssim, total_mse, total_snr = 0, 0, 0
    
    while True:
        source = cv2.imread(f'{frames_dir}/original{frame_idx}.png')
        target = cv2.imread(f'{frames_dir}/encoded{frame_idx}.png')

        if source is None or target is None:
            break

        total_ssim += metrikz.ssim(source, target)
        total_mse += metrikz.mse(source, target)
        total_snr += metrikz.snr(source, target)
        frame_idx += 1

    num_frames = frame_idx - 1

    # Resultats
    if num_frames > 0:
        print("-" * 30)
        print(f"RESULTATS MPEG-2 Q={q_scale}")
        print(f"Frames analitzats: {num_frames}")
        print(f"MSE Mitjà:  {total_mse / num_frames:.4f}")
        print(f"SNR Mitjà:  {total_snr / num_frames:.4f} dB")
        print(f"SSIM Mitjà: {total_ssim / num_frames:.4f}")
        print("-" * 30)
    else:
        print(f"Error en analitzar els frames de MPEG-2.")

if __name__ == '__main__':
    # Defineix paths
    print(f"Directori de treball actual: {os.getcwd()}")
    mi_ffmpeg = './Practica-2/ffmpeg.exe'
    mi_video = './Practica-2/videos/ducks_take_off_420_720p50.y4m'
    
    q = 2 # 2, 10, 20 31
    
    #comprimir_y_analizar_h261(mi_video, q, mi_ffmpeg)

    comprimir_y_analizar_mpeg2(mi_video, q, mi_ffmpeg)


    #codigo de marco:
    
# if __name__ == '__main__':
    
#     ruta_actual = os.getcwd()
#     print(f"Estic treballant a: {ruta_actual}")

# 	#Configurem arxius d'entrada i sortida
#     video_input_file = './Practica-2/videos/flower_garden_422_ntsc.y4m' # Posem aqui el nom del vídeo que volem comprimir
#     q_scale = '31'                                       # Probar amb diferents valors de Q-scale (2,10,20,31)
#     video_output_file = f'./video_comprimido_q{q_scale}.avi'

#     # Creem carpeta per guardar els frames si no existeix
#     if not os.path.exists('./frames'):
#         os.makedirs('./frames')

#     # COMPRIMIM EL VÍDEO
#     print(f"Comprimim vídeo a H.261 amb Q-scale {q_scale}...")
#     command = [
#         './Practica-2/ffmpeg.exe', '-y', '-an', 
#         '-i', video_input_file, 
#         '-s', 'cif',             
#         '-q:v', q_scale, 
#         '-vcodec', 'h261', 
#         video_output_file,
#     ]
#     utility.execute_command(command)

#     # EXTRACCIÓ DE FRAMES
#     print("Extrayent frames del vídeo ORIGINAL...")
#     command = [
#          './Practica-2/ffmpeg.exe', '-y', 
#          '-i', video_input_file,
#          '-s', 'cif',            
#          './frames/original%d.png', 
#      ]
#     utility.execute_command(command)        

#     print("Extrayent frames del vídeo CODIFICAT...")
#     command = [
#          './Practica-2/ffmpeg.exe', '-y', 
#          '-i', video_output_file,
#          './frames/encoded%d.png', 
#      ]
#     utility.execute_command(command)
    
#     # CALCUL DE LES METRIQUES
#     print("Calculant les mètriques de qualitat per cada frame...")
    
#     frame_idx = 1
#     total_ssim, total_mse, total_snr = 0, 0, 0
    
#     # llEGIM ELS FRAMES FINS QUE NO QUEDIN MÉS
#     while True:
#         source = cv2.imread(f'./frames/original{frame_idx}.png')
#         target = cv2.imread(f'./frames/encoded{frame_idx}.png')

#         # Si no hi ha més frames, sortim del bucle
#         if source is None or target is None:
#             break

#         # Sumem a les metriques totals
#         total_ssim += metrikz.ssim(source, target) # Calcula la similitud estructural entre les dues imatges. El valor oscil·la entre -1 i 1, on 1 indica que les imatges són idèntiques. Com més proper a 1, millor qualitat té la imatge codificada.
#         total_mse += metrikz.mse(source, target) # Resta valor pixel original - pixel codificat, eleva al quadrat, i fa la mitjana de tots els pixels. Quant més alt sigui el MSE, pitjor qualitat té la imatge codificada.
#         total_snr += metrikz.snr(source, target) # Calcula la relació entre el senyal original i el soroll introduït per la codificació. Es mesura en decibels (dB). Com més alt sigui el SNR, millor qualitat té la imatge codificada.
        
#         frame_idx += 1

#     num_frames = frame_idx - 1

#     # Resultats finals
#     if num_frames > 0:
#         print("\n" + "="*40)
#         print(f"RESULTATS PER Q-SCALE {q_scale}")
#         print("="*40)
#         print(f"Total de frames analitzats: {num_frames}")
#         print(f"MSE Mitjà:  {total_mse / num_frames:.4f}")
#         print(f"SNR Mitjà:  {total_snr / num_frames:.4f} dB")
#         print(f"SSIM Mitjà: {total_ssim / num_frames:.4f}")
#         print("="*40)
#     else:
#         print("Error: No s'han pogut analitzar els frames. Comprova que els vídeos s'han processat correctament.")
