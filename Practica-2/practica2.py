import os
import cv2
import metrikz
import utility
import matplotlib.pyplot as plt

#----- H261 -----#
def comprimir_y_analizar_h261(video_input_file, q_scale, ffmpeg_path='./ffmpeg.exe'):

    # Comprimeix un vídeo a H.261, extreu els frames i calcula mètriques de qualitat.
    # Configuració de rutes
    video_output_file = f'./Practica-2/video_comprimido_h621.avi'
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
    # total_ssim, total_mse, total_snr = 0, 0, 0
    
    while True:
        source = cv2.imread(f'{frames_dir}/original{frame_idx}.png')
        target = cv2.imread(f'{frames_dir}/encoded{frame_idx}.png')

        if source is None or target is None:
            break

        # total_ssim += metrikz.ssim(source, target)
        # total_mse += metrikz.mse(source, target)
        # total_snr += metrikz.snr(source, target)
        frame_idx += 1

    num_frames = frame_idx - 1

    # -- Resultatss
    if num_frames > 0:
        print("-" * 30)
        print(f"RESULTATS FINALS Q={q_scale}")
        print(f"Frames: {num_frames}")
        # print(f"MSE Mitjà:  {total_mse / num_frames:.4f}")
        # print(f"SNR Mitjà:  {total_snr / num_frames:.4f} dB")
        # print(f"SSIM Mitjà: {total_ssim / num_frames:.4f}")
        print("-" * 30)

         # -- Resultatss
        if os.path.exists(video_output_file):
            return os.path.getsize(video_output_file)
        else:
            print(f"Error: No s'ha creat el fitxer per a Q={q_scale}")
            return None


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
    mse_history = []
    ssim_history = []
    snr_history = []
    
    while True:
        source = cv2.imread(f'{frames_dir}/original{frame_idx}.png')
        target = cv2.imread(f'{frames_dir}/encoded{frame_idx}.png')

        if source is None or target is None:
            break

        total_ssim += metrikz.ssim(source, target)
        total_mse += metrikz.mse(source, target)
        total_snr += metrikz.snr(source, target)

        mse_history.append(metrikz.mse(source, target))
        ssim_history.append(metrikz.ssim(source, target))
        snr_history.append(metrikz.snr(source, target))
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
        if os.path.exists(video_output_file):
            return os.path.getsize(video_output_file), mse_history, ssim_history, snr_history
        else:
            print(f"Error: No s'ha creat el fitxer per a Q={q_scale}")
            return None
        
        
def generate_plots(all_results):
    """
    all_results: A dictionary where keys are video names and values are 
    dictionaries containing the lists of MSE, SSIM, and SNR.
    """
    metrics = ['MSE', 'SSIM', 'SNR']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for video_name, data in all_results.items():
            plt.plot(data[metric.lower()], label=video_name)
        
        plt.title(f'Figura: {metric} vs #quadre')
        plt.xlabel('Número de Quadre (#)')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        # plt.savefig(f'./Practica-2/figura_{metric.lower()}.png') # Optional: save to disk
        plt.show()

if __name__ == '__main__':
    # Defineix paths
    print(f"Directori de treball actual: {os.getcwd()}")

    videos_dataset = [
        './Practica-2/videos/hall_monitor_cif.y4m',
        './Practica-2/videos/ducks_take_off_420_720p50.y4m',
        './Practica-2/videos/flower_garden_422_ntsc.y4m',
        './Practica-2/videos/galleon_422_ntsc.y4m',
        './Practica-2/videos/pamphlet_cif.y4m'
    ]

    results_for_plotting = {}
    chosen_q = 8

    mi_ffmpeg = './Practica-2/ffmpeg.exe'
    # mi_video = './Practica-2/videos/hall_monitor_cif.y4m'
    #q_values = [2, 10, 20, 31] # 2, 10, 20 31
    q_values = [10]

    for mi_video in videos_dataset:
        mida_orig = os.path.getsize(mi_video)
        for q in q_values:
            # mida_com = comprimir_y_analizar_h261(mi_video, i, mi_ffmpeg)
                mida_com, mse_l, ssim_l, snr_l = comprimir_y_analizar_mpeg2(mi_video, q, mi_ffmpeg)



                if mida_com:
                # Conversió a MB
                    mida_mb = mida_com / (1024 * 1024)
                    ratio = mida_orig / mida_com

                    results_for_plotting[mi_video] = {
                    'mse': mse_l,
                    'ssim': ssim_l,
                    'snr': snr_l
                }
                    
                    print(f"Q-Scale: {q} | Mida: {mida_mb:.2f} MB | Ratio: {ratio:.2f}:1\n")
                else: print("Error")

    # Generate the 3 Figures required in section 3.1
    generate_plots(results_for_plotting)
