import os
import cv2
import metrikz
import utility
import pylab
import numpy as np
import matplotlib.pyplot as plt

def get_video_metrics(folder, video_name, codec_name, q_val):
    # Path to the input file inside the Videos folder
    real_video_folder = os.getcwd() + "/" + folder
    video_in = os.path.join(real_video_folder, video_name)
    
    # Check if the file actually exists before trying to compress it
    if not os.path.exists(video_in):
        print(f"Error: File not found at {video_in}")
        return None, None, None

    video_out = f"compressed_{codec_name}_q{q_val}_{video_name}.avi"
    
    # 1. Compress Video
    # Note: If you get a FileNotFoundError here again, 
    # make sure ffmpeg is installed and added to your Windows PATH.
    cmd_compress = [
        'ffmpeg', 
        '-y',
        '-i', video_in,
        '-q:v', str(q_val), 
        '-vcodec', codecs[codec_name],
        '-an', video_out
    ]
    utility.execute_command(cmd_compress)
    
    # 2. Get Compression Ratio
    original_size = os.path.getsize(video_in)
    compressed_size = os.path.getsize(video_out)
    ratio = original_size / compressed_size
    
    # 3. Extract Frames
    utility.execute_command(['ffmpeg', '-y', '-i', video_in, './frames/orig_%d.png'])
    utility.execute_command(['ffmpeg', '-y', '-i', video_out, './frames/enc_%d.png'])
    
    # 4. Calculate MSE
    mse_list = []
    frame_idx = 1
    while True:
        orig_path = f'./frames/orig_{frame_idx}.png'
        enc_path = f'./frames/enc_{frame_idx}.png'
        
        if not os.path.exists(orig_path) or not os.path.exists(enc_path):
            break
            
        img_orig = cv2.imread(orig_path)
        img_enc = cv2.imread(enc_path)
        
        # H.261 requires dimensions to be multiples of 16; resizing for comparison
        if img_orig.shape != img_enc.shape:
            img_enc = cv2.resize(img_enc, (img_orig.shape[1], img_orig.shape[0]))
            
        mse_list.append(metrikz.mse(img_orig, img_enc))
        frame_idx += 1
        
    return compressed_size, ratio, np.mean(mse_list)

if __name__ == '__main__':
	# Noms dels arxius d'entrada i sortida.
    video_folder = 'Practica-2/Videos'
    videos = [
        'ducks_take_off_420_720p50.y4m', 
        'flower_garden_422_ntsc.y4m', 
        'galleon_422_ntsc.y4m', 
        'hall_monitor_cif.y4m', 
        'pamphlet_cif.y4m'
    ]

    q_scales = [2, 8, 15, 31]
    codecs = {'h261': 'h261', 'mpeg2': 'mpeg2video'}

    # Ensure directory for frames exists
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

        # --- Execution ---
    print(f"{'Video':<25} | {'Codec':<6} | {'Q':<2} | {'Size(KB)':<10} | {'Ratio':<6} | {'Avg MSE'}")
    print("-" * 80)

    for v in videos:
        for c_key in codecs:
            for q in q_scales:
                size, ratio, avg_mse = get_video_metrics(video_folder, v, c_key, q)
                if size is not None:
                    print(f"{v[:25]:<25} | {c_key:<6} | {q:<2} | {size/1024:<10.1f} | {ratio:<6.2f} | {avg_mse:.2f}")

	# # Comanda per a la compresio a H.261
    # command = [
	#     'ffmpeg',
	#     '-y',
	#     '-an',
	#     '-i', video_input_file, 
	#     '-q:v', '15',
	#     '-vcodec', 'h261',
	#     video_output_file,
	# ]

	# # Executem la comanda
    # utility.execute_command(command)

	# # Comanda per a extraure els quadres (frames) del video original
    # command = [
	#      'ffmpeg',
	#      '-y',
	#      '-i', video_input_file,
	#      './frames/original%d.png', 
	#  ]

    # utility.execute_command(command)        

	# # Comanda per a extreure els quadres del video codificat
    # command = [
	#      'ffmpeg',
	#      '-y',
	#      '-i', video_output_file,
	#      './frames/encoded%d.png', 
	#  ]

    # utility.execute_command(command)
	
	# # Exemple per lleguir 1 imagatge i comparala amb la codificada, i calcular la metrica de SSIM entre les dues
    # source = cv2.imread('./frames/original' + str(1) + '.png')
    # target = cv2.imread('./frames/encoded' + str(1) + '.png')

    # print(metrikz.ssim(source, target))
