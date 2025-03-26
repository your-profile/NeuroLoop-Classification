import numpy as np
import pandas as pd
import os
import uuid
import datetime
import serial
from pathlib import Path
from collections import defaultdict
from threading import Thread, Lock
import pynput.keyboard

HEADER = ['T', 'PH-D24', 'PH-D23', 'PH-D4', 'PH-D3', 
               'PH-C24', 'PH-C23', 'PH-C4', 'PH-C3', 
               'PH-B22', 'PH-B21', 'PH-B2', 'PH-B1', 
               'PH-A22', 'PH-A21', 'PH-A2', 'PH-A1', 
               'AC-D24', 'AC-D23', 'AC-D4', 'AC-D3', 
               'AC-C24', 'AC-C23', 'AC-C4', 'AC-C3', 
               'AC-B22', 'AC-B21', 'AC-B2', 'AC-B1', 
               'AC-A22', 'AC-A21', 'AC-A2', 'AC-A1', 
               'A1-HbO', 'A1-Hb', 'A2-HbO', 'A2-Hb', 
               'B1-HbO', 'B1-Hb', 'B2-HbO', 'B2-Hb', 
               'C3-HbO', 'C3-Hb', 'C4-HbO', 'C4-Hb', 
               'D3-HbO', 'D3-Hb', 'D4-HbO', 'D4-Hb']

MARKERS = {
    'h': "HEADBAND_ON",
    'b': "TASK_BEGIN", #Starts at Begin Game page
    'm': "MOVEMENT", 
    'r': "REST_START", #Rests between trials
    'x': "MISTAKEN_MARKER", 
    'e': "TASK_END",
    'q': "QUIT"
}

lock = Lock()
marker_atom = 0
quit_atom = False

def on_press(key):    
    global marker_atom, quit_atom, lock, MARKERS

    try: 
        key = key.char
    except:
        print(f"special key pressed (ignoring).")
        return

    with lock:
        if key == 'q':
            quit_atom = True
            print(f"Q pressed. Quitting.")  
            return False

        marker_atom = MARKERS[key] if key in MARKERS else key
        print(f"{datetime.datetime.now().strftime('%m/%d %H:%M:%S')} Inserting Marker: {marker_atom}")  
    

def mainloop (ser, save_path): 
    global marker_atom, quit_atom, lock

    buffer = defaultdict(list)
   
    while True:
        
        with lock:
            if quit_atom:
                return

        #
        # reads until '\n'
        #
        byte_str = ser.read_until()
        
        #
        # commonly the first packet isn't read from the start so utf-8 decoding will fail
        # also, sometimes we get a bad packet without the full set of pairs
        #
        try:  
            packet = byte_str.decode('utf-8').strip()                                                
            packet = [x.split('=') for x in packet.split()] # [X=1, Z=2, ...] => [[X,1, [Z,2] ...]
            assert(len(packet) == 49)
            for key, value in packet:
                buffer[key].append(value)
            buffer['TIME'].append(datetime.datetime.now().timestamp() * 1000)
            
            with lock:
                buffer['MARKER'].append(marker_atom)
                marker_atom = 0
        
        except (UnicodeDecodeError, AssertionError):
            print(f"bad packet at time {datetime.datetime.now()}")
            continue
        
        #
        # save the valid packet
        # pandas to csv is slow af, so only use it when creating the new file. 
        #
        num_packets = len(buffer[HEADER[0]])
        if num_packets >= 5:

            if not os.path.exists(save_path):
                df = pd.DataFrame(buffer)                                  
                df.to_csv(save_path, mode='w', index=False)
            else:                    
                with open(save_path, 'a') as f:
                    for i in range(num_packets):
                        for j, key in enumerate(HEADER):
                            f.write(f'{buffer[key][i]},')
                        f.write(f'{buffer["TIME"][i]},{buffer["MARKER"][i]}\n')                        
            
            buffer = defaultdict(list)
            
if __name__ == '__main__':

    if not os.path.exists('./data'):
        os.mkdir('./data')

    # file save information    
    pid = str(uuid.uuid4()).split('-')[-1]
    date = datetime.datetime.now().strftime('%m_%d')
    file_name = '_'.join([pid, date])
    save_path = os.path.join('./data', file_name + '.fnirs')    
    
    ser = serial.Serial(port='COM3',
                        baudrate=115200,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS)
    
    fnirsT = Thread(target=mainloop, args=[ser, save_path])
    fnirsT.start()
   
    with pynput.keyboard.Listener(on_press=lambda x: on_press(x)) as listener:
        listener.join()   

    fnirsT.join()