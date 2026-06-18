# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:48:15 2026

@author: Technikum-Admin
"""
#%% Imports
from queue import Empty
import numpy as np
import socket
from time import sleep
try:
    from pymodbus.client import ModbusUdpClient as ModbusClient
except ImportError as IE:
    print(f"{IE}")


# Imports from SendValveData by Binder
from UDP_CONNECTION import send_udp_telegram

# Imports from Arduino Connection
from ARDUINO_CONNECTION import create_serial, send_arduino_telegram, close_serial

#%% UDP
def remap_activation(test_activation):
    """
    Remap a full activation array according to the base pattern.
    test_activation: 1D numpy array of length n_nozzles (0/1 or 0/255)
    Returns: remapped_activation array (same shape)
    """
    # Base pattern for 16 nozzles (1-based)
    base_pattern = np.array([8,7,6,5,4,3,2,1,16,15,14,13,12,11,10,9])
    remapped_activation = np.zeros_like(test_activation, dtype=np.uint8)
    
    # Convert 255→1 if needed
    # test_activation = np.where(test_activation == 255, 1, test_activation)
    
    # Find indices of active nozzles
    active_indices = np.where(test_activation == 1)[0]
    
    # Remap each active nozzle
    for idx in active_indices:
        block = idx // 16
        offset = idx % 16
        phys_nozzle = base_pattern[offset] + block*16 - 1  # zero-based
        remapped_activation[phys_nozzle] = 1
    
    return remapped_activation

def shift_inwards(arr):
    N = len(arr)
    LEFT_SHIFT = 4
    RIGHT_SHIFT = 5

    output = np.zeros_like(arr)

    active = np.where(arr == 1)[0]

    new_indices = []
    for i in active:
        if i < LEFT_SHIFT:                # left boundary
            ni = i + LEFT_SHIFT
        elif i >= N - RIGHT_SHIFT:        # right boundary
            ni = i - RIGHT_SHIFT
        else:
            ni = i                        # center stays
    
        if 0 <= ni < N:
            new_indices.append(ni)

    output[new_indices] = 1
    return output

def nozzle_control_UDP(NOZZLE_ACTIVATION_QUEUE, STOP_FLAG): 
    # Ziel-IP und Port
    TARGET_IP = "172.22.0.2" # IP For Local testing
    TARGET_PORT = 1234
    # IP ADRESSE DES WÜRFELS MUSS AUF
    # 172.22.0.9
    # MIT SUBNETZMASKE
    # 255.255.0.0
    # GESETZT SEIN
    
    # Socket erstellen
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #sock.bind(("", TARGET_PORT))
    
    data_header_bytes = bytes([0x80]) * 12
    
    while not STOP_FLAG.is_set():
        try:
            nozzle_activation = NOZZLE_ACTIVATION_QUEUE.get(timeout=0.05)
            nozzle_activation = (nozzle_activation == 255).astype(np.uint8)
            
            
            for line in nozzle_activation:
                #print(str(line).replace(" ",""))
                #line[31] = 0 # Deactivate a single nozzle
                shifted_line = shift_inwards(line)
                remapped_line = remap_activation(shifted_line)
                
                data_valves_bytes = np.packbits(remapped_line.astype(np.uint8), bitorder='big').tobytes()
                message = data_header_bytes + data_valves_bytes
                   
                send_udp_telegram(TARGET_IP, TARGET_PORT, message, sock)
                
                #time.sleep(0.001)  # Pausiert für das Intervall
               
        except KeyboardInterrupt:
            STOP_FLAG.set()
        except Empty:  
            pass
        except:
            STOP_FLAG.set()
        
    STOP_FLAG.set()
    sock.close()

def bits_to_bytes(bit_string: str):
    """Konvertiert eine Bitzeichenkette in einen Hex-String."""
    # Entferne Leerzeichen
    bit_string = bit_string.replace(" ", "")

    # Prüfen, ob Länge ein Vielfaches von 8 ist
    if len(bit_string) % 8 != 0:
        raise ValueError("Die Länge des Bitstrings muss ein Vielfaches von 8 sein.")

    # Umwandlung in eine Byte-Liste
    byte_array = [int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8)]

    # Umwandlung in einen Hex-String ohne Präfix
    hex_string = ''.join(format(byte, '02X') for byte in byte_array)

    return hex_string  # Rückgabe als Hex-String
    
#%% ARDUINO
def nozzle_control_ARDUINO(NOZZLE_ACTIVATION_QUEUE, STOP_FLAG):
    ser = None
    last_bits = None
    # Arduino serial settings
    ARDUINO_PORT = "/dev/ttyCH341USB0"
    ARDUINO_BAUDRATE = 500000
    N_NOZZLES = 8

    try:
        ser = create_serial(ARDUINO_PORT, ARDUINO_BAUDRATE)
        print(f"Arduino connected on {ARDUINO_PORT} @ {ARDUINO_BAUDRATE} baud")
    except Exception as e:
        print(f"Failed to open Arduino serial port: {e}")
        STOP_FLAG.set()
        return

    while not STOP_FLAG.is_set():
        try:
            nozzle_activation = NOZZLE_ACTIVATION_QUEUE.get(timeout=0.01)
            nozzle_activation = np.where(nozzle_activation == 255, 1, 0).astype(np.uint8)

            for line in nozzle_activation:
                if STOP_FLAG.is_set():
                    break

                line = np.asarray(line).flatten()
                if line.size < N_NOZZLES:
                    padded = np.zeros(N_NOZZLES, dtype=np.uint8)
                    padded[:line.size] = line
                    line = padded
                elif line.size > N_NOZZLES:
                    line = line[:N_NOZZLES]

                serial_bits = ''.join(str(int(v)) for v in line)

                if serial_bits != last_bits:
                    send_arduino_telegram(serial_bits, ser)
                    last_bits = serial_bits

        except Empty:
            pass
        except KeyboardInterrupt:
            STOP_FLAG.set()
        except Exception as e:
            print(f"Error in nozzle_control: {e}")
            STOP_FLAG.set()

    if ser is not None:
        close_serial(ser)
        
#%% MODBUS
# NEEDS pymodbus version 3.5.4!!!
# Globale Variablen
Out_On = False
Out_Off = True

def ClearAllOutputs(client):
    print("Setting all Outputs to 0..."),
    #GPIO -Outputs
    for addr in range(0, 24):
        client.write_coil(addr, Out_Off)

    #DOUT -Outputs
    for addr in range(100, 100 + 1*48):
        client.write_coil(addr, Out_Off)
        client.write_coil(addr+1*48, Out_Off)

    print("done.")
        
def nozzle_control_MODBUS(NOZZLE_ACTIVATION_QUEUE, STOP_FLAG):
    # --------------------------------------
    # SETUP
    IP_ADDRESS = "192.168.2.4"
    IP_PORT = 502
    QUEUE_TIMEOUT = 0.01    
    DO_TESTSHOTS = False
    DO_TESTSHOT = True
    # --------------------------------------
    # ESTABLISH CONNECTION
    try:
        client = ModbusClient(IP_ADDRESS, port=IP_PORT)

        print("\n***************")
        print("* Modbus Test *")
        print("***************")
        print(f"Connecting to {IP_ADDRESS}:{IP_PORT} ...")
        if not client.connect():
            raise ConnectionError(f"Could not connect to {IP_ADDRESS}:{IP_PORT}")

        ClearAllOutputs(client)
        
        #TestShots
        if DO_TESTSHOT:
            nozzle_address = 40
            SleepTime = 1
            print(f"Doing Testshot on nozze #{nozzle_address}")
            client.write_coil(100 + nozzle_address, Out_On)
            sleep(SleepTime)
            
            client.write_coil(100 + nozzle_address , Out_Off)
            sleep(SleepTime)
        
        #TestShots
        if DO_TESTSHOTS:
            SleepTime = 0.05
            for nozzle_address in range(80):
                print(f"Doing Testshot on nozze #{nozzle_address}")
                client.write_coil(100 + nozzle_address, Out_On)
                sleep(SleepTime)
                
                client.write_coil(100 + nozzle_address , Out_Off)
                sleep(SleepTime)

    except Exception as e:
        print(f"Failed to open MODBUS connection: {e}")
        STOP_FLAG.set()
        return


    #--------------------------------------
    # ACTIVATION LOOP
    try:
        while not STOP_FLAG.is_set():
            try:
                nozzle_activation = NOZZLE_ACTIVATION_QUEUE.get(timeout=QUEUE_TIMEOUT)
        
                # keep only the newest mask
                while True:
                    try:
                        nozzle_activation = NOZZLE_ACTIVATION_QUEUE.get_nowait()
                    except Empty:
                        break
        
                nozzle_activation = np.asarray(nozzle_activation)
        
                # collapse 12 rows into one nozzle state
                combined = np.any(nozzle_activation == 255, axis=0)
                mapped = np.where(combined, Out_On, Out_Off)
        
                result = client.write_coils(100, [bool(x) for x in mapped])
                if result.isError():
                    print("write_coils failed")
        
            except Empty:
                pass
            except KeyboardInterrupt:
                STOP_FLAG.set()
            except Exception as e:
                print(f"Error in nozzle_control: {e}")
                STOP_FLAG.set()
    finally:
        try:
            if client is not None:
                ClearAllOutputs(client)
                client.close()
                print("MODBUS connection closed.")
        except Exception as e:
            print(f"Error while closing MODBUS connection: {e}")
    
    
#%% SIMULATED
def nozzle_control_SIMULATED(NOZZLE_ACTIVATION_QUEUE, STOP_FLAG):
    from queue import Empty
    import numpy as np
    import pygame

    N_NOZZLES = 10

    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((620, 180))
    pygame.display.set_caption("Simulated Ejection Nozzles")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Segoe UI", 20)
    small_font = pygame.font.SysFont("Segoe UI", 15)

    active = np.zeros(N_NOZZLES, dtype=bool)

    while not STOP_FLAG.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                STOP_FLAG.set()

        try:
            latest = None
            while True:
                latest = NOZZLE_ACTIVATION_QUEUE.get_nowait()
        except Empty:
            pass

        if latest is not None:
            arr = np.asarray(latest)

            if arr.ndim == 2:
                active = np.any(arr == 255, axis=0)
            else:
                active = arr == 255

            active = active[:N_NOZZLES]

        screen.fill((18, 20, 25))

        title = font.render("Simulated Ejection Nozzles", True, (235, 238, 245))
        screen.blit(title, (20, 15))

        for i in range(N_NOZZLES):
            x = 35 + i * 58
            y = 75

            color = (220, 68, 55) if i < len(active) and active[i] else (80, 85, 95)

            pygame.draw.circle(screen, color, (x, y), 18)
            pygame.draw.circle(screen, (180, 185, 195), (x, y), 18, 2)

            label = small_font.render(str(i), True, (235, 238, 245))
            screen.blit(label, label.get_rect(center=(x, y + 38)))

        active_ids = [str(i) for i, v in enumerate(active) if v]
        status = "Active: " + (", ".join(active_ids) if active_ids else "none")
        status_text = small_font.render(status, True, (174, 181, 196))
        screen.blit(status_text, (20, 140))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()    
    
    
    
    
    
    
    
    