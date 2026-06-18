# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:12:08 2026

@author: Technikum-Admin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:26:49 2026

@author: admin
"""

#%% Imports and Globals
from time import sleep
from pymodbus.client import ModbusUdpClient as ModbusClient
from NOZZLE_CONTROL_LAYER import ClearAllOutputs
Out_On = False
Out_Off = True
IP_ADDRESS = "192.168.2.4"
IP_PORT = 502

#%% Connection
try:
    client = ModbusClient(IP_ADDRESS, port=IP_PORT)
    print("\n***************")
    print("* Modbus Test *")
    print("***************")
    print(f"Connecting to {IP_ADDRESS}:{IP_PORT} ...")
    if not client.connect():
        raise ConnectionError(f"Could not connect to {IP_ADDRESS}:{IP_PORT}")
    print(f"Connected to {IP_ADDRESS}:{IP_PORT} ...")
except Exception as e:
    print(f"Failed to open MODBUS connection: {e}")

#%% Testshot    
try:
    nozzle_address = 40
    SleepTime = 1
    print(f"Doing Testshot on nozzle #{nozzle_address}")
    client.write_coil(100 + nozzle_address, Out_On)
    sleep(SleepTime)
    
    client.write_coil(100 + nozzle_address , Out_Off)
    sleep(SleepTime)
except Exception as e:
    print(f"Failed to do Testshot: {e}")

finally:
    try:
        if client is not None:
            ClearAllOutputs(client)
            client.close()
            print("MODBUS connection closed.")
    except Exception as e:
        print(f"Error while closing MODBUS connection: {e}")
        