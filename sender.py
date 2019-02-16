import sys
import time
from networktables import NetworkTables

# To see messages from networktables, you must setup logging
import logging

logging.basicConfig(level=logging.DEBUG)

NetworkTables.initialize(server='10.6.12.2')

sd = NetworkTables.getTable("SmartDashboard")

i = 0
while True:
    sd.putNumber("dsTime", i)
    print(i)
    i += 1
    time.sleep(5)