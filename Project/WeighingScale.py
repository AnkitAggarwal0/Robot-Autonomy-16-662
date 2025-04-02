import serial
import re

class WeighingScale: 
    def __init__(self, port='/dev/ttyUSB0'):
        self.serial_port = serial.Serial(port, 9600, timeout=1)

    def read_weight(self):
        self.serial_port.reset_input_buffer()
        serial_data = self.serial_port.readline().decode('utf-8').strip() # readline blocks by default
        if (serial_data != ''):
            weight = float(re.findall(r'-?\d+\.?\d*', serial_data)[0])
            print(f"Weight: {weight} g")
            return weight
        # else:
        #     print("No data recieved")
        #     return -1
    
    def weight_averaged(self, N = 4):
        weights = []
        for i in range(N):
            weights.append(self.read_weight())
        return sum(weights) / len(weights)

if __name__ == "__main__":
    scale = WeighingScale()
    while(1):
        scale.read_weight()
        print("\n")