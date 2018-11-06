import time
from functools import reduce

class Timer:
    def __init__(self, nsteps = 1):
        self.nsteps = nsteps
        self.prevStep = 1
        self.start = time.time()
        self.prev = self.start
        self.laps = []

    def elapsed(self, step = 0):
        # get eleapsed        
        end = time.time()
        prev = self.prev
        self.prev = end
        lap = end - prev
        
        self.laps.append(lap)
        self.prevStep = step

        return lap

    def left(self):
        avLap = reduce((lambda x, y: x + y), self.laps, 0)/self.prevStep
        remainininTime = avLap * (self.nsteps - self.prevStep)
        return time.strftime("%H:%M:%S",  time.gmtime(remainininTime))
