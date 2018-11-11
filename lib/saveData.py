import csv
import numpy
import os

class DataSaver:
    def __init__(self, *titles, filename='outputData', divider =','):
        self.titles = titles
        self.divider = divider
        self.filename = filename + '.csv'

        if not os.path.isfile(self.filename):
            os.makedirs(os.path.dirname(self.filename))
        
        with open(filename + '.csv', 'w') as file:
            file.write(self.divider.join(self.titles))

    def add(self, data, *rest):
        # import pdb; pdb.set_trace()
        with open(self.filename, 'a') as file:
            file.write('\n')
            row = []

            for title in self.titles:
                value = data[title]
                if isinstance(value, int) or isinstance(value, float) or isinstance(value, numpy.int64):
                    strValue = str(value)
                else:
                    strValue =  ' '.join(str(x) for x in value)
                
                row.append( strValue )
            file.write(self.divider.join(row))
