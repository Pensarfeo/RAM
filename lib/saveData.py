import csv

class DataSaver:
    def __init__(self, *titles, filename='outputData'):
        self.titles = titles
        self.filename = filename + '.csv'
        with open(filename + '.csv', 'w') as file:
            writer = file.write(','.join(self.titles))

    def add(self, data, *rest):
        # import pdb; pdb.set_trace()
        with open(self.filename, 'a') as file:
            file.write('\n')
            row = []
            for title in self.titles:
                row.append( str(data[title]) )
            file.write(','.join(row))
